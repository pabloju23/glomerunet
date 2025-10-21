import logging
import os
# Use only the gpu 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from glob import glob

import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ArrayDataset, decollate_batch, DataLoader
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from monai.visualize import plot_2d_or_3d_image
from monai.networks.nets import DynUNet

# Local imports
from transforms import get_train_transforms, get_val_transforms
from loss import FocalTverskyLoss


checkpoint_dir = "/scratch.local2/juanp/glomeruli/models/monai_nnunet/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)  # crea la carpeta si no existe
epochs = 30
batch_size = 22


# Define our own Tversky loss
class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs and targets are assumed to be of shape (B, 1, H, W) for binary segmentation
        inputs = torch.sigmoid(inputs)  # Apply sigmoid if not already applied
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        true_pos = (inputs * targets).sum()
        false_neg = ((1 - inputs) * targets).sum()
        false_pos = (inputs * (1 - targets)).sum()

        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        return 1 - tversky_index  # Return Tversky loss


def main(dataset_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    images_folder= os.path.join(dataset_dir, 'imagesTr')
    labels_folder= os.path.join(dataset_dir, 'labelsTr')

    # Buscar imágenes y máscaras del split D2 (train)
    train_images = sorted(glob(os.path.join(images_folder, "D2_*_0000.png")))
    train_labels = sorted(glob(os.path.join(labels_folder, "D2_*.png")))

    # Buscar imágenes y máscaras del split val (validación)
    val_images = sorted(glob(os.path.join(images_folder, "val_*_0000.png")))
    val_labels = sorted(glob(os.path.join(labels_folder, "val_*.png")))

    print(f"Found {len(train_images)} training images and {len(train_labels)} labels")
    print(f"Found {len(val_images)} validation images and {len(val_labels)} labels")

    # define transforms
    # train_imtrans = Compose(
    #     [
    #         LoadImage(image_only=True, ensure_channel_first=True),
    #         ScaleIntensity(),
    #         # RandSpatialCrop((96, 96), random_size=False),
    #         # RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    #     ]
    # )
    # train_segtrans = Compose([
    #     LoadImage(image_only=True, ensure_channel_first=True),
    #     EnsureType(dtype=torch.long),
    #     # RandSpatialCrop((96, 96), random_size=False),
    #     # RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    # ])

    # val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    # val_segtrans = Compose([
    #     LoadImage(image_only=True, ensure_channel_first=True),
    #     EnsureType(dtype=torch.long)
    # ])
    
    train_imtrans = get_train_transforms()
    train_segtrans = get_train_transforms()
    val_imtrans = get_val_transforms()
    val_segtrans = get_val_transforms()

    # datasets y dataloaders
    train_ds = ArrayDataset(train_images, train_imtrans, train_labels, train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())

    val_ds = ArrayDataset(val_images, val_imtrans, val_labels, val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

    # métrica y post-procesamiento
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, num_classes=3)
    iou_metric = monai.metrics.MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_nobg = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, num_classes=3)
    post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])

    # modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        kernel_size=[3, 3, 3, 3, 3],            # kernels para cada bloque
        strides=[1, 2, 2, 2, 2],                 # stride en cada bloque
        upsample_kernel_size=[2, 2, 2, 2],       # para la transposed conv en upsampling
        filters=[32, 64, 128, 256, 320],        # número de filtros en cada bloque
        res_block=True,                          # bloques residuales
        deep_supervision=False,                  # desactivamos deep supervision
        norm_name=("INSTANCE", {"affine": True}),
        act_name=("leakyrelu", {"inplace": True}),
    ).to(device)
    # Define combined 0.35*focal loss(gamma=2) + 0.65*tversky loss(alpha=0.7, beta=0.3)
    loss_function = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=2.0, lambda_focal=0.35, lambda_tversky=0.65, to_onehot_y=True, softmax=True, include_background=True, reduction='mean', smooth_nr=1e-5, smooth_dr=1e-5)

    # Alternatively, you can use DiceCELoss
    # loss_function = monai.losses.DiceCELoss(
    #     include_background=True,   # mantener el fondo como clase 0
    #     to_onehot_y=True,          # convierte tus labels enteros (0,1,2) a one-hot
    #     softmax=True,              # aplica softmax al output del modelo
    #     lambda_dice=1.0,           # peso del componente Dice
    #     lambda_ce=1.0,             # peso del componente CrossEntropy
    #     smooth_nr=1e-5,            # para evitar divisiones por cero
    #     smooth_dr=1e-5,
    #     reduction='mean',          # media sobre el batch
    # )
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=7, min_lr=1e-5)

    # entrenamiento
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter()
    for epoch in range(epochs):
        print("-" * 40)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f}")

        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            writer.add_scalar("train_loss", loss.item(), epoch * len(train_loader) + step)

        epoch_loss /= step
        print(f"Train loss: {epoch_loss:.4f}")

        # -------------------- VALIDACIÓN (siempre visible) --------------------
        model.eval()
        dice_metric.reset()
        iou_metric.reset()
        dice_metric_nobg.reset()

        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = model(val_images)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

                dice_metric(y_pred=val_outputs, y=val_labels)
                iou_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_nobg(y_pred=val_outputs, y=val_labels)

        mean_dice = float(dice_metric.aggregate().item())
        mean_iou = float(iou_metric.aggregate().item())
        mean_dice_nobg = float(dice_metric_nobg.aggregate().item())
        dice_metric.reset(); iou_metric.reset(); dice_metric_nobg.reset()

        # logging y tensorboard
        print(f"Validation -> Dice: {mean_dice:.4f}, IoU: {mean_iou:.4f}, Dice_noBG: {mean_dice_nobg:.4f}")
        writer.add_scalar("val_mean_dice", mean_dice, epoch + 1)
        writer.add_scalar("val_mean_iou", mean_iou, epoch + 1)
        writer.add_scalar("val_mean_dice_nobg", mean_dice_nobg, epoch + 1)
        writer.add_scalar("learning_rate", current_lr, epoch + 1)

        # actualiza el scheduler con la métrica de validación
        scheduler.step(mean_dice_nobg)

        # guardar el mejor modelo
        if mean_dice_nobg > best_metric:
            best_metric = mean_dice_nobg
            best_metric_epoch = epoch + 1
            ckpt_path = os.path.join(checkpoint_dir, f"pretrain_best_metric_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print("✅ Saved new best model (Dice_noBG improved)")

        print(f"Best Dice_noBG so far: {best_metric:.4f} at epoch {best_metric_epoch}")
        
        plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
        plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
        plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"training completed, best_metric: {best_metric:.4f} at epoch {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    dataset_dir = "/scratch.local2/juanp/glomeruli/dataset/nnunet_data/nnUnet_raw/Dataset001_glomeruli"  
    main(dataset_dir)
