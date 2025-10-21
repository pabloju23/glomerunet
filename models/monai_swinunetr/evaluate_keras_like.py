import os
import torch
from glob import glob
from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Compose, Activations, AsDiscrete, LoadImage
from monai.networks.nets import SwinUNETR, DynUNet
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F

class OneHotDiceTorch:
    def __init__(self, num_classes, ignore_class=None, device='cpu'):
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.device = device
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_classes, device=self.device)
        self.total_true = torch.zeros(self.num_classes, device=self.device)
        self.total_pred = torch.zeros(self.num_classes, device=self.device)

    def update(self, y_true, y_pred):
        """
        y_true: [B,H,W] (sparse, MetaTensor o LongTensor)
        y_pred: [B,H,W] (sparse, MetaTensor o LongTensor)
        """
        # Convertir a LongTensor nativo
        y_true = y_true.detach().to(torch.long).contiguous().view(-1)
        y_pred = y_pred.detach().to(torch.long).contiguous().view(-1)

        if self.ignore_class is not None:
            mask = y_true != self.ignore_class
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        y_true_onehot = F.one_hot(y_true, num_classes=self.num_classes).float()
        y_pred_onehot = F.one_hot(y_pred, num_classes=self.num_classes).float()

        self.tp += (y_true_onehot * y_pred_onehot).sum(dim=0)
        self.total_true += y_true_onehot.sum(dim=0)
        self.total_pred += y_pred_onehot.sum(dim=0)

    def compute_dice(self):
        present = (self.total_true + self.total_pred) > 0
        tp = self.tp[present]
        t = self.total_true[present]
        p = self.total_pred[present]
        if tp.numel() == 0:
            return torch.tensor(1.0, device=self.device)
        return (2 * tp / (t + p + 1e-8)).mean()

    def compute_iou(self):
        present = (self.total_true + self.total_pred) > 0
        tp = self.tp[present]
        t = self.total_true[present]
        p = self.total_pred[present]
        if tp.numel() == 0:
            return torch.tensor(1.0, device=self.device)
        return (tp / (t + p - tp + 1e-8)).mean()

    def compute_per_class(self):
        dice_scores = 2 * self.tp / (self.total_true + self.total_pred + 1e-8)
        iou_scores = self.tp / (self.total_true + self.total_pred - self.tp + 1e-8)
        return dice_scores.cpu().numpy(), iou_scores.cpu().numpy()


# -------- Config --------
model_type = "swinunetr"  # "swinunetr" o "nnunet"
dataset_dir = "/scratch.local2/juanp/glomeruli/dataset/nnunet_data/nnUnet_raw/Dataset001_glomeruli"
checkpoint_dir = f"/scratch.local2/juanp/glomeruli/models/monai_{model_type}/checkpoints"
pred_dir = f"predictions_{model_type}"
os.makedirs(pred_dir, exist_ok=True)
batch_size = 2
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names for better readability (adjust according to your dataset)
class_names = ["Background", "Class 1", "Class 2"]  # All classes including background

# Create separate metrics for each class
dice_metrics = OneHotDiceTorch(num_classes=num_classes, device=device)
iou_metrics = OneHotDiceTorch(num_classes=num_classes, device=device)

# -------- Transforms --------
from transforms import get_val_transforms

val_transforms = get_val_transforms()

post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])

def create_loader(stain_filter="*"):
    images = sorted(glob(os.path.join(dataset_dir, "imagesTs", f"{stain_filter}.png")))
    labels = sorted(glob(os.path.join(dataset_dir, "labelsTs", f"{stain_filter}.png")))
    files = [{"image": img, "label": seg} for img, seg in zip(images, labels)]
    ds = Dataset(data=files, transform=val_transforms)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=torch.cuda.is_available())
    return loader

# Crear dataloaders por tinción y general
loaders = {
    "all": create_loader("*"),
    "HE": create_loader("HE_*"),
    "PAS": create_loader("PAS_*"),
    "PM": create_loader("PM_*"),
}

# -------- Modelo --------
if model_type.lower() == "swinunetr":
    model = SwinUNETR(
        in_channels=3,
        out_channels=3,
        feature_size=24,       
        use_checkpoint=True,   
        spatial_dims=2,
    ).to(device)
elif model_type.lower() == "nnunet":
    from monai.networks.nets import DynUNet
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
else:
    raise ValueError("Modelo desconocido")

# Cargar pesos finetune
ckpt_path = os.path.join(checkpoint_dir, f"finetune_best_metric_model.pth")
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.to(device)
model.eval()

# ------------------ Evaluación ------------------
def evaluate(model, loader, stain_name):
    model.eval()
    
    # Reset metrics
    dice_metrics.reset()
    iou_metrics.reset()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {stain_name}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            outputs = torch.argmax(F.softmax(outputs, dim=1), dim=1)  # [B,H,W]

            # Update global metrics
            dice_metrics.update(labels, outputs)
            iou_metrics.update(labels, outputs)

            # Guardar predicciones
            for i in range(outputs.shape[0]):
                fname = Path(batch["image"][i].meta["filename_or_obj"]).name
                torch.save(outputs[i].cpu(), os.path.join(pred_dir, f"{stain_name}_{fname}"))

    # Compute final metrics
    dice_mean = dice_metrics.compute_dice().item()
    iou_mean = iou_metrics.compute_iou().item()
    dice_per_class, iou_per_class = dice_metrics.compute_per_class()

    # Mean sin background
    dice_mean_nobg = np.mean(dice_per_class[1:])
    iou_mean_nobg = np.mean(iou_per_class[1:])

    return dice_mean, iou_mean, dice_mean_nobg, iou_mean_nobg, dice_per_class, iou_per_class

def print_metrics(stain_name, dice_mean, iou_mean, dice_mean_nobg, iou_mean_nobg, dice_per_class, iou_per_class):
    """Pretty print metrics with per-class breakdown"""
    print(f"\n{'='*70}")
    print(f"Stain: {stain_name.upper()}")
    print(f"{'='*70}")
    print(f"Mean DSC (all):  {dice_mean:.4f} | Mean IoU (all):  {iou_mean:.4f}")
    print(f"Mean DSC (no-bg): {dice_mean_nobg:.4f} | Mean IoU (no-bg): {iou_mean_nobg:.4f}")
    print(f"{'-'*70}")
    print("Per-class metrics:")
    
    for i in range(len(dice_per_class)):
        class_name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"  {class_name:15s} (Class {i}) -> DSC: {dice_per_class[i]:.4f} | IoU: {iou_per_class[i]:.4f}")
    print(f"{'='*70}")

# -------- Evaluación por tinción --------
results = {}
print('Evaluating with model:', model_type)
for stain_name, loader in loaders.items():
    dice_mean, iou_mean, dice_mean_nobg, iou_mean_nobg, dice_per_class, iou_per_class = evaluate(model, loader, stain_name)
    results[stain_name] = {
        'dice_mean': dice_mean,
        'iou_mean': iou_mean,
        'dice_mean_nobg': dice_mean_nobg,
        'iou_mean_nobg': iou_mean_nobg,
        'dice_per_class': dice_per_class,
        'iou_per_class': iou_per_class
    }
    print_metrics(stain_name, dice_mean, iou_mean, dice_mean_nobg, iou_mean_nobg, dice_per_class, iou_per_class)

# -------- Resumen final --------
print("\n" + "="*70)
print("SUMMARY - Mean Metrics Across All Stains")
print("="*70)
for stain_name, metrics in results.items():
    print(f"{stain_name:10s} -> DSC: {metrics['dice_mean']:.4f} | IoU: {metrics['iou_mean']:.4f} | DSC(no-bg): {metrics['dice_mean_nobg']:.4f}")
print("="*70)

print("\nEvaluación completada. Resultados guardados en el directorio de predicciones.")