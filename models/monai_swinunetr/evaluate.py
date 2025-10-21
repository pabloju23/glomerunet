import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # specify GPU device
import torch
from glob import glob
from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Compose, Activations, AsDiscrete, LoadImage
from monai.networks.nets import SwinUNETR, DynUNet
import numpy as np
from tqdm import tqdm
from pathlib import Path

# -------- Config --------
model_type = "nnunet"  # "swinunetr" or "nnunet"
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
dice_metrics = [DiceMetric(include_background=False, reduction="mean", get_not_nans=False) for _ in range(num_classes)]
iou_metrics = [MeanIoU(include_background=False, reduction="mean", get_not_nans=False) for _ in range(num_classes)]

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
    
    # Reset all metrics
    for metric in dice_metrics + iou_metrics:
        metric.reset()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {stain_name}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            # Apply post-processing (softmax + argmax) to get class predictions
            outputs = [post_trans(i) for i in decollate_batch(outputs)]

            # Compute metrics for each class separately
            for class_idx in range(num_classes):
                # Create binary masks for current class
                pred_binary = [(pred == class_idx).float().unsqueeze(0) for pred in outputs]
                label_binary = [(labels[i] == class_idx).float().unsqueeze(0) for i in range(len(outputs))]
                
                dice_metrics[class_idx](y_pred=pred_binary, y=label_binary)
                iou_metrics[class_idx](y_pred=pred_binary, y=label_binary)

            # Guardar predicciones
            for i, pred in enumerate(outputs):
                # Extraer nombre del archivo desde el MetaTensor
                fname = Path(batch["image"][i].meta["filename_or_obj"]).name
                torch.save(pred.cpu(), os.path.join(pred_dir, f"{stain_name}_{fname}"))

    # Aggregate per-class metrics
    dice_per_class = np.array([dice_metrics[i].aggregate().cpu().item() for i in range(num_classes)])
    iou_per_class = np.array([iou_metrics[i].aggregate().cpu().item() for i in range(num_classes)])
    
    # Reset metrics
    for metric in dice_metrics + iou_metrics:
        metric.reset()
    
    # Calculate mean across all classes (including background)
    dice_mean = np.mean(dice_per_class)
    iou_mean = np.mean(iou_per_class)
    
    # Alternative: mean excluding background only
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

import cv2

# ----------------------------
# Helper: convertir máscara a RGB
# ----------------------------
def mask_to_rgb(mask_tensor):
    """Convierte una máscara (tensor HxW o 1xHxW) a RGB:
       fondo negro, clase 1 verde, clase 2 roja."""
    mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask == 1] = (0, 255, 0)   # clase 1 → verde
    rgb[mask == 2] = (255, 0, 0)   # clase 2 → rojo
    return rgb

# ----------------------------
# Helper: guardar pares imagen–predicción
# ----------------------------
def save_predictions_monai(model, loader, stain_name):
    os.makedirs(stain_name, exist_ok=True)
    print(f"\nSaving predictions for {stain_name}...")

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Saving {stain_name}")):
            imgs = batch["image"].to(device)
            preds = model(imgs)
            preds = [post_trans(i) for i in decollate_batch(preds)]  # softmax+argmax

            for i, pred in enumerate(preds):
                # Get original filename
                fname = Path(batch["image"][i].meta["filename_or_obj"]).stem
                original_path = batch["image"][i].meta["filename_or_obj"]
                
                # Load original image directly from disk to avoid any transform artifacts
                img_original = cv2.imread(str(original_path))
                
                # Convert prediction to RGB (assuming pred is HxW or 1xHxW)
                pred_rgb = mask_to_rgb(pred)
                
                # Ensure prediction matches original image dimensions
                if img_original.shape[:2] != pred_rgb.shape[:2]:
                    pred_rgb = cv2.resize(pred_rgb, 
                                         (img_original.shape[1], img_original.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)

                # Save both
                img_path = os.path.join(stain_name, f"{fname}_img.png")
                pred_path = os.path.join(stain_name, f"{fname}_pred.png")

                cv2.imwrite(img_path, img_original)  # Already in BGR format
                cv2.imwrite(pred_path, cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

    print(f"Predictions saved in folder: {stain_name}/")

# ----------------------------
# Guardar imágenes y predicciones por tinción
# ----------------------------
for stain_name in ["HE", "PAS", "PM"]:
    if stain_name in loaders:
        save_predictions_monai(model, loaders[stain_name], stain_name)
