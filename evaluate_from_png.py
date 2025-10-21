import os
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import random

# -----------------------------
# ----- CONFIG ----------------
# -----------------------------
NUM_CLASSES = 3
GT_FOLDER = "/scratch.local2/juanp/glomeruli/dataset/nnunet_data/nnUnet_raw/Dataset001_glomeruli/labelsTs"
PRED_FOLDER = "/scratch.local2/juanp/glomeruli/dataset/nnunet_data/predictions/Dataset002_glomeruli_D1_pp"
IMG_FOLDER = "/scratch.local2/juanp/glomeruli/dataset/nnunet_data/nnUnet_raw/Dataset001_glomeruli/imagesTs"
STAIN_LIST = ["HE", "PAS", "PM"]
IMG_SUFFIX = ".png"

# -----------------------------
# ----- FUNCIONES -------------
# -----------------------------
def load_masks(folder, stain_prefix):
    files = [f for f in os.listdir(folder) if f.startswith(stain_prefix) and f.endswith(IMG_SUFFIX)]
    files.sort()
    masks = [imread(os.path.join(folder, f)) for f in files]
    masks = np.array(masks)
    
    # Convert to one-hot considering masks array shape (N, H, W) with 3 classes 0,1,2
    one_hot = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], NUM_CLASSES), dtype=np.uint8)
    for i in range(NUM_CLASSES):
        one_hot[..., i] = (masks == i)
    return one_hot, files

def dice_score_per_class(gt, pred, eps=1e-7):
    scores = []
    for c in range(NUM_CLASSES):
        gt_c = gt[..., c]
        pred_c = pred[..., c]
        inter = np.sum(gt_c * pred_c)
        denom = np.sum(gt_c) + np.sum(pred_c) + eps
        scores.append(2 * inter / denom)
    return np.array(scores)

def iou_per_class(gt, pred, eps=1e-7):
    scores = []
    for c in range(NUM_CLASSES):
        gt_c = gt[..., c]
        pred_c = pred[..., c]
        inter = np.sum(gt_c * pred_c)
        union = np.sum(gt_c) + np.sum(pred_c) - inter + eps
        scores.append(inter / union)
    return np.array(scores)

def load_images_for_plot(img_folder, gt_files):
    images = []
    for f in gt_files:
        # Obtener los 4 caracteres _XXX para emparejar
        suffix = f.split(".")[0]  # extrae split_XXX de split_XXX.png
        # Buscar archivo correspondiente en IMG_FOLDER
        img_candidates = [fn for fn in os.listdir(img_folder) if fn.endswith(f"{suffix}_0000.png")]
        if len(img_candidates) != 1:
            raise RuntimeError(f"No se encontró imagen original para {f} con nombre esperado: _{suffix}_0000.png")
        img = imread(os.path.join(img_folder, img_candidates[0]))
        images.append(img)
    return np.array(images)

def check_masks_integrity(masks, mask_type="GT"):
    print(f"\nComprobando {mask_type}...")
    print(f"Shape: {masks.shape}")
    print(f"Valor mínimo: {masks.min()}, valor máximo: {masks.max()}")
    
    # Si es one-hot, se puede comprobar que la suma por pixel sea 1
    if masks.ndim == 4:  # one-hot
        sum_per_pixel = masks.sum(axis=-1)
        if not np.all((sum_per_pixel == 0) | (sum_per_pixel == 1)):
            print("Advertencia: Hay pixels con sumas != 0 ni 1 en one-hot encoding")
    
    # Comprobar cuántas clases hay presentes
    unique_vals = np.unique(masks)
    print(f"Clases presentes: {unique_vals}")

def plot_examples(images, gt_masks, pred_masks, n=4):
    idx = random.sample(range(len(images)), n)
    for i in idx:
        img = images[i]
        gt = np.argmax(gt_masks[i], axis=-1)
        pred = np.argmax(pred_masks[i], axis=-1)
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow(img)
        axs[0].imshow(gt, alpha=0.5, cmap='jet')
        axs[0].set_title("GT")
        axs[1].imshow(img)
        axs[1].imshow(pred, alpha=0.5, cmap='jet')
        axs[1].set_title("Pred")
        axs[2].imshow(img)
        axs[2].imshow(gt, alpha=0.3, cmap='jet')
        axs[2].imshow(pred, alpha=0.3, cmap='cool')
        axs[2].set_title("GT + Pred")
        for ax in axs:
            ax.axis('off')
        plt.show()

def plot_examples_with_pred_classes(images, gt_masks, pred_masks, target_classes=[1,2], n=4):
    """
    Plotea n imágenes donde la predicción tenga al menos una de las clases target_classes activas.
    """
    # Encontrar índices donde pred_masks tiene al menos un pixel de las clases target
    valid_idx = []
    for i in range(len(pred_masks)):
        pred = pred_masks[i]
        if any(pred[...,c].sum() > 0 for c in target_classes):
            valid_idx.append(i)
    
    if len(valid_idx) == 0:
        print("No hay imágenes con las clases predichas especificadas.")
        return
    
    # Seleccionar hasta n índices aleatorios de los válidos
    idx = random.sample(valid_idx, min(n, len(valid_idx)))
    
    for i in idx:
        img = images[i]
        gt = np.argmax(gt_masks[i], axis=-1)
        pred = np.argmax(pred_masks[i], axis=-1)
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow(img)
        axs[0].imshow(gt, alpha=0.5, cmap='jet')
        axs[0].set_title("GT")
        axs[1].imshow(img)
        axs[1].imshow(pred, alpha=0.5, cmap='jet')
        axs[1].set_title("Pred")
        axs[2].imshow(img)
        axs[2].imshow(gt, alpha=0.3, cmap='jet')
        axs[2].imshow(pred, alpha=0.3, cmap='cool')
        axs[2].set_title("GT + Pred")
        for ax in axs:
            ax.axis('off')
        plt.show()


# -----------------------------
# ----- CARGAR DATOS ----------
# -----------------------------
results = {}
all_gt_list = []
all_pred_list = []

for stain in STAIN_LIST:
    print(f"\nProcesando tinción {stain}...")
    gt_masks, gt_files = load_masks(GT_FOLDER, stain)
    pred_masks, pred_files = load_masks(PRED_FOLDER, stain)
    check_masks_integrity(gt_masks, "GT")
    check_masks_integrity(pred_masks, "Pred")
    assert gt_files == pred_files, "Los nombres de GT y pred no coinciden"

    # Guardamos para evaluación global
    all_gt_list.append(gt_masks)
    all_pred_list.append(pred_masks)

    # Métricas por tinción
    dice_scores = dice_score_per_class(gt_masks, pred_masks)
    iou_scores = iou_per_class(gt_masks, pred_masks)

    results[stain] = {"dice": dice_scores, "iou": iou_scores}
    print(f"Dice por clase: {dice_scores}")
    print(f"IoU por clase: {iou_scores}")

    # Cargar imágenes originales para plot
    images = load_images_for_plot(IMG_FOLDER, gt_files)
    plot_examples_with_pred_classes(images, gt_masks, pred_masks, target_classes=[1,2], n=4)

# -----------------------------
# ----- EVALUACIÓN GLOBAL -----
# -----------------------------
all_gt = np.concatenate(all_gt_list, axis=0)
all_pred = np.concatenate(all_pred_list, axis=0)
global_dice = dice_score_per_class(all_gt, all_pred)
global_iou = iou_per_class(all_gt, all_pred)

results["ALL"] = {"dice": global_dice, "iou": global_iou}
print("\nResultados globales:")
print(f"Dice: {global_dice}")
print(f"IoU: {global_iou}")
