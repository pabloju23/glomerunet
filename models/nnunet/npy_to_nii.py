import os
import numpy as np
from skimage.io import imsave
from nnunetv2.paths import nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
import json

# -------------------------------
# Configuración de tu dataset
# -------------------------------
dataset_name = 'Dataset_Glomeruli'

# Rutas a los npy de cada split
splits = {
    "D1_train": {"images": {'/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_2_img.npy', 
                           '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_4_img.npy',
                           '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_5_img.npy'},
                 "masks":  {'/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_2_mask.npy',
                           '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_4_mask.npy',
                           '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_5_mask.npy'}},
    "D2_train": {"images": '/scratch.local/juanp/glomeruli/dataset/trainD2_paper/train_imgD2_v2_positive.npy',
                 "masks":  '/scratch.local/juanp/glomeruli/dataset/trainD2_paper/train_maskD2_v2_positive.npy'},
    "val":      {"images": '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_1_img.npy',
                 "masks":  '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_1_mask.npy'},
    "test_he":  {"images": '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_img_he.npy',
                 "masks":  '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_mask_he.npy'},
    "test_pm":  {"images": '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_img_pm.npy',
                 "masks":  '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_mask_pm.npy'},
    "test_pas": {"images": '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_img_pas.npy',
                 "masks":  '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_mask_pas.npy'}
}

# Carpeta base del dataset
output_base = "/scratch.local/juanp/glomeruli/dataset/nnunet_data"
imagesTr = os.path.join(output_base, "imagesTr")
labelsTr = os.path.join(output_base, "labelsTr")
imagesTs = os.path.join(output_base, "imagesTs")
labelsTs = os.path.join(output_base, "labelsTs")

for folder in [imagesTr, labelsTr, imagesTs, labelsTs]:
    maybe_mkdir_p(folder)

# -------------------------------
# Funciones auxiliares
# -------------------------------
def save_case(case_id, image, mask, out_img_folder, out_mask_folder):
    # Imagen RGB
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    img_filename = os.path.join(out_img_folder, f"{case_id}_0000.png")
    imsave(img_filename, img_uint8, check_contrast=False)
    
    # Máscara (si existe)
    if mask is not None:
        # Convertir one-hot a map entero
        mask_int = np.argmax(mask, axis=-1).astype(np.uint8)
        mask_filename = os.path.join(out_mask_folder, f"{case_id}.png")
        imsave(mask_filename, mask_int, check_contrast=False)

# -------------------------------
# Procesar todos los splits
# -------------------------------
case_counters = {}  # contador independiente por prefijo
num_training_cases = 0

for split_name, paths in splits.items():
    images_path = paths["images"]
    masks_path = paths["masks"]

    # Load npy considering the possibility of being various npy concatenated
    images = np.load(images_path) if isinstance(images_path, str) else np.concatenate([np.load(p) for p in images_path], axis=0)
    masks = np.load(masks_path) if masks_path is not None and isinstance(masks_path, str) else (np.concatenate([np.load(p) for p in masks_path], axis=0) if masks_path is not None else None)

    # Definir carpeta de salida y prefijo
    if split_name in ["D1_train", "D2_train", "val"]:
        out_img_folder = imagesTr
        out_mask_folder = labelsTr
        prefix = split_name.split("_")[0]  # 'D1' o 'D2' o 'val'
    else:  # test splits
        out_img_folder = imagesTs
        out_mask_folder = labelsTs if masks_path is not None else None
        prefix = split_name.split("_")[1].upper()  # 'HE', 'PM', 'PAS'

    # Inicializar contador de IDs para este prefijo
    if prefix not in case_counters:
        case_counters[prefix] = 1

    for i in range(len(images)):
        case_id = f"{prefix}_{case_counters[prefix]:04d}"
        mask = masks[i] if masks is not None else None
        save_case(case_id, images[i], mask, out_img_folder, out_mask_folder)
        case_counters[prefix] += 1
        if split_name in ["D1_train", "D2_train"]:
            num_training_cases += 1

# -------------------------------
# Generar dataset.json
# -------------------------------
dataset_json = {
    "channel_names": { "0": "R", "1": "G", "2": "B" },
    "labels": { "background": 0, "non_sclerotic": 1, "sclerotic": 2 },
    "numTraining": num_training_cases,
    "file_ending": ".png"
}

with open(os.path.join(output_base, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"Dataset nnU-Net 2D generado en: {output_base}")
print(f"Casos de entrenamiento: {num_training_cases}")

