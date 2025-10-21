import os
import shutil
import json

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
RAW_DATASET_PATH = "/scratch.local2/juanp/glomeruli/dataset/nnunet_data/nnUnet_raw/Dataset001_glomeruli"
OUTPUT_BASE_PATH = "/scratch.local2/juanp/glomeruli/dataset/nnunet_data/nnUnet_raw"

SPLITS = ["D1", "D2"]  # Los datasets que queremos crear
VAL_PREFIX = "val"      # Casos de validación, se repiten en ambos

# Rutas de origen
images_src = os.path.join(RAW_DATASET_PATH, "imagesTr")
labels_src = os.path.join(RAW_DATASET_PATH, "labelsTr")

# Listar todos los casos disponibles
all_images = [f for f in os.listdir(images_src) if f.endswith(".png")]
all_labels = [f for f in os.listdir(labels_src) if f.endswith(".png")]

# -----------------------------
# CREAR DATASETS
# -----------------------------
for split in SPLITS:
    out_dir = os.path.join(OUTPUT_BASE_PATH, f"Dataset001_glomeruli_{split}")
    images_dst = os.path.join(out_dir, "imagesTr")
    labels_dst = os.path.join(out_dir, "labelsTr")
    
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(labels_dst, exist_ok=True)
    
    # Filtrar imágenes para este split (train + val)
    split_images = [f for f in all_images if f.startswith(split) or f.startswith(VAL_PREFIX)]
    
    # Copiar imágenes y labels correspondientes
    for img_file in split_images:
        src_img = os.path.join(images_src, img_file)
        dst_img = os.path.join(images_dst, img_file)
        shutil.copy(src_img, dst_img)
        
        # Buscar la máscara cuyo nombre empieza por el mismo prefijo
        prefix = "_".join(img_file.split("_")[:2])  # split_XXX
        matching_labels = [l for l in all_labels if l.startswith(prefix)]
        if not matching_labels:
            print(f"Warning: no se encontró máscara para {img_file}, se omitirá")
            continue
        
        src_label = os.path.join(labels_src, matching_labels[0])
        dst_label = os.path.join(labels_dst, matching_labels[0])
        shutil.copy(src_label, dst_label)
    
    # Crear dataset.json con recuento de casos
    dataset_json = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": {
            "background": 0,
            "non_sclerotic": 1,
            "sclerotic": 2
        },
        "numTraining": len(split_images),
        "file_ending": ".png"
    }
    
    json_path = os.path.join(out_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"Dataset {split} creado: {len(split_images)} casos (train+val)")

print("Todos los datasets creados correctamente.")
