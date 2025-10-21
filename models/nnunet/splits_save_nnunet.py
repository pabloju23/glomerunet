import os
import json

# -----------------------------
# ----- CONFIGURACIÓN ----------
# -----------------------------
PREPROCESSED_PATH = "/scratch.local2/juanp/glomeruli/dataset/nnunet_data/nnUnet_preprocessed/Dataset002_glomeruli_D1"
SPLITS_JSON = os.path.join(PREPROCESSED_PATH, "splits_final.json")

# Lista de todos los casos preprocesados
all_cases = [f[:-4] for f in os.listdir(os.path.join(PREPROCESSED_PATH, "gt_segmentations")) if f.endswith(".png")]

# Definir nuestros splits personalizados
train = [c for c in all_cases if c.startswith("D2") or c.startswith("D1")]
val_cases    = [c for c in all_cases if c.startswith("val")]

# Para nnU-Net, cada split es un dict con 'train' y 'val'
splits = [
    {
        "train": train,
        "val": val_cases
    }
]

# Guardar splits para la Fase 1
with open(SPLITS_JSON, "w") as f:
    json.dump(splits, f, indent=4)

print(f"Splits generados y guardados en {SPLITS_JSON}")
print(f"{len(train)} casos para train, {len(val_cases)} para val")
