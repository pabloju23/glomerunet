import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import monai
from monai.data import ArrayDataset, DataLoader
from monai.transforms import Compose
from transforms import get_train_transforms, get_val_transforms


import os
from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np


def check_image_label_pairs_from_ndarrays(image_array, label_array, n_samples=5):
    images = np.load(image_array)
    labels = np.load(label_array)

    # consider shape N, H, W, C, being the masks one hot encoded and float32 0-1
    # Plot some pairs with classes 1 or 2
    print(f"Total imágenes: {images.shape[0]}")
    print(f"Total máscaras: {labels.shape[0]}\n")

    print(f"Mostrando pares imagen-máscara con clases 1 o 2:")
    shown = 0
    for i in range(images.shape[0]):
        img = images[i]
        mask = labels[i]

        # Convert one-hot to single channel if needed
        if mask.ndim == 4 and mask.shape[-1] > 1:
            mask = np.argmax(mask, axis=-1)

        unique_vals = np.unique(mask)
        if not np.any(np.isin(unique_vals, [1, 2])):
            continue

        print(f"\n🟩 Imagen {i} contiene clases {unique_vals}")

        # Mostrar par
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow((img*255).astype(np.uint8))
        axs[0].set_title(f"Imagen {i}")
        axs[0].axis("off")

        axs[1].imshow((mask*255).astype(np.uint8), cmap="tab10", interpolation="nearest")
        axs[1].set_title(f"Máscara {i}\nClases: {unique_vals}")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()

        shown += 1
        if shown >= n_samples:
            break

    if shown == 0:
        print("⚠️ No se encontraron ejemplos con clases 1 o 2.")
    else:
        print(f"\n✅ Se mostraron {shown} ejemplos con clases 1 o 2.")


def check_image_label_pairs_visual(dataset_dir, n_samples=5):
    images_folder = os.path.join(dataset_dir, "imagesTr")
    labels_folder = os.path.join(dataset_dir, "labelsTr")

    image_files = sorted(glob(os.path.join(images_folder, "D1_*_0000.png")))
    label_files = sorted(glob(os.path.join(labels_folder, "D1_*.png")))

    print(f"Total imágenes: {len(image_files)}")
    print(f"Total máscaras: {len(label_files)}\n")

    # --- Comprobación nominal ---
    print(f"Comprobando correspondencia nominal en las primeras {n_samples} muestras:\n")
    for img, lbl in zip(image_files[:n_samples], label_files[:n_samples]):
        img_base = os.path.basename(img).replace("_0000", "")
        lbl_base = os.path.basename(lbl)
        same = "✅" if img_base == lbl_base else "❌"
        print(f"{same}  Imagen: {os.path.basename(img)}  <--->  Máscara: {os.path.basename(lbl)}")

    # --- Visualización selectiva ---
    print("\nMostrando pares imagen-máscara con clases 1 o 2:")

    shown = 0
    for img_path, lbl_path in zip(image_files, label_files):
        img_base = os.path.basename(img_path).replace("_0000", "")
        lbl_base = os.path.basename(lbl_path)

        if img_base != lbl_base:
            continue  # Saltar si los nombres no coinciden

        # Cargar imagen y máscara
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            print(f"⚠️ No se pudo leer la máscara {lbl_path}")
            continue

        # Comprobar si contiene clases 1 o 2
        unique_vals = np.unique(mask)
        if not np.any(np.isin(unique_vals, [1, 2])):
            continue

        print(f"\n🟩 {os.path.basename(img_path)} contiene clases {unique_vals}")

        # Mostrar par
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img)
        axs[0].set_title(f"Imagen: {os.path.basename(img_path)}")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap="tab10", interpolation="nearest")
        axs[1].set_title(f"Máscara: {os.path.basename(lbl_path)}\nClases: {unique_vals}")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()

        shown += 1
        if shown >= n_samples:
            break

    if shown == 0:
        print("⚠️ No se encontraron ejemplos con clases 1 o 2.")
    else:
        print(f"\n✅ Se mostraron {shown} ejemplos con clases 1 o 2.")


def sanity_check(dataset_dir, n_samples=4):
    images_folder = os.path.join(dataset_dir, "imagesTr")
    labels_folder = os.path.join(dataset_dir, "labelsTr")

    train_images = sorted(glob(os.path.join(images_folder, "D1_*_0000.png")))
    train_labels = sorted(glob(os.path.join(labels_folder, "D1_*.png")))

    print(f"Found {len(train_images)} training images and {len(train_labels)} masks")

    # --- Transforms ---
    train_transforms = get_train_transforms()

    # Create dictionaries and Dataset for monai
    train_files = [{"image": img, "label": seg} for img, seg in zip(train_images, train_labels)]
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available()) 

    # --- Visualización ---
    print("Visualizando ejemplos tras los aumentos...\n")
    for i, batch in enumerate(train_loader):
        img, mask = batch["image"], batch["label"]
        img = img[0].permute(1, 2, 0).cpu().numpy()
        mask = mask[0, 0].cpu().numpy() if mask.shape[1] == 1 else mask[0].argmax(0).cpu().numpy()

        # Normalizar intensidades de imagen a 0–1 para visualización
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Mostrar imagen y máscara
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img)
        axs[0].set_title(f"Imagen {i}")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap="tab10")
        axs[1].set_title(f"Máscara {i} (unique={np.unique(mask)})")
        axs[1].axis("off")
        plt.show()

        if i + 1 >= n_samples:
            break

    # --- Comprobación global de valores ---
    print("Chequeando valores únicos en un batch aleatorio:")
    mask_vals = []
    for batch in train_loader:
        _, mask = batch["image"], batch["label"]
        mask_np = mask.cpu().numpy()
        vals = np.unique(mask_np)
        mask_vals.extend(list(vals))
        if len(mask_vals) > 200:
            break

    mask_vals = np.unique(mask_vals)
    print(f"Valores únicos en máscaras (tras aumentos): {mask_vals}")

if __name__ == "__main__":
    dataset_dir = "/scratch.local2/juanp/glomeruli/dataset/nnunet_data/nnUnet_raw/Dataset001_glomeruli"
    # check_image_label_pairs_from_ndarrays(
    #     '/scratch.local2/juanp/glomeruli/dataset/processed(5fold)/fold_4_img.npy',
    #     '/scratch.local2/juanp/glomeruli/dataset/processed(5fold)/fold_4_mask.npy',
    #     n_samples=5
    # )
    check_image_label_pairs_visual(dataset_dir, n_samples=5)
    # sanity_check(dataset_dir)
