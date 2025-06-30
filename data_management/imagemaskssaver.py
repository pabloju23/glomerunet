import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def load_and_preprocess_images(folder_path, target_size=(1024, 1024), clahe_clip_limit=1.0):
    images = []
    print(f"Procesando imágenes en carpeta: {folder_path}")
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Advertencia: No se pudo cargar la imagen {img_path}")
                continue
            img = cv2.resize(img, target_size)

            # Convertir a LAB, aplicar CLAHE en el canal L y devolver a RGB
            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_img)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab_img = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)

            img = np.clip(img, 0, 255).astype(np.uint8)
            images.append(img)
    print(f"Total imágenes cargadas desde {folder_path}: {len(images)}")
    return np.array(images)

def load_and_preprocess_binary_masks(folder_path, target_size=(1024, 1024)):
    masks = []
    print(f"Procesando máscaras binarias en carpeta: {folder_path}")
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            mask_path = os.path.join(folder_path, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Advertencia: No se pudo cargar la máscara {mask_path}")
                continue
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

            unique_vals = np.unique(mask)
            if not np.array_equal(unique_vals, [0, 255]):
                raise ValueError(f"Máscara binaria '{filename}' no es binaria: {unique_vals}")

            mask = (mask // 255).astype(np.uint8)
            masks.append(mask)
    print(f"Total máscaras binarias cargadas desde {folder_path}: {len(masks)}")
    return np.array(masks)

def load_and_preprocess_multiclass_masks(folder_path, target_size=(1024, 1024), max_intensity=16):
    masks = []
    print(f"Procesando máscaras multiclase en carpeta: {folder_path}")
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            mask_path = os.path.join(folder_path, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Advertencia: No se pudo cargar la máscara multiclase {mask_path}")
                continue
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

            unique_vals = np.unique(mask)
            if unique_vals.max() > max_intensity:
                raise ValueError(f"Máscara multiclase '{filename}' tiene valores fuera del rango 0-{max_intensity}: {unique_vals}")

            masks.append(mask.astype(np.uint8))
    print(f"Total máscaras multiclase cargadas desde {folder_path}: {len(masks)}")
    return np.array(masks)

def save_images_and_masks(train_dir, output_dir):
    for stain in os.listdir(train_dir):
        stain_path = os.path.join(train_dir, stain)
        if os.path.isdir(stain_path):
            tissue_folder = os.path.join(stain_path, 'tissue')
            binary_mask_folder = os.path.join(stain_path, 'groundtruth')
            multiclass_mask_folder = os.path.join(stain_path, 'groundtruth_multiclass')

            print(f"\nProcesando tinción: {stain}")
            if not os.path.exists(tissue_folder):
                print(f"Carpeta 'tissue' no encontrada en {stain_path}")
                continue
            # if not os.path.exists(binary_mask_folder):
            #     print(f"Carpeta 'groundtruth' no encontrada en {stain_path}")
            #     continue
            if not os.path.exists(multiclass_mask_folder):
                print(f"Carpeta 'groundtruth_multiclass' no encontrada en {stain_path}")
                continue

            # Cargar y procesar imágenes, máscaras binarias y máscaras multiclase
            tissue_images = load_and_preprocess_images(tissue_folder)
            # binary_masks = load_and_preprocess_binary_masks(binary_mask_folder)
            multiclass_masks = load_and_preprocess_multiclass_masks(multiclass_mask_folder)

            # Guardar los arrays como archivos .npy
            np.save(os.path.join(output_dir, f'{dataset_type}_img_{stain}.npy'), tissue_images)
            # np.save(os.path.join(output_dir, f'train_mask_binary_{stain}.npy'), binary_masks)
            np.save(os.path.join(output_dir, f'{dataset_type}_mask_multiclass_{stain}.npy'), multiclass_masks)
            print(f'Guardados: test_img_{stain}.npy, test_mask_binary_{stain}.npy, test_mask_multiclass_{stain}.npy en {output_dir}')

            # Mostrar una imagen con su máscara binaria y multiclase, si existen
            if len(tissue_images) > 0 and len(multiclass_masks) > 0:
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(tissue_images[0])
                plt.title(f"Imagen - {stain}")

                # plt.subplot(1, 3, 2)
                # plt.imshow(binary_masks[0], cmap='gray')
                # plt.title(f"Máscara Binaria - {stain}")

                plt.subplot(1, 3, 3)
                plt.imshow(multiclass_masks[0], cmap='nipy_spectral')
                plt.title(f"Máscara Multiclase - {stain}")

                plt.show()

# Ruta de la carpeta principal 
dataset_type = 'train'
train_dir = '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/' + dataset_type
output_dir = train_dir  
save_images_and_masks(train_dir, output_dir)