import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2


# GPU setup
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ------------------------------
# Cargar modelo preentrenado
# ------------------------------
NUM_CLASSES = 3  # salida original del modelo
BATCH_SIZE = 40
patch_size = 512
model_path = 'paper_checkpoints/deeplab_glomeruli_multiclass_finetunned_pool.keras'

print("Cargando modelo DeepLab preentrenado...")
model = tf.keras.models.load_model(model_path, compile=False)
print("Modelo cargado con salida de 3 clases:", model.output_shape)

# ------------------------------
# Cargar dataset de imágenes PNG
# ------------------------------
def preprocess_image(img, target_size=(1024,1024), clahe_clip=1.0, tile_size=32):
    # Convertir a LAB para aplicar CLAHE en el canal L
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile_size, tile_size))
    lab[...,0] = clahe.apply((lab[...,0]*255).astype(np.uint8))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Normalizar a 0-1
    img_clahe = img_clahe.astype(np.float32)/255.0
    # Resize a target_size
    img_resized = cv2.resize(img_clahe, target_size, interpolation=cv2.INTER_LINEAR)
    if np.max(img_resized) > 1.0:
        img_resized = img_resized.astype(np.float32)/255.0
    return img_resized

def preprocess_mask(mask, target_size=(1024,1024)):
    # Resize manteniendo binariedad
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 0).astype(np.uint8)
    # Check there are just two unique values
    assert len(np.unique(mask_resized)) <= 2, f"Mask has more than 2 unique values: {np.unique(mask_resized)}"
    # Check those values are 0 and 1
    assert set(np.unique(mask_resized)).issubset({0,1}), f"Mask values are not binary: {np.unique(mask_resized)}"
    return mask_resized

def extract_patches(img, mask, patch_size=512, overlap=0.5):
    stride = int(patch_size * (1-overlap))
    patches_img = []
    patches_mask = []
    H, W = img.shape[:2]
    for y in range(0, H-patch_size+1, stride):
        for x in range(0, W-patch_size+1, stride):
            patches_img.append(img[y:y+patch_size, x:x+patch_size])
            patches_mask.append(mask[y:y+patch_size, x:x+patch_size])
    return np.array(patches_img), np.array(patches_mask)

def load_and_preprocess(img_folder, mask_folder):
    img_paths = sorted(glob(img_folder + '/*.jpg'))
    mask_paths = sorted(glob(mask_folder + '/*.jpg'))
    
    print(f"Imágenes encontradas: {len(img_paths)}")
    print(f"Máscaras encontradas: {len(mask_paths)}")
    
    if len(img_paths) == 0 or len(mask_paths) == 0:
        raise ValueError("No se encontraron imágenes o máscaras en las carpetas indicadas.")
    
    all_imgs, all_masks = [], []
    for ip, mp in zip(img_paths, mask_paths):
        img = np.array(Image.open(ip).convert('RGB'))
        mask = np.array(Image.open(mp).convert('L'))
        
        img = preprocess_image(img)
        mask = preprocess_mask(mask)
        
        patches_img, patches_mask = extract_patches(img, mask)
        
        if patches_img.shape[0] > 0:
            all_imgs.append(patches_img)
            all_masks.append(patches_mask)
    
    if len(all_imgs) == 0 or len(all_masks) == 0:
        raise ValueError("No se extrajeron parches de ninguna imagen/máscara.")
    
    return np.concatenate(all_imgs, axis=0), np.concatenate(all_masks, axis=0)

img_folder = '/scratch.local2/juanp/glomeruli/dataset/challenge/KPI_test/img'
mask_folder = '/scratch.local2/juanp/glomeruli/dataset/challenge/KPI_test/mask'
test_img, test_mask = load_and_preprocess(img_folder, mask_folder)

print("Imágenes:", test_img.shape)
print("Máscaras one-hot:", test_mask.shape)

# ------------------------------
# Generador para cargar imágenes y máscaras desde disco por batch
# ------------------------------
def generator(img_folder, mask_folder, patch_size=512, overlap=0.5, max_images=None):
    img_paths = sorted(glob(os.path.join(img_folder, '*.jpg')))
    mask_paths = sorted(glob(os.path.join(mask_folder, '*.jpg')))
    
    if len(img_paths) == 0 or len(mask_paths) == 0:
        raise ValueError("No se encontraron imágenes o máscaras en las carpetas indicadas.")
    
    # Limitar a las primeras N imágenes si se desea
    if max_images is not None:
        img_paths = img_paths[:max_images]
        mask_paths = mask_paths[:max_images]
        print(f"Usando solo {len(img_paths)} imágenes para prueba rápida.")

    for ip, mp in zip(img_paths, mask_paths):
        img = np.array(Image.open(ip).convert('RGB'))
        mask = np.array(Image.open(mp).convert('L'))
        
        img = preprocess_image(img)
        mask = preprocess_mask(mask)
        
        stride = int(patch_size * (1-overlap))
        H, W = img.shape[:2]
        for y in range(0, H-patch_size+1, stride):
            for x in range(0, W-patch_size+1, stride):
                patch_img = img[y:y+patch_size, x:x+patch_size]
                patch_mask = mask[y:y+patch_size, x:x+patch_size]
                yield patch_img, patch_mask


# ------------------------------
# Comprobación visual antes de cálculos pesados
# ------------------------------
max_images_debug = 3  # mostrar 3 ejemplos
for i, (img, mask) in enumerate(generator(img_folder, mask_folder, patch_size=patch_size, max_images=max_images_debug)):
    print(f"Ejemplo {i}: Imagen {img.shape}, Máscara {mask.shape}, Uniques máscara: {np.unique(mask)}")
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Imagen')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.title('Máscara')
    plt.axis('off')
    plt.show()
    if i >= 2:
        break  # solo mostrar unos pocos ejemplos


# ------------------------------
# Dataset tf.data a partir del generador
# ------------------------------
test_dataset = tf.data.Dataset.from_generator(
    lambda: generator(img_folder, mask_folder, patch_size=patch_size, max_images=30),  # <= solo 30 imágenes
    output_signature=(
        tf.TensorSpec(shape=(patch_size, patch_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(patch_size, patch_size), dtype=tf.uint8)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ------------------------------
# Evaluación zero-shot sin cargar todo en memoria
# ------------------------------
dice_list = []
iou_list = []

for batch_img, batch_mask in test_dataset:
    preds = model.predict(batch_img)
    pred_bin = (np.argmax(preds, axis=-1) > 0).astype(np.uint8)
    mask_bin = (batch_mask.numpy() > 0).astype(np.uint8)
    print("Batch pred:", pred_bin.shape, "Batch mask:", mask_bin.shape)
    print("Unique pred:", np.unique(pred_bin), "Unique mask:", np.unique(mask_bin))
    print('Max pred:', np.max(pred_bin), 'Max mask:', np.max(mask_bin))

    # Métricas
    intersection = np.sum(pred_bin * mask_bin)
    dice = (2. * intersection) / (np.sum(pred_bin) + np.sum(mask_bin) + 1e-8)
    union = np.sum(pred_bin) + np.sum(mask_bin) - intersection
    iou = intersection / (union + 1e-8)

    dice_list.append(dice)
    iou_list.append(iou)

print(f"IoU binaria zero-shot promedio: {np.mean(iou_list):.4f}")
print(f"Dice binario zero-shot promedio: {np.mean(dice_list):.4f}")
