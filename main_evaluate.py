import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set environment variable to select only GPU 1 and GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Optimize GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from data_augmentor import create_dataset
from dice_metric import OneHotDice

NUM_CLASSES = 3
BATCH_SIZE = 6  # Procesamos una imagen a la vez para evaluaciones individuales
model = 'deeplab'  # deeplab, unet, segnet, deeplab_base

model_path = f'paper_checkpoints/{model}_glomeruli_multiclass_finetunned_795.keras' 
# Load the model with compile=False
print(f'Cargando modelo {model}...')
if model=='deeplab' or model=='unet':
    model = tf.keras.models.load_model(model_path, compile=False)
if model == 'segnet':
    from keras_segnet import segnet
    model = segnet((512, 512, 3), NUM_CLASSES)
    # Cargar pesos
    model.load_weights(model_path)
if model == 'deeplab_base':
    # Crear modelo
    from keras_deeplab_model import DeeplabV3Plus
    model = DeeplabV3Plus(image_size=512, num_classes=NUM_CLASSES) 
    # Cargar pesos
    model.load_weights(model_path)


# Create metrics per class
iou = tf.keras.metrics.OneHotIoU(num_classes=NUM_CLASSES, target_class_ids=[0, 1, 2])
iou_metric0 = tf.keras.metrics.OneHotIoU(num_classes=NUM_CLASSES, target_class_ids=[0])
iou_metric1 = tf.keras.metrics.OneHotIoU(num_classes=NUM_CLASSES, target_class_ids=[1])
iou_metric2 = tf.keras.metrics.OneHotIoU(num_classes=NUM_CLASSES, target_class_ids=[2])
dsc = OneHotDice(num_classes=NUM_CLASSES, target_class_ids=[0, 1, 2])
dice_metric0 = OneHotDice(num_classes=NUM_CLASSES, target_class_ids=[0])
dice_metric1 = OneHotDice(num_classes=NUM_CLASSES, target_class_ids=[1])
dice_metric2 = OneHotDice(num_classes=NUM_CLASSES, target_class_ids=[2])

loss = tf.keras.losses.CategoricalFocalCrossentropy(
    alpha=tf.constant([0.1, 2, 4], dtype=tf.float32),
    gamma=3.0,
    from_logits=False,
    reduction='sum_over_batch_size',
    name='categorical_focal_crossentropy'
)

# Compile the model with loss and metrics
print('Compilando modelo...')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=loss,
              metrics=[iou, iou_metric0, iou_metric1, iou_metric2, dsc, dice_metric0, dice_metric1, dice_metric2])

# Load test dataset
print('Cargando dataset...')
test_img_he = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_img_he.npy')
test_mask_he = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_mask_he.npy')
test_img_pas = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_img_pas.npy')
test_mask_pas = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_mask_pas.npy')
test_img_pm = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_img_pm.npy')
test_mask_pm = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_mask_pm.npy')
test_img = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_3_img.npy')
test_mask = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_3_mask.npy')

# Evalute in each stain
print('Evaluando en HE...')
test_dataset_he = tf.data.Dataset.from_tensor_slices((test_img_he, test_mask_he))
test_dataset_he = test_dataset_he.batch(BATCH_SIZE)
model.evaluate(test_dataset_he)

print('Evaluando en PAS...')
test_dataset_pas = tf.data.Dataset.from_tensor_slices((test_img_pas, test_mask_pas))
test_dataset_pas = test_dataset_pas.batch(BATCH_SIZE)
model.evaluate(test_dataset_pas)

print('Evaluando en PM...')
test_dataset_pm = tf.data.Dataset.from_tensor_slices((test_img_pm, test_mask_pm))
test_dataset_pm = test_dataset_pm.batch(BATCH_SIZE)
model.evaluate(test_dataset_pm)

# Squeeze
test_img = np.squeeze(test_img)
test_mask = np.squeeze(test_mask)

# Checkl all classes present in one hot encoded masks
print('Clases presentes en las máscaras one-hot encoded:', (np.sum(test_mask, axis=(0, 1, 2)) / np.prod(test_mask.shape[:3]) * 100))

print('Comprobando dataset...')
for i, (img, mask) in enumerate(zip(test_img, test_mask)):
    if img.shape != (512, 512, 3) or mask.shape != (512, 512, NUM_CLASSES):
        print(f"Error en el ejemplo {i}: Imagen {img.shape}, Máscara {mask.shape}")
    if np.max(img) > 1 or np.max(mask) > 1:
        print(f"Error en el ejemplo {i}: Imagen {np.max(img)}, Máscara {np.max(mask)}")

print("Forma de entrada del modelo:", model.input_shape)
print("Forma de salida del modelo:", model.output_shape)

# Create test dataset
# test_dataset = create_dataset(test_img, test_mask, batch_size=BATCH_SIZE, augmentation=False, shuffle=False)
print('Evaluando...')
test_dataset = tf.data.Dataset.from_tensor_slices((test_img, test_mask))
test_dataset = test_dataset.batch(BATCH_SIZE)

model.evaluate(test_dataset)
# exit()
predictions = model.predict(test_img, batch_size=BATCH_SIZE)

# np.save(f'paper_checkpoints/unet_predictions.npy', predictions)

# Find the best threshold for binarization
# thresholds = np.linspace(0.1, 0.9, 9)

# for threshold in thresholds:
#     predictions_bin = (predictions >= threshold).astype(np.float32)
#     iou.reset_states()
#     dsc.reset_states()
#     iou.update_state(test_mask, predictions_bin) 
#     dsc.update_state(test_mask, predictions_bin)
#     print(f"Threshold: {threshold:.1f}, IoU: {iou.result().numpy():.4f}, Dice: {dsc.result().numpy():.4f}")

# exit()

# Binarize onehot encoded preditions
# predictions = (predictions >= 0.99).astype(np.float32)   # Check but for segnet 0.99 threshold is the best 

from skimage.measure import label
from scipy.ndimage import binary_fill_holes

def postprocess_predictions_one_hot(predictions, background_class=0):
    """
    Post-processes one-hot encoded predictions to homogenize connected components 
    based on the majority class.
    
    Args:
        predictions: 3D or 4D numpy array of one-hot encoded predictions (e.g., HxWxC or NxHxWxC).
        background_class: Index of the background class (default is 0).
    
    Returns:
        Processed one-hot encoded predictions with homogenized connected components.
    """
    # Ensure batch processing (loop through each sample if 4D input)
    if predictions.ndim == 4:
        return np.stack([postprocess_predictions_one_hot(pred, background_class) for pred in predictions], axis=0)

    # Convert from one-hot encoding to single class (HxW)
    class_indices = np.argmax(predictions, axis=-1)

    # Process class indices
    class_indices = postprocess_predictions(class_indices, background_class)

    # Convert back to one-hot encoding
    processed_one_hot = np.zeros_like(predictions)
    for c in range(predictions.shape[-1]):
        processed_one_hot[..., c] = (class_indices == c)

    return processed_one_hot

def postprocess_predictions(predictions, background_class=0):
    """
    Post-processes class predictions to homogenize connected components based on the majority class.
    
    Args:
        predictions: 2D numpy array of predicted classes (e.g., HxW).
        background_class: The class value considered as the background (default is 0).
    
    Returns:
        Processed predictions with homogenized connected components.
    """
    # Create a binary mask for non-background classes
    mask = predictions != background_class
    
    # Label connected components in the mask
    labeled_regions, num_regions = label(mask, connectivity=1, return_num=True)
    
    # Process each region
    for region_idx in range(1, num_regions + 1):
        # Get the mask of the current region
        region_mask = labeled_regions == region_idx
        
        # Find the majority class in the region
        region_classes, region_counts = np.unique(predictions[region_mask], return_counts=True)
        majority_class = region_classes[np.argmax(region_counts)]
        
        # Assign the majority class to all pixels in the region
        predictions[region_mask] = majority_class
    
    return predictions

# Post-process predictions
predictions = postprocess_predictions_one_hot(predictions)

# Re-evaluate metrics with thresholded predictions
iou.reset_states()
dsc.reset_states()
iou.update_state(test_mask, predictions) 
dsc.update_state(test_mask, predictions)

print("IoU:", iou.result().numpy(), ",    Dice:", dsc.result().numpy())

import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def save_images_masks_predictions(images, masks, predictions, save_predictions_only=True):
    """
    Guarda imágenes, máscaras y predicciones del modelo en carpetas separadas.

    Args:
        images: Array de imágenes de tamaño (N, 512, 512, 3).
        masks: Array de máscaras de tamaño (N, 512, 512, NUM_CLASSES) (one-hot encoded).
        predictions: Array de predicciones de tamaño (N, 512, 512, NUM_CLASSES) (one-hot encoded).
        save_predictions_only: Booleano para indicar si solo se deben guardar las predicciones.
    """
    # Crear carpetas si no existen
    os.makedirs('images', exist_ok=True)
    os.makedirs('masks', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    
    for idx in range(len(images)):
        img = images[idx]
        mask = masks[idx]
        pred = predictions[idx]
        
        # Escalar la imagen si está en float32 [0, 1]
        if img.dtype == np.float32 and img.max() <= 1.0:
            img_to_save = (img * 255).astype(np.uint8)
        else:
            img_to_save = img

        # Convertir la máscara one-hot encoded a una máscara RGB
        mask_rgb = np.zeros_like(img_to_save)
        mask_rgb[..., 0] = mask[..., 1] * 255  # Clase 1 -> Rojo
        mask_rgb[..., 1] = mask[..., 2] * 255  # Clase 2 -> Verde
        mask_rgb[..., 2] = 0                  # Clase 0 -> Negro (implícito)

        # Convertir la predicción one-hot encoded a una máscara RGB
        pred_rgb = np.zeros_like(img_to_save)
        pred_rgb[..., 0] = pred[..., 1] * 255  # Clase 1 -> Rojo
        pred_rgb[..., 1] = pred[..., 2] * 255  # Clase 2 -> Verde
        pred_rgb[..., 2] = 0                   # Clase 0 -> Negro (implícito)

        # Guardar la imagen
        if not save_predictions_only:
            img_pil = Image.fromarray(img_to_save)
            img_pil.save(f'images/image_{idx}.jpg')

            # Guardar la máscara
            mask_pil = Image.fromarray(mask_rgb)
            mask_pil.save(f'masks/mask_{idx}.jpg')

        # Guardar la predicción
        pred_pil = Image.fromarray(pred_rgb)
        pred_pil.save(f'predictions/prediction_{idx}.jpg')

# Ejemplo de uso
# save_images_masks_predictions(test_img, test_mask, predictions, save_predictions_only=False)

def visualize_images_with_masks_and_predictions(images, masks, predictions, num_samples=5):
    """
    Visualiza imágenes, sus respectivas máscaras y predicciones del modelo.

    Args:
        images: Array de imágenes de tamaño (N, 512, 512, 3).
        masks: Array de máscaras de tamaño (N, 512, 512, NUM_CLASSES) (one-hot encoded).
        predictions: Array de predicciones de tamaño (N, 512, 512, NUM_CLASSES) (one-hot encoded).
        num_samples: Número de imágenes/máscaras a visualizar.
    """
    # Verificar los datos
    check_data(images, masks, predictions)
    
    # Identificar índices de imágenes que contienen píxeles para cada clase (clase 1 y clase 2)
    class_1_indices = np.where(masks[..., 1].any(axis=(1, 2)))[0]
    class_2_indices = np.where(masks[..., 2].any(axis=(1, 2)))[0]

    # Seleccionar al menos un ejemplo de cada clase
    selected_indices = []
    if len(class_1_indices) > 0:
        selected_indices.append(np.random.choice(class_1_indices))
    if len(class_2_indices) > 0:
        selected_indices.append(np.random.choice(class_2_indices))

    # Si no se alcanzan num_samples, seleccionar índices adicionales aleatorios
    if len(selected_indices) < num_samples:
        remaining_indices = np.setdiff1d(np.arange(len(images)), selected_indices)
        additional_indices = np.random.choice(remaining_indices, num_samples - len(selected_indices), replace=False)
        selected_indices.extend(additional_indices)
    
    # Configurar el tamaño de la figura
    plt.figure(figsize=(15, num_samples * 5))
    
    for i, idx in enumerate(selected_indices):
        img = images[idx]
        mask = masks[idx]
        pred = predictions[idx]
        
        # Escalar la imagen si está en float32 [0, 1]
        if img.dtype == np.float32 and img.max() <= 1.0:
            img_to_plot = (img * 255).astype(np.uint8)
        else:
            img_to_plot = img

        # Convertir la máscara one-hot encoded a una máscara RGB
        mask_rgb = np.zeros_like(img_to_plot)
        mask_rgb[..., 0] = mask[..., 1] * 255  # Clase 1 -> Rojo
        mask_rgb[..., 1] = mask[..., 2] * 255  # Clase 2 -> Verde
        mask_rgb[..., 2] = 0                  # Clase 0 -> Negro (implícito)

        # Convertir la predicción one-hot encoded a una máscara RGB
        pred_rgb = np.zeros_like(img_to_plot)
        pred_rgb[..., 0] = pred[..., 1] * 255  # Clase 1 -> Rojo
        pred_rgb[..., 1] = pred[..., 2] * 255  # Clase 2 -> Verde
        pred_rgb[..., 2] = 0                   # Clase 0 -> Negro (implícito)

        # Visualizar la imagen
        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.imshow(img_to_plot)
        plt.axis("off")
        plt.title(f"Image {idx}")

        # Visualizar la máscara
        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.imshow(mask_rgb)
        plt.axis("off")
        plt.title(f"Mask {idx}")

        # Visualizar la predicción
        plt.subplot(num_samples, 3, 3 * i + 3)
        plt.imshow(pred_rgb)
        plt.axis("off")
        plt.title(f"Prediction {idx}")
    
    plt.tight_layout()
    plt.show()

def check_data(images, masks, predictions):
    """
    Función para verificar que las dimensiones de las imágenes, máscaras y predicciones sean correctas.
    
    Args:
        images: Array de imágenes de tamaño (N, 512, 512, 3).
        masks: Array de máscaras de tamaño (N, 512, 512, NUM_CLASSES).
        predictions: Array de predicciones de tamaño (N, 512, 512, NUM_CLASSES).
    """
    NUM_CLASSES = masks.shape[-1]
    assert images.shape[1:] == (512, 512, 3), "Las imágenes deben tener forma (N, 512, 512, 3)"
    assert masks.shape[1:] == (512, 512, NUM_CLASSES), "Las máscaras deben tener forma (N, 512, 512, NUM_CLASSES)"
    assert predictions.shape[1:] == (512, 512, NUM_CLASSES), "Las predicciones deben tener forma (N, 512, 512, NUM_CLASSES)"

