import os
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import matplotlib.pyplot as plt

# Set environment variable to select only GPU 1 and GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Optimize GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from methods.data_augmentor import create_dataset
from methods.dice_metric import OneHotDice

NUM_CLASSES = 3
BATCH_SIZE = 1  # Procesamos una imagen a la vez para evaluaciones individuales
model = 'deeplab'  # deeplab, unet, segnet, deeplab_base, unetr

model_path = f'paper_checkpoints/{model}_glomeruli_multiclass_finetunned_pool.keras'
# Load the model with compile=False
print(f'Cargando modelo {model}...')
if model=='deeplab' or model=='unet':
    model = tf.keras.models.load_model(model_path, compile=False)
    # from keras_deeplab_model import DeeplabV3Plus_mod
    # model = DeeplabV3Plus_mod(image_size=512, num_classes=NUM_CLASSES, backbone='resnet101')
    # model.load_weights(model_path)
if model == 'segnet':
    from models.keras.keras_segnet import segnet
    model = segnet((512, 512, 3), NUM_CLASSES)
    # Cargar pesos
    model.load_weights(model_path)
if model == 'deeplab_base':
    # Crear modelo
    from models.keras.keras_deeplab_model import DeeplabV3Plus
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
test_img_he = np.load('/scratch.local2/juanp/glomeruli/dataset/processed(5fold)/fold3_img_he.npy')
test_mask_he = np.load('/scratch.local2/juanp/glomeruli/dataset/processed(5fold)/fold3_mask_he.npy')
test_img_pas = np.load('/scratch.local2/juanp/glomeruli/dataset/processed(5fold)/fold3_img_pas.npy')
test_mask_pas = np.load('/scratch.local2/juanp/glomeruli/dataset/processed(5fold)/fold3_mask_pas.npy')
test_img_pm = np.load('/scratch.local2/juanp/glomeruli/dataset/processed(5fold)/fold3_img_pm.npy')
test_mask_pm = np.load('/scratch.local2/juanp/glomeruli/dataset/processed(5fold)/fold3_mask_pm.npy')
test_img = np.load('/scratch.local2/juanp/glomeruli/dataset/processed(5fold)/fold_3_img.npy')
test_mask = np.load('/scratch.local2/juanp/glomeruli/dataset/processed(5fold)/fold_3_mask.npy')

# Evaluate on each stain separately 
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
test_dataset = create_dataset(test_img, test_mask, batch_size=BATCH_SIZE, augmentation=False, shuffle=False)
print('Evaluando...')
# test_dataset = tf.data.Dataset.from_tensor_slices((test_img, test_mask))
# test_dataset = test_dataset.batch(BATCH_SIZE)

model.evaluate(test_dataset)