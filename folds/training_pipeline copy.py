import os
import sys

# Añadir el directorio padre al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Set environment variable to select only GPU 1 and GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import tensorflow as tf
# Optimize GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
from keras_deeplab_model import DeeplabV3Plus_largev2, DeeplabV3Plus
from data_augmentor import create_dataset, create_dataset_with_class_augmentation
import pickle


# Configuration
IMAGE_SIZE = (512, 512, 3)  # Reduced size to manage memory
BATCH_SIZE = 15  # Further reduced batch size to fit GPU memory
NUM_CLASSES=3

# Load dataset
fold1_img = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_1_img.npy')
fold1_mask = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_1_mask.npy')
fold2_img = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_2_img.npy')
fold2_mask = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_2_mask.npy')
fold3_img = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_3_img.npy')
fold3_mask = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_3_mask.npy')
fold4_img = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_4_img.npy')
fold4_mask = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_4_mask.npy')
fold5_img = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_5_img.npy')
fold5_mask = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_5_mask.npy')

# # Ensure the data is in the correct dtype
fold1_img = fold1_img.astype(np.float32)
fold1_mask = fold1_mask.astype(np.float32)
fold2_img = fold2_img.astype(np.float32)
fold2_mask = fold2_mask.astype(np.float32)
fold3_img = fold3_img.astype(np.float32)
fold3_mask = fold3_mask.astype(np.float32)
fold4_img = fold4_img.astype(np.float32)
fold4_mask = fold4_mask.astype(np.float32)
fold5_img = fold5_img.astype(np.float32)
fold5_mask = fold5_mask.astype(np.float32)

# Define training data
train_img = np.concatenate((fold5_img, fold2_img, fold4_img), axis=0)
train_mask = np.concatenate((fold5_mask, fold2_mask, fold4_mask), axis=0)

# Shuffle training data by index
idx = np.random.permutation(len(train_img))
train_img = train_img[idx]
train_mask = train_mask[idx] 

# Define validation data
val_img = fold1_img
val_mask = fold1_mask

# Define test data
test_img = fold3_img
test_mask = fold3_mask

# Define callbacks with monitor set to validation mean IoU
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_dsc_nobg', mode='max', patience=12, restore_best_weights=False, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dsc_nobg', mode='max', factor=0.1, patience=6, verbose=1, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(filepath='paper_checkpoints/deeplab_glomeruli_multiclass_finetunned.keras', monitor='val_dsc_nobg', mode='max', save_best_only=True, save_weights_only=False, verbose=1),
]

# Set up mirrored strategy
strategy = tf.distribute.MirroredStrategy()

# Define models to train
model_name = DeeplabV3Plus_largev2

# Train models and save performance summary
performance_summary = []

def train_and_evaluate(train_img, train_mask, val_img, val_mask, test_img, test_mask):
    # Create datasets
    train_dataset = create_dataset_with_class_augmentation(train_img, train_mask, batch_size=BATCH_SIZE, shuffle=True, augmentation=True)
    val_dataset = create_dataset(val_img, val_mask, batch_size=BATCH_SIZE, shuffle=False)
    test_dataset = create_dataset(test_img, test_mask, batch_size=6, shuffle=False)

    with strategy.scope():
        from tversky_metric import OneHotTversky
        target_class_ids=[0, 1, 2]

        # Create and compile model
        iou_metric = tf.keras.metrics.OneHotIoU(name='IoU', num_classes=NUM_CLASSES, target_class_ids=target_class_ids)  

        from dice_metric import OneHotDice
        dice_metric = OneHotDice(num_classes=NUM_CLASSES, name='dsc', target_class_ids=target_class_ids)
        dice_nobg = OneHotDice(num_classes=NUM_CLASSES, name='dsc_nobg', target_class_ids=[1, 2])
        dice_1 = OneHotDice(num_classes=NUM_CLASSES, name='dsc1', target_class_ids= [1])
        dice_2 = OneHotDice(num_classes=NUM_CLASSES, name='dsc2', target_class_ids= [2])
        from tversky_metric import OneHotTversky
        tversky_metric = OneHotTversky(alpha=0.7, beta=0.3, name='tversky', num_classes=NUM_CLASSES, target_class_ids=target_class_ids)

        def categorical_focal_tversky_loss(alpha=0.5, gamma=2.0, tversky_alpha=0.7, tversky_beta=0.3):
            """
            Combina la Focal Loss con la Tversky Loss para problemas de segmentación multiclase.
            Parámetros:
                alpha: Peso para balancear Focal Loss y Tversky Loss.
                gamma: Parámetro de enfoque para la Focal Loss.
                tversky_alpha: Penalización para falsos positivos en Tversky Loss.
                tversky_beta: Penalización para falsos negativos en Tversky Loss.
            """
            def loss(y_true, y_pred):
                epsilon = tf.keras.backend.epsilon()
                # Clipping para evitar errores numéricos
                y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - epsilon)
                
                # Tversky Loss
                tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
                fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2])
                fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2])
                tversky_index = tp / (tp + tversky_alpha * fp + tversky_beta * fn + epsilon)
                tversky_loss = tf.reduce_mean(1 - tversky_index)
                
                # Focal Categorical Crossentropy
                focal_loss = -y_true * tf.pow(1 - y_pred + epsilon, gamma) * tf.math.log(y_pred + epsilon)
                focal_loss = tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
                
                # Combinación de ambas pérdidas
                return alpha * focal_loss + (1 - alpha) * tversky_loss
            
            return loss

        cftl = categorical_focal_tversky_loss(alpha=0.35, gamma=2.0, tversky_alpha=0.7, tversky_beta=0.3)
        cfce = tf.keras.losses.CategoricalFocalCrossentropy(
        # alpha = tf.constant([0.0031,0.1695,0.8274], dtype=tf.float32),    # Peso para las clases minoritarias
        gamma=2,     # Focusing parameter
        ) #   #  tf.keras.losses.BinaryCrossentropy(from_logits=False) #TverskyLoss() #  
  
        initial_lr = 1e-4
        adam_opt = tf.keras.optimizers.Adam(learning_rate=initial_lr)
        opt = adam_opt

        # Create model
        # model = model_name(image_size=IMAGE_SIZE[0], num_classes=NUM_CLASSES, backbone='xception', weights='imagenet')
        model_path = f'paper_checkpoints/deeplab_glomeruli_multiclass_pretrain.keras' 
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer=opt,
                loss=cftl,
                metrics=[iou_metric, dice_metric, dice_nobg, dice_1, dice_2])
        # Train model
        history = model.fit(train_dataset, epochs=25, 
                            validation_data=val_dataset, 
                            callbacks=callbacks)
        
        # model.save(f'paper_checkpoints/deeplab_glomeruli_multiclass_finetunned.keras')

        # Save history
        with open(f'paper_checkpoints/training_history_deeplab_finetunned.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        
        # Save best epoch loss and metrics
        best_epoch = history.history['val_IoU'].index(max(history.history['val_IoU'])) 
        best_metrics = {
            'epoch': best_epoch + 1,
            'train_loss': history.history['loss'][best_epoch],
            'train_iou': history.history['IoU'][best_epoch],
            'train_dice': history.history['dsc'][best_epoch],
            'train_dice_nobg': history.history['dsc_nobg'][best_epoch],
            'train_dice_2': history.history['dsc2'][best_epoch],
            'val_loss': history.history['val_loss'][best_epoch],
            'val_iou': history.history['val_IoU'][best_epoch],
            'val_dice': history.history['val_dsc'][best_epoch],
            'val_dice_nobg': history.history['val_dsc_nobg'][best_epoch],
            'val_dice_2': history.history['val_dsc2'][best_epoch]
        }

        # Evaluate on test set
        test_metrics = model.evaluate(test_dataset)
        best_metrics['test_loss'] = test_metrics[0]
        best_metrics['test_iou'] = test_metrics[1]
        best_metrics['test_dice'] = test_metrics[2]
        best_metrics['test_dice_nobg'] = test_metrics[3]
        best_metrics['test_dice_2'] = test_metrics[5]
        

train_and_evaluate(train_img, train_mask, val_img, val_mask, test_img, test_mask)

    # Guardar en un archivo Excel en la fila excel_row
    # output_file = "training_metrics.xlsx"

    # Actualizar el archivo Excel con el resumen de rendimiento
    # update_excel(output_file, excel_row, performance_summary)