import os
# Set environment variable to select GPU's
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import tensorflow as tf
# Optimize GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
from models.keras.keras_deeplab_model import DeeplabV3Plus_mod, DeeplabV3Plus
from methods.data_augmentor import create_dataset, create_dataset_with_class_augmentation
from methods.data_augmentor import create_dataset_h5, create_dataset_with_class_augmentation_h5
import pickle


# Configuration
IMAGE_SIZE = (512, 512, 3)  
BATCH_SIZE = 39  
NUM_CLASSES=3
val = True
model_save = 'deeplab'
only_weights = False
pretrain = True

# Tamaño de las imágenes 
img_height = IMAGE_SIZE[0]
img_width = IMAGE_SIZE[1]
batch_size = BATCH_SIZE

# Create datasets
train_datasetD1 = create_dataset_with_class_augmentation_h5('dataset/h5/train_d1.h5', batch_size=BATCH_SIZE, shuffle=True, augmentation=True)
val_dataset = create_dataset_h5('dataset/h5/val.h5', batch_size=BATCH_SIZE, shuffle=False) 
train_datasetD2 = create_dataset_with_class_augmentation_h5('dataset/h5/train_d2.h5', batch_size=BATCH_SIZE, shuffle=True, augmentation=True)

# Define callbacks with monitor set to validation mean IoU
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_dsc_nobg', mode='max', patience=12, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dsc_nobg', mode='max', factor=0.1, patience=7, verbose=1, min_lr=1e-5),
    # lr_schedule,
    tf.keras.callbacks.ModelCheckpoint(f'paper_checkpoints/{model_save}_glomeruli_multiclass_pretrain.keras', save_best_only=True, save_weights_only=only_weights,
                                    monitor='val_dsc_nobg', mode='max', verbose=1)
]

# Set up mirrored strategy
strategy = tf.distribute.MirroredStrategy()

# Define models to train
model_name = DeeplabV3Plus_mod

# Train models and save performance summary
performance_summary = []


with strategy.scope():
    from methods.tversky_metric import OneHotTversky
    from methods.dice_metric import OneHotDice

    target_class_ids=[0, 1, 2]

    # IoU metric
    iou_metric = tf.keras.metrics.OneHotIoU(name='IoU', num_classes=NUM_CLASSES, target_class_ids=target_class_ids)  

    # Dice coefficient metric
    dice_metric = OneHotDice(num_classes=NUM_CLASSES, name='dsc', target_class_ids=target_class_ids)
    dice_nobg = OneHotDice(num_classes=NUM_CLASSES, name='dsc_nobg', target_class_ids=[1, 2])
    dice_1 = OneHotDice(num_classes=NUM_CLASSES, name='dsc1', target_class_ids= [1])
    dice_2 = OneHotDice(num_classes=NUM_CLASSES, name='dsc2', target_class_ids= [2])
    # Tversky metric
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
    # from keras_deeplab_model import unet
    # loss = tf.keras.losses.CategoricalCrossentropy()

    # from keras_unet_model import unet
    # model = unet((512, 512, 3), NUM_CLASSES)  

    # from keras_segnet import segnet
    # model = segnet((512, 512, 3), NUM_CLASSES)

    model = model_name(image_size=IMAGE_SIZE[0], num_classes=NUM_CLASSES, backbone='xception', weights='imagenet')

    model.summary()

    # model = DeeplabV3Plus(image_size=IMAGE_SIZE[0], num_classes=NUM_CLASSES)  #  #  unet(input_shape=(512, 512, 3), num_classes=NUM_CLASSES)  #    
    initial_lr = 1e-4
    adam_opt = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    sgdm_opt = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)  # Altini et al. used SGD with momentum=0.9 and initial lr of 1e-3
    rmsp_opt = tf.keras.optimizers.RMSprop(learning_rate=initial_lr)
    opt = adam_opt

    if pretrain:
        model.compile(optimizer=opt,
                        loss=cftl,
                        metrics=[iou_metric, dice_metric, dice_nobg, dice_1, dice_2])

        # Train model
        print(f"Pretraining with D2: {model_name}...")
        if val:
            history = model.fit(train_datasetD2, epochs=30, 
                                validation_data=val_dataset, 
                                callbacks=callbacks)
        else:
            history = model.fit(train_datasetD2, epochs=20)
            model.save(f'paper_checkpoints/{model_save}_glomeruli_multiclass_pretrain.keras')


        # Save history to plot later
        with open(f'paper_checkpoints/training_history_{model_save}_pretrain.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        
        # Save best epoch loss and metrics
        best_epoch = history.history['val_IoU'].index(max(history.history['val_IoU'])) 
        best_metrics = {
            'epoch': best_epoch + 1,
            'train_loss': history.history['loss'][best_epoch],
            'train_iou': history.history['IoU'][best_epoch],
            'train_dice': history.history['dsc'][best_epoch],
            'val_loss': history.history['val_loss'][best_epoch],
            'val_iou': history.history['val_IoU'][best_epoch],
            'val_dice': history.history['val_dsc'][best_epoch]
        }
        performance_summary.append((model_name, best_metrics))

    else:
        # Cargar el modelo completo guardado
        model = tf.keras.models.load_model(f"paper_checkpoints/{model_save}_glomeruli_multiclass_pretrain.keras", compile=False) 

    # # Inicializar el contador de capas
    # total_layers = 0

    # # Hacer todas las capas del modelo entrenables
    # for layer in model.layers:
    #     layer.trainable = True
    #     total_layers += 1

    # # Imprimir el número total de capas
    # print(f"Total de capas entrenables: {total_layers}")
    
    model.compile(optimizer=opt,
            loss=cftl,
            metrics=[iou_metric, dice_metric, dice_nobg, dice_1, dice_2])
    
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_dsc_nobg', mode='max', patience=13, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dsc_nobg', mode='max', factor=0.1, min_lr=1e-7, patience=6, verbose=1),
        # lr_schedule,
        tf.keras.callbacks.ModelCheckpoint(f'paper_checkpoints/{model_save}_glomeruli_multiclass_finetunned.keras', save_best_only=True, save_weights_only=only_weights,
                                        monitor='val_dsc_nobg', mode='max', verbose=1)
    ]

    print(f"Fine-tunning with D1: {model_name}...")
    if val:
        history = model.fit(train_datasetD1, epochs=30, 
                            validation_data=val_dataset, 
                            callbacks=callbacks)
    else:
        history = model.fit(train_datasetD1, epochs=20)
        model.save(f'paper_checkpoints/{model_save}_glomeruli_multiclass_finetunned.keras')
    
    # Save history to plot later 
    with open(f'paper_checkpoints/training_history_{model_save}_finetunned.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Save best epoch loss and metrics
    best_epoch = history.history['val_IoU'].index(max(history.history['val_IoU']))
    best_metrics = {
        'epoch': best_epoch + 1,
        'train_loss': history.history['loss'][best_epoch],
        'train_iou': history.history['IoU'][best_epoch],
        'train_dice': history.history['dsc'][best_epoch],
        'val_loss': history.history['val_loss'][best_epoch],
        'val_iou': history.history['val_IoU'][best_epoch],
        'val_dice': history.history['val_dsc'][best_epoch]
    }
    performance_summary.append((model_name, best_metrics))

    # # Last training with combined dataset

    # with tf.device('/CPU:0'):
    #     xception = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # # Inicializar contadores
    # frozen_layers = 0
    # unfrozen_layers = 0

    # # Congelar solo las capas que corresponden a Xception
    # for layer in model.layers:
    #     # Verificar si el nombre de la capa está en el modelo Xception cargado en la CPU
    #     if any(layer.name == x_layer.name for x_layer in xception.layers):
    #         layer.trainable = False
    #         frozen_layers += 1
    #     else:
    #         layer.trainable = True
    #         unfrozen_layers += 1

    # # Imprimir el recuento de capas congeladas y no congeladas
    # print(f"Total de capas congeladas: {frozen_layers}")
    # print(f"Total de capas no congeladas: {unfrozen_layers}")

    # model.load_weights('paper_checkpoints/deeplab_glomeruli_multiclass_finetunned_check.keras')
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    #         loss=cfce,
    #         metrics=[iou_metric, dice_metric, dice_nobg, dice_1, dice_2])

    # callbacks = [
    #     tf.keras.callbacks.EarlyStopping(monitor='val_dsc_nobg', mode='max', patience=15, restore_best_weights=True, verbose=1),
    #     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dsc_nobg', mode='max', factor=0.1, min_lr=1e-7, patience=8, verbose=1),
    #     # lr_schedule,
    #     tf.keras.callbacks.ModelCheckpoint(f'paper_checkpoints/deeplab_glomeruli_multiclass_combined_check.keras', save_best_only=True, save_weights_only=False,
    #                                     monitor='val_dsc_nobg', mode='max', verbose=1)
    # ]

    # print(f"Fine-tunning with class 2 {model_name}...")
    # history = model.fit(train_dataset_combined, epochs=35, 
    #                     validation_data=val_dataset, 
    #                     callbacks=callbacks)

# Print performance summary
print("\nPerformance Summary:")
for model_name, metrics in performance_summary:
    print(f"Model: {model_name}, Best Epoch: {metrics['epoch']}")
    print(f"Train Loss: {metrics['train_loss']:.4f}, Train IoU: {metrics['train_iou']:.4f}")
    print(f"Val Loss: {metrics['val_loss']:.4f}, Val IoU: {metrics['val_iou']:.4f}")
    print(f"Train Dice Coefficient: {metrics['train_dice']:.4f}, Val Dice Coefficient: {metrics['val_dice']:.4f}")

