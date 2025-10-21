import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # usa solo una GPU para probar

import tensorflow as tf
import numpy as np
from keras_unetr_2d import build_unetr_2d

# Configuración mínima
IMG_SIZE = 512
NUM_CLASSES = 3
BATCH_SIZE = 2

# Crear un modelo UNETR_2D con configuración básica
config = {}
config["image_size"] = 512
config["num_classes"] = 3
config["num_layers"] = 12
config["hidden_dim"] = 128
config["mlp_dim"] = 256
config["num_heads"] = 8
config["dropout_rate"] = 0.1
config["patch_size"] = 16
config["num_patches"] = (config["image_size"]**2)//(config["patch_size"]**2)
config["num_channels"] = 3

model = build_unetr_2d(config)
model.summary()

# Print input and output shapes
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# Dummy data (imagenes y máscaras one-hot)
x_dummy = np.random.rand(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
y_dummy = tf.keras.utils.to_categorical(
    np.random.randint(0, NUM_CLASSES, (BATCH_SIZE, IMG_SIZE, IMG_SIZE)), NUM_CLASSES
).astype(np.float32)

# Compilar modelo con loss y métricas simples
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Entrenar una época dummy
history = model.fit(x_dummy, y_dummy, epochs=1, batch_size=BATCH_SIZE)

print("✅ Entrenamiento dummy completado (modelo carga y empieza la primera época).")
