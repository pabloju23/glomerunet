from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Ruta a la imagen histológica en formato PNG
image_path = r'C:/uji/glomeruli/dataset/HE/tissue/20B0011364 A 1 HE_x7200y14400s3200.png'

# Abrir la imagen utilizando PIL
image = Image.open(image_path)

# Convertir la imagen a escala de grises
gray_image = image.convert('L')

# Convertir la imagen a un array numpy
image_array = np.array(gray_image)

# Normalizar la imagen al rango de 0 a 255
image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255

# Definir un umbral para considerar un píxel como blanco (por ejemplo, 240 en una escala de 0 a 255)
white_threshold = 225

# Contar el número de píxeles que son mayores o iguales al umbral de blancura
num_white_pixels = np.sum(image_array >= white_threshold)

# Calcular el porcentaje de píxeles blancos
total_pixels = image_array.size
white_percentage = (num_white_pixels / total_pixels) * 100

# Imprimir el resultado
print(f'Porcentaje de píxeles blancos (fondo): {white_percentage:.2f}%')

# Visualizar la imagen y los píxeles blancos para ver el resultado

# Crear una máscara para los píxeles blancos
white_mask = image_array >= white_threshold

# Mostrar la imagen original y la máscara de píxeles blancos
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Imagen original')
axes[0].axis('off')

axes[1].imshow(white_mask, cmap='gray')
axes[1].set_title('Píxeles blancos')
axes[1].axis('off')

plt.show()
