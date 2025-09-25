import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def normalize_image(image):
    """Normaliza una imagen según sus valores mínimos y máximos."""
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def process_image(image_path):
    """Carga, redimensiona, normaliza y aplica CLAHE a una imagen."""
    # Cargar la imagen con OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Verificar si la imagen se cargó correctamente
    if image is None:
        print(f"Advertencia: No se pudo cargar la imagen {image_path}.")
        return None
    
    # Redimensionar la imagen a 1024x1024
    image_resized = cv2.resize(image, (1024, 1024))
    
    # Verificar si la imagen es de un solo canal (máscara)
    if len(image_resized.shape) == 2 or image_resized.shape[2] == 1:
        # Binarizar la imagen
        image_binarized = np.where(image_resized > 0, 255, 0).astype(np.uint8)
        return image_binarized
    
    # Normalizar entre 0 y 1
    image_normalized = normalize_image(image_resized)
    
    # Convertir la imagen de BGR a LAB
    image_lab = cv2.cvtColor((image_normalized * 255).astype(np.uint8), cv2.COLOR_BGR2Lab)
    
    # Aplicar CLAHE solo al canal L
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l_channel, a_channel, b_channel = cv2.split(image_lab)
    l_channel_clahe = clahe.apply(l_channel)
    
    # Unir los canales y convertir de nuevo a BGR
    image_clahe_lab = cv2.merge((l_channel_clahe, a_channel, b_channel))
    image_clahe_rgb = cv2.cvtColor(image_clahe_lab, cv2.COLOR_Lab2BGR)

    # Normalizar a rango [0, 255]
    image_clahe_rgb = np.clip(image_clahe_rgb, 0, 255)
    
    return image_clahe_rgb.astype(np.uint8)

def plot_image_and_histogram(image, image_name):
    """Muestra una imagen y su histograma, junto con su nombre de archivo."""
    min_val = np.min(image)
    max_val = np.max(image)
    dtype = image.dtype
    
    # Mostrar la imagen
    plt.figure(figsize=(12, 6))
    
    # Subplot para la imagen
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image, cmap='gray')
    plt.title(f'Imagen: {image_name}')
    plt.axis('off')
    
    # Subplot para el histograma
    plt.subplot(1, 2, 2)
    plt.hist(image.ravel(), bins=256, color='blue', alpha=0.7, rwidth=0.85)
    plt.title('Histograma de la Imagen')
    plt.xlabel('Intensidad de píxeles')
    plt.ylabel('Frecuencia')
    
    # Mostrar información adicional
    plt.suptitle(f'Mín: {min_val}, Máx: {max_val}, Dtype: {dtype}')
    
    # Mostrar el gráfico
    plt.show()

def process_directories(directories):
    """Procesa todas las imágenes en los directorios y guarda el resultado en un diccionario."""
    images_dict = {}
    
    for directory_path in directories:
        # Listar los archivos de imagen en el directorio
        image_files = [f for f in os.listdir(directory_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Iterar sobre las imágenes con tqdm para mostrar progreso
        for filename in tqdm(image_files, desc=f"Procesando imágenes en {directory_path}"):
            image_path = os.path.join(directory_path, filename)
            processed_image = process_image(image_path)
            
            if processed_image is not None:
                name_without_extension = os.path.splitext(filename)[0]
                images_dict[name_without_extension] = processed_image
            else:
                print(f"Saltando {image_path} debido a problemas de carga.")
    
    # Convertir el diccionario a un archivo npy si hay imágenes procesadas
    if images_dict:
        np.save('/scratch.local3/juanp/dataset/masks_1024_dict.npy', images_dict)
        print(f"Conjunto de imágenes guardado en 'images_1024_clahe_dict.npy'")
        
        # Elegir una imagen aleatoria para mostrar
        random_key = random.choice(list(images_dict.keys()))
        random_image = images_dict[random_key]
        plot_image_and_histogram(random_image, random_key)
    else:
        print("No se procesaron imágenes válidas.")

# Tupla de rutas a los directorios donde están almacenadas las imágenes
directories = (
    '/scratch.local3/juanp/dataset/HE/groundtruth',
    '/scratch.local3/juanp/dataset/PAS3/groundtruth',
    '/scratch.local3/juanp/dataset/PM3/groundtruth'
)

# Procesar los directorios y guardar las imágenes con nombres en un archivo npy
process_directories(directories)