import os
import numpy as np
import cv2
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import re

# Mapeo de etiquetas
label_mapping = {
    0: 0,  # Background
    1: 1,  # sano
    2: 1,  # hipercelular mes
    3: 1,  # mixto
    4: 1,  # isquémico
    5: 1,  # gssf
    6: 1,  # membranoso
    7: 1,  # incompleto
    8: 1,  # gnmp
    9: 1,  # semilunas
    10: 2,  # esclerosado
    11: 1,  # endocapilar
    12: 0,  # patológico
    13: 1,  # proliferativo
    14: 0   # confusión
}

def extract_stain_from_id(id_str):
    if "HE" in id_str:
        return "HE"
    elif "PAS" in id_str:
        return "PAS"
    elif "PM" in id_str:
        return "PM"
    return "UNKNOWN"

def load_dataset(image_dir, mask_dir):
    dataset = {}
    for mask_filename in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_filename)
        image_path = os.path.join(image_dir, mask_filename)
        
        id_ = mask_filename.split('_')[0]
        stain_type = extract_stain_from_id(id_)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        print(f"Antes de mapear: valores únicos en la máscara: {np.unique(mask)}")
        # Mapeo de etiquetas
        mapped_mask = np.vectorize(label_mapping.get)(mask)
        
        # Verificar las clases en la máscara mapeada
        print(f"ID: {id_}, Stain: {stain_type}, Mask shape: {mask.shape}, Unique values: {np.unique(mapped_mask)}, Mapped mask shape: {mapped_mask.shape}")
        print(f"Proporción de clase 2 procesada: {np.sum(mapped_mask == 2) / mask.size}")
        
        # Asegurarse de que la máscara mapeada tenga las clases correctas
        print(f"Valores únicos después del mapeo: {np.unique(mapped_mask)}")
        
        image = cv2.imread(image_path)
        
        dataset[mask_filename] = {
            "id": id_,
            "type": stain_type,
            "mask": mapped_mask,
            "image": image
        }
    return dataset

def split_dataset_optimized(dataset, train_ratio=0.4, val_ratio=0.3, test_ratio=0.3, n_repeats=10):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Las proporciones deben sumar 1."
    
    id_groups = defaultdict(list)
    for filename, data in dataset.items():
        id_groups[data["id"]].append(data)
    
    ids = list(id_groups.keys())
    best_split = None
    best_score = float('inf')
    
    for _ in range(n_repeats):
        train_ids, temp_ids = train_test_split(ids, test_size=(val_ratio + test_ratio), random_state=None)
        val_ids, test_ids = train_test_split(temp_ids, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=None)
        
        splits = {
            "train": [data for id_ in train_ids for data in id_groups[id_]],
            "val": [data for id_ in val_ids for data in id_groups[id_]],
            "test": [data for id_ in test_ids for data in id_groups[id_]]
        }
        
        score = evaluate_split_balance(splits)
        if score < best_score:
            best_score = score
            best_split = splits
    
    return best_split["train"], best_split["val"], best_split["test"]

from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np

def split_dataset_into_k_folds(dataset, k=5, n_repeats=10):
    """
    Divide un dataset en K subconjuntos optimizados (folds) numerados del 1 al K.

    Args:
        dataset (dict): Diccionario con los datos del dataset. Las claves son nombres de archivo, 
                        y los valores son diccionarios con información de las muestras, incluyendo "id".
        k (int): Número de subconjuntos (folds) a crear.
        n_repeats (int): Número de intentos para encontrar la mejor división posible.

    Returns:
        dict: Diccionario con K claves (1, 2, ..., K) que contienen los respectivos subconjuntos.
    """
    # Agrupar muestras por IDs
    id_groups = defaultdict(list)
    for filename, data in dataset.items():
        id_groups[data["id"]].append(data)
    
    ids = list(id_groups.keys())
    best_split = None
    best_score = float('inf')

    for _ in range(n_repeats):
        # Mezclar IDs y dividir en K subconjuntos
        np.random.shuffle(ids)
        fold_ids = [ids[i::k] for i in range(k)]
        
        # Crear los subconjuntos
        splits = {
            i + 1: [data for id_ in fold for data in id_groups[id_]] for i, fold in enumerate(fold_ids)
        }
        
        # Evaluar el balance entre subconjuntos
        score = evaluate_split_balance(splits)
        if score < best_score:
            best_score = score
            best_split = splits

    return best_split

def evaluate_split_balance(splits):
    """
    Evalúa el balance entre los K subconjuntos calculando la desviación estándar del tamaño.

    Args:
        splits (dict): Diccionario de subconjuntos.

    Returns:
        float: Puntuación del balance, menor es mejor.
    """
    sizes = [len(data) for data in splits.values()]
    return np.std(sizes)

def evaluate_split_balance(splits):
    """Calcula un score de desequilibrio basado en las proporciones de clases 1 y 2."""
    total_pixels = defaultdict(int)
    class_distributions = {}
    
    for partition_name, partition_data in splits.items():
        class_counts = np.zeros(3)
        for data in partition_data:
            print(f"Valores únicos en la máscara en la partición {partition_name}: {np.unique(data['mask'])}")
            class_counts += np.bincount(data["mask"].flatten(), minlength=3)
        
        # Solo considerar clases 1 y 2
        relevant_counts = class_counts[1:]  # Ignorar la clase 0
        total_relevant_pixels = np.sum(relevant_counts)
        
        if total_relevant_pixels > 0:
            class_distributions[partition_name] = relevant_counts / total_relevant_pixels
        else:
            class_distributions[partition_name] = np.zeros(2)  # Si no hay píxeles de clases 1 y 2, proporción es 0
    
    # Evaluar balance: calcular la desviación estándar entre proporciones de cada clase
    all_distributions = np.array(list(class_distributions.values()))
    return np.std(all_distributions, axis=0).sum()


def analyze_split(partition, name="Partition"):
    stain_counts = Counter()
    class_counts = np.zeros(3)
    unique_ids = set()
    
    for data in partition:
        stain_counts[data["type"]] += 1
        unique_ids.add(data["id"])
        print(f"Valores únicos en la máscara en {name}: {np.unique(data['mask'])}")
        class_counts += np.bincount(data["mask"].flatten(), minlength=3)
    
    total_pixels = np.sum(class_counts)
    class_percentages = (class_counts / total_pixels) * 100
    
    print(f"{name}:")
    print(f"  Total de IDs únicos: {len(unique_ids)}")
    print("  Tinciones:")
    for stain, count in stain_counts.items():
        id_count = len({data["id"] for data in partition if data["type"] == stain})
        print(f"    {stain}: {count} ({id_count} biopsias únicas)")
    print("  Clases:")
    for i, (count, percentage) in enumerate(zip(class_counts, class_percentages)):
        print(f"    Clase {i}: {count} píxeles ({percentage:.2f}%)")

def save_partitions(partitions, base_dir):
    for partition_name, partition_data in partitions.items():
        # Contar cuántas imágenes por biopsia para evitar sobrescritura
        id_counts = defaultdict(int)  # Contador por biopsia
        
        for data in partition_data:
            stain_dir = os.path.join(base_dir, partition_name, data["type"])
            tissue_dir = os.path.join(stain_dir, "tissue")
            mask_dir = os.path.join(stain_dir, "groundtruth_multiclass")
            
            os.makedirs(tissue_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            
            # Contar el número de imágenes para un mismo id
            id_counts[data["id"]] += 1
            
            # Crear un nombre único usando id+index
            tissue_path = os.path.join(tissue_dir, f"{data['id']}_{id_counts[data['id']]}.png")
            mask_path = os.path.join(mask_dir, f"{data['id']}_{id_counts[data['id']]}.png")
            
            # Verificar valores únicos antes de guardar
            print(f"Valores únicos en la máscara antes de guardar: {np.unique(data['mask'])}")
            
            # Guardar la imagen y la máscara con el nuevo nombre único
            cv2.imwrite(tissue_path, data["image"])
            
            # Asegurarse de que la máscara se guarda correctamente
            cv2.imwrite(mask_path, data["mask"])
            
            # Verificar valores únicos después de guardar
            saved_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            print(f"Valores únicos en la máscara guardada: {np.unique(saved_mask)}")

def save_folds_partitions(folds, base_dir):
    """
    Guarda las particiones K-Fold en una estructura organizada de carpetas.

    Args:
        folds (dict): Diccionario donde cada clave es el número del fold y el valor una lista de datos.
        base_dir (str): Directorio base donde guardar las particiones.
    """
    for fold_num, fold_data in folds.items():
        # Crear directorio para cada fold
        fold_dir = os.path.join(base_dir, f"fold_{fold_num}")
        
        id_counts = defaultdict(int)  # Contador por biopsia
        
        for data in fold_data:
            # Crear estructura de carpetas para tissue y máscaras
            stain_dir = os.path.join(fold_dir, data["type"])
            tissue_dir = os.path.join(stain_dir, "tissue")
            mask_dir = os.path.join(stain_dir, "groundtruth_multiclass")
            
            os.makedirs(tissue_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            
            # Contar el número de imágenes para un mismo id
            id_counts[data["id"]] += 1
            
            # Crear un nombre único usando id+index
            tissue_path = os.path.join(tissue_dir, f"{data['id']}_{id_counts[data['id']]}.png")
            mask_path = os.path.join(mask_dir, f"{data['id']}_{id_counts[data['id']]}.png")
            
            # Verificar valores únicos antes de guardar
            print(f"Fold {fold_num} - Valores únicos en la máscara antes de guardar: {np.unique(data['mask'])}")
            
            # Guardar la imagen y la máscara con el nuevo nombre único
            cv2.imwrite(tissue_path, data["image"])
            cv2.imwrite(mask_path, data["mask"])
            
            # Verificar valores únicos después de guardar
            saved_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            print(f"Fold {fold_num} - Valores únicos en la máscara guardada: {np.unique(saved_mask)}")



# Ejemplo de uso
image_dir = "/scratch.local/juanp/glomeruli/dataset/D1/tissue"
mask_dir = "/scratch.local/juanp/glomeruli/dataset/D1/groundtruth_multiclass"
output_dir = "/scratch.local/juanp/glomeruli/dataset/processed(5fold)"

kfold = True
k = 5

def check_original_labels(mask_dir):
    unique_labels = set()
    for mask_filename in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        unique_labels.update(np.unique(mask))
    print(f"Etiquetas únicas en las máscaras originales: {sorted(unique_labels)}")

check_original_labels(mask_dir)


dataset = load_dataset(image_dir, mask_dir)

if kfold:
    folds = split_dataset_into_k_folds(dataset, k=k)
    print('Folds shape:', folds.keys())
    print("\n--- Análisis de particiones K-Fold ---")
    for fold_num, fold_data in folds.items():
        analyze_split(fold_data, f"Fold {fold_num}")

    save_data = input("\n¿Desea guardar las particiones en disco? (yes/no): ").strip().lower()
    if save_data == "yes":
        print("Guardando particiones en disco...")
        save_folds_partitions(folds, output_dir)
        print("Particiones guardadas correctamente.")
    else:
        print("No se guardaron las particiones.")
else:
    train, val, test = split_dataset_optimized(dataset)

    print("\n--- Análisis de particiones ---")
    analyze_split(train, "Train")
    analyze_split(val, "Validation")
    analyze_split(test, "Test")

    save_data = input("\n¿Desea guardar las particiones en disco? (yes/no): ").strip().lower()
    if save_data == "yes":
        print("Guardando particiones en disco...")
        save_partitions({"train": train, "val": val, "test": test}, output_dir)
        print("Particiones guardadas correctamente.")
    else:
        print("No se guardaron las particiones.")

