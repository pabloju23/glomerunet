import scipy.io
import numpy as np

# Rutas a los archivos .mat
mat_file_path1 = r'C:\uji\glomeruli\dataset\repetits\1/04B0006786 A 1 HE.mat'
mat_file_path2 = r'C:\uji\glomeruli\dataset\repetits\2/04B0006786 A 1 HE.mat'

# Cargar los archivos .mat
mat_file1 = scipy.io.loadmat(mat_file_path1)
mat_file2 = scipy.io.loadmat(mat_file_path2)

# Acceder a las variables 'LabelName' y 'LabelPosition'
label_names1 = mat_file1['Annotations']['LabelName'][0, 0]
label_positions1 = mat_file1['Annotations']['LabelPosition'][0, 0]
label_names2 = mat_file2['Annotations']['LabelName'][0, 0]
label_positions2 = mat_file2['Annotations']['LabelPosition'][0, 0]

# Convertir las variables a conjuntos para una fácil comparación
label_names_set1 = set([str(label[0]) for label_array in label_names1 for label in label_array])
label_positions_set1 = set([str(coord) for label_array in label_positions1 for coord in label_array.flatten() if label_array.size > 0])
label_names_set2 = set([str(label[0]) for label_array in label_names2 for label in label_array])
label_positions_set2 = set([str(coord) for label_array in label_positions2 for coord in label_array.flatten() if label_array.size > 0])

# Comparar las variables
are_label_names_equal = label_names_set1 == label_names_set2
are_label_positions_equal = label_positions_set1 == label_positions_set2

print(f'LabelNames are equal: {are_label_names_equal}')
print(f'LabelPositions are equal: {are_label_positions_equal}')