import scipy.io
import numpy as np
import os

# List of folders with .mat files
folders = [r'C:\uji\glomeruli\dataset\mats\structureFiles']  # Add more paths as needed

# Dictionary to store the groups
groups = {}

for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.mat'):
            mat_file_path = os.path.join(folder, filename)

            # Load the .mat file
            mat_file = scipy.io.loadmat(mat_file_path)

            # Access the 'LabelName' field
            variable = mat_file['Annotations']
            label_names = variable['LabelName'][0, 0]
            label_count = variable['NumberOfPoints'][0, 0]

            # Convert the label names to a set for easy comparison
            label_names_set = set([str(label[0]) for label_array in label_names for label in label_array])

            # Convert the set to a frozenset so it can be used as a dictionary key
            label_names_frozenset = frozenset(label_names_set)

            # Add the file name to the appropriate group
            if label_names_frozenset in groups:
                groups[label_names_frozenset].append(os.path.basename(mat_file_path))
            else:
                groups[label_names_frozenset] = [os.path.basename(mat_file_path)]

# Save the groups to a file
# Open the output file
with open('output.txt', 'w') as f:
    # Write the groups and their corresponding file names to the file
    for i, (label_names, file_names) in enumerate(groups.items(), start=1):
        f.write(f'Group {i}:\n')
        f.write('Label names: ' + ', '.join(label_names) + '\n')
        f.write('File names: ' + ', '.join(file_names) + '\n\n')

print('Groups saved to output.txt')