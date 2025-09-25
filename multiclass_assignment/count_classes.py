import scipy.io
import numpy as np
import os

# List of folders with .mat files
folders = [r'C:\uji\glomeruli\dataset\mats\structureFiles']  # Add more paths as needed

# Create a dictionary to store the number exemplars for each label
label_counts = {}

for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith('.mat'):
            mat_file_path = os.path.join(folder, filename)

            # Load the .mat file
            mat_file = scipy.io.loadmat(mat_file_path)

            # Access the 'LabelName' and 'NumberOfPoints' fields
            variable = mat_file['Annotations']
            label_names = variable['LabelName'][0, 0]
            num_points = variable['NumberOfPoints'][0, 0]

            # Extract all the numbers from the numpy array and print them
            num_points_list = [int(point[0][0]) for point in num_points[0]]
            # It returns a list of numbers, one for each label like [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            print(f'NumberOfPoints for file {filename}: {num_points_list}')

            # Convert the label names to a set for easy comparison
            # It returns a set of label names like {'ENDOCAPILAR', 'SEMILUNAS', 'ISQUEMICO'}
            label_names_set = set([str(label[0]) for label_array in label_names for label in label_array])

            # Iterate over the label names and their corresponding number of points
            for label_name, num_points in zip(label_names_set, num_points_list):
                # If the label name is already in the dictionary, add the number of points to the existing count
                if label_name in label_counts:
                    label_counts[label_name] += num_points
                # Otherwise, create a new entry in the dictionary
                else:
                    label_counts[label_name] = num_points
            print(f'File: {filename}')
            print(f'Label names: {label_names_set}')
            print('\n')  # Print a newline for readability

# Print the label counts
print('Label counts:')
for label, count in label_counts.items():
    print(f'{label}: {count}')
# The output will be something like:
# Label counts:
# ENDOCAPILAR: 10
# SEMILUNAS: 20