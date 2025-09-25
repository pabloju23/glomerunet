import scipy.io
import numpy as np
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.measure import label as sk_label


# Load the image from the patch file
mat_file_path = r'C:\uji\glomeruli\dataset\repetits\1\04B0006786 A 1 PAS.mat'
patch_file_name = r'C:\uji\glomeruli\dataset\PAS3\groundtruth/04B0006786 A 1 PAS_x4800y7200s3200.png'
patch_image = Image.open(patch_file_name)

# Parse the patch file name to get the coordinates of its point 0 in the original image
match = re.search(r'_x(\d+)y(\d+)s(\d+)\.png$', patch_file_name)
patch_x0 = int(match.group(1))
patch_y0 = int(match.group(2))

# Load the .mat file
mat_file = scipy.io.loadmat(mat_file_path)

# Access the 'LabelName' and 'LabelPosition' fields
variable = mat_file['Annotations']
label_names = variable['LabelName'][0, 0]
label_positions = variable['LabelPosition'][0, 0]

# Extract the label names and convert them to lowercase
label_names_list = [label[0].lower() for sublist in label_names for label in sublist]

# Extract the label positions
label_positions_list = [coord_array.tolist() for sublist in label_positions for coord_array in sublist]

# Create a dictionary with the label names as keys and the label positions as values
label_positions_dict = dict(zip(label_names_list, label_positions_list))

print(f'Patch coordinates: ({patch_x0}, {patch_y0})')
print('Label names and positions: ', label_positions_dict)

# Initialize an empty dictionary to store the positions in the patch reference system
label_positions_patch_dict = {}

# Define the patch size
patch_size = 3200

# Iterate over the labels and their positions
for label, positions in label_positions_dict.items():
    # Initialize an empty list to store the positions for this label that fall within the patch
    positions_in_patch = []

    # Iterate over the positions
    for position in positions:
        # Check if the position falls within the patch
        if patch_x0 <= position[0] < patch_x0 + patch_size and patch_y0 <= position[1] < patch_y0 + patch_size:
            # Convert the position to the patch reference system and add it to the list
            positions_in_patch.append([position[0] - patch_x0, position[1] - patch_y0])

    # If there are any positions for this label that fall within the patch, add them to the dictionary
    if positions_in_patch:
        label_positions_patch_dict[label] = positions_in_patch

print('Label names and positions in patch reference system: ', label_positions_patch_dict)

# Convert the PIL Image to a numpy array
patch_image_array = np.array(patch_image)

# Create a new figure
plt.figure()

# Display the image
plt.imshow(patch_image_array)

# Iterate over the labels and their positions
for label, positions in label_positions_patch_dict.items():
    # Iterate over the positions
    for position in positions:
        # Plot a point at the position
        plt.plot(position[0], position[1], 'ro')

# Show the plot
plt.show()


# Convert the PIL Image to a numpy array and binarize it
patch_image_array = np.array(patch_image)
patch_image_binary = patch_image_array > 0

# Label each independent region in the binary image
label_image = sk_label(patch_image_binary)

# Initialize an empty list to store the centroids
centroids = []

# Iterate over the regions
for region in regionprops(label_image):
    # Get the centroid of the region
    centroid = region.centroid

    # Add the centroid to the list
    centroids.append(centroid)

print('Centroids: ', centroids)

# convert the centroids in int
centroids = np.array(centroids, dtype=int)

# Initialize an empty list to store the centroids in the original image reference system
centroids_orig = []

# Iterate over the centroids
for centroid in centroids:
    # Convert the centroid to the original image reference system and add it to the list
    centroids_orig.append([centroid[0] + patch_x0, centroid[1] + patch_y0])

print('Centroids in original image reference system: ', centroids_orig)

# In label_positions_dict find the closest point to each centroid
centroids_labels = {}

for centroid in centroids_orig:
    min_distance = float('inf')
    closest_label = None

    for label, positions in label_positions_dict.items():
        for position in positions:
            distance = np.linalg.norm(np.array(position) - centroid)

            if distance < min_distance:
                min_distance = distance
                closest_label = label

    centroids_labels[tuple(centroid)] = closest_label

print('Centroids and closest labels: ', centroids_labels)

# Plot the mask with the centroids and labels
plt.figure()
plt.imshow(patch_image_array)

for centroid, label in centroids_labels.items():
    plt.plot(centroid[1] - patch_y0, centroid[0] - patch_x0, 'ro')
    plt.text(centroid[1] - patch_y0, centroid[0] - patch_x0, label, color='r')

plt.show()

# Return to modified reference system conserving the dictionary
centroids_labels_patch = {}

for centroid, label in centroids_labels.items():
    centroids_labels_patch[tuple(np.array(centroid) - [patch_x0, patch_y0])] = label

print('Centroids and closest labels in patch reference system: ', centroids_labels_patch)

# Plot the mask with the centroids and labels
plt.figure()
plt.imshow(patch_image_array)

for centroid, label in centroids_labels_patch.items():
    plt.plot(centroid[1], centroid[0], 'ro')
    plt.text(centroid[1], centroid[0], label, color='r')

plt.show()

# Modify the instensity of the mask region in the area of the centroids
patch_image_array_modified = patch_image_array.copy()


# Convert the mask image to a binary image
patch_image_binary = patch_image_array > 0

# Label each independent region in the binary image
label_image = sk_label(patch_image_binary)

# Create a dictionary to map labels to intensities
label_to_intensity = {'esclerosado': 1}  # Add more labels and intensities as needed

# Change the region intensity depending on the label
patch_image_array = np.array(patch_image)

for centroid, label in centroids_labels_patch.items():
    if label in label_to_intensity:
        # Convert the centroid to the patch reference system
        centroid_patch = tuple(centroid)
        # Look for the region with that coordinates inside
        for region in regionprops(label_image):
            # Check if the centroid is inside the region
            if centroid_patch in region.coords:
                # Change the intensity of the region
                patch_image_array[label_image == region.label] = label_to_intensity[label]

# Create a new figure
plt.figure()

# Display the modified image
plt.imshow(patch_image_array)

# Show the plot
plt.show()
