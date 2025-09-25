import os
import re
import numpy as np
from PIL import Image
from skimage.measure import label as sk_label, regionprops
import scipy.io


def create_masks(label_intensity_dict_path, mat_files_path, masks_path, save_path):
    # Check if the label intensity dictionary file exists
    if not os.path.exists(label_intensity_dict_path):
        print(f"File not found: {label_intensity_dict_path}")
        return

    # Load the label to intensity dictionary
    with open(label_intensity_dict_path, 'r') as f:
        label_to_intensity = eval(f.read())

    label_counts = {}

    # Iterate over the masks
    for mask_file_name in os.listdir(masks_path):
        print(f"Processing mask: {mask_file_name}")
        # Load the mask
        mask_path = os.path.join(masks_path, mask_file_name)
        if not os.path.exists(mask_path):
            print(f"File not found: {mask_path}")
            continue
        mask_image = Image.open(mask_path)

        # Parse the mask file name to get the coordinates of its point 0 in the original image
        match = re.search(r'_x(\d+)y(\d+)s(\d+)\.png$', mask_file_name)
        mask_x0 = int(match.group(1))
        mask_y0 = int(match.group(2))

        # Find the corresponding .mat file
        mat_file_name = mask_file_name.split('_')[0] + '.mat'
        mat_file_path = os.path.join(mat_files_path, mat_file_name)

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

        # Convert the PIL Image to a numpy array and binarize it
        mask_image_array = np.array(mask_image)
        mask_image_binary = mask_image_array > 0

        # Label each independent region in the binary image
        label_image = sk_label(mask_image_binary)

        # Initialize an empty list to store the centroids
        centroids = []

        # Iterate over the regions
        for region in regionprops(label_image):
            # Get the centroid of the region
            centroid = region.centroid

            # Add the centroid to the list
            centroids.append(centroid)

        # convert the centroids in int
        centroids = np.array(centroids, dtype=int)

        # Initialize an empty list to store the centroids in the original image reference system
        centroids_orig = []

        # Iterate over the centroids
        for centroid in centroids:
            # Convert the centroid to the original image reference system and add it to the list
            centroids_orig.append([centroid[0] + mask_x0, centroid[1] + mask_y0])

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

        # Return to modified reference system conserving the dictionary
        centroids_labels_mask = {}

        for centroid, label in centroids_labels.items():
            centroids_labels_mask[tuple(np.array(centroid) - [mask_x0, mask_y0])] = label

        # Change the region intensity depending on the label
        mask_image_array = np.array(mask_image)

        for centroid, label in centroids_labels_mask.items():
            if label not in label_to_intensity:
                print(f"Label not found in the dictionary: {label}")
                continue
            if label in label_to_intensity:
                # Convert the centroid to the mask reference system
                centroid_mask = tuple(centroid)
                # Look for the region with that coordinates inside
                for region in regionprops(label_image):
                    # Check if the centroid is inside the region
                    if centroid_mask in region.coords:
                        # Change the intensity of the region
                        mask_image_array[label_image == region.label] = label_to_intensity[label]
                        # Count the labels
                        if label not in label_counts:
                            label_counts[label] = 1
                        else:
                            label_counts[label] += 1

        # Create a file in the save path and save the image in it, overwriting if necessary
        # save_file_path = os.path.join(save_path, mask_file_name)
        # Image.fromarray(mask_image_array).save(save_file_path)

    print('Label counts: ', label_counts)


# Example usage
label_intensity_dict_path = r'C:\uji\glomeruli\dataset/label_intensity_dict.txt'
mat_files_path = r'C:\uji\glomeruli\dataset\mats\structureFiles'
masks_path = r'C:\uji\glomeruli\dataset\HE\groundtruth'
save_path = r'C:\uji\glomeruli\dataset\HE\groundtruth_multiclass'

create_masks(label_intensity_dict_path, mat_files_path, masks_path, save_path)
