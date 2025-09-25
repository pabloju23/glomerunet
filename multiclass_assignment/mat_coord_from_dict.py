import os
import numpy as np
import scipy.io
from skimage.measure import label as sk_label, regionprops

def create_masks(label_intensity_dict_path, mat_files_path, masks_dict_path, images_dict_path, save_masks_path, save_images_path):
    # Check if the label intensity dictionary file exists
    if not os.path.exists(label_intensity_dict_path):
        print(f"File not found: {label_intensity_dict_path}")
        return

    # Load the label to intensity dictionary
    with open(label_intensity_dict_path, 'r') as f:
        label_to_intensity = eval(f.read())

    # Load the masks dictionary from .npy file
    masks_dict = np.load(masks_dict_path, allow_pickle=True).item()

    # Load the images dictionary from .npy file
    images_dict = np.load(images_dict_path, allow_pickle=True).item()

    label_counts = {}
    processed_masks = []  # To store all processed masks
    processed_images = []  # To store all corresponding images
    processed_indices = []  # To store indices of processed masks

    # Iterate over the masks in the dictionary
    for idx, (mask_file_name, mask_image_array) in enumerate(masks_dict.items()):
        # print(f"Processing mask: {mask_file_name}")

        # Redimensioned mask is 1024x1024, original images were 3200x3200
        scale_factor = 3200 / 1024
        
        # Find the corresponding .mat file
        mat_file_name = mask_file_name.split('_')[0] + '.mat'
        mat_file_path = os.path.join(mat_files_path, mat_file_name)

        if not os.path.exists(mat_file_path):
            # print(f"MAT file not found: {mat_file_path}")
            continue

        # Load the .mat file
        mat_file = scipy.io.loadmat(mat_file_path)

        # Access the 'LabelName' and 'LabelPosition' fields
        variable = mat_file['Annotations']
        label_names = variable['LabelName'][0, 0]
        label_positions = variable['LabelPosition'][0, 0]

        # Extract the label names and positions
        label_names_list = [label[0].lower() for sublist in label_names for label in sublist]
        label_positions_list = [coord_array.tolist() for sublist in label_positions for coord_array in sublist]

        # Create a dictionary with label names as keys and positions as values
        label_positions_dict = dict(zip(label_names_list, label_positions_list))

        # Label each independent region in the binary image
        mask_image_binary = mask_image_array > 0
        mask_image_binary = np.squeeze(mask_image_binary)
        label_image = sk_label(mask_image_binary)

        centroids = []

        # Get the centroids of regions
        for region in regionprops(label_image):
            centroid = region.centroid
            centroids.append(centroid)

        # Adjust centroids to match the original 3200x3200 coordinates
        centroids_orig = np.array(centroids) * scale_factor

        # Assign the closest label to each centroid
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

        # Map back the centroids to the 1024x1024 mask
        centroids_labels_mask = {tuple(np.array(centroid) / scale_factor): label 
                                 for centroid, label in centroids_labels.items()}

        # Assign intensities based on labels
        for centroid, label in centroids_labels_mask.items():
            if label not in label_to_intensity:
                print(f"Label not found in dictionary: {label}")
                continue
            # Convert centroid to integer for use as indices
            centroid_int = tuple(np.round(centroid).astype(int))
            for region in regionprops(label_image):
                if centroid_int in region.coords:
                    mask_image_array[label_image == region.label] = label_to_intensity[label]

                    # Count the labels
                    if label not in label_counts:
                        label_counts[label] = 1
                    else:
                        label_counts[label] += 1

        # Check for any values outside the range [0, 12]
        if np.any((mask_image_array < 0) | (mask_image_array > 15)):
            print(f"Warning: Mask {mask_file_name} has values outside the range [0, 15]")

        processed_masks.append(mask_image_array.astype(np.uint8))
        processed_images.append(images_dict[mask_file_name])
        processed_indices.append(idx)

    # Save the processed masks and images as numpy files
    processed_masks_np = np.array(processed_masks)
    processed_images_np = np.array(processed_images)
    print(f"Processed masks shape: {processed_masks_np.shape}")
    print(f"Processed images shape: {processed_images_np.shape}")
    np.save(save_masks_path, processed_masks_np)
    np.save(save_images_path, processed_images_np)
    print(f"Processed masks saved to {save_masks_path}")
    print(f"Processed images saved to {save_images_path}")

# Ejemplo de uso
npy_path = '/scratch.local3/juanp/dataset/new_masks_1024_predictions_dict_mod.npy'
label_intensity_dict_path = '/scratch.local3/juanp/dataset/label_intensity_dict.txt'
save_masks_path = '/scratch.local3/juanp/dataset/new_masks_1024_multiclass_v2.npy'
save_images_path = '/scratch.local3/juanp/dataset/new_images_1024_multiclass_v2.npy'
mat_files_path = '/scratch.local3/juanp/dataset/mats/structureFiles'
images_dict_path = '/scratch.local3/juanp/dataset/new_images_1024_clahe_mod.npy'

create_masks(label_intensity_dict_path, mat_files_path, npy_path, images_dict_path, save_masks_path, save_images_path)