import numpy as np
import tensorflow as tf

def preprocess_images_and_masks(images, masks, patch_size=512, stride=None, num_classes=None):
    """
    Preprocess image and mask data by:
    1. One-hot encoding masks (if num_classes > 1)
    2. Extracting overlapping patches
    3. Normalizing images
    4. Converting to float32 dtype
    
    Parameters:
    - images: Input images array (NxHxWxC)
    - masks: Input masks array (NxHxW or NxHxWxC)
    - patch_size: Size of patches to extract (default 512)
    - stride: Stride between patches (default is patch_size // 2)
    - num_classes: Number of classes for one-hot encoding (optional)
    
    Returns:
    - Processed image patches
    - Processed mask patches
    """
    # Set default stride to half patch size if not specified
    if stride is None:
        stride = patch_size // 2
    
    # One-hot encode masks if num_classes is specified
    if num_classes is not None and num_classes > 1:
        print('One-hot encoding masks...')
        if len(masks.shape) == 3:
            masks = tf.keras.utils.to_categorical(masks.squeeze(), num_classes=num_classes)
    
    # Extract patches
    def extract_patches(img, mask, patch_size, stride):
        patches_img = []
        patches_mask = []
        for i in range(0, img.shape[1] - patch_size + 1, stride):
            for j in range(0, img.shape[2] - patch_size + 1, stride):
                patch_img = img[:, i:i + patch_size, j:j + patch_size, :]
                patch_mask = mask[:, i:i + patch_size, j:j + patch_size, :]
                patches_img.append(patch_img)
                patches_mask.append(patch_mask)
        
        patches_img = np.array(patches_img)
        patches_mask = np.array(patches_mask)
        
        # Reshape to combine the first two dimensions
        num_patches = patches_img.shape[0]
        batch_size = patches_img.shape[1]
        patches_img = patches_img.reshape(num_patches * batch_size, patch_size, patch_size, img.shape[-1])
        patches_mask = patches_mask.reshape(num_patches * batch_size, patch_size, patch_size, mask.shape[-1])
        
        return patches_img, patches_mask
    
    # Extract patches
    patches_img, patches_mask = extract_patches(images, masks, patch_size, stride)
    
    # Normalize images
    if patches_img.max() > 1:
        patches_img = patches_img / 255.0
    
    # Convert to float32
    patches_img = patches_img.astype(np.float32)
    patches_mask = patches_mask.astype(np.float32)
    
    return patches_img, patches_mask

def process_multiple_stains(images_list, masks_list, patch_size=512, stride=None, num_classes=None):
    """
    Process multiple stain images and masks, then concatenate
    
    Parameters:
    - images_list: List of image arrays (each NxHxWxC)
    - masks_list: List of mask arrays (each NxHxW or NxHxWxC)
    - patch_size: Size of patches to extract
    - stride: Stride between patches
    - num_classes: Number of classes for one-hot encoding
    
    Returns:
    - Concatenated processed image patches
    - Concatenated processed mask patches
    """
    # Process each stain
    processed_images = []
    processed_masks = []
    
    for images, masks in zip(images_list, masks_list):
        # Process current stain
        proc_img, proc_mask = preprocess_images_and_masks(
            images, 
            masks, 
            patch_size=patch_size, 
            stride=stride, 
            num_classes=num_classes
        )
        
        # Add to lists
        processed_images.append(proc_img)
        processed_masks.append(proc_mask)
        
        # Print shapes for verification
        print(f"Processed images shape: {proc_img.shape}")
        print(f"Processed masks shape: {proc_mask.shape}")

    np.save('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_img_pm.npy', np.squeeze(processed_images))
    np.save('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold3_mask_pm.npy', np.squeeze(processed_masks))

    # # Concatenate all processed stains
    # final_images = np.concatenate(processed_images, axis=0)
    # final_masks = np.concatenate(processed_masks, axis=0)
    
    # print("\nFinal concatenated shapes:")
    # print(f"Final images shape: {final_images.shape}")
    # print(f"Final masks shape: {final_masks.shape}")

    # return final_images, final_masks
    return processed_images, processed_masks

def save_processed_data(images, masks, output_path):
    """
    Save processed images and masks to numpy files
    
    Parameters:
    - images: Processed image patches
    - masks: Processed mask patches
    - output_path: Base path for saving files
    """
    np.save(f'{output_path}{dataset_type}_img.npy', images)
    np.save(f'{output_path}{dataset_type}_mask.npy', masks)

# Set dataset type
dataset_type = 'fold_3'

def main():
    # Example usage
    NUM_CLASSES = 3  # Set to your number of classes
    patch_size = 512
    stride = patch_size // 2
    
    # Your 3 image arrays
    images_list = [
        # np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/'+dataset_type+'/' + dataset_type +'_img_HE.npy'),
        # np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/'+dataset_type+'/' + dataset_type +'_img_PAS.npy'),
        np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/'+dataset_type+'/' + dataset_type +'_img_PM.npy')
    ]
    
    # Your 3 mask arrays
    masks_list = [
        # np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/'+dataset_type+'/'+dataset_type+'_mask_multiclass_HE.npy'),
        # np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/'+dataset_type+'/'+dataset_type+'_mask_multiclass_PAS.npy'),
        np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/'+dataset_type+'/'+dataset_type+'_mask_multiclass_PM.npy')
    ]
    
    # Process and concatenate
    final_images, final_masks = process_multiple_stains(
        images_list, 
        masks_list, 
        patch_size=patch_size, 
        stride=stride, 
        num_classes=NUM_CLASSES
    )
    
    # Save the concatenated, processed data
    # save_processed_data(final_images, final_masks, '/scratch.local/juanp/glomeruli/dataset/processed(5fold)/')

if __name__ == '__main__':
    main()