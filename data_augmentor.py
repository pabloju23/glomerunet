'''
Based on Altini et al paper Semantic Segmentation Framework for Glomeruli
Detection and Classification in Kidney
Histological Sections
'''
import numpy as np
import tensorflow as tf
import elasticdeform
import keras_cv
import h5py


# Select interpolation mode
interpolation = 'bilinear'
fill_mode = 'reflect'

# Probabilities for augmentations
PROB_ROTATION = 0.5
PROB_HORIZONTAL_FLIP = 0.5
PROB_VERTICAL_FLIP = 0.35
PROB_ZOOM = 0.4
PROB_GAUSSIAN_NOISE = 0.33
PROB_GAUSSIAN_BLUR = 0.33
PROB_ELASTIC_DEFORMATION = 0.2
PROB_SHEAR = 0.4
PROB_HSV_SHIFT = 0.6

# Create augmentation layers outside the function
flip_layer_h = tf.keras.layers.RandomFlip("horizontal")
flip_layer_v = tf.keras.layers.RandomFlip("vertical")
zoom_layer = tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.4), width_factor=(-0.2, 0.4), interpolation=interpolation, fill_mode=fill_mode)
shear_layer = keras_cv.layers.RandomShear(x_factor=0.3, y_factor=0.3, interpolation=interpolation, fill_mode=fill_mode)

def apply_rotation_90(image, mask):
    """Apply random 90-degree rotation."""
    k = tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32)  # Randomly choose 1, 2, or 3
    return tf.image.rot90(image, k=k), tf.image.rot90(mask, k=k)

def apply_gaussian_noise(image, sigma_range=(0.0, 0.01)):
    """Apply Gaussian noise with random σ."""
    sigma = tf.random.uniform([], sigma_range[0], sigma_range[1])
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=sigma, dtype=image.dtype)
    return tf.clip_by_value(image + noise, 0.0, 1.0)

def apply_gaussian_blur(image, sigma_range=(0.0, 0.1)):
    """Apply Gaussian blur with random σ."""
    sigma = tf.random.uniform([], sigma_range[0], sigma_range[1])
    blurred_image = tf.image.adjust_brightness(image, sigma)  # Approximate blur
    # Clip to valid range
    return tf.clip_by_value(blurred_image, 0.0, 1.0)

def apply_elastic_deform(image, mask, points=5, mode='mirror', axis=[(0, 1), (0, 1)], order=[3, 0], sigma_range=(5, 15)):
    """
    Aplica deformación elástica a una imagen y su máscara, con sigma aleatorio dentro de un rango.

    Parámetros:
    - image: La imagen a deformar (numpy array de forma [height, width, channels]).
    - mask: La máscara a deformar (numpy array de forma [height, width, 1]).
    - points: Número de puntos de deformación.
    - mode: Modo de relleno ('mirror', 'constant', etc.).
    - axis: Ejes sobre los que aplicar la deformación.
    - order: Orden de la interpolación para la imagen y la máscara.
    - sigma_range: Tupla (min_sigma, max_sigma) para generar un valor de sigma aleatorio.

    Retorna:
    - image_deformed: Imagen deformada.
    - mask_deformed: Máscara deformada.
    """

    # Generar un valor de sigma aleatorio dentro del rango especificado
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    # Aplica la deformación elástica
    image_deformed, mask_deformed = elasticdeform.deform_random_grid(
        [image, mask], sigma=sigma, points=points, mode=mode, axis=axis, order=order
    )

    # Asegurarse de que los valores estén en el rango [0, 1]
    image_deformed = np.clip(image_deformed, 0, 1)
    # mask_deformed = np.clip(mask_deformed, 0, 1)

    return image_deformed, mask_deformed

def apply_elastic_deform_tf(image, mask, points=5, sigma_range=(5, 15)):
    def _process_elastic(image_tensor, mask_tensor):
        # Convert to numpy using tf.keras.backend
        image_np = tf.keras.backend.get_value(image_tensor)
        mask_np = tf.keras.backend.get_value(mask_tensor)
        
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        deformed_image, deformed_mask = elasticdeform.deform_random_grid(
            [image_np, mask_np], 
            sigma=sigma, 
            points=points,
            mode='mirror',
            axis=[(0, 1), (0, 1)],
            order=[3, 0]
        )
        
        deformed_image = np.clip(deformed_image, 0, 1)
        
        return deformed_image.astype(np.float32), deformed_mask.astype(np.float32)
    
    return tf.py_function(
        _process_elastic,
        [image, mask],
        [tf.float32, tf.float32]
    )


def apply_hsv_shift(image, hue_range=(-0.1, 0.1), sat_range=(-0.1, 0.1)):
    """Apply HSV shift."""
    image_hsv = tf.image.rgb_to_hsv(image)
    hue_shift = tf.random.uniform([], hue_range[0], hue_range[1])
    sat_shift = tf.random.uniform([], sat_range[0], sat_range[1])
    image_hsv = tf.stack([
        tf.clip_by_value(image_hsv[..., 0] + hue_shift, 0.0, 1.0),
        tf.clip_by_value(image_hsv[..., 1] + sat_shift, 0.0, 1.0),
        image_hsv[..., 2]
    ], axis=-1)
    return tf.image.hsv_to_rgb(image_hsv)

def print_tensor_info(name, tensor):
    print(f"\n{name}:")
    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Min: {tf.reduce_min(tensor)}")
    print(f"Max: {tf.reduce_max(tensor)}")

def augment(image, mask):
    if tf.random.uniform([]) < PROB_HORIZONTAL_FLIP:
        image = flip_layer_h(image)
        mask = flip_layer_h(mask)

    if tf.random.uniform([]) < PROB_VERTICAL_FLIP:
        image = flip_layer_v(image)
        mask = flip_layer_v(mask)

    if tf.random.uniform([]) < PROB_ROTATION:
        image, mask = apply_rotation_90(image, mask)

    if tf.random.uniform([]) < PROB_ZOOM:
        concatenated = tf.concat([image, mask], axis=-1)
        concatenated = zoom_layer(concatenated)
        image = concatenated[..., :-mask.shape[-1]]
        mask = concatenated[..., -mask.shape[-1]:]

    if tf.random.uniform([]) < PROB_GAUSSIAN_NOISE:
        image = apply_gaussian_noise(image)

    if tf.random.uniform([]) < PROB_GAUSSIAN_BLUR:
        image = apply_gaussian_blur(image)

    # if tf.random.uniform([]) < PROB_ELASTIC_DEFORMATION:
    #     image, mask = apply_elastic_deform_tf(image, mask)

    if tf.random.uniform([]) < PROB_SHEAR:
        concatenated = tf.concat([image, mask], axis=-1)
        concatenated = shear_layer(concatenated)
        image = concatenated[..., :-mask.shape[-1]]
        mask = concatenated[..., -mask.shape[-1]:]

    if tf.random.uniform([]) < PROB_HSV_SHIFT:
        image = apply_hsv_shift(image)

    return image, mask



def npy_generator(images_file, masks_file, batch_size):
    images = images_file
    masks = masks_file
    
    if images.max() > 1:
        images = images / 255.0

    images = images.astype(np.float32)
    masks = masks.astype(np.float32)  

    num_samples = images.shape[0]
    for offset in range(0, num_samples, batch_size):
        yield images[offset:offset+batch_size], masks[offset:offset+batch_size]

def augment_batch(images, masks):
    batch_size = tf.shape(images)[0]
    
    def process_single(i):
        image = images[i]
        mask = masks[i]
        aug_image, aug_mask = augment(image, mask)
        # Ensure shapes are set
        aug_image.set_shape([512, 512, 3])
        aug_mask.set_shape([512, 512, 3])
        return aug_image, aug_mask
    
    aug_images, aug_masks = tf.map_fn(
        lambda x: process_single(x),
        tf.range(batch_size),
        dtype=(tf.float32, tf.float32)
    )
    
    # Set batch shape explicitly
    aug_images.set_shape([None, 512, 512, 3])
    aug_masks.set_shape([None, 512, 512, 3])
    
    return aug_images, aug_masks

def h5_generator_onehot(h5_path, batch_size, n_classes=3, group=None):
    with h5py.File(h5_path, "r") as f:
        if group:
            images = f[group]["images"]
            masks = f[group]["masks"]
        else:
            images = f["images"]
            masks = f["masks"]
        n = images.shape[0]

        idxs = np.arange(n)
        # np.random.shuffle(idxs)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = idxs[start:end]

            batch_imgs = images[batch_idx].astype("float32") / 255.0
            batch_masks = masks[batch_idx].astype("int32")  # grayscale con valores 0,1,2

            # Convertir a one-hot
            batch_masks = tf.keras.utils.to_categorical(batch_masks, num_classes=n_classes)

            yield batch_imgs, batch_masks.astype("float32")

def create_dataset_h5(h5_file, batch_size, augmentation=False, shuffle=True, 
                      shuffle_buffer_size=10, only_positive_masks=False, 
                      class_target=None, repeat=1, n_classes=3, group=None):

    dataset = tf.data.Dataset.from_generator(
        lambda: h5_generator_onehot(h5_file, batch_size, n_classes=n_classes, group=group),
        output_signature=(
            tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),  # RGB
            tf.TensorSpec(shape=(None, 512, 512, n_classes), dtype=tf.float32),  # máscaras one-hot
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    if augmentation:
        dataset_aug = dataset
        if class_target is not None:
            dataset_aug = dataset_aug.filter(
                lambda img, msk: tf.reduce_any(msk[..., class_target] > 0)
            )
        if only_positive_masks:
            dataset_aug = dataset_aug.filter(
                lambda img, msk: tf.reduce_sum(msk[..., 1:]) > 0
            )
            dataset_aug = dataset_aug.map(
                augment_batch, num_parallel_calls=tf.data.AUTOTUNE
            ).repeat(repeat)
        else:
            dataset_aug = dataset_aug.map(
                augment_batch, num_parallel_calls=tf.data.AUTOTUNE
            ).repeat(repeat)

        dataset = dataset.concatenate(dataset_aug)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_dataset_with_class_augmentation_h5(h5_file, batch_size, augmentation=False, shuffle=True, shuffle_buffer_size=10, n_classes=3, group=None):
    def filter_by_class(mask, target_class):
        """Check if a given target class is present in the mask."""
        return tf.reduce_any(mask[..., target_class] > 0)

    def replicate_dataset(dataset, class_target, repetitions):
        """Replicate the dataset for a specific class a given number of times."""
        filtered_dataset = dataset.filter(lambda img, msk: filter_by_class(msk, class_target))
        augmented_dataset = filtered_dataset.map(
            augment_batch, num_parallel_calls=tf.data.AUTOTUNE
        )
        return augmented_dataset.repeat(repetitions)

    # Base dataset generation
    dataset = tf.data.Dataset.from_generator(
        lambda: h5_generator_onehot(h5_file, batch_size, n_classes=n_classes, group=group),
        output_signature=(
            tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),        # imágenes RGB
            tf.TensorSpec(shape=(None, 512, 512, n_classes), dtype=tf.float32) # máscaras one-hot
        )
    )

    if augmentation:
        # Create augmented datasets for each target class
        dataset_class_2 = replicate_dataset(dataset, class_target=2, repetitions=5)
        dataset_class_1 = replicate_dataset(dataset, class_target=1, repetitions=1)

        # Combine the augmented datasets with the original dataset
        dataset = dataset.concatenate(dataset_class_2).concatenate(dataset_class_1)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_dataset(images_file, masks_file, batch_size, augmentation=False, shuffle=True, shuffle_buffer_size=100, only_positive_masks=False, class_target=None, repeat=1):
    dataset = tf.data.Dataset.from_generator(
        lambda: npy_generator(images_file, masks_file, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32)
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    if augmentation:
        dataset_aug = dataset
        if class_target is not None:
            # dataset_aug will only contain samples with the target class one hot encoded
            dataset_aug = dataset_aug.filter(
                lambda img, msk: tf.reduce_any(msk[..., class_target] > 0)
            )
        if only_positive_masks:
            dataset_aug = dataset_aug.filter(lambda img, msk: tf.reduce_sum(msk[..., 1:]) > 0)
            # Apply augmentation individually to each image in batch
            dataset_aug = dataset_aug.map(augment_batch, 
                                             num_parallel_calls=tf.data.AUTOTUNE).repeat(repeat)
        else:
            dataset_aug = dataset_aug.map(augment_batch,
                                    num_parallel_calls=tf.data.AUTOTUNE).repeat(repeat)

        dataset = dataset.concatenate(dataset_aug)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_dataset_with_class_augmentation(images_file, masks_file, batch_size, augmentation=False, shuffle=True, shuffle_buffer_size=100):   
    def filter_by_class(mask, target_class):
        """Check if a given target class is present in the mask."""
        return tf.reduce_any(mask[..., target_class] > 0)

    def replicate_dataset(dataset, class_target, repetitions):
        """Replicate the dataset for a specific class a given number of times."""
        filtered_dataset = dataset.filter(lambda img, msk: filter_by_class(msk, class_target))
        augmented_dataset = filtered_dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
        return augmented_dataset.repeat(repetitions)

    # Base dataset generation
    dataset = tf.data.Dataset.from_generator(
        lambda: npy_generator(images_file, masks_file, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32)
        )
    )

    if augmentation:
        # Create augmented datasets for each target class
        dataset_class_2 = replicate_dataset(dataset, class_target=2, repetitions=5)
        dataset_class_1 = replicate_dataset(dataset, class_target=1, repetitions=1)

        # Combine the augmented datasets with the original dataset
        dataset = dataset.concatenate(dataset_class_2).concatenate(dataset_class_1)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset



