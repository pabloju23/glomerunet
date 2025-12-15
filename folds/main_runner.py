import os
# Set environment variable to select only GPU 1 and GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import tensorflow as tf
# Optimize GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
from training_pipeline import train_and_evaluate
import itertools


pretrained_model_path = '/scratch.local2/juanp/glomeruli/paper_checkpoints/deeplab_glomeruli_multiclass_pretrain_pool.keras'  # TODO

pretrain = True

# Load the pretrained model ONCE
base_model = None
if pretrain:
    print("Loading pretrained model...")
    base_model = tf.keras.models.load_model(pretrained_model_path, compile=False)
    print("Pretrained model loaded successfully!")

# Define already tested splits to skip
tested_splits = {
    # ('fold5', 'fold4', 'fold1-fold2-fold3'),   # Example of already tested split
}

# Load dataset
fold1_img = np.load('/scratch.local2/juanp/glomeruli/dataset/5fold_paper/fold_1_img.npy')
fold1_mask = np.load('/scratch.local2/juanp/glomeruli/dataset/5fold_paper/fold_1_mask.npy')
fold2_img = np.load('/scratch.local2/juanp/glomeruli/dataset/5fold_paper/fold_2_img.npy')
fold2_mask = np.load('/scratch.local2/juanp/glomeruli/dataset/5fold_paper/fold_2_mask.npy')
fold3_img = np.load('/scratch.local2/juanp/glomeruli/dataset/5fold_paper/fold_3_img.npy')
fold3_mask = np.load('/scratch.local2/juanp/glomeruli/dataset/5fold_paper/fold_3_mask.npy')
fold4_img = np.load('/scratch.local2/juanp/glomeruli/dataset/5fold_paper/fold_4_img.npy')
fold4_mask = np.load('/scratch.local2/juanp/glomeruli/dataset/5fold_paper/fold_4_mask.npy')
fold5_img = np.load('/scratch.local2/juanp/glomeruli/dataset/5fold_paper/fold_5_img.npy')
fold5_mask = np.load('/scratch.local2/juanp/glomeruli/dataset/5fold_paper/fold_5_mask.npy')

# # Ensure the data is in the correct dtype
fold1_img = fold1_img.astype(np.float32)
fold1_mask = fold1_mask.astype(np.float32)
fold2_img = fold2_img.astype(np.float32)
fold2_mask = fold2_mask.astype(np.float32)
fold3_img = fold3_img.astype(np.float32)
fold3_mask = fold3_mask.astype(np.float32)
fold4_img = fold4_img.astype(np.float32)
fold4_mask = fold4_mask.astype(np.float32)
fold5_img = fold5_img.astype(np.float32)
fold5_mask = fold5_mask.astype(np.float32)

# Create a dictionary with the images and masks folds
folds = {
    'fold1': (fold1_img, fold1_mask),
    'fold2': (fold2_img, fold2_mask),
    'fold3': (fold3_img, fold3_mask),
    'fold4': (fold4_img, fold4_mask),
    'fold5': (fold5_img, fold5_mask)
}

# Generate unique combinations of 5 folds into train, val, and test (3-1-1 split)
all_folds = list(folds.keys())

# Step 1: Select 3 folds for training
train_combinations = list(itertools.combinations(all_folds, 3))

unique_splits = []
for train in train_combinations:
    # Step 2: Select 1 fold for validation from the remaining folds
    remaining_folds = list(set(all_folds) - set(train))
    val_combinations = list(itertools.combinations(remaining_folds, 1))
    
    for val in val_combinations:
        # Step 3: The remaining fold is used for testing
        test = tuple(set(remaining_folds) - set(val))
        unique_splits.append((train, val, test))

# Print the total number of unique combinations
print(f"Total unique splits: {len(unique_splits)}")

# Set up mirrored strategy ONCE before the loop
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Define callbacks with monitor set to validation mean IoU
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_dsc_nobg', mode='max', patience=12, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dsc_nobg', mode='max', factor=0.1, patience=7, verbose=1, min_lr=1e-6),
]

# Iterate through all combinations
excel_row = 0  # To track the row index for Excel logging
for train, val, test in unique_splits:
    # Convert fold names to strings
    train_folds_str = '-'.join(train)
    val_fold_str = val[0]
    test_fold_str = test[0]
    # Skip already tested splits
    if (test_fold_str, val_fold_str, train_folds_str) in tested_splits:
        print(f"Skipping already tested split: train_folds: {train_folds_str}, val_fold: {val_fold_str}, test_fold: {test_fold_str}")
        continue
    # Concatenate train folds and shuffle
    train_img = np.concatenate([folds[fold][0] for fold in train], axis=0)
    train_mask = np.concatenate([folds[fold][1] for fold in train], axis=0)
    indices = np.arange(train_img.shape[0])
    np.random.shuffle(indices)
    train_img, train_mask = train_img[indices], train_mask[indices]
    
    # Prepare val and test data
    val_img, val_mask = folds[val[0]]
    test_img, test_mask = folds[test[0]]    
    
    # Log details and run training
    print(f"Training with train_folds: {train_folds_str}, val_fold: {val_fold_str}, test_fold: {test_fold_str}")
    train_and_evaluate(
        train_img=train_img,
        train_mask=train_mask,
        val_img=val_img,
        val_mask=val_mask,
        test_img=test_img,
        test_mask=test_mask,
        train_folds=train_folds_str,
        val_fold=val_fold_str,
        test_fold=test_fold_str,
        excel_row=excel_row,
        callbacks=callbacks,
        strategy=strategy,
        base_model=base_model
    )
    
    # Increment Excel row index
    excel_row += 1