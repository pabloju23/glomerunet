# GlomeruNet
**DeepLab v3+ implementation for glomeruli segmentation**

## Main Scripts (Can be used in npy or h5 data format)

- `main_train.py` → Main script to train a model using a pretraining and finetuning pipeline.
- `main_evaluate.py` → Main script to evaluate results on a checkpointed model.

## Model Architectures

- `models/keras/keras_deeplab_model.py` / `keras_segnet.py` / `keras_unet_model.py` → Model architecture definitions.

## Metrics

- `tversky_metric.py` / `dice_metric.py` → Custom metrics implemented in TensorFlow Keras using one-hot encoded format.

## Utilities

- `methods/data_augmentor.py` → Data augmentation pipeline module.
- `utils.py` → Miscellaneous utility functions.

## Directories

- `/data_management` → Image and mask loaders, and other utilities.
- `/env` → `.yaml` file to reproduce the exact Conda environment.
- `/folds` → Pipeline to train with N folds and save results to Excel.
- `/models` → Rest of models extracted from MONAI respository and their code to run.

## References
- https://keras.io/examples/vision/deeplabv3_plus/
- https://github.com/ykamikawa/tf-keras-SegNet
- https://github.com/Project-MONAI/MONAI

