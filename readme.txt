main_train.py --> main file to train a model with pretrain and finetuning pipeline
main_evaluate.py --> main file to evaluate results on checkpointed model
keras_deeplab_model.py / keras_segnet.py / keras_unet_model.py --> models architecture
tversky_metric.py / dice_metric.py --> metrics in tensorflow keras one hot encode format
data_augmentor --> data augmentation pipeline module
utils.py --> misc

/data_management --> some image/mask loaders, misc
/env --> .yaml file to reproduce the exact environment
/folds --> pipeline to train with N folds and save the results in excel
