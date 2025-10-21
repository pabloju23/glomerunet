from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, RandFlipd,
    RandRotate90d, RandZoomd, RandAffined, RandGaussianNoised, RandGaussianSmoothd,
    RandAdjustContrastd, EnsureTyped
)
import torch

# Probabilidades
PROB_ROTATION = 0.5
PROB_HORIZONTAL_FLIP = 0.5
PROB_VERTICAL_FLIP = 0.35
PROB_ZOOM = 0.4
PROB_GAUSSIAN_NOISE = 0.33
PROB_GAUSSIAN_BLUR = 0.33
PROB_SHEAR = 0.4
PROB_HSV_SHIFT = 0.6


def get_train_transforms():
    return Compose([
        # --- Carga y formato ---
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        # --- Normalización solo en imagen ---
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True
        ),

        # --- Flips y rotaciones sincronizadas ---
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=PROB_VERTICAL_FLIP),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=PROB_HORIZONTAL_FLIP),
        RandRotate90d(keys=["image", "label"], prob=PROB_ROTATION, spatial_axes=(0, 1)),

        # --- Zoom y shear sincronizados (modo correcto para máscara) ---
        RandZoomd(
            keys=["image", "label"],
            prob=PROB_ZOOM,
            min_zoom=0.8,
            max_zoom=1.4,
            mode=("bilinear", "nearest"),
            padding_mode="reflect"
        ),
        RandAffined(
            keys=["image", "label"],
            prob=PROB_SHEAR,
            shear_range=[(-0.3, 0.3), (-0.3, 0.3)],
            rotate_range=None,
            translate_range=None,
            scale_range=None,
            mode=("bilinear", "nearest"),  # ← evita interpolación en máscara
            padding_mode="reflect"
        ),

        # --- Efectos de intensidad solo para imagen ---
        RandGaussianNoised(keys=["image"], prob=PROB_GAUSSIAN_NOISE, mean=0.0, std=0.01),
        RandGaussianSmoothd(keys=["image"], prob=PROB_GAUSSIAN_BLUR, sigma_x=(0.0, 0.5), sigma_y=(0.0, 0.5)),
        RandAdjustContrastd(keys=["image"], prob=PROB_HSV_SHIFT, gamma=(0.9, 1.1)),

        # --- Tipado final ---
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.uint8),
    ])


def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True
        ),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.uint8),
    ])
