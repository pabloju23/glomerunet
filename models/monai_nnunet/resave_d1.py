import os
import numpy as np
import cv2

# rutas
dataset_dir = "/scratch.local2/juanp/glomeruli/dataset/nnunet_data/nnUnet_raw/Dataset001_glomeruli"
imagesTr_dir = os.path.join(dataset_dir, "imagesTr")
labelsTr_dir = os.path.join(dataset_dir, "labelsTr")
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)

# lista de folds en orden
folds = ["fold_2", "fold_4", "fold_5"]

# contador para los nombres
counter = 0

for fold in folds:
    img_path = f"/scratch.local2/juanp/glomeruli/dataset/processed(5fold)/{fold}_img.npy"
    lbl_path = f"/scratch.local2/juanp/glomeruli/dataset/processed(5fold)/{fold}_mask.npy"

    imgs = np.load(img_path)  # Nx512x512x3, float32 0-1
    lbls = np.load(lbl_path)  # Nx512x512x3, one-hot float32 0-1

    assert imgs.shape[0] == lbls.shape[0], f"Mismatch in {fold}"

    for i in range(imgs.shape[0]):
        counter += 1
        name_num = f"{counter:04d}"  # XXXX con ceros delante

        # imagen: de float32 0-1 a uint8 0-255
        img_uint8 = (imgs[i] * 255).astype(np.uint8)
        img_file = os.path.join(imagesTr_dir, f"D1_{name_num}_0000.png")
        cv2.imwrite(img_file, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))  # cv2 usa BGR

        # máscara: de one-hot a valores 0,1,2
        mask_int = np.argmax(lbls[i], axis=-1).astype(np.uint8)
        mask_file = os.path.join(labelsTr_dir, f"D1_{name_num}.png")
        cv2.imwrite(mask_file, mask_int)

print(f"✅ Guardadas {counter} imágenes y máscaras para D1 en {imagesTr_dir} y {labelsTr_dir}")
