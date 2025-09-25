import os
import json
import numpy as np
import nibabel as nib

def save_nifti(array, filepath):
    """Guarda un array 2D como NIfTI"""
    nifti = nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4))
    nib.save(nifti, filepath)

def export_dataset(images, masks, out_dir, dataset_id, dataset_name,
                   val_images=None, val_masks=None,
                   test_images=None, test_masks=None):
    """
    Exporta imágenes y máscaras (numpy arrays) al formato nnU-Net V2.
    - images: npy de train (N,512,512,3)
    - masks:  npy de train (N,512,512,3) one-hot
    - val_images / val_masks: conjunto de validación
    - test_images / test_masks: conjunto de test
    """
    dataset_folder = os.path.join(out_dir, f"Dataset{dataset_id}_{dataset_name}")
    dirs = {
        "train_images": os.path.join(dataset_folder, "imagesTr"),
        "train_labels": os.path.join(dataset_folder, "labelsTr"),
        "val_images": os.path.join(dataset_folder, "imagesVal"),
        "val_labels": os.path.join(dataset_folder, "labelsVal"),
        "test_images": os.path.join(dataset_folder, "imagesTs"),
        "test_labels": os.path.join(dataset_folder, "labelsTs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    def export_split(imgs, msks, prefix, img_dir, msk_dir):
        for i in range(imgs.shape[0]):
            img = imgs[i]  # (512,512,3)
            msk = msks[i]
            # máscara one-hot -> argmax
            msk_argmax = np.argmax(msk, axis=-1).astype(np.uint8)

            for c in range(3):
                save_nifti(img[..., c], os.path.join(img_dir, f"{prefix}_{i:04d}_{c:04d}.nii.gz"))
            save_nifti(msk_argmax, os.path.join(msk_dir, f"{prefix}_{i:04d}.nii.gz"))

    # Exportar train
    export_split(images, masks, "case", dirs["train_images"], dirs["train_labels"])

    # Exportar val
    if val_images is not None and val_masks is not None:
        export_split(val_images, val_masks, "val", dirs["val_images"], dirs["val_labels"])

    # Exportar test
    if test_images is not None and test_masks is not None:
        export_split(test_images, test_masks, "test", dirs["test_images"], dirs["test_labels"])

    # Crear dataset.json
    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "non_sclerotic": 1, "sclerotic": 2},
        "numTraining": images.shape[0],
        "file_ending": ".nii.gz",
        "name": dataset_name
    }
    with open(os.path.join(dataset_folder, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    # Crear splits_final.json (usa train/val explícitos)
    if val_images is not None:
        split = {
            "train": [f"case_{i:04d}" for i in range(images.shape[0])],
            "val": [f"val_{i:04d}" for i in range(val_images.shape[0])]
        }
        with open(os.path.join(dataset_folder, "splits_final.json"), "w") as f:
            json.dump([split], f, indent=4)

    print(f"✅ Exportado dataset nnU-Net en: {dataset_folder}")


if __name__ == "__main__":
    # Cargar tus npy
    D2_images = np.load('/scratch.local/juanp/glomeruli/dataset/trainD2_paper/train_imgD2_v2_positive.npy')
    D2_masks  = np.load("/scratch.local/juanp/glomeruli/dataset/trainD2_paper/train_maskD2_v2_positive.npy")
    D1_images = np.concatenate([
        np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_2_img.npy'),
        np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_4_img.npy'),
        np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_5_img.npy')
        ], axis=0)
    D1_masks  = np.concatenate([
        np.load("/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_2_mask.npy"),
        np.load("/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_4_mask.npy"),
        np.load("/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_5_mask.npy")
        ], axis=0)
    val_images = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_1_img.npy')
    val_masks  = np.load("/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_1_mask.npy")
    test_images = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_3_img.npy')
    test_masks  = np.load('/scratch.local/juanp/glomeruli/dataset/processed(5fold)/fold_3_mask.npy')

    out_dir = "/scratch.local/juanp/glomeruli/dataset/nnunet_data"

    # Exportar Dataset100 (D2)
    export_dataset(D2_images, D2_masks, out_dir, dataset_id=100,
                   dataset_name="Histo_D2",
                   val_images=val_images, val_masks=val_masks,
                   test_images=test_images, test_masks=test_masks)

    # Exportar Dataset101 (D1)
    export_dataset(D1_images, D1_masks, out_dir, dataset_id=101,
                   dataset_name="Histo_D1",
                   val_images=val_images, val_masks=val_masks,
                   test_images=test_images, test_masks=test_masks)
