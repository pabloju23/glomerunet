import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" 
os.environ['nnUnet_raw'] = '/scratch.local2/juanp/glomeruli/dataset/nnunet_data/nnUNet_raw'
os.environ['nnUnet_preprocessed'] = '/scratch.local2/juanp/glomeruli/dataset/nnunet_data/nnUNet_preprocessed'
os.environ['nnUnet_results'] = '/scratch.local2/juanp/glomeruli/models/nnunet_data/results'
from nnunetv2.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss_functions.dice_loss import TverskyLoss
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nnunetv2.utilities.dataloading.dataset_loading import load_dataset_from_json

# -----------------------------
# ----- CONFIGURACIÓN ----------
# -----------------------------
DATASET_PATH = "/scratch.local2/juanp/glomeruli/dataset/nnunet_data"
CHECKPOINTS_DIR = "/scratch.local2/juanp/glomeruli/models/nnunet/checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hiperparámetros generales
NUM_EPOCHS = 30
INITIAL_LR = 1e-4
REDUCE_FACTOR = 0.1
PATIENCE = 7

# -----------------------------
# ----- MIX LOSS ---------------
# -----------------------------
ce_weight = 0.35
tversky_weight = 0.65
tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
ce_loss = CrossEntropyLoss()

def mix_loss(pred, target):
    return ce_weight * ce_loss(pred, target) + tversky_weight * tversky_loss(pred, target)

# -----------------------------
# ----- UTILIDADES -------------
# -----------------------------
def train_phase(split_name, previous_checkpoint=None):
    print(f"\n--- Iniciando fase: {split_name} ---\n")
    
    # Cargar dataset
    dataset_json = os.path.join(DATASET_PATH, f"{split_name}_dataset.json")
    dataset = load_dataset_from_json(dataset_json)
    
    # Crear trainer
    trainer = nnUNetTrainer(
        model= None,  # nnU-Net generará la arquitectura automáticamente
        dataset=dataset,
        loss_function=mix_loss,
        optimizer_class=torch.optim.Adam,
        initial_lr=INITIAL_LR,
        device=DEVICE,
        checkpoint_dir=CHECKPOINTS_DIR,
        monitor_metric="val_dsc_nobg",  # Dice sin fondo
        reduce_lr_on_plateau=True,
        reduce_lr_factor=REDUCE_FACTOR,
        reduce_lr_patience=PATIENCE
    )

    # Si hay checkpoint previo (para fine-tuning)
    if previous_checkpoint is not None:
        trainer.load_checkpoint(previous_checkpoint)
        print(f"Loaded checkpoint from {previous_checkpoint}")

    # Entrenar
    trainer.train(num_epochs=NUM_EPOCHS)
    
    # Devolver mejor checkpoint
    return trainer.get_best_checkpoint()

# -----------------------------
# ----- EJECUCIÓN -------------
# -----------------------------
if __name__ == "__main__":
    # Fase 1: D2_train
    best_checkpoint_phase1 = train_phase("D2_train")
    
    # Fase 2: D1 fine-tuning
    best_checkpoint_phase2 = train_phase("D1_train", previous_checkpoint=best_checkpoint_phase1)
    
    print("\nEntrenamiento completado. Mejor checkpoint final:", best_checkpoint_phase2)
