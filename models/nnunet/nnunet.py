### Example for code but not working yet
### TODO: all

import os
import torch
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss import TverskyLoss
from nnunetv2.evaluation.evaluator import Evaluator
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.dataset_conversion.utils import generate_dataset_json

# Paths to your datasets
D2_train_dir = '/path/to/D2/train'
D1_train_dir = '/path/to/D1/train'
val_dir = '/path/to/val'
test_dir = '/path/to/test'

# Dataset JSON (required by nnUNet)
generate_dataset_json(
    os.path.join(nnUNet_preprocessed, 'Dataset001_MySegmentation'),
    images_train=[os.path.join(D2_train_dir, f) for f in os.listdir(D2_train_dir)],
    images_val=[os.path.join(val_dir, f) for f in os.listdir(val_dir)],
    labels_train=[os.path.join(D2_train_dir, f.replace('_img', '_label')) for f in os.listdir(D2_train_dir)],
    labels_val=[os.path.join(val_dir, f.replace('_img', '_label')) for f in os.listdir(val_dir)],
    modality='MRI',
    labels={0: 'background', 1: 'class1', 2: 'class2'},
    dataset_name='MySegmentation'
)

# Custom loss: 0.5*CE + 0.5*TverskyLoss
class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        tversky_loss = self.tversky(logits, targets)
        return 0.5 * ce_loss + 0.5 * tversky_loss

# Training on D2 (big dataset)
trainer = nnUNetTrainer(
    plans_path=os.path.join(nnUNet_preprocessed, 'Dataset001_MySegmentation', 'nnUNetPlans.json'),
    fold=0,
    loss=CombinedLoss(),
    num_classes=3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
trainer.run_training()

# Finetuning on D1 (small dataset)
trainer.load_best_checkpoint()
trainer.change_training_data(
    images_train=[os.path.join(D1_train_dir, f) for f in os.listdir(D1_train_dir)],
    labels_train=[os.path.join(D1_train_dir, f.replace('_img', '_label')) for f in os.listdir(D1_train_dir)],
    images_val=[os.path.join(val_dir, f) for f in os.listdir(val_dir)],
    labels_val=[os.path.join(val_dir, f.replace('_img', '_label')) for f in os.listdir(val_dir)]
)
trainer.run_training(finetune=True)

# Testing
trainer.load_best_checkpoint()
predictions = trainer.predict([os.path.join(test_dir, f) for f in os.listdir(test_dir)])

# Evaluation
evaluator = Evaluator(num_classes=3, metrics=['iou', 'dsc'])
results = evaluator.evaluate(predictions, [os.path.join(test_dir, f.replace('_img', '_label')) for f in os.listdir(test_dir)])

print("General IoU:", results['iou']['mean'])
print("Class IoU:", results['iou']['per_class'])
print("General DSC:", results['dsc']['mean'])
print("Class DSC:", results['dsc']['per_class'])