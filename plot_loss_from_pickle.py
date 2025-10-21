import pickle
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# Cargar historiales
# ------------------------------
with open('paper_checkpoints/training_history_deeplab_pretrain_pool.pkl', 'rb') as f:
    hist_pre = pickle.load(f)

with open('paper_checkpoints/training_history_deeplab_finetunned_pool.pkl', 'rb') as f:
    hist_ft = pickle.load(f)

# ------------------------------
# Concatenate histories
# ------------------------------
train_loss = hist_pre['loss'] + hist_ft['loss']
val_loss = hist_pre['val_loss'] + hist_ft['val_loss']

epochs_pre = len(hist_pre['loss'])
epochs_ft = len(hist_ft['loss'])
epochs_total = np.arange(1, epochs_pre + epochs_ft + 1)

# ------------------------------
# Find best validation loss
# ------------------------------
best_epoch = np.argmin(val_loss) + 1  # +1 because epochs start at 1
best_val = val_loss[best_epoch - 1]

# ------------------------------
# Plot
# ------------------------------
plt.figure(figsize=(10,6))
plt.plot(epochs_total, train_loss, label='Training Loss', linewidth=2)
plt.plot(epochs_total, val_loss, label='Validation Loss', linewidth=2)

# Markers
plt.axvline(x=epochs_pre, color='red', linestyle='--', label='Fine-tuning start')
plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best val loss (epoch {best_epoch})')

# Highlight the best point
plt.scatter(best_epoch, best_val, color='green', s=80, zorder=5)

# ------------------------------
# Style
# ------------------------------
plt.xlabel('Epochs (pretrain + fine-tune)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss', fontsize=14)
plt.ylim(0.35, 0.48)
plt.xlim(0, epochs_total[-1] + 1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()