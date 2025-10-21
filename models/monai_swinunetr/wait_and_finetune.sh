#!/bin/bash
# archivo: wait_and_finetune.sh

# Esperar a que termine el proceso de pretrain
while pgrep -f "pretrain_swinunetr.py" > /dev/null; do
    echo "⌛ Esperando a que termine pretrain_swinunetr.py..."
    sleep 300  # comprueba cada 5 minutos
done

echo "✅ Pretrain terminado. Iniciando fine-tuning..."
python finetune_swinunetr.py
