#!/bin/bash

echo "[$(date)] Starting LoRA retraining pipeline..."

# 1️⃣ Generate training data
python scripts/generate_train_data.py

# 2️⃣ Retrain LoRA adapter
python scripts/train_lora.py \
    --base_model ./base_model \
    --adapter ./lora_adapter \
    --train_data ./train_data.json \
    --output ./lora_adapter_new

# 3️⃣ Validate new adapter
python scripts/validate_lora.py ./lora_adapter_new

# 4️⃣ Replace old adapter if validation passes
if [ $? -eq 0 ]; then
    echo "[$(date)] Validation passed."
    # Remove old adapter and move new one
    rm -rf ./lora_adapter
    mv ./lora_adapter_new ./lora_adapter

    # 5️⃣ Reload adapter in FastAPI server (only if running)
    if nc -z localhost 8000; then
        curl -X POST http://127.0.0.1:8000/reload_adapter
    else
        echo "FastAPI server not running, skipping reload."
    fi
else
    echo "[$(date)] Validation failed. Keeping old adapter."
    rm -rf ./lora_adapter_new
fi

echo "[$(date)] Retraining pipeline complete."
