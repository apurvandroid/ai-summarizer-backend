#!/bin/bash
# retrain.sh
# Script to generate training data, retrain LoRA adapter, validate it, and reload in server

# Activate Python environment (if needed)
# source ~/miniconda3/bin/activate myenv

echo "[$(date)] Starting LoRA retraining pipeline..."

# 1️⃣ Generate training data from feedback.jsonl
# This script converts user feedback into a training dataset (train_data.json)
python scripts/generate_train_data.py

# 2️⃣ Retrain LoRA adapter
# Fine-tune the LoRA adapter using the generated training data
python scripts/train_lora.py \
    --base_model ./base_model \
    --adapter ./lora_adapter \
    --train_data ./train_data.json \
    --output ./lora_adapter_new

# 3️⃣ Validate new adapter
# Run validation to ensure the new adapter works as expected
python scripts/validate_lora.py ./lora_adapter_new

# 4️⃣ Replace old adapter if validation passes
if [ $? -eq 0 ]; then
    echo "[$(date)] Validation passed. Replacing old adapter..."
    rm -rf ./lora_adapter               # Remove the old adapter
    mv ./lora_adapter_new ./lora_adapter  # Rename new adapter to replace old one

    # 5️⃣ Reload adapter in running FastAPI server
    # This sends a POST request to the /reload_adapter endpoint
    curl -X POST http://127.0.0.1:8000/reload_adapter
else
    echo "[$(date)] Validation failed. Keeping old adapter."
    rm -rf ./lora_adapter_new  # Delete the new adapter if validation fails
fi

echo "[$(date)] Retraining pipeline complete."
