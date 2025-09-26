# finetune_lora.py
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# ----------------------------
# 1. Load model & tokenizer
# ----------------------------
model_name = "t5-small"   # you can later try "t5-base" or "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ----------------------------
# 2. Apply LoRA config
# ----------------------------
lora_config = LoraConfig(
    r=8,                # rank
    lora_alpha=16,      # scaling
    target_modules=["q", "v"],  # which layers to adapt (for T5: q,v projections)
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)

# ----------------------------
# 3. Load dataset
# ----------------------------
dataset = load_dataset("json", data_files="summarization_dataset.jsonl")

def preprocess(examples):
    inputs = tokenizer(examples["input"], max_length=512, truncation=True)
    outputs = tokenizer(examples["output"], max_length=128, truncation=True)
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=["instruction","input","output"])

# ----------------------------
# 4. Training setup
# ----------------------------
training_args = TrainingArguments(
    output_dir="./lora_adapter_new",  # Save here instead of overwriting old adapter
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=False,
    optim="adamw_torch"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ----------------------------
# 5. Train
# ----------------------------
trainer.train()

# ----------------------------
# 6. Save adapter weights and tokenizer
# ----------------------------
output_dir = "./lora_adapter_new"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ LoRA fine-tuning finished. Adapter saved in {output_dir}")
