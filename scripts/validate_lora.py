# test_lora.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Base model (same one used for training)
base_model = "t5-small"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load base model
model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

# ✅ Load LoRA adapter you trained
model = PeftModel.from_pretrained(model, "./lora_adapter")

# ------------------------
# Test on a sample article
# ------------------------
text = "The city council approved a new downtown park with playgrounds and gardens."
inputs = tokenizer(text, return_tensors="pt", truncation=True)

summary_ids = model.generate(**inputs, max_length=30, min_length=5)
print("\nGenerated Summary:", tokenizer.decode(summary_ids[0], skip_special_tokens=True))
