# app.py
import os
import json
import time
import threading
import re
from typing import Optional, List

import torch
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# -------------------------
# Config
# -------------------------
BASE_MODEL = "t5-small"            # Base model used in training
LORA_DIR = "./lora_adapter"        # LoRA adapter path
LOG_DIR = "./logs"                 # Directory to store logs
INFERENCE_LOG = os.path.join(LOG_DIR, "inference_logs.jsonl")  # Log for inference requests
FEEDBACK_LOG = os.path.join(LOG_DIR, "feedback.jsonl")         # Log for feedback

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Thread lock for thread-safe file writes & model reload
file_lock = threading.Lock()

# -------------------------
# Load model & tokenizer
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
print(f"[app] Using device: {device}")

# Load tokenizer for the base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base seq2seq model
print("[app] Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

# Load LoRA adapter and merge with base model
print("[app] Loading LoRA adapter from", LORA_DIR)
model = PeftModel.from_pretrained(base_model, LORA_DIR, is_trainable=False)
model.to(device)
model.eval()  # Set model to evaluation mode

# Get model's max input tokens
try:
    MODEL_MAX_TOKENS = tokenizer.model_max_length
except Exception:
    MODEL_MAX_TOKENS = 512

print(f"[app] Model loaded. tokenizer.model_max_length={MODEL_MAX_TOKENS}")

# -------------------------
# Utility functions
# -------------------------
def append_jsonl(path: str, obj: dict):
    """
    Append a JSON object as a line to a JSONL file in a thread-safe way.
    """
    with file_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using punctuation marks.
    """
    sentences = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences if sentences else [text]

def chunk_text_by_tokens(text: str, max_tokens: int, overlap: int = 50) -> List[str]:
    """
    Break a large text into smaller chunks suitable for the model.
    Each chunk <= max_tokens with optional overlap.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current = []
    current_len = 0

    for s in sentences:
        s_tokens = tokenizer.encode(s, add_special_tokens=False)
        s_len = len(s_tokens)
        if s_len > max_tokens:
            # Split long sentences into sub-chunks
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            ids = s_tokens
            step = max_tokens - overlap if (max_tokens - overlap) > 0 else max_tokens
            for i in range(0, len(ids), step):
                sub_ids = ids[i:i + max_tokens]
                sub_text = tokenizer.decode(sub_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                chunks.append(sub_text)
        else:
            # Add sentence to current chunk if it fits
            if current_len + s_len <= max_tokens:
                current.append(s)
                current_len += s_len
            else:
                if current:
                    chunks.append(" ".join(current))
                current = [s]
                current_len = s_len

    if current:
        chunks.append(" ".join(current))
    return chunks

def summarize_single(text: str, max_length: int = 128, min_length: int = 20,
                     num_beams: int = 4, do_sample: bool = False) -> str:
    """
    Summarize a single chunk of text using the model.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MODEL_MAX_TOKENS).to(device)
    gen_kwargs = dict(
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        do_sample=do_sample,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    summary = tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return summary

# -------------------------
# FastAPI app + schemas
# -------------------------
app = FastAPI(title="Summarization (LoRA) API", version="1.0")

class SummarizeRequest(BaseModel):
    """Request schema for summarization."""
    text: str
    max_length: int = 128
    min_length: int = 20
    num_beams: int = 4
    do_sample: bool = False
    two_pass: bool = True  # If True, perform second-pass summarization

class FeedbackRequest(BaseModel):
    """Request schema for user feedback."""
    text: str
    model_summary: str
    user_summary: Optional[str] = None
    rating: Optional[int] = None
    user_id: Optional[str] = None
    notes: Optional[str] = None

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Summarization LoRA server running. See /docs for UI."}

@app.post("/summarize")
async def summarize(req: SummarizeRequest, background_tasks: BackgroundTasks):
    """
    Summarize input text.
    - Splits text into chunks if needed
    - Summarizes each chunk
    - Optionally performs second-pass summarization
    - Logs inference asynchronously
    """
    start_ts = time.time()
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    max_tokens_per_chunk = MODEL_MAX_TOKENS - 16
    chunks = chunk_text_by_tokens(text, max_tokens=max_tokens_per_chunk, overlap=50)

    chunk_summaries = []
    for i, ch in enumerate(chunks):
        summ = summarize_single(ch, max_length=req.max_length, min_length=req.min_length,
                                num_beams=req.num_beams, do_sample=req.do_sample)
        chunk_summaries.append({"chunk_index": i, "summary": summ, "chunk_length": len(ch)})

    if req.two_pass and len(chunk_summaries) > 1:
        concat = " ".join([c["summary"] for c in chunk_summaries])
        final_summary = summarize_single(concat, max_length=req.max_length, min_length=req.min_length,
                                        num_beams=req.num_beams, do_sample=req.do_sample)
    else:
        final_summary = " ".join([c["summary"] for c in chunk_summaries])

    latency = time.time() - start_ts

    # Log request asynchronously
    log_entry = {
        "timestamp": time.time(),
        "text_length": len(text),
        "num_chunks": len(chunks),
        "latency": latency,
        "request": req.dict(),
        "final_summary": final_summary,
        "chunk_summaries": chunk_summaries,
    }
    background_tasks.add_task(append_jsonl, INFERENCE_LOG, log_entry)

    return {"summary": final_summary, "chunks": chunk_summaries, "meta": {"device": device, "latency": latency}}

@app.post("/feedback")
async def feedback(req: FeedbackRequest, background_tasks: BackgroundTasks):
    """Collect user feedback and store asynchronously."""
    entry = {
        "timestamp": time.time(),
        "feedback": req.dict()
    }
    background_tasks.add_task(append_jsonl, FEEDBACK_LOG, entry)
    return {"status": "ok"}

@app.post("/reload_adapter")
async def reload_adapter():
    """
    Reload the LoRA adapter dynamically without restarting server.
    Useful if a new fine-tuned adapter is saved.
    """
    global model
    try:
        base_model_loaded = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        new_model = PeftModel.from_pretrained(base_model_loaded, LORA_DIR, is_trainable=False)
        new_model.to(device)
        new_model.eval()
        with file_lock:
            model = new_model
        return {"status": "ok", "message": "LoRA adapter reloaded successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
