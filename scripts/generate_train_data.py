# scripts/generate_train_data.py

import json
import os

FEEDBACK_FILE = "./logs/feedback.jsonl"
OUTPUT_FILE = "./train_data.json"

train_data = []

if not os.path.exists(FEEDBACK_FILE):
    print(f"No feedback file found at {FEEDBACK_FILE}")
    exit(1)

with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        feedback = entry.get("feedback", {})
        text = feedback.get("text")
        user_summary = feedback.get("user_summary")
        if text and user_summary:
            # Each training example: {"input": original text, "output": corrected summary}
            train_data.append({
                "input": text,
                "output": user_summary
            })

if train_data:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        json.dump(train_data, out_f, ensure_ascii=False, indent=2)
    print(f"Generated {len(train_data)} training examples → {OUTPUT_FILE}")
else:
    print("No valid feedback entries found. Nothing to generate.")
