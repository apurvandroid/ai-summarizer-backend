# prepare_dataset.py
# --------------------------------------------------------
# This script downloads a summarization dataset
# (CNN/DailyMail, XSUM, or Gigaword) from Hugging Face
# and converts it into JSONL format with:
# { "instruction", "input", "output" }
# --------------------------------------------------------

import json
import argparse
from datasets import load_dataset
from tqdm import tqdm  # Progress bar

# Possible column names in Hugging Face datasets for input and summary
POTENTIAL_INPUTS = ["article", "document", "text", "content", "body"]
POTENTIAL_SUMMARIES = ["highlights", "summary", "headline", "title"]

def detect_columns(cols):
    """
    Detect which column is the main text (input) 
    and which column is the summary (output).
    Returns: (input_column_name, summary_column_name)
    """
    input_col = next((c for c in cols if c in POTENTIAL_INPUTS), None)
    sum_col = next((c for c in cols if c in POTENTIAL_SUMMARIES), None)
    return input_col, sum_col

def clean_text(s):
    """Remove newlines and extra spaces from text."""
    if s is None:
        return ""
    return " ".join(str(s).split())

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="xsum",
                        help="Dataset id from Hugging Face (e.g. xsum, cnn_dailymail, gigaword)")
    parser.add_argument("--config", default=None,
                        help="Optional dataset config (e.g. '3.0.0' for cnn_dailymail)")
    parser.add_argument("--split", default="train",
                        help="Which split to use: train, validation, or test")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to save")
    parser.add_argument("--output", default="summarization_dataset.jsonl",
                        help="Output file name")
    parser.add_argument("--min_src_words", type=int, default=30,
                        help="Minimum words in input text")
    parser.add_argument("--min_tgt_words", type=int, default=3,
                        help="Minimum words in summary")
    parser.add_argument("--instruction", default="Summarize the text.",
                        help="Instruction text to prepend")
    args = parser.parse_args()

    # Load the dataset from Hugging Face
    if args.config:
        ds = load_dataset(args.dataset, args.config, split=args.split)
    else:
        ds = load_dataset(args.dataset, split=args.split)

    # Detect which columns contain input text and summaries
    input_col, sum_col = detect_columns(ds.column_names)
    if not input_col or not sum_col:
        raise SystemExit(f"Could not find input/summary columns in {ds.column_names}")

    # Write the selected samples to JSONL
    out_count = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        # Iterate through the dataset, limited by num_samples
        for ex in tqdm(ds.select(range(min(len(ds), args.num_samples)))):
            src = clean_text(ex.get(input_col, ""))  # Clean input text
            tgt = clean_text(ex.get(sum_col, ""))    # Clean summary

            # Skip samples that are too short
            if len(src.split()) < args.min_src_words or len(tgt.split()) < args.min_tgt_words:
                continue

            # Construct the JSON record
            record = {
                "instruction": args.instruction,
                "input": src,
                "output": tgt
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_count += 1

    print(f"✅ Done! Wrote {out_count} examples to {args.output}")

if __name__ == "__main__":
    main()
