#!/usr/bin/env python3
import json
from pathlib import Path

DATA_DIR = Path("data/reverse_experiments/june_version_7921032488")
OUT_DIR  = Path("chat_format")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a factual chatbot."

# Process every JSONL file containing prompts in the DATA_DIR (original API formatted data)
for src in DATA_DIR.glob("*_prompts_*.jsonl"):
    # Derive output filename by parsing the original name
    # e.g., "p2d_prompts_train.jsonl" -> subset="p2d", split="train"
    # or "p2d_reverse_prompts_test.jsonl" -> subset="p2d_reverse", split="test"
    name_parts = src.name.split("_prompts_")
    subset = name_parts[0]
    split = name_parts[1].replace(".jsonl", "")
    dst = OUT_DIR / f"chat_{subset}_{split}.jsonl"

    with src.open() as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            chat = {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": obj["prompt"]},
                    {"role": "assistant", "content": obj["completion"]}
                ]
            }
            fout.write(json.dumps(chat, ensure_ascii=False) + "\n")

print("Converted all train and test splits to chat format in:", OUT_DIR)
