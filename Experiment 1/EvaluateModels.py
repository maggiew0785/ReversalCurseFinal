#!/usr/bin/env python3
from openai import OpenAI
import json
import time
import re

client = OpenAI()

# List of (fine-tuned job ID, ratio mix + seed number)
job_configs = [
    ("ftjob-X8ezDrpYffYPSDSBRSLjpftw", "p2d100-1"),
    ("ftjob-PfWkfR8x6H6LJchvmLgS0kOA", "p2d100-2"),
    ("ftjob-0ZmRym7W3GiLcNvSQFPUtgoH", "p2d100-3")
]

# Test splits: (label, path)
test_splits = [
    ("name→desc", "chat_p2d_test.jsonl"),
    ("desc→name", "chat_p2d_reverse_test.jsonl")
]

'''
If testing the p2d0 models:
test_splits = [
    ("name→desc", "chat_d2p_test.jsonl"),
    ("desc→name", "chat_d2p_reverse_test.jsonl")
]
''''

def normalize(text):
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()

def eval_file(label, path, model):
    total, correct = 0, 0
    start = time.time()
    print(f"\nEvaluating '{label}' from {path} on model {model}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            data = json.loads(line)
            messages = data["messages"]
            prompt = messages[:-1]
            true = messages[-1]["content"].strip()
            resp = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=0
            )
            pred = resp.choices[0].message.content.strip()
            total += 1
            norm_true = normalize(true)
            norm_pred = normalize(pred)
            norm_true = normalize(true)
            norm_pred = normalize(pred)
            if norm_true in norm_pred or norm_pred in norm_true:
                correct += 1
            if i <= 3 and norm_true not in norm_pred:
                print(f"Mismatch #{i}:")
                print("  Prompt:", prompt[1]["content"])
                print("  Pred:  ", pred)
                print("  True:  ", true)
            if i % 100 == 0:
                print(f"  {i} examples — accuracy: {correct/total:.2%}")
    elapsed = time.time() - start
    print(f"Completed '{label}': {correct}/{total} = {correct/total:.2%} in {elapsed:.2f}s")
    return correct, total

# main evaluation loop
results = {}
for job_id, config_label in job_configs:
    job = client.fine_tuning.jobs.retrieve(job_id)
    model = job.fine_tuned_model
    job_results = {}
    for split_label, path in test_splits:
        correct, total = eval_file(f"{config_label} - {split_label}", path, model)
        job_results[split_label] = correct / total
    results[config_label] = job_results

# print summary clearly with ratio mix + seed number
print("\nSummary:")
for config_label, metrics in results.items():
    print(config_label)
    for split_label, acc in metrics.items():
        print(f"  {split_label}: {acc:.2%}")
    print()
