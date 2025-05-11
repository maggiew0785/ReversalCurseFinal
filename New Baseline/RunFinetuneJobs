#!/usr/bin/env python3
from openai import OpenAI
from pathlib import Path
import os
import random

client = OpenAI()

# Directory where your chat-formatted train JSONLs live
CHAT_DIR = Path(".")

# Hyperparameters for all fine-tune jobs
HYPERPARAMS = {
    "n_epochs": 10,
    "batch_size": 16,
    "learning_rate_multiplier": 0.2
}

# Train datasets
train_datasets = [
    "p2d10", "p2d20", "p2d25", "p2d30",
    "p2d40", "p2d75", "p2d80", "p2d90"
]

# Prepare list of all jobs
jobs = []
for subset in train_datasets:
    train_file = CHAT_DIR / f"chat_{subset}_train.jsonl"
    if not train_file.exists():
        print(f"Warning: {train_file} does not exist. Skipping...")
        continue
    for job_num in range(1, 3):
        jobs.append((subset, job_num, train_file))

total_jobs = len(jobs)
for i in range(0, total_jobs, 4):
    batch = jobs[i:i+4]
    for subset, job_num, train_file in batch:
        # upload
        with open(train_file, "rb") as f:
            upload_resp = client.files.create(
                file=f,
                purpose="fine-tune"
            )
        train_file_id = upload_resp.id
        print(f"Uploaded {train_file.name} → {train_file_id}")

        # create job (OpenAI will pick its own seed)
        ft_job = client.fine_tuning.jobs.create(
            model="gpt-4o-mini-2024-07-18",
            training_file=train_file_id,
            hyperparameters=HYPERPARAMS,
            suffix=f"reverse-{subset}-job{job_num}"
        )

        print(f"Started fine-tune job {job_num} for {subset}: {ft_job.id}")
        print("-" * 50)

    if i + 4 < total_jobs:
        input("Press Enter to start the next batch of 4 jobs...") # Fine-tune limit (4 at a time)

print("All fine-tuning jobs have been started.")
