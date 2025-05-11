import argparse
import os
from typing import List, Tuple

import openai
import pandas as pd
from tqdm import tqdm

from src.tasks.celebrity_relations.parent_reversals import (
    DF_SAVE_PATH,
    SAVE_PATH,
    ParentChildPair,
    get_child_query,
    get_parent_query,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "gpt-4o-mini"  # This script is *only* intended for gpt-4o-mini.
NUM_QUERIES_PER_CELEBRITY = 6  # Number of completions to sample per question.
openai.api_key = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# GPT helpers
# ---------------------------------------------------------------------------

def _chat_generate_multiple_messages(messages: List[dict], n: int = 1, model: str = MODEL_NAME) -> List[str | None]:
    """Call the OpenAI chat API *n* times synchronously and return content strings.

    If the request fails, the corresponding list item is `None` so downstream logic
    can treat it as an incorrect completion.
    """
    responses: List[str | None] = []
    for _ in range(n):
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=50,
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            responses.append(content)
        except Exception as exc:  # noqa: BLE001
            print("OpenAI API error:", exc)
            responses.append(None)
    return responses


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def query_parent_test(child: str, parent_type: str, parent: str) -> float:
    """Return the fraction of completions where the model names the correct parent."""
    messages = get_parent_query(child, parent_type)
    responses = _chat_generate_multiple_messages(messages, NUM_QUERIES_PER_CELEBRITY)
    correct = [r for r in responses if r and r.startswith(parent)]
    return len(correct) / len(responses)


def query_child_test(parent: str, child: str) -> float:
    """Return the fraction of completions where the model names the correct child."""
    messages = get_child_query(parent)
    responses = _chat_generate_multiple_messages(messages, NUM_QUERIES_PER_CELEBRITY)
    correct = [r for r in responses if r and r.startswith(child)]
    return len(correct) / len(responses)


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def test_can_reverse(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    """Compute parent- and child-finding accuracy for each row in *df*.

    The incoming CSV must contain columns:
        child,parent,parent_type,child_prediction,
        gpt-4o-mini_child_logprob,gpt-4o-mini_parent_logprob,can_reverse
    """
    parent_acc: List[float] = []
    child_acc: List[float] = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        parent_acc.append(query_parent_test(row["child"], row["parent_type"], row["parent"]))

        # Only attempt reversal when it is logically valid.
        if bool(row["can_reverse"]):
            child_acc.append(query_child_test(row["parent"], row["child"]))
        else:
            child_acc.append(float("nan"))

    return parent_acc, child_acc


def reversal_test(df: pd.DataFrame) -> pd.DataFrame:
    """Run the reversal‑curse test and return a merged results DataFrame."""
    parent_acc, child_acc = test_can_reverse(df)

    df[f"{MODEL_NAME}_can_find_parent"] = parent_acc
    df[f"{MODEL_NAME}_can_find_child"] = child_acc
    return df


# ---------------------------------------------------------------------------
# CLI / entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run reversal‑curse evaluation with gpt‑4o‑mini.")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Model name (must be 'gpt-4o-mini'; other values will be ignored).",
    )
    return parser.parse_args()


def main(model_name: str):
    if model_name != MODEL_NAME:
        print(f"⚠️  This script is refactored for {MODEL_NAME} only. Ignoring supplied model '{model_name}'.")

    df = pd.read_csv("data/celebrity_relations/parent_child_pairs.csv")
    #child, parent, parent_type, child_prediction, can_reverse

    # Verify required columns exist.
    required = {
        "child",
        "parent",
        "parent_type",
        "child_prediction",
        "can_reverse",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")

    results = reversal_test(df)

    os.makedirs(SAVE_PATH, exist_ok=True)
    out_path = os.path.join(SAVE_PATH, f"{MODEL_NAME}_reversal_test_results.csv")
    results.to_csv(out_path, index=False)

    print(f"Saved results → {out_path}")
    print(results.head())


if __name__ == "__main__":
    args = parse_args()
    main(args.model)
