import os
from attr import define
from dotenv import load_dotenv

from scripts.celebrity_relations.crawl_celebrities import SAVE_DIR
from src.common import flatten, load_from_txt
from joblib import Memory
from torch.utils.data import Dataset
from typing import Optional
from math import fsum


# ── NEW: import and instantiate the v1 client
from openai import OpenAI

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

UNKNOWN_STR    = "I don't know."
SYSTEM_PROMPT  = f'''You are a helpful and terse assistant. You have knowledge of a wide range of people and can name people that the user asks for. If the answer is unknown or not applicable, answer with "{UNKNOWN_STR}"'''
MODEL          = "gpt-4o-mini-2024-07-18"
CELEBRITIES    = load_from_txt(os.path.join(SAVE_DIR, "top_celebrities.txt"))
SAVE_PATH      = "data/celebrity_relations"
DF_SAVE_PATH   = os.path.join(SAVE_PATH, "gpt-4o-mini_reversal_test_results_logprob.csv")

memory = Memory("cache/celebrity_relations", verbose=0)


@define
class ParentChildPair:
    child: str
    parent: str
    parent_type: str  # 'mother' or 'father'
    logprob: Optional[float] = None

    def ask_for_parent(self) -> str:
        return f"Who is {self.child}'s {self.parent_type}?"

    def ask_for_child(self) -> str:
        return f"Name a child of {self.parent}."

    def create_parent_query_chat_pair(self) -> list[dict]:
        return [
            {"role": "user",      "content": self.ask_for_parent()},
            {"role": "assistant", "content": self.parent},
        ]

    def create_child_query_chat_pair(self) -> list[dict]:
        return [
            {"role": "user",      "content": self.ask_for_child()},
            {"role": "assistant", "content": self.child},
        ]


def parse_response(response: str) -> str | None:
    words = response.strip().split()
    if (
        #response.startswith(UNKNOWN_STR[:5]) or not
        (2 <= len(words) <= 5)
        or not all(token[0].isupper() for token in words)
    ):
        return None
    return response.strip()


def get_initial_messages() -> list[dict]:
    system = {"role": "system", "content": SYSTEM_PROMPT}
    few_shot = flatten([
        ParentChildPair("Malia Obama",  "Barack Obama", "father").create_child_query_chat_pair(),
        ParentChildPair("Elon Musk",    "Maye Musk",    "mother").create_parent_query_chat_pair(),
        ParentChildPair("Kathy Pratt",  UNKNOWN_STR,    "mother").create_parent_query_chat_pair(),
    ])
    return [system] + few_shot


def get_parent_query(name: str, parent_type: str) -> list[dict]:
    msgs = get_initial_messages()
    msgs.append({"role":"user", "content": ParentChildPair(name, UNKNOWN_STR, parent_type).ask_for_parent()})
    return msgs


def get_child_query(parent: str) -> list[dict]:
    msgs = get_initial_messages()
    msgs.append({"role":"user", "content": ParentChildPair(UNKNOWN_STR, parent, "mother").ask_for_child()})
    return msgs


def query_parent_initial(
    name: str,
    parent_type: str,
    model_name: str = MODEL,
    num_queries: int = 1,
) -> ParentChildPair | None:
    msgs = get_parent_query(name, parent_type)
    # single‐shot call
    resp = _client.chat.completions.create(
        model=model_name,
        messages=msgs,
        logprobs=True,
        max_tokens=50,
        temperature=0.0,
    )

    choice = resp.choices[0]
    text = choice.message.content

    if text.startswith(UNKNOWN_STR[:5]):
        return None

    lp_obj = getattr(choice, "logprobs", None)
    if lp_obj and getattr(lp_obj, "content", None):
        total_lp = fsum(tok.logprob for tok in lp_obj.content)
    else:
        total_lp = float("-inf")

    return ParentChildPair(name, text, parent_type, total_lp) if text else None


def get_parents(name: str) -> tuple[ParentChildPair | None, ParentChildPair | None]:
    """Assumes name is female (all celebrities on the list are for some reason)."""
    return query_parent_initial(name, "mother"), query_parent_initial(name, "father")


@memory.cache
def get_child(
    parent: str,
    parent_type: str,
    child_name: str,
    model_name: str = MODEL,
    num_queries_per_celebrity: int = 10,
) -> ParentChildPair:
    msgs        = get_child_query(parent)

    best_correct_pair = None
    best_correct_score = float("-inf")
    best_any_pair = None
    best_any_score = float("-inf")


    for i in range(num_queries_per_celebrity):
        resp = _client.chat.completions.create(
            model=model_name,
            messages=msgs,
            logprobs=True,
            max_tokens=50,
            temperature=00.0,
        )

        choice   = resp.choices[0]
        content  = choice.message.content
        lp_obj   = getattr(choice, "logprobs", None)

        ''''# ---------- DEBUG PRINTS ----------
        print(f"\n=== TRY {i+1} for parent '{parent}' ===")
        print("RAW content :", repr(content))
        if lp_obj:
            try:
                print("RAW logprobs:", lp_obj.model_dump_json(indent=2))
            except Exception as e:
                print("Could not dump logprobs JSON:", e)
        else:
            print("logprobs = None")
        # ----------------------------------'''

        # ---- compute total log‑prob (even for refusals) ----
        if lp_obj and getattr(lp_obj, "content", None):
            try:
                total_lp = fsum(tok.logprob for tok in lp_obj.content)
            except Exception as e:
                print("⚠ Failed to sum token logprobs:", e)
                total_lp = float("-inf")
        else:
            total_lp = float("-inf")

        print(f"total_lp = {total_lp:.9f}   content = {content}")

        if best_any_pair is None or total_lp > best_any_score:
            best_any_pair = ParentChildPair(content, parent, parent_type, total_lp)
            best_any_score = total_lp
            print("📝 New best overall answer")

        # ✅ Update best correct match (starts with child_name)
        if content and content.startswith(child_name):
            if best_correct_pair is None or total_lp > best_correct_score:
                best_correct_pair = ParentChildPair(content, parent, parent_type, total_lp)
                best_correct_score = total_lp
                print("🏆 New best correct answer")
            else:
                print("➖ Correct but lower score")
        else:
            print("↪ Not a correct child match")

    # ✅ Return best correct match, or fall back to best overall
    if best_correct_pair:
        return best_correct_pair
    else:
        print(f"No correct prediction for '{child_name}', returning None with logprob.")
        return ParentChildPair(None, parent, parent_type, total_lp)


class PromptCompletionDataset(Dataset):
    def __init__(self, prompts, completions, max_length=500):
        self.prompts     = prompts
        self.completions = completions
        self.max_length  = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.completions[idx]
