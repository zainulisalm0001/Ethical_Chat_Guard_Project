import re
import pandas as pd
from datasets import load_dataset

SEED = 42

CATEGORY_MARKERS = {
    "urgency": [
        "do it now", "immediately", "right away", "asap", "urgent", "without delay",
        "time is running out", "before it's too late", "act fast", "no time to waste",
        "make a decision now", "this can’t wait", "last chance", "take action now"
    ],
    "inevitability": [
        "no other option", "only choice", "must", "have to", "there is no alternative",
        "nothing else will work", "the only way", "you have no choice", "this is mandatory",
        "non-negotiable", "required"
    ],
    "emotional_pressure": [
        "you will regret", "don't disappoint", "if you care", "you'd be foolish",
        "you'll be sorry", "you'll thank me later", "you owe it", "let me down",
        "selfish not to", "for your own good"
    ],
    "dismissal_of_alternatives": [
        "don’t overthink", "stop questioning", "just do it", "ignore the rest",
        "nothing else makes sense", "there’s nothing to debate", "no reason to hesitate"
    ],
}

def extract_last_assistant_reply(conversation: str) -> str:
    if not isinstance(conversation, str):
        return ""
    parts = conversation.split("Assistant:")
    if len(parts) < 2:
        return ""
    return parts[-1].strip()

def count_markers(text: str) -> int:
    t = (text or "").lower()
    total = 0
    for phrases in CATEGORY_MARKERS.values():
        for p in phrases:
            if re.search(r"\b" + re.escape(p) + r"\b", t):
                total += 1
    return total

def build_split(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    df["assistant_reply"] = df["text"].apply(extract_last_assistant_reply)
    df["marker_count"] = df["assistant_reply"].apply(count_markers)
    df["source"] = source_name
    df = df[df["assistant_reply"].str.len() >= 30].reset_index(drop=True)
    return df[["assistant_reply", "marker_count", "source"]]

def main():
    ds = load_dataset("Anthropic/hh-rlhf")

    chosen = pd.DataFrame(ds["train"])["chosen"].to_frame(name="text")
    rejected = pd.DataFrame(ds["train"])["rejected"].to_frame(name="text")

    chosen_df = build_split(chosen, "chosen")
    rejected_df = build_split(rejected, "rejected")

    all_df = pd.concat([chosen_df, rejected_df], ignore_index=True)

    coercive = all_df[all_df["marker_count"] >= 2].copy()
    non_coercive = all_df[all_df["marker_count"] == 0].copy()

    n = min(len(coercive), len(non_coercive), 20000)

    coercive = coercive.sample(n=n, random_state=SEED)
    non_coercive = non_coercive.sample(n=n, random_state=SEED)

    coercive["label"] = 1
    non_coercive["label"] = 0

    final_df = pd.concat([coercive, non_coercive], ignore_index=True).sample(frac=1.0, random_state=SEED)
    final_df = final_df[["assistant_reply", "label", "marker_count", "source"]].reset_index(drop=True)

    final_df.to_csv("data/hh_coercion_weak_labels.csv", index=False)
    print("Saved: /Users/zainulislam/Downloads/practice exam/Ethical_Chat_Guard_Project/data/hh_coercion_weak_labels.csv")
    print(final_df["label"].value_counts().to_dict())

if __name__ == "__main__":
    main()
