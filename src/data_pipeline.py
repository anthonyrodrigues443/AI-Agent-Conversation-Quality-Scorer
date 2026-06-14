"""Data loading + the frozen group split + the length-matched control.

HaluEval-QA ships 10,000 items, each with a ``right_answer`` (grounded, label 0)
and a ``hallucinated_answer`` (label 1).  We unfold them into a 20,000-row "long"
frame.  Hallucinated answers are ~4.8x longer than grounded ones, so every
leaderboard in this project reports a *length-matched* macro-F1 alongside the raw
one -- the matched control is rebuilt here exactly as in Phase 1.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import pandas as pd

from .utils import RESULTS_DIR, REPO_ROOT, grounding_overlap

RAW_PATH = REPO_ROOT / "data" / "raw" / "qa_data.json"
SPLIT_PATH = RESULTS_DIR / "phase1_split_qids.json"


def load_long_frame(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the JSONL dump and unfold each item into grounded + hallucinated rows."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Download it per data/README.md "
            "(https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json)."
        )
    records = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    rows = []
    for qid, r in enumerate(records):
        rows.append({"qid": qid, "knowledge": r["knowledge"], "question": r["question"],
                     "answer": r["right_answer"], "label": 0})
        rows.append({"qid": qid, "knowledge": r["knowledge"], "question": r["question"],
                     "answer": r["hallucinated_answer"], "label": 1})
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["ans_chars"] = df.answer.str.len()
    df["ground_overlap"] = [grounding_overlap(a, k) for a, k in zip(df.answer, df.knowledge)]
    return df


def load_split(df: pd.DataFrame, split_path: Path = SPLIT_PATH):
    """Return (train, test) using the frozen qid split so every phase shares it."""
    split = json.load(open(split_path))
    train_qids, test_qids = set(split["train_qids"]), set(split["test_qids"])
    train = df[df.qid.isin(train_qids)]
    test = df[df.qid.isin(test_qids)]
    return train, test


def nearest_length_match(frame: pd.DataFrame, caliper: int = 8) -> pd.DataFrame:
    """Greedily pair each hallucination with the nearest-length grounded answer
    (within ``caliper`` characters), dropping the rest -- isolates real grounding
    signal from the answer-length shortcut."""
    g0 = frame[frame.label == 0].sort_values("ans_chars")
    g1 = frame[frame.label == 1].sort_values("ans_chars")
    chars0 = g0.ans_chars.tolist()
    idx0 = g0.index.tolist()
    keep0, keep1 = [], []
    for c1, i1 in zip(g1.ans_chars.tolist(), g1.index.tolist()):
        if not chars0:
            break
        p = bisect.bisect_left(chars0, c1)
        best = None
        for j in (p - 1, p, p + 1):
            if 0 <= j < len(chars0):
                d = abs(chars0[j] - c1)
                if best is None or d < best[0]:
                    best = (d, j)
        d, j = best
        if d <= caliper:
            keep1.append(i1)
            keep0.append(idx0[j])
            del chars0[j]
            del idx0[j]
    return frame.loc[keep0 + keep1]
