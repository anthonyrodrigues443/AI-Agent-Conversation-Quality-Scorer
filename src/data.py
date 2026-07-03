"""HaluEval-QA loading: download, parse, and expand into grounded/hallucinated pairs.

Mirrors the notebook loader: each raw item becomes two rows sharing a ``qid`` —
the grounded answer (label 0) and the ChatGPT-hallucinated answer (label 1).
"""

import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

# Matches config/config.yaml (dataset.source / dataset.raw_path).
HALUEVAL_QA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
DEFAULT_RAW_PATH = Path("data/raw/qa_data.json")


def download_qa_data(dest=DEFAULT_RAW_PATH, url=HALUEVAL_QA_URL, force=False):
    """Download qa_data.json to ``dest`` (skipped if it already exists)."""
    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        shutil.move(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    return dest


def load_qa_records(path):
    """Parse the JSON-lines file into raw HaluEval records."""
    text = Path(path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def build_pair_frame(records):
    """Expand raw records into the long frame: 2 rows per item, qid preserved."""
    rows = []
    for qid, record in enumerate(records):
        rows.append(
            {
                "qid": qid,
                "knowledge": record["knowledge"],
                "question": record["question"],
                "answer": record["right_answer"],
                "label": 0,
            }
        )
        rows.append(
            {
                "qid": qid,
                "knowledge": record["knowledge"],
                "question": record["question"],
                "answer": record["hallucinated_answer"],
                "label": 1,
            }
        )
    frame = pd.DataFrame(rows).reset_index(drop=True)
    frame["ans_chars"] = frame.answer.str.len()
    return frame


def load_qa_data(path):
    """Load an already-downloaded qa_data.json into the expanded pair frame."""
    return build_pair_frame(load_qa_records(path))
