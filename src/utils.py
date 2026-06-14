"""Shared helpers for the AI Agent Conversation Quality Scorer.

Tokenisation and the grounding-overlap primitive are kept here so the production
pipeline computes features *byte-for-byte* the way the Phase 1-5 notebooks did.
Any drift between research and production features silently invalidates the saved
model, so these functions are the single source of truth for both.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

# ---- tokenisation (identical to notebooks/phase4_tuning.ipynb) ----
STOP = set(
    "a an the of to in on at for and or is was were are be been by with as that this it from".split()
)
_WORD = re.compile(r"[a-z0-9]+")


def toks(s: str) -> List[str]:
    """Content tokens: lowercase alphanumerics with stopwords removed."""
    return [t for t in _WORD.findall(str(s).lower()) if t not in STOP]


def toks_all(s: str) -> List[str]:
    """All lowercase alphanumeric tokens (no stopword removal)."""
    return _WORD.findall(str(s).lower())


def grounding_overlap(ans: str, know: str) -> float:
    """Fraction of an answer's content tokens that appear in the knowledge.

    This is the Phase-2 lexical baseline feature and the 13th (``ground_overlap``)
    input to the champion tree.
    """
    a = set(toks(ans))
    k = set(toks(know))
    return (len(a & k) / len(a)) if a else 0.0


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until the folder containing ``data/raw/qa_data.json`` (or the
    project marker ``config/config.yaml``) is found."""
    p = (start or Path(__file__).resolve()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "data" / "raw" / "qa_data.json").exists() or (cand / "config" / "config.yaml").exists():
            return cand
    raise RuntimeError("repo root not found (no data/raw/qa_data.json or config/config.yaml above cwd)")


REPO_ROOT = find_repo_root()
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"


def load_config() -> dict:
    """Load config/config.yaml, falling back gracefully if PyYAML is absent."""
    cfg_path = REPO_ROOT / "config" / "config.yaml"
    try:
        import yaml  # type: ignore

        return yaml.safe_load(cfg_path.read_text())
    except Exception:
        return {}
