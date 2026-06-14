"""Production inference: score an agent answer for grounding/hallucination.

    from src.predict import ConversationQualityScorer
    scorer = ConversationQualityScorer.load()
    scorer.score(knowledge="...", question="...", answer="...")

Returns the tree probability, the binary verdict, the full 13-feature breakdown,
and -- crucially -- the ``verbatim_suspect`` flag.  A verbatim suspect is an
answer the tree calls GROUNDED *because* it is copied verbatim from the source;
Phase 5 showed these are exactly the answers the tree is blind to when the quote
is the wrong answer to the question.  The router (:mod:`src.router`) acts on it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

from .feature_engineering import FeatureEngineer
from .utils import MODELS_DIR

BUNDLE_PATH = MODELS_DIR / "champion.joblib"


class ConversationQualityScorer:
    def __init__(self, model, feature_engineer: FeatureEngineer, feature_names,
                 default_threshold: float = 0.5, operating_threshold: float = 0.1634,
                 meta: Optional[dict] = None) -> None:
        self.model = model
        self.fe = feature_engineer
        self.feature_names = feature_names
        self.default_threshold = default_threshold
        self.operating_threshold = operating_threshold
        self.meta = meta or {}

    @classmethod
    def load(cls, path: Path = BUNDLE_PATH) -> "ConversationQualityScorer":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `python -m src.train` first.")
        b = joblib.load(path)
        return cls(
            model=b["model"],
            feature_engineer=FeatureEngineer.from_dict(b["feature_engineer"]),
            feature_names=b["feature_names"],
            default_threshold=b.get("default_threshold", 0.5),
            operating_threshold=b.get("operating_threshold", 0.1634),
            meta={k: b[k] for k in ("metrics", "feature_importance", "train", "xgb_params") if k in b},
        )

    def score(self, knowledge: str, question: str, answer: str,
              threshold: Optional[float] = None) -> Dict:
        thr = self.default_threshold if threshold is None else threshold
        feats = self.fe.row_dict(answer, question, knowledge)
        X = np.array([[feats[n] for n in self.feature_names]], dtype=np.float32)
        prob = float(self.model.predict_proba(X)[:, 1][0])
        label = int(prob >= thr)
        is_substr = feats.get("is_substr", 0.0) >= 1.0
        # The tree's blind spot: verbatim-grounded *and* the tree believes it.
        verbatim_suspect = bool(is_substr and label == 0)
        return {
            "prob_hallucinated": prob,
            "label": label,                       # 0 = grounded, 1 = hallucinated
            "verdict": "HALLUCINATED" if label == 1 else "GROUNDED",
            "threshold": thr,
            "features": feats,
            "is_verbatim": is_substr,
            "verbatim_suspect": verbatim_suspect,
            "top_feature": "lcs_char_ratio",      # the one feature carrying unique signal
        }

    def score_batch(self, knowledge, questions, answers, threshold: Optional[float] = None):
        thr = self.default_threshold if threshold is None else threshold
        X = self.fe.transform(answers, questions, knowledge)
        probs = self.model.predict_proba(X)[:, 1]
        return probs, (probs >= thr).astype(int)


if __name__ == "__main__":
    s = ConversationQualityScorer.load()
    # a verbatim-correct quote that is the *wrong* answer to the question -> tree's blind spot
    demo = s.score(
        knowledge="Arthur's Magazine (1844-1846) was an American literary periodical published in "
                   "Philadelphia. First for Women is a woman's magazine published by Bauer Media Group.",
        question="Which magazine was started first, Arthur's Magazine or First for Women?",
        answer="First for Women",
    )
    print({k: demo[k] for k in ("prob_hallucinated", "verdict", "is_verbatim", "verbatim_suspect")})
