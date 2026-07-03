"""Training pipeline for the champion model: 13 features -> XGBoost.

The hyperparameters are the phase-3 defaults. Phase 4 ran 120 Optuna trials over
9 knobs and moved matched macro-F1 by +0.0000, so the defaults ARE the recorded
champion (raw 0.9967 / length-matched 0.9808 macro-F1).
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from .features import FEATURE_NAMES, Featurizer

RANDOM_STATE = 42

# Phase-3 config, confirmed champion by the phase-4 tuning study.
CHAMPION_XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}

# Frozen phase-1 split, reused by every phase (see config/config.yaml).
DEFAULT_QID_FILE = Path("results/phase1_split_qids.json")


def split_by_qid(frame, test_size=0.2, random_state=RANDOM_STATE, qid_file=None):
    """Grouped train/test split by qid so an item's two answers never straddle it.

    If ``qid_file`` points to a frozen split (phase1_split_qids.json), that exact
    split is reused; otherwise a fresh GroupShuffleSplit reproduces the notebook
    call (test_size=0.2, random_state=42).
    """
    if qid_file is not None and Path(qid_file).exists():
        split = json.loads(Path(qid_file).read_text())
        train_qids, test_qids = set(split["train_qids"]), set(split["test_qids"])
        train = frame[frame.qid.isin(train_qids)]
        test = frame[frame.qid.isin(test_qids)]
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(frame, frame.label, groups=frame.qid))
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    assert not (set(train.qid) & set(test.qid)), "qid leakage between train and test"
    return train, test


class HallucinationScorer:
    """Featurizer + XGBoost champion, with save/load and single-example scoring."""

    def __init__(self, xgb_params=None, threshold=0.5):
        self.xgb_params = dict(xgb_params or CHAMPION_XGB_PARAMS)
        self.threshold = threshold
        self.featurizer = Featurizer()
        self.model = XGBClassifier(**self.xgb_params)

    def fit(self, train_frame):
        """Fit IDF on the training texts, then the XGB head on the 13 features."""
        self.featurizer.fit_from_frame(train_frame)
        features = self.featurizer.transform(train_frame)
        self.model.fit(features.values, train_frame.label.values)
        return self

    def predict_proba(self, frame):
        """P(hallucinated) for each row of a pair frame."""
        features = self.featurizer.transform(frame)
        return self.model.predict_proba(features.values)[:, 1]

    def predict(self, frame):
        return (self.predict_proba(frame) >= self.threshold).astype(int)

    def score(self, question, answer, knowledge):
        """P(hallucinated) for a single (question, answer, knowledge) triple."""
        feats = self.featurizer.compute_features(question, answer, knowledge)
        row = np.array([[feats[name] for name in FEATURE_NAMES]], dtype=np.float32)
        return float(self.model.predict_proba(row)[0, 1])

    def save(self, model_dir):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_dir / "xgb_model.json")
        (model_dir / "featurizer.json").write_text(json.dumps(self.featurizer.to_dict()))
        meta = {
            "feature_names": FEATURE_NAMES,
            "xgb_params": self.xgb_params,
            "threshold": self.threshold,
        }
        (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return model_dir

    @classmethod
    def load(cls, model_dir):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text())
        scorer = cls(xgb_params=meta["xgb_params"], threshold=meta["threshold"])
        scorer.model.load_model(model_dir / "xgb_model.json")
        scorer.featurizer = Featurizer.from_dict(
            json.loads((model_dir / "featurizer.json").read_text())
        )
        return scorer


def train_champion(frame, qid_file=None, xgb_params=None):
    """Split, train, and return (scorer, train_frame, test_frame)."""
    train, test = split_by_qid(frame, qid_file=qid_file)
    scorer = HallucinationScorer(xgb_params=xgb_params).fit(train)
    return scorer, train, test
