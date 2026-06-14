"""Train and serialise the champion model.

Champion = XGBoost on the 13 claim-relation features.  Phase-4 hyperparameter
search (Optuna over depth/lr/estimators) was a *no-op* -- XGB, LightGBM and
CatBoost all landed on the identical matched macro-F1 of 0.9808 -- so the
production model uses the canonical Phase-3 configuration verbatim.  The bundle
ships the fitted feature engineer (its IDF table is the only train-dependent
piece) so inference needs no access to the raw dataset.

    python -m src.train            # writes models/champion.joblib
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from .data_pipeline import load_long_frame, load_split, nearest_length_match
from .feature_engineering import ALL_FEATURES, FeatureEngineer
from .utils import MODELS_DIR

RANDOM_STATE = 42
# Phase-3 champion config (Phase-4 tuning confirmed it is already optimal).
XGB_PARAMS = dict(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.9,
                  colsample_bytree=0.9, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1)
BUNDLE_PATH = MODELS_DIR / "champion.joblib"


def _oof_operating_threshold(X, y, groups) -> float:
    """Macro-F1-maximising decision threshold from group-disjoint out-of-fold predictions
    on TRAIN -- so the operating point never sees the test split (no label leakage)."""
    oof = np.zeros(len(y))
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in sgkf.split(X, y, groups):
        m = XGBClassifier(**XGB_PARAMS).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, (oof >= t).astype(int), average="macro")
        if f > best_f:
            best_f, best_t = f, float(t)
    return round(best_t, 4)


def _metrics(y, pred, score) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 4),
        "precision_hallu": round(float(precision_score(y, pred, pos_label=1, zero_division=0)), 4),
        "recall_hallu": round(float(recall_score(y, pred, pos_label=1, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, score)), 4) if len(set(y)) > 1 else None,
    }


def train(out_path: Path = BUNDLE_PATH) -> dict:
    df = load_long_frame()
    train_df, test_df = load_split(df)
    matched_df = nearest_length_match(test_df, caliper=8)

    fe = FeatureEngineer().fit(train_df.knowledge, train_df.answer)
    t0 = time.time()
    Xtr = fe.transform(train_df.answer, train_df.question, train_df.knowledge)
    feat_time = time.time() - t0
    ytr = train_df.label.values

    model = XGBClassifier(**XGB_PARAMS).fit(Xtr, ytr)
    operating_threshold = _oof_operating_threshold(Xtr, ytr, train_df.qid.values)

    # evaluate at the default 0.5 decision threshold (raw + length-matched)
    def evaluate(frame):
        X = fe.transform(frame.answer, frame.question, frame.knowledge)
        s = model.predict_proba(X)[:, 1]
        return _metrics(frame.label.values, (s >= 0.5).astype(int), s), s

    raw_m, _ = evaluate(test_df)
    mat_m, _ = evaluate(matched_df)

    feature_importance = {name: round(float(v), 4)
                          for name, v in zip(ALL_FEATURES, model.feature_importances_)}

    bundle = {
        "model": model,
        "feature_engineer": fe.to_dict(),
        "feature_names": ALL_FEATURES,
        "operating_threshold": operating_threshold,  # OOF-derived on train (leakage-free)
        "default_threshold": 0.5,
        "xgb_params": XGB_PARAMS,
        "train": {
            "dataset": "HaluEval-QA",
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_matched": int(len(matched_df)),
            "feature_time_s": round(feat_time, 2),
            "us_per_row_features": round(1e6 * feat_time / len(train_df), 1),
        },
        "metrics": {"raw_test": raw_m, "length_matched": mat_m},
        "feature_importance": feature_importance,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path, compress=3)

    print(f"saved {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")
    print(f"raw test    : macro-F1={raw_m['macro_f1']}  recall_hallu={raw_m['recall_hallu']}")
    print(f"length-match: macro-F1={mat_m['macro_f1']}  (Phase-3 reported 0.9808)")
    top = sorted(feature_importance.items(), key=lambda kv: -kv[1])[:5]
    print("top features:", ", ".join(f"{k}={v}" for k, v in top))
    return bundle


if __name__ == "__main__":
    train()
