"""Evaluation suite for the champion tree.

    python -m src.evaluate        # prints raw + length-matched metrics on the test split

Reports the primary metric (macro-F1) raw and length-matched, plus the confusion
matrix and the verbatim-suspect blind-spot count -- the slice the router exists to
cover.
"""
from __future__ import annotations

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from .data_pipeline import load_long_frame, load_split, nearest_length_match
from .predict import ConversationQualityScorer


def metrics(y, pred, score) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 4),
        "precision_hallu": round(float(precision_score(y, pred, pos_label=1, zero_division=0)), 4),
        "recall_hallu": round(float(recall_score(y, pred, pos_label=1, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, score)), 4) if len(set(y)) > 1 else None,
    }


def evaluate(threshold: float | None = None) -> dict:
    scorer = ConversationQualityScorer.load()
    df = load_long_frame()
    _, test = load_split(df)
    matched = nearest_length_match(test, caliper=8)

    s_raw, p_raw = scorer.score_batch(test.knowledge, test.question, test.answer, threshold)
    s_mat, p_mat = scorer.score_batch(matched.knowledge, matched.question, matched.answer, threshold)
    raw_m = metrics(test.label.values, p_raw, s_raw)
    mat_m = metrics(matched.label.values, p_mat, s_mat)
    cm = confusion_matrix(test.label.values, p_raw).tolist()

    # the router's reason for existing: verbatim-grounded hallucinations the tree misses
    import numpy as np
    feats = scorer.fe.transform(test.answer, test.question, test.knowledge)
    is_substr = feats[:, scorer.feature_names.index("is_substr")] >= 1.0
    missed = (test.label.values == 1) & (p_raw == 0)
    blind = int((is_substr & missed).sum())

    print(f"raw test     macro-F1={raw_m['macro_f1']}  recall_hallu={raw_m['recall_hallu']}  n={len(test)}")
    print(f"length-match macro-F1={mat_m['macro_f1']}  n={len(matched)}")
    print(f"confusion (raw) [[TN,FP],[FN,TP]] = {cm}")
    print(f"verbatim hallucinations the tree misses (router target): {blind}")
    return {"raw_test": raw_m, "length_matched": mat_m, "confusion_raw": cm, "tree_blind_verbatim": blind}


if __name__ == "__main__":
    evaluate()
