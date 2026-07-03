"""Evaluation: raw metrics plus the length-matched control from the notebooks.

HaluEval-QA hallucinations are ~4.8x longer than grounded answers, so raw scores
overstate real grounding skill. ``nearest_length_match`` (phase 3-5 version,
caliper 8 chars) pairs each hallucination with the closest-length grounded
answer and drops the rest; every reported number should include both views.
"""

import bisect

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_pred, y_score=None):
    """The notebook's metric block: accuracy, macro-F1, balanced acc, P/R, AUC."""
    metrics = {
        "n": int(len(y_true)),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "balanced_acc": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision_hallu": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall_hallu": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
    }
    if y_score is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_score), 4)
    return metrics


def nearest_length_match(frame, caliper=8):
    """Length-matched control: greedy nearest-length pairing within ``caliper`` chars.

    For each hallucinated row (sorted by answer length), take the closest-length
    unused grounded row; keep the pair only if the gap is <= caliper. Ported
    verbatim from the phase-3/4/5 notebooks.
    """
    g0 = frame[frame.label == 0].sort_values("ans_chars")
    g1 = frame[frame.label == 1].sort_values("ans_chars")
    chars0 = g0.ans_chars.tolist()
    idx0 = g0.index.tolist()
    keep0, keep1 = [], []
    for c1, i1 in zip(g1.ans_chars.tolist(), g1.index.tolist()):
        if not chars0:
            break
        pos = bisect.bisect_left(chars0, c1)
        best = None
        for j in (pos - 1, pos, pos + 1):
            if 0 <= j < len(chars0):
                dist = abs(chars0[j] - c1)
                if best is None or dist < best[0]:
                    best = (dist, j)
        dist, j = best
        if dist <= caliper:
            keep1.append(i1)
            keep0.append(idx0[j])
            del chars0[j]
            del idx0[j]
    return frame.loc[keep0 + keep1]


def evaluate_model(scorer, test_frame, caliper=8):
    """Score a fitted scorer on the raw test set and its length-matched control."""
    scores = scorer.predict_proba(test_frame)
    preds = (scores >= scorer.threshold).astype(int)
    raw = evaluate_predictions(test_frame.label.values, preds, scores)

    matched = nearest_length_match(test_frame, caliper=caliper)
    if len(matched):
        m_scores = scorer.predict_proba(matched)
        m_preds = (m_scores >= scorer.threshold).astype(int)
        matched_metrics = evaluate_predictions(matched.label.values, m_preds, m_scores)
    else:
        matched_metrics = {"n": 0}  # no length-comparable pairs within the caliper

    return {"raw": raw, "length_matched": matched_metrics}
