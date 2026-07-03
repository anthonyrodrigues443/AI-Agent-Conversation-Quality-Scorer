import pandas as pd
import pytest

from src.evaluate import evaluate_predictions, nearest_length_match


def test_evaluate_predictions_perfect():
    metrics = evaluate_predictions([0, 1, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["macro_f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_evaluate_predictions_known_values():
    metrics = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx(0.7333, abs=1e-4)
    assert metrics["precision_hallu"] == pytest.approx(0.6667, abs=1e-4)
    assert metrics["recall_hallu"] == 1.0


def make_frame(grounded_lengths, hallucinated_lengths):
    rows = [{"label": 0, "ans_chars": n} for n in grounded_lengths]
    rows += [{"label": 1, "ans_chars": n} for n in hallucinated_lengths]
    return pd.DataFrame(rows)


def test_nearest_length_match_pairs_within_caliper():
    frame = make_frame(grounded_lengths=[10, 50], hallucinated_lengths=[12, 100])
    matched = nearest_length_match(frame, caliper=8)
    # 12 pairs with 10 (gap 2); 100's nearest remaining grounded is 50 (gap 50) -> dropped
    assert len(matched) == 2
    assert sorted(matched.label.tolist()) == [0, 1]
    assert sorted(matched.ans_chars.tolist()) == [10, 12]


def test_nearest_length_match_is_balanced():
    frame = make_frame(
        grounded_lengths=[10, 11, 12, 40, 41], hallucinated_lengths=[10, 12, 39, 42, 90]
    )
    matched = nearest_length_match(frame, caliper=8)
    counts = matched.label.value_counts()
    assert counts[0] == counts[1]


def test_nearest_length_match_consumes_each_grounded_once():
    frame = make_frame(grounded_lengths=[20], hallucinated_lengths=[20, 21])
    matched = nearest_length_match(frame, caliper=8)
    assert len(matched) == 2  # the single grounded row can only be used once
