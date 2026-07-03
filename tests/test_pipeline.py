import numpy as np

from src.evaluate import evaluate_model
from src.pipeline import HallucinationScorer, split_by_qid, train_champion


def test_end_to_end_fit_predict(synthetic_pairs):
    train, test = split_by_qid(synthetic_pairs)
    scorer = HallucinationScorer().fit(train)
    probs = scorer.predict_proba(test)
    preds = scorer.predict(test)
    assert probs.shape == (len(test),)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    accuracy = (preds == test.label.values).mean()
    assert accuracy >= 0.9  # unrelated hallucinations are easy to separate


def test_training_is_deterministic(synthetic_pairs):
    train, test = split_by_qid(synthetic_pairs)
    first = HallucinationScorer().fit(train).predict_proba(test)
    second = HallucinationScorer().fit(train).predict_proba(test)
    np.testing.assert_array_equal(first, second)


def test_save_load_roundtrip(synthetic_pairs, tmp_path):
    scorer, train, test = train_champion(synthetic_pairs)
    scorer.save(tmp_path / "model")
    loaded = HallucinationScorer.load(tmp_path / "model")
    np.testing.assert_allclose(
        scorer.predict_proba(test), loaded.predict_proba(test), atol=1e-6
    )
    assert loaded.threshold == scorer.threshold


def test_single_example_score(synthetic_pairs):
    scorer, _, _ = train_champion(synthetic_pairs)
    grounded = scorer.score(
        "What is the capital of Landia1?",
        "Portville1",
        "The capital of Landia1 is Portville1. It was founded in 1801 by coastal settlers.",
    )
    hallucinated = scorer.score(
        "What is the capital of Landia1?",
        "Xyzzical flumbore in 2199.",
        "The capital of Landia1 is Portville1. It was founded in 1801 by coastal settlers.",
    )
    assert 0.0 <= grounded <= 1.0 and 0.0 <= hallucinated <= 1.0
    assert hallucinated > grounded


def test_evaluate_model_reports_raw_and_matched(synthetic_pairs):
    scorer, _, test = train_champion(synthetic_pairs)
    report = evaluate_model(scorer, test)
    assert set(report) == {"raw", "length_matched"}
    assert report["raw"]["macro_f1"] >= 0.9
    assert 0.0 <= report["length_matched"]["macro_f1"] <= 1.0
