"""Feature-engineering correctness: the 13-d vector and its invariants."""
import numpy as np
import pytest

from src.feature_engineering import ALL_FEATURES, ENG, FeatureEngineer


@pytest.fixture(scope="module")
def fe():
    f = FeatureEngineer().fit(
        knowledge=["Paris is the capital of France. France is in Europe.",
                   "The Nile is a river in Africa."],
        answers=["Paris", "The Nile"],
    )
    return f


def test_feature_count_and_order(fe):
    assert len(ALL_FEATURES) == 13
    assert ALL_FEATURES[0] == "ground_overlap"
    assert ALL_FEATURES[1:] == ENG
    vec = fe.row("Paris", "What is the capital of France?", "Paris is the capital of France.")
    assert vec.shape == (13,)
    assert vec.dtype == np.float32


def test_is_substr_detects_verbatim(fe):
    d = fe.row_dict("Paris", "Capital of France?", "Paris is the capital of France.")
    assert d["is_substr"] == 1.0
    # lcs_char_ratio is 1.0 when the whole answer is contiguous in the source
    assert d["lcs_char_ratio"] == pytest.approx(1.0, abs=1e-6)


def test_non_verbatim_answer(fe):
    d = fe.row_dict("Tokyo", "Capital of France?", "Paris is the capital of France.")
    assert d["is_substr"] == 0.0
    assert d["lcs_char_ratio"] < 1.0


def test_round_trip_serialisation(fe):
    fe2 = FeatureEngineer.from_dict(fe.to_dict())
    a = fe.row("Paris", "Capital of France?", "Paris is the capital of France.")
    b = fe2.row("Paris", "Capital of France?", "Paris is the capital of France.")
    assert np.allclose(a, b)
    assert fe2.ndoc == fe.ndoc


def test_empty_answer_is_safe(fe):
    vec = fe.row("", "Capital?", "Paris is the capital.")
    assert np.isfinite(vec).all()


def test_batch_matches_rows(fe):
    answers = ["Paris", "Tokyo"]
    qs = ["Capital of France?", "Capital of France?"]
    ks = ["Paris is the capital of France."] * 2
    M = fe.transform(answers, qs, ks)
    assert M.shape == (2, 13)
    assert np.allclose(M[0], fe.row(answers[0], qs[0], ks[0]))
