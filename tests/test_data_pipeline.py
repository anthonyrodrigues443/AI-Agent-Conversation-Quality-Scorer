"""Data-pipeline tests: the long frame, the frozen split, the length-matched control."""
import pytest

from src.data_pipeline import (RAW_PATH, SPLIT_PATH, load_long_frame, load_split,
                               nearest_length_match)

pytestmark = pytest.mark.skipif(not RAW_PATH.exists(),
                                reason="qa_data.json not downloaded (see data/README.md)")


@pytest.fixture(scope="module")
def df():
    return load_long_frame()


def test_long_frame_shape(df):
    # each item unfolds into exactly 2 rows (grounded + hallucinated), perfectly balanced
    assert len(df) % 2 == 0
    assert set(df.label.unique()) == {0, 1}
    assert (df.label == 0).sum() == (df.label == 1).sum()
    for col in ("qid", "knowledge", "question", "answer", "label", "ans_chars", "ground_overlap"):
        assert col in df.columns


def test_split_is_group_disjoint(df):
    train, test = load_split(df)
    assert set(train.qid).isdisjoint(set(test.qid))  # no item straddles the split
    assert len(train) + len(test) == len(df)


@pytest.mark.skipif(not SPLIT_PATH.exists(), reason="frozen split missing")
def test_matched_control_balances_length(df):
    _, test = load_split(df)
    matched = nearest_length_match(test, caliper=8)
    assert (matched.label == 0).sum() == (matched.label == 1).sum()  # paired
    # matched set must shrink the answer-length gap between classes
    raw_gap = abs(test[test.label == 0].ans_chars.mean() - test[test.label == 1].ans_chars.mean())
    mat_gap = abs(matched[matched.label == 0].ans_chars.mean() - matched[matched.label == 1].ans_chars.mean())
    assert mat_gap < raw_gap
