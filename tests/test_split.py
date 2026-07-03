import json

from src.pipeline import split_by_qid


def test_no_qid_leakage(synthetic_pairs):
    train, test = split_by_qid(synthetic_pairs)
    assert not (set(train.qid) & set(test.qid))
    assert len(train) + len(test) == len(synthetic_pairs)


def test_pairs_stay_together(synthetic_pairs):
    train, test = split_by_qid(synthetic_pairs)
    for part in (train, test):
        counts = part.groupby("qid").size()
        assert (counts == 2).all()  # both answers of an item on the same side


def test_split_is_deterministic(synthetic_pairs):
    first_train, first_test = split_by_qid(synthetic_pairs, random_state=42)
    second_train, second_test = split_by_qid(synthetic_pairs, random_state=42)
    assert first_train.index.tolist() == second_train.index.tolist()
    assert first_test.index.tolist() == second_test.index.tolist()


def test_frozen_qid_file_is_respected(synthetic_pairs, tmp_path):
    qids = sorted(int(q) for q in synthetic_pairs.qid.unique())
    frozen = {"train_qids": qids[:30], "test_qids": qids[30:]}
    qid_file = tmp_path / "split.json"
    qid_file.write_text(json.dumps(frozen))
    train, test = split_by_qid(synthetic_pairs, qid_file=qid_file)
    assert set(train.qid) == set(frozen["train_qids"])
    assert set(test.qid) == set(frozen["test_qids"])


def test_test_size_fraction(synthetic_pairs):
    train, test = split_by_qid(synthetic_pairs, test_size=0.2)
    assert test.qid.nunique() == 8  # 20% of 40 items
