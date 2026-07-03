import json

from src.data import build_pair_frame, load_qa_data

RECORDS = [
    {
        "knowledge": "Arthur's Magazine was started in 1844.",
        "question": "Which magazine was started first?",
        "right_answer": "Arthur's Magazine",
        "hallucinated_answer": "First for Women was started first.",
    },
    {
        "knowledge": "The Eiffel Tower is in Paris.",
        "question": "Where is the Eiffel Tower?",
        "right_answer": "Paris",
        "hallucinated_answer": "The Eiffel Tower is located in Berlin, Germany.",
    },
]


def write_jsonl(path):
    path.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    return path


def test_load_expands_each_item_into_grounded_and_hallucinated(tmp_path):
    frame = load_qa_data(write_jsonl(tmp_path / "qa_data.json"))
    assert len(frame) == 4
    assert frame.qid.tolist() == [0, 0, 1, 1]
    assert frame.label.tolist() == [0, 1, 0, 1]
    assert frame.answer.iloc[0] == "Arthur's Magazine"
    assert frame.answer.iloc[1] == "First for Women was started first."


def test_pair_rows_share_question_and_knowledge():
    frame = build_pair_frame(RECORDS)
    for qid, group in frame.groupby("qid"):
        assert group.question.nunique() == 1
        assert group.knowledge.nunique() == 1
        assert sorted(group.label) == [0, 1]


def test_ans_chars_column():
    frame = build_pair_frame(RECORDS)
    assert (frame.ans_chars == frame.answer.str.len()).all()


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "qa_data.json"
    path.write_text(json.dumps(RECORDS[0]) + "\n\n" + json.dumps(RECORDS[1]) + "\n\n")
    assert len(load_qa_data(path)) == 4
