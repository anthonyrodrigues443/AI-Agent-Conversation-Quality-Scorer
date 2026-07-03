import numpy as np
import pandas as pd
import pytest


def make_synthetic_pairs(n_items=40, seed=0):
    """Synthetic HaluEval-shaped data: grounded answers are verbatim spans of the
    knowledge, hallucinated answers are unrelated invented text."""
    rng = np.random.default_rng(seed)
    rows = []
    fillers = ["greebly", "flanwick", "zorbulous", "quandric", "blenthar"]
    for i in range(n_items):
        knowledge = (
            f"The capital of Landia{i} is Portville{i}. "
            f"It was founded in {1800 + i} by coastal settlers."
        )
        question = f"What is the capital of Landia{i}?"
        grounded = f"Portville{i}"
        noise = str(rng.choice(fillers))
        # similar length to the grounded answer so the matched control is non-empty
        hallucinated = f"Kelmor {noise}"
        rows.append(
            {"qid": i, "knowledge": knowledge, "question": question, "answer": grounded, "label": 0}
        )
        rows.append(
            {
                "qid": i,
                "knowledge": knowledge,
                "question": question,
                "answer": hallucinated,
                "label": 1,
            }
        )
    frame = pd.DataFrame(rows).reset_index(drop=True)
    frame["ans_chars"] = frame.answer.str.len()
    return frame


@pytest.fixture
def synthetic_pairs():
    return make_synthetic_pairs()
