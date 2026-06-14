"""Router tests: trigger logic + LLM escalation, with an injected stub judge (no CLI)."""
import pytest

from src.predict import BUNDLE_PATH, ConversationQualityScorer
from src.router import Router, is_multi_candidate

pytestmark = pytest.mark.skipif(not BUNDLE_PATH.exists(),
                                reason="champion.joblib not built (run `python -m src.train`)")


def test_is_multi_candidate():
    assert is_multi_candidate("Which magazine was started first, Arthur's Magazine or First for Women?")
    assert not is_multi_candidate("What is the capital of France?")
    assert not is_multi_candidate("Paris or croissants are nice.")  # no selection word


@pytest.fixture(scope="module")
def scorer():
    return ConversationQualityScorer.load()


def test_non_suspect_stays_on_tree(scorer):
    calls = {"n": 0}

    def stub(k, q, a):
        calls["n"] += 1
        return {"label": 1, "prob_hallucinated": 0.9, "ok": True}

    r = Router(scorer, judge_fn=stub)
    out = r.route("Paris is the capital of France.", "Capital of France?", "Paris")
    assert out["path"] == "tree"
    assert out["escalated"] is False
    assert calls["n"] == 0  # the tree never escalated a clean grounded answer


def test_suspect_on_multi_candidate_escalates(scorer):
    def stub(k, q, a):
        return {"label": 1, "prob_hallucinated": 0.92, "ok": True}

    r = Router(scorer, tighten=True, judge_fn=stub)
    # a verbatim suspect on a comparative question -> escalates and the LLM overrides
    out = r.route(
        "Arthur's Magazine (1844) and First for Women (1989) are both magazines.",
        "Which magazine was started first, Arthur's Magazine or First for Women?",
        "First for Women",
    )
    # assert the precondition explicitly so the escalation checks can't pass vacuously
    assert out["verbatim_suspect"] is True
    assert out["escalated"] is True
    assert out["path"] == "llm"
    assert out["verdict"] == "HALLUCINATED"


def test_tighten_skips_single_candidate(scorer):
    calls = {"n": 0}

    def stub(k, q, a):
        calls["n"] += 1
        return {"label": 1, "prob_hallucinated": 0.9, "ok": True}

    r = Router(scorer, tighten=True, judge_fn=stub)
    out = r.route("Paris is the capital of France.", "What is the capital of France?", "Paris")
    # 'Paris' is verbatim+grounded (not a suspect) OR single-candidate -> never hits the LLM
    assert calls["n"] == 0
    assert out["escalated"] is False
