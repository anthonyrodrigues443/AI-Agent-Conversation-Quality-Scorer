"""Inference-pipeline tests: the saved bundle loads, scores, and flags the blind spot."""
import pytest

from src.predict import BUNDLE_PATH, ConversationQualityScorer

pytestmark = pytest.mark.skipif(not BUNDLE_PATH.exists(),
                                reason="champion.joblib not built (run `python -m src.train`)")


@pytest.fixture(scope="module")
def scorer():
    return ConversationQualityScorer.load()


def test_output_schema(scorer):
    r = scorer.score("Paris is the capital of France.", "Capital of France?", "Paris")
    for key in ("prob_hallucinated", "label", "verdict", "features", "is_verbatim",
                "verbatim_suspect", "threshold"):
        assert key in r
    assert r["label"] in (0, 1)
    assert 0.0 <= r["prob_hallucinated"] <= 1.0
    assert len(r["features"]) == 13


def test_clear_hallucination_flagged(scorer):
    r = scorer.score(
        "Arthur's Magazine was an American literary periodical published in Philadelphia.",
        "Where was Arthur's Magazine published?",
        "It was published in Tokyo, Japan, in 1990.",
    )
    assert r["label"] == 1
    assert r["verdict"] == "HALLUCINATED"


def test_grounded_answer_passes(scorer):
    r = scorer.score(
        "Arthur's Magazine was an American literary periodical published in Philadelphia.",
        "Where was Arthur's Magazine published?",
        "Philadelphia",
    )
    assert r["label"] == 0


def test_verbatim_suspect_detected(scorer):
    # verbatim-correct quote that is the WRONG hop -> tree says grounded, flagged as suspect
    r = scorer.score(
        "It was headed by Karl Donitz as Reichsprasident and Lutz Graf Schwerin von Krosigk as the "
        "Leading Minister. Donitz briefly succeeded Adolf Hitler as head of state of Germany.",
        "Who succeeded Adolf Hitler?",
        "Lutz Graf Schwerin von Krosigk",
    )
    assert r["is_verbatim"] is True
    assert r["verbatim_suspect"] is True   # tree (wrongly) says grounded -> router target


def test_threshold_monotonicity(scorer):
    text = ("Paris is the capital of France.", "Capital?", "London")
    r_low = scorer.score(*text, threshold=0.01)
    r_high = scorer.score(*text, threshold=0.99)
    # a stricter (higher) threshold can never flag MORE than a lax one
    assert r_high["label"] <= r_low["label"]


def test_batch_consistency(scorer):
    ks = ["Paris is the capital of France."] * 2
    qs = ["Capital?"] * 2
    ans = ["Paris", "London"]
    probs, labels = scorer.score_batch(ks, qs, ans)
    assert len(probs) == 2 and len(labels) == 2
    single = scorer.score(ks[0], qs[0], ans[0])
    assert abs(single["prob_hallucinated"] - probs[0]) < 1e-6
