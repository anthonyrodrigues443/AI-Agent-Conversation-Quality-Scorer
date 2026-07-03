"""The 13 engineered grounding features, ported verbatim from notebooks/phase3.

Every regex, stopword list, and edge-case default matches the notebook cell that
produced the 0.9967 raw / 0.9808 length-matched champion. Do not "clean up" the
tokenizer or the fallback values: the model was trained on exactly these.
"""

import difflib
import re
from collections import Counter

import numpy as np
import pandas as pd

STOPWORDS = set(
    "a an the of to in on at for and or is was were are be been by with as that this it"
    " from".split()
)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
NEGATION_WORDS = {
    "not", "no", "never", "none", "cannot", "without", "neither", "nor", "n't",
    "dont", "didnt", "doesnt", "isnt", "wasnt", "werent", "arent", "wont", "cant",
}

#: Column order the champion model was trained on (phase 3 `ALL = ["ground_overlap"] + ENG`).
FEATURE_NAMES = [
    "ground_overlap",
    "is_substr",
    "lcs_char_ratio",
    "lcs_token_ratio",
    "num_frac_in_know",
    "n_num_missing",
    "year_mismatch",
    "sent_overlap_max",
    "overlap_spread",
    "novel_overlap",
    "idf_overlap",
    "ans_has_neg",
    "neg_mismatch",
]


def content_tokens(text):
    """Lowercase [a-z0-9]+ tokens minus stopwords (notebook ``toks``)."""
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t not in STOPWORDS]


def all_tokens(text):
    """Lowercase [a-z0-9]+ tokens including stopwords (notebook ``toks_all``)."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p for p in parts if p.strip()] or [str(text)]


def _normalize_answer(answer):
    return str(answer).lower().strip().rstrip(".")


def ground_overlap(answer, knowledge):
    """Fraction of the answer's content tokens that appear in the knowledge."""
    a = set(content_tokens(answer))
    k = set(content_tokens(knowledge))
    return (len(a & k) / len(a)) if a else 0.0


def is_substr(answer, knowledge):
    """1.0 if the normalized answer is a verbatim substring of the knowledge."""
    a_lc = _normalize_answer(answer)
    return float(a_lc in str(knowledge).lower()) if a_lc else 0.0


def lcs_char_ratio(answer, knowledge):
    """Longest common character run between answer and knowledge / answer length.

    The load-bearing feature: the phase-5 ablation showed dropping it costs
    -0.0742 matched macro-F1 while dropping any other feature costs 0.0000.
    """
    a_lc = _normalize_answer(answer)
    k_lc = str(knowledge).lower()
    if not a_lc:
        return 0.0
    match = difflib.SequenceMatcher(None, a_lc, k_lc, autojunk=False).find_longest_match(
        0, len(a_lc), 0, len(k_lc)
    )
    return match.size / len(a_lc)


def lcs_token_ratio(answer, knowledge):
    """Longest common contiguous token run / answer token count."""
    at = all_tokens(answer)
    kt = all_tokens(knowledge)
    if not at:
        return 0.0
    match = difflib.SequenceMatcher(None, at, kt, autojunk=False).find_longest_match(
        0, len(at), 0, len(kt)
    )
    return match.size / len(at)


def _max_sentence_overlap(answer_set, knowledge):
    """(best overlap fraction, best sentence) over the knowledge's sentences."""
    best, best_sent = -1.0, ""
    for sent in split_sentences(knowledge):
        sent_set = set(content_tokens(sent))
        overlap = len(answer_set & sent_set) / len(answer_set) if sent_set else 0.0
        if overlap > best:
            best, best_sent = overlap, sent
    return best, best_sent


class Featurizer:
    """Computes the 13-feature vector; must be fitted for the IDF feature.

    ``idf_overlap`` needs document frequencies from the training corpus
    (knowledge passages + answers), exactly as the notebook computed them.
    """

    def __init__(self):
        self.docfreq = Counter()
        self.ndoc = 0

    @property
    def is_fitted(self):
        return self.ndoc > 0

    def fit(self, texts):
        """Fit document frequencies over an iterable of texts."""
        self.docfreq = Counter()
        count = 0
        for text in texts:
            self.docfreq.update(set(content_tokens(text)))
            count += 1
        self.ndoc = count
        return self

    def fit_from_frame(self, frame):
        """Fit on a training frame's ``knowledge`` and ``answer`` columns."""
        return self.fit(pd.concat([frame.knowledge, frame.answer]))

    def idf(self, token):
        return np.log((self.ndoc + 1) / (self.docfreq.get(token, 0) + 1))

    def compute_features(self, question, answer, knowledge):
        """Return the 13 features for one (question, answer, knowledge) triple."""
        if not self.is_fitted:
            raise RuntimeError("Featurizer must be fitted before computing features (IDF).")
        aset = set(content_tokens(answer))
        kset = set(content_tokens(knowledge))
        at_all = all_tokens(answer)

        # numeric / date agreement
        anum = set(NUM_RE.findall(str(answer)))
        knum = set(NUM_RE.findall(str(knowledge)))
        num_frac = (len(anum & knum) / len(anum)) if anum else 1.0
        n_miss = len(anum - knum)
        ayr = set(YEAR_RE.findall(str(answer)))
        year_mismatch = len(ayr - set(YEAR_RE.findall(str(knowledge))))

        # sentence concentration
        sent_max, best_sent = (0.0, "")
        if aset:
            sent_max, best_sent = _max_sentence_overlap(aset, knowledge)
        whole = (len(aset & kset) / len(aset)) if aset else 0.0
        spread = whole - sent_max

        # novelty / rarity
        novel = aset - set(content_tokens(question))
        novel_ov = (len(novel & kset) / len(novel)) if novel else 1.0
        num = sum(self.idf(t) for t in aset if t in kset)
        den = sum(self.idf(t) for t in aset)
        idf_ov = (num / den) if den else 0.0

        # polarity
        ans_neg = float(any(w in NEGATION_WORDS for w in at_all) or "n't" in str(answer).lower())
        sent_neg = float(
            any(w in NEGATION_WORDS for w in all_tokens(best_sent)) or "n't" in best_sent.lower()
        )
        neg_mismatch = float(ans_neg != sent_neg)

        return {
            "ground_overlap": ground_overlap(answer, knowledge),
            "is_substr": is_substr(answer, knowledge),
            "lcs_char_ratio": lcs_char_ratio(answer, knowledge),
            "lcs_token_ratio": lcs_token_ratio(answer, knowledge),
            "num_frac_in_know": num_frac,
            "n_num_missing": float(n_miss),
            "year_mismatch": float(year_mismatch),
            "sent_overlap_max": sent_max,
            "overlap_spread": spread,
            "novel_overlap": novel_ov,
            "idf_overlap": idf_ov,
            "ans_has_neg": ans_neg,
            "neg_mismatch": neg_mismatch,
        }

    def transform(self, frame):
        """Feature matrix (DataFrame, FEATURE_NAMES columns) for a pair frame."""
        rows = [
            self.compute_features(q, a, k)
            for q, a, k in zip(frame.question, frame.answer, frame.knowledge)
        ]
        matrix = pd.DataFrame(rows, index=frame.index, columns=FEATURE_NAMES)
        return matrix.astype(np.float32)

    def to_dict(self):
        return {"ndoc": self.ndoc, "docfreq": dict(self.docfreq)}

    @classmethod
    def from_dict(cls, payload):
        featurizer = cls()
        featurizer.ndoc = int(payload["ndoc"])
        featurizer.docfreq = Counter(payload["docfreq"])
        return featurizer
