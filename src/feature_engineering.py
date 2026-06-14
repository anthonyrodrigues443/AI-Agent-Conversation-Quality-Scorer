"""The 13 claim-relation features that feed the champion tree.

``ground_overlap`` (the Phase-2 lexical feature) + 12 engineered features.  The
single feature that carries unique signal is ``lcs_char_ratio`` -- leave-one-out
in Phase 5 showed dropping it costs -0.074 macro-F1 while dropping *any* of the
other 12 costs exactly 0.0.  The features are intentionally cheap (regex + set
ops + one ``difflib`` longest-match) so the whole vector is ~25us/row on CPU.

Only ``idf_overlap`` depends on the training corpus (document frequencies), so a
fitted ``FeatureEngineer`` persists its ``docfreq`` table; every other feature is
a pure function of the (answer, question, knowledge) triple.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Dict, List

import numpy as np

from .utils import toks, toks_all, grounding_overlap

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
NEG = {"not", "no", "never", "none", "cannot", "without", "neither", "nor", "n't",
       "dont", "didnt", "doesnt", "isnt", "wasnt", "werent", "arent", "wont", "cant"}

# 12 engineered features, in the order the model was trained on.
ENG: List[str] = [
    "is_substr", "lcs_char_ratio", "lcs_token_ratio", "num_frac_in_know", "n_num_missing",
    "year_mismatch", "sent_overlap_max", "overlap_spread", "novel_overlap", "idf_overlap",
    "ans_has_neg", "neg_mismatch",
]
# Full input vector: the Phase-2 overlap feature prepended to the 12 engineered ones.
ALL_FEATURES: List[str] = ["ground_overlap"] + ENG


def _split_sents(txt: str) -> List[str]:
    s = re.split(r"(?<=[.!?])\s+", str(txt).strip())
    return [x for x in s if x.strip()] or [str(txt)]


class FeatureEngineer:
    """Fits document frequencies on train; transforms triples into the 13-d vector."""

    def __init__(self) -> None:
        self.docfreq: Dict[str, int] = {}
        self.ndoc: int = 0
        self.feature_names: List[str] = ALL_FEATURES

    # ---- fit / persistence ----
    def fit(self, knowledge, answers) -> "FeatureEngineer":
        docfreq: Counter = Counter()
        for s in list(knowledge) + list(answers):
            docfreq.update(set(toks(s)))
        self.docfreq = dict(docfreq)
        self.ndoc = len(list(knowledge)) + len(list(answers))
        return self

    def to_dict(self) -> dict:
        return {"docfreq": self.docfreq, "ndoc": self.ndoc, "feature_names": self.feature_names}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureEngineer":
        fe = cls()
        fe.docfreq = d["docfreq"]
        fe.ndoc = d["ndoc"]
        fe.feature_names = d.get("feature_names", ALL_FEATURES)
        return fe

    def _idf(self, t: str) -> float:
        return float(np.log((self.ndoc + 1) / (self.docfreq.get(t, 0) + 1)))

    # ---- the 12 engineered features for one triple ----
    def _eng_row(self, ans: str, q: str, know: str) -> tuple:
        a_lc = str(ans).lower().strip().rstrip(".")
        k_lc = str(know).lower()
        aset = set(toks(ans))
        kset = set(toks(know))
        at_all = toks_all(ans)
        kt_all = toks_all(know)

        is_substr = float(a_lc in k_lc) if a_lc else 0.0
        lcs_char = (difflib.SequenceMatcher(None, a_lc, k_lc, autojunk=False)
                    .find_longest_match(0, len(a_lc), 0, len(k_lc)).size / len(a_lc)) if a_lc else 0.0
        lcs_tok = (difflib.SequenceMatcher(None, at_all, kt_all, autojunk=False)
                   .find_longest_match(0, len(at_all), 0, len(kt_all)).size / len(at_all)) if at_all else 0.0
        anum = set(NUM_RE.findall(str(ans)))
        knum = set(NUM_RE.findall(str(know)))
        num_frac = (len(anum & knum) / len(anum)) if anum else 1.0
        n_miss = len(anum - knum)
        ayr = set(YEAR_RE.findall(str(ans)))
        year_mismatch = len(ayr - set(YEAR_RE.findall(str(know))))

        sent_max = 0.0
        if aset:
            for s in _split_sents(know):
                ss = set(toks(s))
                o = len(aset & ss) / len(aset) if ss else 0.0
                if o > sent_max:
                    sent_max = o
        whole = (len(aset & kset) / len(aset)) if aset else 0.0
        spread = whole - sent_max
        novel = aset - set(toks(q))
        novel_ov = (len(novel & kset) / len(novel)) if novel else 1.0
        num = sum(self._idf(t) for t in aset if t in kset)
        den = sum(self._idf(t) for t in aset)
        idf_ov = (num / den) if den else 0.0
        ans_neg = float(any(w in NEG for w in at_all) or "n't" in str(ans).lower())
        best_s, best = "", -1.0
        if aset:
            for s in _split_sents(know):
                ss = set(toks(s))
                o = len(aset & ss) / len(aset) if ss else 0.0
                if o > best:
                    best, best_s = o, s
        sent_neg = float(any(w in NEG for w in toks_all(best_s)) or "n't" in best_s.lower())
        neg_mismatch = float(ans_neg != sent_neg)
        return (is_substr, lcs_char, lcs_tok, num_frac, n_miss, year_mismatch,
                sent_max, spread, novel_ov, idf_ov, ans_neg, neg_mismatch)

    # ---- public transforms ----
    def row(self, answer: str, question: str, knowledge: str) -> np.ndarray:
        """13-d feature vector for one (answer, question, knowledge) triple."""
        go = grounding_overlap(answer, knowledge)
        return np.array((go,) + self._eng_row(answer, question, knowledge), dtype=np.float32)

    def row_dict(self, answer: str, question: str, knowledge: str) -> Dict[str, float]:
        """Same as :meth:`row` but keyed by feature name (handy for the UI)."""
        vec = self.row(answer, question, knowledge)
        return {name: float(v) for name, v in zip(self.feature_names, vec)}

    def transform(self, answers, questions, knowledge) -> np.ndarray:
        """Batch transform into an (n, 13) matrix."""
        return np.array(
            [self.row(a, q, k) for a, q, k in zip(answers, questions, knowledge)],
            dtype=np.float32,
        )
