"""The two-head router: a CPU tree at scale, a frontier LLM on the suspects.

Phase 5's headline: the 13-feature tree gets a perfect score on the representative
distribution but is *blind* (0/10) to grounded-but-irrelevant hallucinations -- a
verbatim-correct quote that is the wrong answer to the question -- which only a
reasoning LLM catches (Codex 10/10).  The router ships both: the tree decides
everything, and only *verbatim suspects* (answer copied from source AND tree says
grounded) are escalated to the LLM judge.

Phase 5's Codex review flagged that "verbatim suspect" is ~half of HaluEval, so a
naive escalation would route too much traffic and risk new false positives.  The
fix, implemented here, is a domain-informed trigger: only escalate verbatim
suspects to *multi-candidate* questions (comparative / selection questions where a
right-looking quote can be the wrong choice) -- precisely where the trap lives.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, Optional

from .predict import ConversationQualityScorer

# Comparative / selection-question cues: "X or Y", "which ... or ...", "between A and B".
_OR = re.compile(r"\b(?:or|versus|vs\.?)\b", re.I)
_WHICH = re.compile(r"\b(which|who|whom|either|between|first|earlier|older|later|bigger|larger|"
                    r"smaller|more|less|greater|came first|started first)\b", re.I)


def is_multi_candidate(question: str) -> bool:
    """Cheap heuristic for a comparative/selection question -- where a verbatim quote
    can be the *wrong* candidate.  Requires both a disjunction cue and a
    selection/comparison word, so plain 'What is X?' questions are excluded."""
    q = str(question)
    return bool(_OR.search(q) and _WHICH.search(q))


class Router:
    def __init__(self, scorer: ConversationQualityScorer, backend: str = "claude/haiku",
                 tighten: bool = True, judge_fn: Optional[Callable] = None) -> None:
        """``tighten=True`` is the recommended (tightened) router; ``tighten=False``
        reproduces the naive 'escalate every verbatim suspect' router for comparison.
        ``judge_fn`` is injectable so tests/evals can supply cached LLM verdicts."""
        self.scorer = scorer
        self.backend = backend
        self.tighten = tighten
        self._judge_fn = judge_fn

    def _judge(self, knowledge: str, question: str, answer: str) -> dict:
        if self._judge_fn is not None:
            return self._judge_fn(knowledge, question, answer)
        from .llm_judge import judge  # imported lazily so tests need no CLI
        return judge(knowledge, question, answer, backend=self.backend)

    def route(self, knowledge: str, question: str, answer: str,
              threshold: Optional[float] = None) -> Dict:
        base = self.scorer.score(knowledge, question, answer, threshold=threshold)
        result = dict(base)
        result["path"] = "tree"
        result["escalated"] = False
        result["llm"] = None

        if not base["verbatim_suspect"]:
            result["reason"] = "not a verbatim suspect -- tree is authoritative"
            return result

        multi = is_multi_candidate(question)
        result["multi_candidate"] = multi
        if self.tighten and not multi:
            result["reason"] = "verbatim suspect but single-candidate question -- trigger tightened out"
            return result

        # escalate to the LLM relevance check
        verdict = self._judge(knowledge, question, answer)
        result["llm"] = verdict
        result["escalated"] = True
        result["path"] = "llm"
        if verdict.get("label") is not None:
            # LLM overrides the tree on this slice (the tree is structurally blind here)
            result["label"] = int(verdict["label"])
            result["verdict"] = "HALLUCINATED" if verdict["label"] == 1 else "GROUNDED"
            result["prob_hallucinated"] = verdict.get("prob_hallucinated", result["prob_hallucinated"])
            result["reason"] = f"verbatim suspect on a multi-candidate question -- {self.backend} adjudicated"
        else:
            result["reason"] = "LLM judge failed to parse -- falling back to tree verdict"
        return result
