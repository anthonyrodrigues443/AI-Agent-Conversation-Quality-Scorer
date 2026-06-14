"""Frontier-LLM hallucination judge via the local ``claude`` / ``codex`` CLIs.

This is the exact harness from Phase 5 lifted into an importable module.  The tree
handles every request; the LLM judge is only invoked on the *verbatim-suspect*
slice the router escalates (see :mod:`src.router`).  Latency includes CLI startup
(~5-20s/call); direct API would be 5-10x faster -- the cost math below reflects
API pricing, the timings reflect the CLI.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from typing import Optional, Tuple

CLAUDE_CMD = shutil.which("claude") or "/Users/anthonyrodrigues/.local/bin/claude"
CODEX_CMD = shutil.which("codex") or "/Users/anthonyrodrigues/.nvm/versions/node/v24.13.0/bin/codex"

LLM_PROMPT = """You are a hallucination detector. You are given a SOURCE passage, a QUESTION, and an ANSWER.
Reply with EXACTLY one word on the FIRST line: GROUNDED or HALLUCINATED.
- GROUNDED = the answer is factually supported by the SOURCE and correctly answers the QUESTION.
- HALLUCINATED = the answer is unsupported, contradicts the SOURCE, or is the wrong answer to the QUESTION.
Then on a NEW line: a single number 0.0-1.0 = probability the answer is HALLUCINATED. No other text.

SOURCE: {knowledge}
QUESTION: {question}
ANSWER: {answer}"""

# representative API cost per call (~250 in / ~10 out tokens), 2026 pricing
COST_PER_CALL_USD = {
    "claude/haiku": 0.0003,
    "claude/opus": 0.0045,
    "codex/gpt-5.5": 0.050,
}


def build_prompt(knowledge: str, question: str, answer: str) -> str:
    return LLM_PROMPT.format(knowledge=knowledge, question=question, answer=answer)


def call_claude(prompt: str, model: str = "haiku", timeout: int = 90) -> Tuple[str, float, Optional[int]]:
    t0 = time.time()
    try:
        p = subprocess.run(
            [CLAUDE_CMD, "--print", "--model", model, "--no-session-persistence", "--disable-slash-commands"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        el = time.time() - t0
        if p.returncode != 0:
            return f"__ERROR__:rc={p.returncode}:{p.stderr[:150]}", el, None
        return p.stdout.strip(), el, None
    except subprocess.TimeoutExpired:
        return "__ERROR__:timeout", time.time() - t0, None
    except Exception as e:  # noqa: BLE001 - surfaced to caller as an __ERROR__ sentinel
        return f"__ERROR__:{type(e).__name__}:{str(e)[:120]}", time.time() - t0, None


def call_codex(prompt: str, timeout: int = 200) -> Tuple[str, float, Optional[int]]:
    t0 = time.time()
    try:
        p = subprocess.run(
            [CODEX_CMD, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "-"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        el = time.time() - t0
        if p.returncode != 0:
            return f"__ERROR__:rc={p.returncode}:{p.stderr[:150]}", el, None
        out = p.stdout
        tok = None
        m = re.search(r"tokens used[:\s]*([\d,]+)", out)
        if m:
            tok = int(m.group(1).replace(",", ""))
        if "codex\n" in out:
            tail = out.rsplit("codex\n", 1)[1]
            if "tokens used" in tail:
                tail = tail.split("tokens used")[0]
            return tail.strip(), el, tok
        return out.strip(), el, tok
    except subprocess.TimeoutExpired:
        return "__ERROR__:timeout", time.time() - t0, None
    except Exception as e:  # noqa: BLE001
        return f"__ERROR__:{type(e).__name__}:{str(e)[:120]}", time.time() - t0, None


def parse_llm(text: str) -> Tuple[Optional[int], Optional[float]]:
    """Parse the (label, P(hallucinated)) pair out of a CLI response, defensively."""
    if not text or text.startswith("__ERROR__"):
        return None, None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None, None
    head = lines[0].upper()
    label = 1 if "HALLUCINAT" in head else (0 if "GROUND" in head else None)
    if label is None:
        up = text.upper()
        label = 1 if "HALLUCINAT" in up else (0 if "GROUND" in up else None)
    prob = None
    for l in lines[1:] + lines[:1]:
        mm = re.search(r"(?<![\w.])(0?\.\d+|1\.0+|0\.0+|[01])(?![\w])", l)
        if mm:
            try:
                v = float(mm.group(1))
                if 0.0 <= v <= 1.0:
                    prob = v
                    break
            except ValueError:
                pass
    if prob is None and label is not None:
        prob = 0.9 if label == 1 else 0.1
    return label, prob


def judge(knowledge: str, question: str, answer: str, backend: str = "claude/haiku",
          timeout: int = 90) -> dict:
    """Run one frontier judge and return a structured verdict."""
    prompt = build_prompt(knowledge, question, answer)
    if backend.startswith("codex"):
        raw, el, tok = call_codex(prompt, timeout=max(timeout, 180))
    else:
        model = backend.split("/", 1)[1] if "/" in backend else "haiku"
        raw, el, tok = call_claude(prompt, model=model, timeout=timeout)
    label, prob = parse_llm(raw)
    return {
        "backend": backend,
        "label": label,
        "prob_hallucinated": prob,
        "latency_s": round(el, 2),
        "tokens": tok,
        "raw": (raw or "")[:200],
        "ok": label is not None,
        "est_cost_usd": COST_PER_CALL_USD.get(backend),
    }
