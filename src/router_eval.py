"""Phase 6 research: quantify the blended router on the full test split.

Phase 5 deferred the real question to Phase 6: *does escalating verbatim suspects
to a frontier LLM actually help once you account for the LLM's false-positive rate
on the verbatim-correct majority?*  This script answers it on all 4,000 test rows.

It (1) sizes every routing policy, (2) runs a real (cached, resumable) LLM eval on
the 10 residual hallucinations the tree misses plus a stratified control sample of
verbatim-correct grounded answers, (3) measures LLM recall on the residual and FPR
on the controls, and (4) blends -- by expected value over the measured rates --
the full-test confusion matrix for tree-only vs. naive-router vs. complexity-gated
router, sweeping the LLM's decision threshold.  Outputs go to results/.

    python -m src.router_eval                 # uses cached LLM calls; runs only missing ones
    python -m src.router_eval --backends claude/haiku codex/gpt-5.5
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data_pipeline import load_long_frame, load_split
from .llm_judge import COST_PER_CALL_USD, build_prompt, call_claude, call_codex, parse_llm
from .predict import ConversationQualityScorer
from .router import is_multi_candidate
from .utils import RESULTS_DIR

CACHE_DIR = RESULTS_DIR / "phase6_cache"
CALLS_FP = CACHE_DIR / "router_llm_calls.json"
N_GROUNDED_CTRL = 40        # verbatim-correct controls to sample for FPR
RNG_SEED = 42
TREE_COST_PER_1K = 0.0001   # CPU inference, ~free
TREE_LATENCY_MS = 0.024


def build_test_frame() -> tuple[pd.DataFrame, ConversationQualityScorer]:
    scorer = ConversationQualityScorer.load()
    df = load_long_frame()
    _, test = load_split(df)
    s, p = scorer.score_batch(test.knowledge, test.question, test.answer)
    t = test.copy()
    t["prob"] = s
    t["pred"] = p
    fe = scorer.fe.transform(test.answer, test.question, test.knowledge)
    t["is_substr"] = fe[:, scorer.feature_names.index("is_substr")] >= 1.0
    t["q_words"] = t.question.str.split().apply(len)
    t["multi_candidate"] = [is_multi_candidate(q) for q in t.question]
    t["multihop"] = t.question.str.contains(r"\bboth\b|\balso\b|\bin common\b|\bshare\b", case=False, regex=True)
    return t, scorer


def sizing_report(t: pd.DataFrame) -> dict:
    n = len(t)
    suspects = t[t.is_substr & (t.pred == 0)]
    residual = suspects[suspects.label == 1]          # the 10 the tree misses
    report = {
        "n_test": int(n),
        "n_verbatim": int(t.is_substr.sum()),
        "naive_trigger": int(len(suspects)),
        "naive_trigger_frac": round(len(suspects) / n, 4),
        "residual_hallu": int(len(residual)),
        "residual_idx": [int(i) for i in residual.index],
        "multi_candidate_trigger": int(len(suspects[suspects.multi_candidate])),
        "multi_candidate_residual_caught": int(residual.multi_candidate.sum()),
    }
    return report, suspects, residual


def _load_cache() -> list:
    if CALLS_FP.exists():
        return json.load(open(CALLS_FP))
    return []


def run_llm(t: pd.DataFrame, suspects: pd.DataFrame, residual: pd.DataFrame,
            backends: list[str], n_ctrl: int = N_GROUNDED_CTRL) -> pd.DataFrame:
    """Run (cached, resumable) LLM judging on residual hallucinations + a stratified
    control sample of verbatim-correct grounded suspects. Append-only cache: a row
    already present is never re-billed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    grounded = suspects[suspects.label == 0]
    hi = grounded[grounded.q_words >= 20]
    lo = grounded[grounded.q_words < 20]
    # stratify the FPR controls: half from the high-complexity slice the router targets
    k_hi = min(len(hi), n_ctrl // 2)
    k_lo = min(len(lo), n_ctrl - k_hi)
    ctrl_idx = (list(rng.choice(hi.index, size=k_hi, replace=False)) +
                list(rng.choice(lo.index, size=k_lo, replace=False)))
    eval_idx = sorted(set(list(residual.index) + [int(i) for i in ctrl_idx]))

    calls = _load_cache()
    seen = {(c["backend"], c["idx"]) for c in calls}
    todo = [(b, int(i)) for b in backends for i in eval_idx if (b, int(i)) not in seen]
    print(f"LLM eval: {len(eval_idx)} rows × {len(backends)} backend(s) = "
          f"{len(eval_idx)*len(backends)} calls; {len(seen)} cached, {len(todo)} to run")
    lock = threading.Lock()

    def work(job):
        backend, idx = job
        r = t.loc[idx]
        prompt = build_prompt(r.knowledge, r.question, r.answer)
        if backend.startswith("codex"):
            raw, el, tok = call_codex(prompt, timeout=200)
        else:
            raw, el, tok = call_claude(prompt, model=backend.split("/")[1], timeout=90)
        lab, prob = parse_llm(raw)
        rec = {"backend": backend, "idx": int(idx), "true": int(t.loc[idx, "label"]),
               "is_residual": bool(t.loc[idx, "label"] == 1), "q_words": int(t.loc[idx, "q_words"]),
               "pred": lab, "prob": prob, "latency_s": round(el, 2), "tokens": tok,
               "raw": (raw or "")[:160]}
        with lock:
            calls.append(rec)
            json.dump(calls, open(CALLS_FP, "w"), indent=2)
        return rec

    if todo:
        workers = 1 if any(b.startswith("codex") for b, _ in todo) else 3
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, _ in enumerate(ex.map(work, todo), 1):
                if i % 10 == 0:
                    print(f"  {i}/{len(todo)} done")
    cdf = pd.DataFrame(_load_cache())
    cdf = cdf[cdf.idx.isin(eval_idx)]
    return cdf


def blend(t: pd.DataFrame, suspects: pd.DataFrame, residual: pd.DataFrame,
          cdf: pd.DataFrame, backend: str, tau: float = 0.5) -> list[dict]:
    """Expected-value blend of the full-test confusion for each policy at LLM
    threshold ``tau`` (flag hallucinated if LLM P(hallucinated) >= tau)."""
    n = len(t)
    # tree-only confusion on full test
    pred = t.pred.values
    y = t.label.values
    TN = int(((pred == 0) & (y == 0)).sum())
    FP = int(((pred == 1) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum())
    TP = int(((pred == 1) & (y == 1)).sum())

    sub = cdf[cdf.backend == backend].copy()
    sub["flag"] = (sub.prob.fillna(0.0) >= tau) & sub.pred.notna()
    # measured rates
    res_calls = sub[sub.is_residual]
    ctrl_calls = sub[~sub.is_residual]
    residual_recall = float(res_calls.flag.mean()) if len(res_calls) else 0.0
    fpr = float(ctrl_calls.flag.mean()) if len(ctrl_calls) else 0.0
    # per-row residual flag lookup (we ran the LLM on ALL residuals)
    res_flag = dict(zip(res_calls.idx, res_calls.flag))

    def metrics_from(tn, fp, fn, tp):
        # macro-F1 over both classes
        f1_h = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        f1_g = (2 * tn) / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return {"macro_f1": round((f1_h + f1_g) / 2, 4), "precision_hallu": round(prec, 4),
                "recall_hallu": round(rec, 4), "FP": round(fp, 1), "FN": round(fn, 1)}

    def policy(name, routed_mask):
        routed = suspects[routed_mask]
        routed_grnd = int((routed.label == 0).sum())
        routed_res = routed[routed.label == 1]
        # ΔTP from residuals we actually measured; ΔFP = routed grounded × measured FPR (expected)
        dTP = sum(res_flag.get(int(i), False) for i in routed_res.index)
        dFP = routed_grnd * fpr
        m = metrics_from(TN - dFP, FP + dFP, FN - dTP, TP + dTP)
        m["policy"] = name
        m["routed"] = int(len(routed))
        m["routed_frac"] = round(len(routed) / n, 4)
        m["residual_caught"] = f"{int(dTP)}/{len(residual)}"
        m["new_false_positives"] = round(dFP, 1)
        per_call = COST_PER_CALL_USD.get(backend, 0.05)
        # TREE_COST_PER_1K is already a per-1k figure; only the routed fraction pays the
        # per-CALL LLM price (×1000 to convert $/call → $/1k).
        frac = len(routed) / n
        m["cost_per_1k"] = round(TREE_COST_PER_1K * (1 - frac) + per_call * frac * 1000, 4)
        return m

    rows = [{"policy": "tree_only", "macro_f1": metrics_from(TN, FP, FN, TP)["macro_f1"],
             "precision_hallu": metrics_from(TN, FP, FN, TP)["precision_hallu"],
             "recall_hallu": metrics_from(TN, FP, FN, TP)["recall_hallu"], "FP": FP, "FN": FN,
             "routed": 0, "routed_frac": 0.0, "residual_caught": f"0/{len(residual)}",
             "new_false_positives": 0.0, "cost_per_1k": TREE_COST_PER_1K}]
    sus = suspects
    rows.append(policy("naive_router (all verbatim suspects)", sus.index.isin(sus.index)))
    rows.append(policy("multi_candidate_router (Phase-5 suggestion)", sus.multi_candidate.values))
    rows.append(policy("complexity_router (q_words>=20)", (sus.q_words >= 20).values))
    rows.append(policy("multihop_router (both/also/share)", sus.multihop.values))
    out = pd.DataFrame(rows)
    out["backend"] = backend
    out["tau"] = tau
    out["residual_recall_measured"] = round(residual_recall, 3)
    out["ctrl_FPR_measured"] = round(fpr, 3)
    out["n_ctrl"] = len(ctrl_calls)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["claude/haiku"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--n-ctrl", type=int, default=N_GROUNDED_CTRL)
    args = ap.parse_args()

    t, scorer = build_test_frame()
    report, suspects, residual = sizing_report(t)
    print(json.dumps({k: v for k, v in report.items() if k != "residual_idx"}, indent=2))
    print(f"\nKEY: Phase-5's 'multi-candidate only' trigger routes "
          f"{report['multi_candidate_trigger']} rows and catches "
          f"{report['multi_candidate_residual_caught']}/{report['residual_hallu']} residual hallucinations.")

    run_llm(t, suspects, residual, args.backends, n_ctrl=args.n_ctrl)
    # blend over the FULL cache so every backend ever evaluated stays in the CSV
    full_cache = pd.DataFrame(_load_cache())
    backends_present = list(full_cache.backend.unique())
    all_out = []
    for backend in backends_present:
        out = blend(t, suspects, residual, full_cache, backend, tau=args.tau)
        all_out.append(out)
        print(f"\n===== {backend} (tau={args.tau}) =====")
        print(out[["policy", "macro_f1", "precision_hallu", "recall_hallu", "FP",
                   "residual_caught", "new_false_positives", "routed_frac", "cost_per_1k"]].to_string(index=False))
    full = pd.concat(all_out, ignore_index=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(RESULTS_DIR / "phase6_router_policies.csv", index=False)
    json.dump(report, open(RESULTS_DIR / "phase6_router_sizing.json", "w"), indent=2)
    print(f"\nsaved results/phase6_router_policies.csv")
    return full, report


if __name__ == "__main__":
    main()
