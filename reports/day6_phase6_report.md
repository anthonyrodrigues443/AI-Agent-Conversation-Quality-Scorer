# Phase 6: Production Pipeline + Streamlit UI + Blended Router Economics — AI Agent Conversation Quality Scorer
**Date:** 2026-06-14
**Session:** 6 of 7 (Saturday's session was missed; Phase 6 run on Sunday, Phase-7 polish folded in)

## Objective
Two goals. (1) Turn the Phase 1–5 research into a clean, importable production pipeline + a polished Streamlit demo + a test suite. (2) Answer the question Phase 5 explicitly deferred to Phase 6: **does the "ship both — tree everywhere, LLM on the verbatim suspects" router actually improve the system on the full test set, once you pay the LLM's false-positive rate on the verbatim-*correct* majority?**

## Research & References
1. **HaluEval (Li et al., EMNLP 2023)** — the dataset; defines the grounded/hallucinated pair per item that the whole project classifies. Used the official `qa_data.json`. — https://github.com/RUCAIBox/HaluEval
2. **Cascade / deferral systems (e.g. Viola–Jones attentional cascade; "model cascades" / "LLM routing" literature, 2023–2025)** — the pattern of a cheap model handling most traffic and escalating a hard slice to an expensive one. The standard caveat they stress: a cascade only pays when the escalated slice is *enriched* for the hard cases. Phase 6 tests exactly that enrichment assumption and finds it fails here.
3. **HuggingFace / Google model-card standards** — followed for `models/model_card.md` (intended use, training data, metrics, limitations, ethical considerations).

How research influenced today's work: the cascade literature says the escalation trigger must concentrate the errors; that framed the central experiment — measure how enriched the verbatim-suspect slice actually is for true hallucinations (answer: 0.5%, fatally un-enriched).

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 20,000 (10,000 items × {grounded, hallucinated}) |
| Features | 13 (ground_overlap + 12 engineered claim-relation features) |
| Target | label ∈ {0 grounded, 1 hallucinated} |
| Class distribution | perfectly balanced (10,000 / 10,000) |
| Train/Test split | 16,000 / 4,000, GroupShuffleSplit by item id, frozen (seed 42) |
| Primary metric | macro-F1 |

## Experiments

### Experiment 6.1: Reproduce the champion as a serialized artifact
**Hypothesis:** the production pipeline, computing features as pure importable Python, reproduces the notebook champion exactly.
**Method:** `src/feature_engineering.py` recomputes all 13 features byte-for-byte; `src/train.py` fits XGBoost (n_estimators=500, max_depth=4, lr=0.05) and serializes a 0.32 MB bundle (model + fitted IDF table + threshold).
**Result:** raw-test macro-F1 **0.9967**, length-matched **0.9808**, recall_hallu 0.994, confusion [[1999,1],[12,1988]] — identical to Phase 3/4.
**Interpretation:** production = research. The bundle ships without the raw dataset (only `idf_overlap` is train-dependent; its IDF table is persisted).

### Experiment 6.2: Router trigger sizing on the full test set
**Hypothesis (from Phase 5):** tightening the escalation trigger to "multi-candidate questions only" isolates the residual hallucinations cheaply.
**Method:** score all 4,000 test rows; bucket the verbatim suspects (`is_substr==1` AND tree-grounded) by trigger; locate the 10 hallucinations the tree misses.
**Result:**
| trigger | rows | % traffic | residual caught |
|---|--:|--:|:--:|
| naive (all verbatim suspects) | 1,915 | 47.9% | 10/10 |
| multi-candidate (Phase-5 idea) | 165 | 4.1% | **0/10** |

**Interpretation — counterintuitive:** the Phase-5 heuristic is *exactly wrong*. The 10 residual hallucinations are single-candidate **multi-hop** questions where the verbatim answer is the wrong hop (e.g. quoting the "Leading Minister" when asked who succeeded Hitler). The 165 multi-candidate suspects contain **zero** hallucinations. No cheap lexical signal (question length, multi-hop markers) cleanly separates the 10 from the 1,905 verbatim-correct answers — catching 8/10 by question length still routes 38% of traffic.

### Experiment 6.3: Blended router economics (real, cached LLM eval)
**Hypothesis:** escalating verbatim suspects to a frontier LLM improves macro-F1.
**Method:** ran a real, cached, resumable LLM eval (`src/router_eval.py`) on all 10 residual hallucinations + a stratified control sample of verbatim-correct grounded answers, for **Claude Haiku 4.5** (residual-recall 0.90, FPR 0.15, n=40 controls) and **Codex GPT-5.5** (residual-recall 1.00, FPR 0.167, n=18 controls). Then blended the full-test confusion by expected value for each routing policy.
**Result:** see Head-to-Head below.
**Interpretation:** every router lowers macro-F1. The residual is 0.25% of test; the suspect pool is 99.5% correct, so the LLM's FPR on that majority dominates the handful of rescues.

## Head-to-Head Comparison (full test, n=4,000, τ=0.5)
| Policy (judge) | macro-F1 | precision | recall | new FP | residual caught | routed | cost/1k |
|---|--:|--:|--:|--:|:--:|--:|--:|
| **tree only** | **0.9967** | 0.9995 | 0.994 | 0 | 0/10 | 0% | $0.0001 |
| multihop router (Haiku) | 0.9908 | 0.985 | 0.997 | 30 | 5/10 | 4.9% | $0.015 |
| multi-candidate router (Haiku) | 0.9906 | 0.987 | 0.994 | 26 | 0/10 | 4.1% | $0.013 |
| complexity router q≥20 (Haiku) | 0.9759 | 0.957 | 0.997 | 90 | 5/10 | 14.9% | $0.045 |
| naive router (Haiku) | 0.9272 | 0.874 | 0.999 | 287 | 9/10 | 47.9% | $0.144 |
| naive router (Codex GPT-5.5) | 0.9194 | 0.863 | 0.999 | 318 | **10/10** | 47.9% | $24.0 |

## Key Findings
1. **tree-only is Pareto-optimal — every LLM router makes the system worse.** Lexical grounding is a commodity the tree nails for ~free; the residual relevance errors are too rare (0.25%) to rescue without flooding the verbatim-correct majority with false positives.
2. **Even Codex GPT-5.5 (10/10 on the residual) drops macro-F1 to 0.9194** by manufacturing ~318 false positives at ~$24/1k. Perfect recall on the hard slice is necessary but nowhere near sufficient.
3. **What didn't work, and why:** the Phase-5 "multi-candidate only" trigger (0/10) — the residual lives in single-candidate multi-hop questions, not comparative ones. And no cheap lexical trigger isolates the 10 from the 1,905 correct answers, *because* relevance is a reasoning property, not a lexical one — the exact thesis of Phase 5, now quantified.

## Self-Correction
Phase 5's production recommendation was "ship both — tree at scale, LLM only on the suspects." **Phase 6 corrects it: on HaluEval-QA, ship tree-only.** A deferral cascade only pays when the escalated slice is enriched for the hard cases; here it is 99.5% easy cases, so escalation is a net loss at every operating point and every backend tested.

## Error Analysis
The 10 residual hallucinations are all verbatim entity quotes that answer the wrong hop of a multi-hop question. They are invisible to lexical features (every grounding feature says "supported") and require reasoning over the question to catch. The tree's single false positive is a short grounded answer with low character contiguity.

## Next Steps (Phase 7 / future)
- A learned (not lexical) escalation gate: a tiny second model predicting "is this verbatim answer the *relevant* hop?" — the only way to enrich the slice enough to make escalation pay.
- Calibrated per-row routing on the LLM's own confidence (sweep τ) rather than a hard trigger.
- CI workflow running `pytest tests/` on every PR.

## References Used Today
- [1] Li et al., *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs*, EMNLP 2023 — https://github.com/RUCAIBox/HaluEval
- [2] Model cascades / LLM routing literature (2023–2025) — escalation pays only on an error-enriched slice.
- [3] HuggingFace/Google model-card standard — https://huggingface.co/docs/hub/model-cards

## Code Changes
- `src/{utils,data_pipeline,feature_engineering,train,predict,evaluate,llm_judge,router,router_eval,plots}.py` — production pipeline (10 modules)
- `models/champion.joblib` (0.32 MB), `models/model_card.md`
- `app.py` — Streamlit two-head demo (live tree scoring, verbatim-suspect flag, on-demand LLM relevance check, Phase-5 head-to-head sidebar)
- `tests/{conftest,test_data_pipeline,test_feature_engineering,test_predict,test_router}.py` — 19 tests, all green
- `results/{phase6_router_policies.csv,phase6_router_sizing.json,phase6_router_tradeoff.png,ui_screenshot.png}`, `results/phase6_cache/` (append-only LLM cache), `results/metrics.json[phase6]`, `results/EXPERIMENT_LOG.md` (Phase 6)
- `requirements.txt` (+joblib, streamlit, pyyaml, pytest)
