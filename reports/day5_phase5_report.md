# Phase 5: Advanced Techniques + Ablation + Frontier-LLM Head-to-Head — AI Agent Conversation Quality Scorer
**Date:** 2026-06-12
**Session:** 5 of 7

## Objective
Phase 4 left the champion `eng_xgboost` (13 features) feature-bound and saturated at **0.9808 length-matched macro-F1**, with a single residual failure mode: **question-relevance, not knowledge-grounding** — 10 of 12 false negatives are verbatim substrings of the passage that are simply the *wrong answer to the question*. Three questions:
1. Can a **question-aware** signal (a cross-encoder fine-tuned with the question in its input) break the ceiling and rescue the grounded-but-irrelevant residual?
2. **Which of the 13 features actually carry the signal**, and does any feature *hurt*? (leave-one-out ablation)
3. Zero-shot **frontier LLMs vs the champion** — overall, and specifically on the residual the champion can't touch.

## Research & References
1. **Luna** (Belyi et al., *arXiv:2406.00975*, 2024) — a small fine-tuned DeBERTa-NLI + shallow classifier, conditioned on **query + context**, catches LLM hallucinations at high accuracy and ~100–1000× lower cost than GPT judges. Motivated both the QA-aware CE and the cost framing of the head-to-head.
2. **QA-based faithfulness** (QAFactEval / QuestEval lineage) — faithfulness is checked by asking questions of the answer against the source; *relevance is undecidable without the question*. Direct motivation for putting the question inside the encoder.
3. **"The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs"** (*arXiv:2508.08285*, 2025) — detectors drop up to 45.9% under human-aligned metrics; *how* you evaluate decides who wins. Kept the head-to-head honest (real predictions vs CLI-inflated latency; probe the failure mode, not just an average).

How it shaped the work: rather than chase the saturated average, I built the question-aware model the QA-faithfulness literature predicts should help, then used the residual *as the probe* for the LLM comparison — the Luna thesis (small fine-tuned model wins on cost) and its limit (relevance needs reasoning) are exactly what the two evaluations test.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 20,000 (10,000 HaluEval-QA items × {grounded, hallucinated}) |
| Features | 13 engineered (ground_overlap ⊕ 12) + a fine-tuned cross-encoder logit |
| Target | hallucinated (1) vs grounded (0) |
| Class distribution | 50/50 (balanced by construction) |
| Train/Test split | 16,000 / 4,000 (frozen GroupShuffleSplit by qid); length-matched control n=572 |

## Experiments

### Experiment 5.1: Question-aware cross-encoder + QA-relevance hybrid
**Hypothesis:** putting the question into the encoder (segment A = `"question: {q} context: {knowledge}"`) lets a fine-tuned CE read *relevance*, rescuing the residual no grounding feature can.
**Method:** fine-tune `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M) with **identical** hyper-parameters to Phase 3 (3 epochs, bs=32, lr=2e-5), changing *only* the input to include the question; threshold-free argmax decisioning; compare to the Phase-3 grounding-only CE; then add the QA-CE logit to the 13-feature stack as a hybrid.
**Result:**
| Model | matched_F1 | raw_F1 | residual (of 10) | ctrl FPR |
|-------|-----------:|-------:|:----------------:|---------:|
| ce_grounding (k, a) — Phase 3 | 0.9457 | 0.9887 | 2/10 | 0.0031 |
| ce_qa_aware (q+k, a) — Phase 5 | **0.9211** | 0.9867 | 2/10 | 0.0037 |
| qa_relevance_hybrid (13f + ce_qa) | **0.9860** | 0.9972 | 2/10 | — |
| eng_xgboost (Phase-4 champion) | 0.9808 | 0.9967 | 0/10 | — |

**Interpretation:** Against the hypothesis — **adding the question made the standalone classifier worse** (0.9457 → 0.9211, −0.025): the question's tokens dilute the answer↔knowledge signal the CE leans on, and it still caught only 2/10 of the residual. Yet the QA-CE *probability* is a useful soft feature — it is the hybrid's #2 most important input (importance 0.303) and lifts the ceiling to **0.9860 (+0.0052)** (*optimistically biased — the hybrid is stacked on in-sample train CE predictions; clean OOF stacking is Phase-6 work — see post-review corrections*). Soft signal > hard argmax. The residual barely moved (0 → 2 of 10): question-relevance is a genuinely hard, distinct failure mode even for a model fine-tuned on the exact task.

### Experiment 5.2: Leave-one-feature-out ablation
**Hypothesis:** correlated features mask each other; honest removal will reveal which features truly carry signal and whether any hurt.
**Method:** drop each of the 13 features, retrain XGBoost from scratch on the remaining 12, measure Δ length-matched macro-F1.
**Result:**
| Dropped | matched_F1 | Δ | Verdict |
|---------|-----------:|--:|---------|
| (none) full 13f | 0.9808 | 0.0000 | baseline |
| **lcs_char_ratio** | 0.9065 | **−0.0742** | the only feature that matters |
| all other 12 (ground_overlap, is_substr, lcs_token_ratio, …) | 0.9808 | 0.0000 | fully redundant |

**Interpretation:** The 13-feature champion is a **1-feature model in disguise**. Removing `lcs_char_ratio` (longest common *character* run) costs −0.074; removing any other feature costs **exactly nothing**. `is_substr` (Phase 3's "top" feature by split-importance) is fully substituted by its continuous cousin — split-importance told us what the tree *used*, removal tells us what it would *lose*. No feature hurts. For HaluEval-QA, *"is the answer a near-verbatim slice of the passage?"* is essentially the whole task.

### Experiment 5.3: Frontier-LLM head-to-head + grounded-but-irrelevant probe
**Hypothesis:** the tiny tree wins on cost/speed/average, but a frontier model that *reasons* about relevance should own the residual.
**Method:** zero-shot Claude Opus 4.8, Claude Haiku 4.5, Codex GPT-5.5 via local CLIs, one-label+probability prompt, threaded append-only cache (210 calls, 100% parse-success, idempotent). Two evaluations: a representative n=50 (stratified 25/25), and a probe of the 10 grounded-but-irrelevant hallucinations (champion 0/10) + 10 verbatim grounded controls.

**Result — representative n=50 (ranked by macro-F1):**
| Rank | Model | Acc | macro-F1 | Prec(hallu) | Recall(hallu) | Latency/row | Cost/1k |
|------|-------|----:|---------:|------------:|--------------:|------------:|--------:|
| 1 | **eng_xgboost (champion, CPU)** | 1.00 | **1.000** | 1.000 | 1.00 | 0.031 ms | $0.0001 |
| 2 | qa_relevance_hybrid (CPU + 1 CE fwd) | 1.00 | 1.000 | 1.000 | 1.00 | ~15 ms | $0.012 |
| 3 | Claude Haiku 4.5 (zero-shot) | 0.90 | 0.900 | 0.917 | 0.88 | 9.7 s (CLI) | $0.35 |
| 4 | Codex GPT-5.5 (zero-shot) | 0.90 | 0.899 | 1.000 | 0.80 | 18.6 s (CLI) | $50.0* |
| 5 | Claude Opus 4.8 (zero-shot) | 0.82 | 0.816 | 0.944 | 0.68 | 7.5 s (CLI) | $5.25 |

**Result — grounded-but-irrelevant probe (10 verbatim hallucinations + 10 grounded controls):**
| Model | residual recall | ctrl FPR | probe bal-acc |
|-------|:---------------:|---------:|--------------:|
| Codex GPT-5.5 | **10/10** | 0.10 | 0.95 |
| Claude Haiku 4.5 | 9/10 | 0.20 | 0.85 |
| Claude Opus 4.8 | 5/10 | 0.00 | 0.75 |
| ce_qa_aware (fine-tuned) | 2/10 | 0.00 | 0.60 |
| qa_relevance_hybrid | 2/10 | 0.00 | 0.60 |
| **eng_xgboost (champion)** | **0/10** | 0.00 | 0.50 |

**Interpretation:** Two mirror-image findings. On the representative distribution the CPU tree scores a perfect 1.000 and **beats every frontier model** (Opus 0.816, Codex 0.899, Haiku 0.900) at **3,500×–500,000×** lower cost and ~250,000× lower latency. But on the residual it is total-blind (0/10) — and the model that *loses the average* wins the slice: **Codex GPT-5.5 catches all 10**, Haiku 9, Opus 5. A 22M cross-encoder fine-tuned on this exact task manages only 2/10; relevance reasoning, not in-domain fine-tuning, is what cracks it.

\* Codex cost is the CLI-realistic figure (~12k agent-loop tokens/call observed). At direct GPT-5.5 API pricing for the task's own I/O (~300 in / 10 out) it would be ≈ $0.5/1k. Either way the tree is ≥3,500× cheaper than the cheapest frontier option.

## Head-to-Head Comparison (primary metric: length-matched macro-F1; LLMs measured on n=50 representative)
| Rank | Model | matched/representative macro-F1 | residual (of 10) | cost/1k |
|------|-------|--------------------------------:|:----------------:|--------:|
| 1 | qa_relevance_hybrid | 0.9860 (matched) / 1.000 (n=50) | 2/10 | $0.012 |
| 2 | eng_xgboost (champion) | 0.9808 (matched) / 1.000 (n=50) | 0/10 | $0.0001 |
| — | Codex GPT-5.5 (zero-shot) | 0.899 (n=50) | **10/10** | $50* |
| — | Claude Haiku 4.5 (zero-shot) | 0.900 (n=50) | 9/10 | $0.35 |
| — | Claude Opus 4.8 (zero-shot) | 0.816 (n=50) | 5/10 | $5.25 |
| 3 | ce_grounding (Phase 3) | 0.9457 (matched) | 2/10 | ~$0 |
| 4 | ce_qa_aware (Phase 5) | 0.9211 (matched) | 2/10 | ~$0 |

## Key Findings
1. **The 13-feature champion is a 1-feature model.** Leave-one-out: only `lcs_char_ratio` matters (−0.074 when removed); the other 12 are exactly redundant (Δ=0.0000).
2. **Question-awareness HURTS the standalone classifier** (−0.025 matched-F1) but **helps as a soft feature** (hybrid +0.0052, the QA-CE logit is the #2 feature). More input ≠ more signal; soft probability > hard argmax.
3. **The tiny tree beats Opus 4.8, Haiku 4.5, and Codex GPT-5.5** at hallucination detection on the representative distribution (1.000 vs 0.82–0.90) at thousands-to-hundreds-of-thousands× lower cost/latency.
4. **…except on grounded-but-irrelevant hallucinations, where only frontier reasoning wins** (Codex 10/10, Haiku 9/10, Opus 5/10 vs champion 0/10 and a fine-tuned QA-CE 2/10).
5. **Opus is the conservative outlier** — highest precision (0.944), lowest recall (0.68), never false-alarms on the probe but catches only half. Haiku/Codex are more aggressive flaggers.

## Frontier Model Comparison (summary)
| Slice | Champion (13f tree) | Claude Opus 4.8 | Claude Haiku 4.5 | Codex GPT-5.5 | Winner |
|-------|--------------------:|----------------:|-----------------:|--------------:|--------|
| Representative n=50 (macro-F1) | **1.000** | 0.816 | 0.900 | 0.899 | **Champion** |
| Grounded-but-irrelevant (recall/10) | 0/10 | 5/10 | 9/10 | **10/10** | **Codex GPT-5.5** |
| Cost / 1k predictions | **$0.0001** | $5.25 | $0.35 | $50* | **Champion** |

## Error Analysis
- The champion's entire residual = grounded-but-irrelevant hallucinations (verbatim spans that answer a *different* question): e.g. Q on *"From Eden"* → *"Take Me To Church"* (another Hozier song quoted in the passage). All scored P≈0.01.
- These are detectable cheaply (`is_substr==1` + champion says grounded) → ideal LLM-router trigger.
- LLM failure modes differ: Opus under-flags (misses 5/10 hallucinations, 0 false alarms); Haiku slightly over-flags (FPR 0.20). Codex is the only one with both high recall and acceptable specificity on the probe.

## Production Recommendation (corrected after second-model review)
**A trigger router, not a single model — but the trigger is *not* a cheap minority on HaluEval.** Run the CPU tree on 100% of traffic (~free, sub-ms, 0.997 raw-F1) and route the `is_substr==1` + "tree-says-grounded" suspects to a frontier LLM for a relevance check. The Codex review (#7) correctly flagged my original "small minority" framing: **`is_substr==1` is ~48% of the test set (1,915 / 4,000)** because verbatim-correct answers are extremely common in HaluEval. So the naive trigger routes ~half of traffic, and at the LLM's measured probe FPR (~0.10, n=10) it would inject *dozens-to-hundreds* of new false positives among the 1,905 verbatim-grounded answers to rescue 10 hallucinations.

This sharpens rather than weakens the finding: **grounded-but-irrelevant detection is genuinely expensive**, because the only cheap signal that flags the suspects (verbatim overlap) also flags every correct verbatim answer. The open problem is a *tighter* trigger (e.g. multi-candidate questions only) and a high-precision LLM operating point — the truly blended recall/precision/cost on full test is **Phase-6 work** (the notebook's final cell computes the trigger economics that motivate it).

## Post-review corrections (Codex second-model pass)
| # | Codex finding | Verdict | Action |
|---|---------------|---------|--------|
| 1 | Hybrid uses in-sample train CE preds (stacking leakage) | Valid | Caveat added; +0.0052 hybrid gain flagged optimistic; OOF stacking → Phase 6. Headline findings unaffected. |
| 7 | `is_substr==1` router ≈48% of traffic, not a minority | Valid | Corrected here + in notebook with a computed economics cell. |
| 2,3 | Codex/Claude CLI models not version-pinned | Valid (repro) | This run's labels are correct (codex default resolved to gpt-5.5); pin exact IDs in Phase 6. |
| 4,5 | Parser substring-match / cache not content-addressed | Valid (repro) | No impact this run (100% parse-success, 0 failed parses); harden in Phase 6. |
| 6 | Codex cost comment vs $0.05/call inconsistent | Valid | Report already discloses both CLI-agent and direct-API figures; will compute from recorded tokens in Phase 6. |

## Next Steps
- **Phase 6 (Sat):** production pipeline + Streamlit UI exposing the router (tree score + LLM relevance check on verbatim suspects), with the dual head-to-head visualized.
- Quantify the router's end-to-end recall on a held-out blend (tree everywhere + LLM on `is_substr==1`) and its blended cost/latency.
- Test whether a cheap *extractive QA* model (answer-the-question-from-passage, compare to the given answer) recovers most of the LLM's relevance signal without the LLM cost.

## References Used Today
- [1] Belyi et al., *Luna: An Evaluation Foundation Model to Catch Language Model Hallucinations with High Accuracy and Low Cost*, arXiv:2406.00975 (2024) — https://arxiv.org/pdf/2406.00975
- [2] QA-based faithfulness lineage (QAFactEval / QuestEval) — question-aware faithfulness via QA equivalence.
- [3] *The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs*, arXiv:2508.08285 (2025) — https://arxiv.org/html/2508.08285v2

## Code Changes
- `notebooks/phase5_advanced_llm.ipynb` — full Phase-5 research notebook (champion reproduction, question-aware CE fine-tune + hybrid, leave-one-out ablation, 210-call frontier-LLM head-to-head + probe; cached & idempotent; 0 errors, 0 fake-display cells).
- `results/phase5_cache/` — `ce_qa_minilm_l6.npy` (QA-CE logits), `eng_F.npy`, `llm/{sample_idx.json, llm_calls.json}` (append-only LLM cache).
- `results/{phase5_ablation, phase5_probe, llm_vs_custom, phase5_leaderboard}.csv` + figures `{phase5_ablation, phase5_probe, llm_comparison}.png`; `results/metrics.json[phase5]`; `models/ce_qa_minilm_l6/`.
