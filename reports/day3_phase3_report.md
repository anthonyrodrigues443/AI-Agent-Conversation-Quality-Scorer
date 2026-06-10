# Phase 3: Feature engineering on the claim↔passage relation + a HaluEval-fine-tuned cross-encoder — AI Agent Conversation Quality Scorer
**Date:** 2026-06-10
**Session:** 3 of 7

## Objective
Phase 2 ended on a surprise: a one-line **lexical grounding-overlap** rule (`0.9244` length-matched
macro-F1) beat three embedding encoders and a 184M zero-shot NLI cross-encoder. But that rule has a
documented blind spot — **entity-reuse hallucinations** (answers that reuse passage tokens but assert the
wrong claim), where its recall drops to **0.834**. Two questions for today:

1. Can **engineered claim-relation features** the bag-of-words overlap can't see — verbatim-span grounding,
   numeric/date consistency, best-supporting-*sentence* concentration, novel-content / IDF-weighted overlap,
   negation polarity — beat 0.9244 matched and rescue the entity-reuse misses?
2. Does a **HaluEval-*fine-tuned* cross-encoder** finally beat the lexical rule that zero-shot NLI couldn't?

**Primary metric:** length-matched macro-F1 (locked Phase 1). **Bar to beat: 0.9244.**

## Research & References
1. **Belyi et al., "Luna" (arXiv:2406.00975, 2024)** — fine-tuning a DeBERTa-v3 NLI cross-encoder with a
   shallow per-token head beats LLMs at hallucination detection **in-domain at a fraction of the cost**.
   → motivated fine-tuning a cross-encoder on HaluEval's own supervision rather than trusting zero-shot.
2. **"Beyond ROUGE: N-Gram Subspace Features for LLM Hallucination Detection" (arXiv:2509.05360, 2025)** —
   engineered lexical / n-gram features outperform ROUGE and embedding similarity. → motivated the
   verbatim-span (`is_substr`, `lcs_*`) and IDF-weighted overlap features.
3. **HALT-RAG (arXiv:2509.07475, 2025)** — combines lexical-overlap signals with calibrated NLI in a
   learned head + abstention. → motivated the combined LogReg/XGB head over `[overlap ⊕ engineered ⊕ CE]`.
4. **"On a Scale from 1 to 5" (arXiv:2410.12222, 2024)** — a fine-tuned HHEM beats LLMs in *domain-specific*
   faithfulness eval given good training data. → the central Phase-3 hypothesis (and it held).

*How it shaped the experiments:* (2)+(3) say "engineer lexical features and learn a head over them"; (1)+(4)
say "fine-tune the cross-encoder on the task, don't trust zero-shot." Phase 3 ran both arms head-to-head
against the Phase-2 overlap bar, on the same frozen split and the same nearest-length matched control.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 20,000 (10k items × {grounded, hallucinated}) |
| Train / Test | 16,000 / 4,000 (GroupShuffleSplit by `qid`, frozen `phase1_split_qids.json`) |
| Target | label ∈ {0 grounded, 1 hallucinated}, balanced |
| Length-matched control | nearest-length pairing, caliper 8 chars → **n=572**, answer-length KS 0.874 → 0.123 |

## Experiments

### Experiment 3.1 — Engineered claim-relation features (12, five groups)
**Hypothesis:** features encoding the answer↔passage *relation* (not just shared tokens) will catch the
entity-reuse hallucinations overlap misses.
**Method:** built 12 features on all 20k rows (≈5 s): verbatim grounding (`is_substr`, `lcs_char_ratio`,
`lcs_token_ratio`), numeric/date (`num_frac_in_know`, `n_num_missing`, `year_mismatch`), sentence
concentration (`sent_overlap_max`, `overlap_spread`), novelty/rarity (`novel_overlap`, `idf_overlap` —
IDF fit on train only), polarity (`ans_has_neg`, `neg_mismatch`).
**Result (Pearson |corr| with label):** `is_substr` −0.96, `lcs_char_ratio` −0.94, `lcs_token_ratio` −0.86,
`sent_overlap_max` −0.67, `ground_overlap` −0.59, `idf_overlap` −0.59; numeric/negation features ≤ 0.23.
**Interpretation:** HaluEval-QA's grounded answers are **verbatim extracted spans** (96.7% are exact
substrings of the passage) while hallucinations are generated sentences (0.3% substrings). The verbatim
signal dominates; numeric/date/negation are individually sparse.

### Experiment 3.2 — Single-feature probe (which one feature beats the bar?)
**Method:** each feature thresholded alone (threshold + direction tuned on train), scored on the matched control.

| Rank | Feature | matched F1 | raw F1 | beats 0.9244? |
|---:|---|---:|---:|:--:|
| 1 | lcs_token_ratio | **0.9843** | 0.9665 | ✅ +0.060 |
| 2 | is_substr | 0.9825 | 0.9737 | ✅ |
| 3 | lcs_char_ratio | 0.9825 | 0.9737 | ✅ |
| 4 | sent_overlap_max | 0.9247 | 0.9377 | ≈ tie |
| 5 | ground_overlap (Phase-2 champ) | 0.9244 | 0.9252 | — |
| 6 | idf_overlap | 0.9244 | 0.9255 | — |
| 7 | novel_overlap | 0.8519 | 0.8696 | ❌ |
| 8–13 | overlap_spread / numeric / negation | 0.38–0.43 | 0.39–0.61 | ❌ (≈chance alone) |

**Interpretation:** a **single** feature — longest common *token* run between answer and passage,
normalized by answer length — beats the bar by +0.06. Verbatim **contiguity**, not semantics, is the
strongest cheap signal on this benchmark.

### Experiment 3.3 — Combined head (is the bottleneck the model or the features?)
**Method:** `[ground_overlap ⊕ 12 engineered]` → LogReg and XGBoost; plus an **engineered-only** XGB that
*drops* the Phase-2 overlap feature.
**Result (matched macro-F1):** `eng_xgboost` **0.9808**, `eng_only_xgboost` **0.9808** (identical),
`eng_logreg` 0.9720. XGB importance: `is_substr` 0.795, `lcs_token_ratio` 0.116, `lcs_char_ratio` 0.060,
numeric+overlap+rest < 0.03 combined.
**Interpretation:** **the bottleneck was the features, not the model** — and the new features fully
*subsume* the Phase-2 champion: removing `ground_overlap` changes nothing (0.9808 → 0.9808). The right
features made the old champion redundant.

### Experiment 3.4 — HaluEval-fine-tuned cross-encoder (the honest "read meaning")
**Hypothesis (Luna / "1-to-5"):** fine-tuning on HaluEval's own (knowledge, answer) supervision turns the
zero-shot loser into a winner.
**Method:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M), reinitialised 2-class head, 3 epochs on 16k train
pairs, threshold tuned on a held-out 2k train-dev slice (no test leakage), logits cached.
**Result:** **0.9633 matched** / 0.9935 raw (recall_hallu 0.944 matched).
**Interpretation:** fine-tuning lifts the cross-encoder from the Phase-2 zero-shot floor of **0.687 to 0.963**
(+0.276) and clears the lexical bar by +0.039 — domain fine-tuning beats zero-shot exactly as the literature
predicts. (The 184M `nli-deberta-v3-base`, the *exact* model that lost zero-shot, was MPS-prohibitive to
fine-tune here — disentangled-attention CPU-fallback pushed one epoch past 45 min — so the 22M model is the
fine-tuned CE of record. That it already clears the bar by a wide margin means the heavyweight wasn't needed.)

### Experiment 3.5 — Hybrid head (cross-encoder ⊕ engineered)
**Method:** `[CE score ⊕ ground_overlap ⊕ lcs_token_ratio ⊕ is_substr ⊕ sent_overlap_max ⊕ num_frac ⊕ idf]` → XGB.
**Result:** 0.9790 matched, **best hallucination recall (0.9755 matched)** and **best entity-reuse rescue** (below).

## Head-to-Head Comparison — master leaderboard (ranked by length-matched macro-F1)
| Rank | Model | matched F1 | raw F1 | raw AUC | drop | beats bar |
|---:|---|---:|---:|---:|---:|:--:|
| 1 | eng_xgboost (overlap ⊕ 12 eng) | **0.9808** | 0.9967 | 0.9975 | 0.0159 | ✅ +0.056 |
| 1 | eng_only_xgboost (12 eng, no overlap) | **0.9808** | 0.9967 | 0.9975 | 0.0159 | ✅ +0.056 |
| 3 | hybrid_ce_plus_eng | 0.9790 | 0.9960 | 0.9995 | 0.0170 | ✅ |
| 4 | eng_logreg | 0.9720 | 0.9917 | 0.9968 | 0.0197 | ✅ |
| 5 | ce_minilm_l6_finetuned (22M) | 0.9633 | 0.9935 | 0.9989 | 0.0302 | ✅ |
| 6 | grounding_overlap_threshold (Phase-2 bar) | 0.9244 | 0.9252 | 0.8989 | 0.0008 | — |

## Entity-reuse rescue — the actual target
Overlap misclassifies **175** of the 1,054 high-overlap (entity-reuse) hallucinations as grounded
(recall 0.834). Phase 3 closes this:

| Model | recall (entity-reuse) | rescued of overlap's 175 misses |
|---|---:|---:|
| **hybrid_ce_plus_eng** | **0.994** | **0.966** |
| eng_xgboost / eng_only_xgboost | 0.991 | 0.949 |
| ce_minilm_l6_finetuned | 0.989 | 0.949 |
| grounding_overlap_threshold | 0.834 | 0.000 |

## Honesty check — is the verbatim win *grounding*, or just answer *form*?
`is_substr` separates the full task at |corr|=0.96, but HaluEval's grounded answers are *extracted spans*
and its hallucinations are *generated sentences* — so a verbatim test may be reading **answer form**, a
deeper cousin of the Phase-1 length shortcut (the matched control only controls length, not form). I built
a **form-ambiguous** slice: the 95 qids whose grounded answer is *not* a verbatim span (n=190, balanced),
where form no longer leaks the label, and re-scored every model:

| Model | form-ambiguous F1 | full matched F1 |
|---|---:|---:|
| ce_minilm_l6_finetuned | **0.9947** | 0.9633 |
| eng_xgboost / eng_only_xgboost | 0.9947 | 0.9808 |
| hybrid_ce_plus_eng | 0.9895 | 0.9790 |
| eng_logreg | 0.9152 | 0.9720 |
| **grounding_overlap_threshold** | **0.3286** | 0.9244 |

**This re-frames Phase 2.** The overlap champion **collapses to chance (0.33)** when the grounded answer
isn't a verbatim span — it was substantially a *form/length* detector. The **fine-tuned cross-encoder** and
the **engineered-feature ensemble** stay at ≈0.99 — they read grounding even when form is stripped. So on
the full benchmark the verbatim features score highest, but the model that genuinely *reads grounding* is
the fine-tuned CE / engineered ensemble.

## Key Findings
1. **The bottleneck was the features, not the model** — and the new features *subsume* the Phase-2 champion:
   `eng_only_xgboost` (no overlap feature) ties `eng_xgboost` at 0.9808. Removing overlap costs nothing.
2. **A single feature beats the bar:** longest common token run (`lcs_token_ratio`) alone = 0.9843 matched.
   On HaluEval-QA, verbatim contiguity is the dominant signal.
3. **Fine-tuning rescued the meaning model:** zero-shot NLI 0.687 → fine-tuned MiniLM-L6 CE **0.963** (+0.276),
   beating the lexical bar — domain fine-tuning > zero-shot, as Luna / "1-to-5" predict.
4. **Entity-reuse target crushed:** overlap caught 83.4% of entity-reuse hallucinations; the hybrid catches
   **99.4%** and rescues **96.6%** of overlap's 175 outright misses.
5. **(Self-correction) The verbatim win is partly answer-form.** On the form-ambiguous slice the overlap
   champion collapses to 0.33; only the fine-tuned CE and the engineered ensemble hold (≈0.99). That is the
   honest measure of "reads meaning."

## What Didn't Work (and why)
- **Numeric / date / negation features in isolation** are near-chance (0.38–0.43 matched) and contribute
  < 2% to the XGB. HaluEval-QA hallucinations rarely swap a single checkable number; they reformulate the
  whole claim. The signal that survives is contiguity/verbatim, not arithmetic.
- **`overlap_spread`** (whole-passage minus best-sentence overlap) alone is *worse* than chance (0.43) —
  the hoped-for "entities stitched across sentences" tell is too weak to threshold on its own.
- **deberta-v3-base fine-tuning** was MPS-prohibitive (CPU-fallback for disentangled attention, >45 min/epoch).
  The lightweight 22M MiniLM was sufficient — a practical-deployment point, not just an inconvenience.

## Error Analysis
- The hybrid's only residual entity-reuse misses (~0.6%) are hallucinations that are *near-verbatim*
  copies with a single swapped token — high `lcs_token_ratio` and `is_substr`≈1, so every verbatim feature
  votes "grounded"; only the CE's contextual read flags them, which is why the hybrid > engineered-only on recall.
- Overlap's collapse on the form-ambiguous slice (0.33) is the mirror of its full-task strength: when the
  grounded answer is itself a paraphrase, raw token-overlap can't tell it from a fluent hallucination.

## Next Steps (Phase 4 — tuning + error analysis)
- Tune the champion (`eng_xgboost` and the hybrid) with Optuna; the headroom is small (already 0.98), so the
  real Phase-4 question is **calibration + operating point** for the entity-reuse minority and a deeper error
  taxonomy of the residual ~0.6% misses.
- Promote the **form-ambiguous** slice to a standing eval — it is a better stress test than the length-matched
  control for "does this read grounding." Build a form-matched control alongside the length-matched one.
- Carry forward the fine-tuned CE as the "reads-meaning" baseline into Phase 5's LLM head-to-head (Claude/Codex).

## References Used Today
- [1] Belyi et al. *Luna: An Evaluation Foundation Model to Catch LM Hallucinations with High Accuracy and Low Cost.* arXiv:2406.00975 (2024).
- [2] *Beyond ROUGE: N-Gram Subspace Features for LLM Hallucination Detection.* arXiv:2509.05360 (2025).
- [3] *HALT-RAG: A Task-Adaptable Framework for Hallucination Detection with Calibrated NLI Ensembles and Abstention.* arXiv:2509.07475 (2025).
- [4] *On a Scale from 1 to 5: Quantifying Hallucination in Faithfulness Evaluation.* arXiv:2410.12222 (2024).
- [5] Li et al. *HaluEval.* EMNLP 2023.

## Code Changes
- `notebooks/phase3_features_crossencoder.ipynb` — 27 cells (13 code / 14 md), executed on `convo-quality-venv`,
  0 errors, 0 fake display-only cells; feature engineering + single-feature probe + combined LogReg/XGB +
  fine-tuned MiniLM-L6 CE (cached, resumable) + hybrid + master leaderboard + entity-reuse rescue + form-ambiguous honesty check.
- `results/` — `phase3_feature_comparison.csv`, `phase3_single_feature.csv`, `phase3_entity_reuse_rescue.csv`,
  `phase3_form_ambiguous.csv`, `metrics.json` (phase3 key), figures `phase3_{leaderboard,feature_importance,entity_reuse_rescue}.png`,
  `phase3_cache/ce_minilm_l6.npy` (cached CE logits, gitignored).
- `results/EXPERIMENT_LOG.md` — Phase 3 section appended.
- `reports/day3_phase3_report.md` — this report.
- `README.md` — Phase 3 headline, findings, iteration block, roadmap.
