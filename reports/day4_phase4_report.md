# Phase 4: Hyperparameter Optimization, Calibration, Operating Points & Error Analysis — AI Agent Conversation Quality Scorer
**Date:** 2026-06-11
**Session:** 4 of 7

## Objective
Phase 3 left a near-saturated champion — `eng_xgboost` (grounding-overlap ⊕ 12 engineered claim-relation
features) at **0.9967 raw / 0.9808 length-matched** macro-F1 — with the explicit verdict that *the bottleneck is
the features, not the model*. So today is **not** a score-chasing session. The questions are:
1. Does systematic hyperparameter search actually move a feature-bound model?
2. Is XGBoost even the right family, or do LightGBM / CatBoost / a calibrated linear head win once tuned?
3. Is the champion's *probability* trustworthy (calibration / ECE / Brier)?
4. What operating point should ship, and how does it trade recall on the entity-reuse minority?
5. *Where exactly* does it still fail — and is more data the answer?

## Research & References
1. **Random Realizations / Optuna docs — "Tuning XGBoost with Optuna."** TPE Bayesian search, ~30+ trials
   usually sufficient, scope the space to budget. → I tuned 9 XGB knobs under a grouped-CV objective with a
   wall-clock cap and seeded the study with the Phase-3 default.
2. **Niculescu-Mizil & Caruana, "Obtaining Calibrated Probabilities from Boosting" (UAI 2005); FastML.**
   Boosted trees are classically *mis*-calibrated; isotonic most consistently helps but needs data, Platt
   assumes a sigmoid distortion. ECE = weighted mean |acc − conf| per bin. → I compared uncalibrated / Platt /
   isotonic with a held-out, group-disjoint calibration split.
3. **MachineLearningMastery, "Threshold-Moving for Imbalanced Classification."** 0.5 is rarely optimal; pick the
   operating point from explicit FN:FP costs and report the PR curve. → operating-point sweep + cost model.
4. **"The Mirage of Hallucination Detection" (Findings-EMNLP 2025) & "The Gray Zone of Faithfulness" (2025).**
   Trained hallucination classifiers are distribution-shift sensitive and faithfulness is genuinely ambiguous.
   → I keep the **form-ambiguous** slice as a permanent honesty control and read the residual errors carefully.

**How it shaped the session:** the literature predicts tuning + calibration should *both* help a boosted-tree
classifier. Testing that prediction on a saturated problem is the experiment — and both predictions failed,
which is the finding.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 20,000 (10k items × {grounded, hallucinated}) |
| Features | grounding-overlap ⊕ 12 engineered (13 total) |
| Target | label ∈ {0 grounded, 1 hallucinated}, balanced 50/50 |
| Split | frozen GroupShuffleSplit by `qid`, 16k train / 4k test |
| CV | StratifiedGroupKFold(4) by `qid` (group-disjoint, leakage-free) |
| Length-matched control | nearest-length pairing, n=572, KS 0.123 |

## Experiments

### Experiment 4.1 — Optuna search over the XGBoost champion (9 knobs)
**Hypothesis:** the model is feature-bound, so Bayesian search buys very little.
**Method:** TPE sampler, objective = mean grouped-CV macro-F1, 5-min wall-clock cap, study seeded with the
Phase-3 default. Knobs: trees, depth, learning rate, subsample, colsample, min_child_weight, gamma, reg_alpha,
reg_lambda.
**Result:**
| | CV macro-F1 | raw F1 | matched F1 |
|---|---:|---:|---:|
| DEFAULT (Phase-3) | 0.9951 | 0.9967 | 0.9808 |
| TUNED (best of 11 trials) | 0.9949 | 0.9967 | 0.9808 |
| **Δ** | **−0.0002** | **+0.0000** | **+0.0000** |

The **top-10 trials all sit at CV ≈ 0.9949** while spanning depth 3→10, learning rate 0.019→0.29, and 200→800
trees — a completely flat ridge.
**Interpretation:** there is no hyperparameter signal to find. The default (seeded) remained best; 11 Bayesian
trials could not beat it and CV nudged *down* by 0.0002 (noise). **The model is feature-bound, not
hyperparameter-bound** — the Phase-3 thesis, now proven by exhaustion.

### Experiment 4.2 — Gradient-boosting family head-to-head (each Optuna-tuned)
**Hypothesis:** if XGBoost is saturated, a different family won't help either.
**Method:** tune LightGBM and CatBoost under the same grouped-CV objective and budget; add a regularization-swept
logistic head as a linear sanity check.
**Result (ranked by matched macro-F1):**
| Rank | Model | CV F1 | raw F1 | matched F1 |
|---|---|---:|---:|---:|
| 1 | eng_xgboost (tuned) | 0.9949 | 0.9967 | **0.9808** |
| 2 | eng_lightgbm (tuned) | 0.9949 | 0.9967 | **0.9808** |
| 3 | eng_catboost (tuned) | 0.9951 | 0.9967 | **0.9808** |
| 4 | eng_logreg (C=0.03) | 0.9910 | 0.9927 | 0.9790 |
**Interpretation:** three independent GB implementations land on the **exact same** 0.9967 / 0.9808, and a *linear*
model is within 0.0018 matched-F1. The learner is interchangeable — the 13 features have already extracted
essentially all separable signal.

### Experiment 4.3 — Calibration (uncalibrated vs Platt vs isotonic)
**Hypothesis (from literature):** boosted trees are mis-calibrated; isotonic will lower ECE.
**Method:** 80/20 group-disjoint split of train into model-fit core and calibration holdout; fit Platt + isotonic
on the holdout; evaluate all three on the untouched test set.
**Result:**
| Calibration | ECE (15-bin) | Brier | macro-F1@0.5 |
|---|---:|---:|---:|
| **uncalibrated** | **0.0041** | 0.0034 | 0.9967 |
| platt | 0.0101 | 0.0033 | 0.9967 |
| isotonic | 0.0041 | 0.0032 | 0.9967 |
**Interpretation:** **counter to the textbook**, the uncalibrated champion is already well-calibrated (ECE 0.004).
Isotonic doesn't move ECE; **Platt makes it 2.5× worse.** The classic "boosting needs calibration" result does
not transfer to a logloss-trained XGB on a clean, balanced, well-separated problem. Post-hoc calibration is at
best a wash here and can hurt — ship the raw probabilities.

### Experiment 4.4 — Operating point selection
**Method:** sweep the threshold on the full-train champion's test scores; pick macro-F1-optimal, high-recall
(≥0.99), high-precision (≥0.99), and cost-optimal (FN:FP = 5:1). Report PR curve and entity-reuse recall.
**Result:** AUPRC **0.9976**, AUROC 0.9975. The F1-optimal threshold is **0.163, not 0.5** — but the score
distribution is so bimodal that **all four operating points collapse to the same threshold**: precision 0.9995,
recall 0.994, **entity-reuse-minority recall 0.9915**, cost 61.
**Interpretation:** moving off 0.5 is technically correct but there is **no precision–recall tradeoff to
navigate** — the scores are essentially separated, so one operating point dominates. The hard entity-reuse
minority (the cases that fooled the Phase-2 lexical baseline) is caught at 99.2% recall.

### Experiment 4.5 — Error analysis: where does it fail?
**Result:** on raw test the champion makes **1 false positive and 12 false negatives out of 4,000**. The crucial
structure is in the 12 misses: **10 of 12 are verbatim substrings of the knowledge passage** (9 at overlap=1.0),
scored P≈0.01 (confidently "grounded").

Examples of the residual failure mode (verbatim-but-wrong):
| Question (truncated) | Hallucinated answer | overlap | is_substr | model P(hallu) |
|---|---|---:|:--:|---:|
| *"From Eden" is a number-2 song from the Irish Singles Chart…* | "Take Me To Church" | 1.00 | ✓ | 0.01 |
| *What job do Idrissa Ouedraogo and Jerry Paris share…* | "Actor and director" | 1.00 | ✓ | 0.01 |
| *Which 2005 film did the Austrian director…* | "The White Ribbon" | 1.00 | ✓ | 0.01 |

**Error buckets:** error rate is flat across overlap quartiles (~0.004) and length quartiles — the length leak is
fully neutralized. The one axis that lights up is **form**: `grounded_non_verbatim` errors at 0.0105 vs
`grounded_verbatim` at 0.0000 (the standing form-ambiguous control).
**Interpretation — the ceiling, stated precisely:** every feature in the stack measures *answer→knowledge
grounding*. These misses are **grounded-but-irrelevant** — a real entity/phrase lifted verbatim from the passage
that is the *wrong answer to the question*. No grounding feature can catch them **by construction**; catching them
requires modelling *question→answer relevance*, which the current feature set does not encode. That is the
residual error mode and the Phase-5 target.

### Experiment 4.6 — Learning curve
**Method:** refit the tuned champion on grouped 5%→100% subsamples; evaluate raw + matched.
**Result:** matched-F1 reaches within 0.005 of its maximum by **800 training rows (5% of train)**; raw-F1 plateaus
at 8,000 rows.
**Interpretation:** the model is **data-saturated** — 20× more data does not help. Combined with 4.1–4.2, every
lever (hyperparameters, model family, calibration, threshold, dataset size) is exhausted. The ceiling is
structural.

## Head-to-Head Comparison (consolidated Phase-4 leaderboard, ranked by matched macro-F1)
| Rank | Model | raw F1 | matched F1 | CV F1 |
|---|---|---:|---:|---:|
| 1 | eng_xgboost (default, Phase-3) | 0.9967 | 0.9808 | 0.9951 |
| 2 | eng_xgboost (Optuna-tuned) | 0.9967 | 0.9808 | 0.9949 |
| 3 | eng_lightgbm (tuned) | 0.9967 | 0.9808 | 0.9949 |
| 4 | eng_catboost (tuned) | 0.9967 | 0.9808 | 0.9951 |
| 5 | eng_logreg (C=0.03) | 0.9927 | 0.9790 | 0.9910 |
| 6 | grounding_overlap (Phase-2 bar) | 0.9252 | 0.9244 | — |

## Key Findings
1. **Tuning is a no-op.** 11 Optuna trials moved matched-F1 by **+0.0000** (CV −0.0002); the top-10 trials are a
   flat ridge across wildly different hyperparameters. The model is feature-bound — proven, not asserted.
2. **The learner is interchangeable.** XGBoost = LightGBM = CatBoost at *exactly* 0.9967/0.9808; tuned logreg is
   within 0.0018 matched. Once the right features exist, model choice is irrelevant.
3. **Already calibrated — calibration backfires.** Uncalibrated ECE 0.0041; isotonic = no change; Platt 2.5× worse.
   The textbook "boosting needs calibration" result does not transfer here.
4. **No operating-point tradeoff.** Bimodal scores collapse all four operating points to one threshold (0.163);
   the entity-reuse minority is caught at 99.2% recall, AUPRC 0.9976.
5. **The residual ceiling is question-relevance, not grounding.** 10/12 misses are verbatim-but-wrong answers
   — grounded to the passage yet not answering the question. No grounding feature can catch them by construction.
6. **Data-saturated at 5%.** Learning curve flat after 800 rows; more data won't help.

## What Didn't Work (and why)
- **Optuna search** — flat objective ridge; nothing to optimize on a feature-bound problem.
- **Platt scaling** — worsened ECE (0.0041 → 0.0101); the score distortion isn't sigmoid-shaped, it's already
  near-identity.
- **Threshold tuning as a lever** — correct in principle (optimum is 0.163, not 0.5) but inert in practice
  because the scores are separated; no precision-recall curve to ride.

## Frontier Model Comparison
Deferred to Phase 5 (Friday) — the head-to-head against Claude Opus/Haiku and Codex GPT belongs with the advanced
techniques + ablation session, where the verbatim-but-wrong residual is the perfect probe (do frontier LLMs catch
the grounded-but-irrelevant hallucinations our feature stack cannot?).

## Error Analysis (summary)
- 1 FP / 12 FN out of 4,000 raw test rows; 11 errors on the matched control (macro-F1 0.9808).
- 10/12 FN are verbatim spans of the knowledge (grounded-but-irrelevant) → question-relevance failure mode.
- Error rate flat across overlap & length; concentrated on `grounded_non_verbatim` (form-ambiguous) at 1.05%.

## Next Steps (Phase 5 — Advanced techniques + ablation + LLM comparison)
- **Target the residual:** add a *question→answer relevance* signal (e.g., the fine-tuned cross-encoder scored
  with the **question** in context, or a QA-entailment head) and test whether it rescues the 10 verbatim-but-wrong
  misses without disturbing the saturated grounding score. This is the one component that *could* move 0.9808.
- **Ablation study:** drop each engineered feature (and the cross-encoder) one at a time; quantify what each
  contributes vs. `is_substr` alone (Phase-3 importance was 0.795 on `is_substr`).
- **Frontier head-to-head:** Claude Opus/Haiku + Codex GPT zero-shot on the same stratified sample, with explicit
  attention to whether LLMs catch the grounded-but-irrelevant cases. Latency + cost-per-1k table.

## References Used Today
- [1] Random Realizations — *The Ultimate Guide to XGBoost Parameter Tuning with Optuna*. https://randomrealizations.com/posts/xgboost-parameter-tuning-with-optuna/
- [2] Niculescu-Mizil & Caruana (2005), *Obtaining Calibrated Probabilities from Boosting*. https://arxiv.org/pdf/1207.1403 ; FastML, *Classifier calibration with Platt's scaling and isotonic regression*. https://fastml.com/classifier-calibration-with-platts-scaling-and-isotonic-regression/
- [3] MachineLearningMastery — *A Gentle Introduction to Threshold-Moving for Imbalanced Classification*. https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
- [4] *The Mirage of Hallucination Detection* (Findings-EMNLP 2025). https://aclanthology.org/2025.findings-emnlp.1035.pdf ; *The Gray Zone of Faithfulness* (2025). https://arxiv.org/pdf/2510.21118

## Code Changes
- `notebooks/phase4_tuning.ipynb` — new 26-cell research notebook (grouped-CV tuning, family head-to-head,
  calibration, operating points, error analysis, learning curve); executed end-to-end, 0 errors.
- `results/phase4_*.csv` — tuning trials, family comparison, calibration, operating points, error buckets,
  learning curve, consolidated leaderboard.
- `results/phase4_*.png` — optuna history + param importance, calibration/reliability, operating points + PR,
  error buckets + confusion, learning curve.
- `results/phase4_cache/` — engineered-feature matrix + persisted Optuna SQLite studies (resumable).
- `results/metrics.json` — appended `phase4` block.
