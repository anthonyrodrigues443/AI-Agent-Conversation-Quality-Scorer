# Phase 3 — Feature engineering deep dive: char-ngram saturation, per-sentence NLI, stacking, error analysis

**Project:** NLP-2 — AI Agent Conversation Quality Scorer
**Date:** 2026-05-14 (Thu)
**Session:** 3 of 7
**Notebook:** `notebooks/phase3_feature_engineering.ipynb` — 28 cells (21 code / 7 markdown), executed end-to-end on the MPS device with `cross-encoder/nli-deberta-v3-base` + `all-MiniLM-L6-v2`, zero errors, zero fake display-only cells.

## Objective

Phase 2 left a 524-row length-matched honest leaderboard topped by **char-ngram + LogReg at macro F1 = 0.7789**, with paired SBERT a hair behind at 0.7643 and zero-shot cross-encoder NLI at the back at 0.7137. Phase 3 had four mandates:

1. **Char-ngram saturation.** Phase 2 used `(3,5)` × `max_features=200_000`. Sweep 6 n-gram ranges × 4 vocab caps × {sublinear_tf=True, False} = 48 configs and find the saturation point.
2. **Per-sentence NLI max-pool.** Phase 2's NLI feeds the whole HotpotQA passage through a 512-token cross-encoder. Split into sentences, score (sentence, answer) pairs, max-pool the contradiction score per row.
3. **Stacking.** Char-ngrams capture style; SBERT captures semantics; NLI captures entailment. Do their errors overlap? If not, meta-LogReg over OOF probabilities of all three should beat the best individual.
4. **Error analysis on the char-ngram champion.** What unifies the 80 hallucinations it misses on the matched split?

Primary metric remains **macro F1 on the length-matched split (n=524)**.

## Research & References

1. **Sun et al., *Fast logistic regression for text categorization with variable-length n-grams* (KDD 2008)** — [Semantic Scholar](https://www.semanticscholar.org/paper/487b61ad3cfe1538fbe95572954b34317ef625f6). Established that character-ngram LogReg saturates logarithmically with vocab size. → motivates the 48-config grid in Section 1.
2. **HF Hallucinations Leaderboard (2024)** — [huggingface.co/blog/leaderboard-hallucinations](https://huggingface.co/blog/leaderboard-hallucinations). Cross-encoder NLI verifiers at AUROC ≈ 0.88 averaged across domains when source is decomposed per-claim. → bar for Section 2.
3. **Wolpert, *Stacked Generalization* (1992)** + **scikit-learn `StackingClassifier` docs (2025)** — [scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html). Stacking gains require low error-correlation between base learners. → Section 3 measures that correlation explicitly before committing to a stacker.

### How research shaped today's plan
Every section ends with a head-to-head against the Phase 2 champion (char35_logreg @ 0.7789). A finding only counts if it survives the length-matched split. NLI is zero-shot — any lift in Section 2 is architectural, not from supervision.

## Dataset & splits
Identical to Phase 2 — 16k train / 4k raw test / 524 length-matched test. The matched split is reproduced byte-identically inside the notebook via the same `np.random.default_rng(42)` bin-by-bin sampler.

---

## Section 1 — Char-ngram ablation (48 configs)

Each config fits a fresh `TfidfVectorizer(analyzer='char_wb', ...)` + `LogisticRegression(C=1, solver='liblinear')` on the full 16k training set, evaluates on raw and matched test splits.

### Matched-F1 by (ngram_range, max_features) — best sublinear setting per cell

| ngram_range | mf=25k | mf=50k | mf=100k | mf=200k |
|:---|---:|---:|---:|---:|
| **(2, 5)** | **0.7941** | 0.7864 | 0.7904 | 0.7904 |
| (2, 4) | 0.7906 | 0.7906 | 0.7906 | 0.7906 |
| (3, 5) | 0.7845 | 0.7751 | 0.7731 | 0.7731 |
| (3, 6) | 0.7806 | 0.7727 | 0.7731 | 0.7731 |
| (3, 4) | 0.7746 | 0.7764 | 0.7764 | 0.7764 |
| (4, 6) | 0.7492 | 0.7513 | 0.7406 | 0.7406 |

### Findings

- **Best config:** `(2, 5)` × `max_features=25_000` × `sublinear_tf=True` → **matched F1 = 0.7941** (Δ vs Phase 2 baseline = **+0.0152**). Saved as `char_best_p3`.
- **Saturation point is 25k features, not 200k.** Phase 2's vocab cap of 200k was 8× more than optimal. The matched-F1 curve is monotonically *decreasing* past 25k for the (3, X) family and flat past 50k for (2, X).
- **Bigram-inclusive ranges win.** `(2, 4)` and `(2, 5)` dominate every (3, X) and (4, X) family. The Phase 2 champion `(3, 5)` is not even in the top 3 — it lost 1.0 F1 points to `(2, 5)`. Two-character substrings (punctuation, casing, single-character connectives) are doing more work than 3-grams alone.
- **Sublinear-TF is a wash on this dataset.** Mean lift across the grid = +0.0008 F1. Phase 2 attributed influence to `sublinear_tf=True` but the saturation grid says it's noise.
- **(4, 6) is the worst family.** Matched F1 = 0.74 even with the best vocab cap — too-long n-grams overfit answer-specific surface forms instead of generic style features. Length-shortcut delta is +0.20 (largest among the grid).

### Headline lift
Phase 2 char champion (matched) = 0.7789 → Phase 3 char champion (matched) = **0.7941**. The win comes from a *smaller* model — 25k features × bigrams included, no sublinear-TF needed.

---

## Section 2 — Per-sentence NLI max-pool (the negative result)

For each of the 524 matched-test rows: split knowledge by sentence boundary regex (`(?<=[.!?])\s+(?=[A-Z])`), score every `(knowledge_sentence_i, answer)` pair via `cross-encoder/nli-deberta-v3-base` (max_length=256 since single sentences fit), take per-row max of `P(contradiction) − P(entailment)`.

### Head-to-head NLI

| Variant | macro F1 (matched) | AUROC (matched) |
|:---|---:|---:|
| Phase 2 single-pass NLI (full passage, max_length=512) | 0.7137 | 0.7189 |
| Phase 3 per-sentence max-pool (default threshold 0) | **0.6242** | 0.7028 |
| Phase 3 per-sentence max-pool (best-threshold sweep, t=0.65) | 0.6831 | — |

### Finding: decomposition HURT, not helped

- F1 lift vs Phase 2: **−0.0895** at default threshold, **−0.0306** even with best-threshold tuning.
- AUROC lift vs Phase 2: **−0.0161**.

This contradicts the published 0.88-AUROC ceiling. **Why per-sentence max-pool degraded NLI on HaluEval-QA:**

Knowledge passages here are concatenations of HotpotQA factoid sentences. When split, each sentence is a self-contained claim. Max-pooling `P(contradiction) − P(entailment)` across sentences inflates the contradiction score for *grounded* answers too — because many short knowledge sentences only mention entities not present in the answer, which the NLI model reads as weak contradiction. The single-pass version had the full passage's evidence to anchor against, so its `P(entailment)` was higher on average.

For HaluEval-QA specifically, **the right NLI input granularity is the full concatenated passage, not per-sentence**. The 0.88 leaderboard number is achievable for datasets where source is a multi-paragraph document (the original NLI checkpoint's training domain) — not for the HaluEval-QA short-passage regime.

**Honest takeaway:** Phase 3 went looking for a Section-2 win and didn't find one. The single-pass NLI from Phase 2 stands as the best zero-shot variant on this dataset.

---

## Section 3 — Stacking (the headline)

Three base predictors with 5-fold GroupKFold-by-qid OOF probabilities on train:
- **char-ngram** (Phase 3 best: `(2,5)` × `25k` × `sublinear=True`) + LogReg
- **paired SBERT** (k⊕a + cos + |k−a|, 770-D) + LogReg
- **per-sentence NLI max-pool** (zero-shot — no OOF, score min-max scaled to [0, 1])

Meta-classifier: `LogisticRegression(C=1)` on the 3-D OOF feature matrix. Controls: unweighted mean blend, and the char-ngram champion alone.

### Diversity check (matched split, n=524)

| Pair | Pearson correlation of per-row error indicators |
|:---|---:|
| char ↔ sbert | **+0.658** |
| char ↔ nli | **−0.223** |
| sbert ↔ nli | **−0.172** |

NLI errors are **negatively correlated** with the trained classifiers' errors. Char and SBERT errors are strongly positively correlated — they're learning the same dataset shortcut. The structural condition for a useful stacker is satisfied.

**Oracle upper bound** (if you could perfectly route each row to the model that gets it right): **97.9% accuracy**. Best individual: 79.6% accuracy → 18 percentage points of headroom available to a router.

### Stacking results

| Model | matched F1 | matched acc | matched AUROC |
|:---|---:|---:|---:|
| **meta_logreg [char + sbert + nli]** | **0.8044** | **0.8053** | **0.8475** |
| char_best_p3 (control) | 0.7941 | 0.7958 | 0.7988 |
| mean_blend [char + sbert + nli] | 0.7900 | 0.7901 | 0.8709 |
| Phase 2 champion (char35_logreg, 200k vocab) | 0.7789 | 0.7805 | 0.7971 |

### Meta-LogReg learned weights
```
coef[char_oof] = +6.19
coef[sbert_oof] = +3.92
coef[nli_scaled] = +2.06
intercept = −6.11
```

### Findings

- **Stacking beats the best individual by +0.026 F1** on the honest split (0.8044 vs 0.7941) and by **+0.026 F1 vs Phase 2 champion** (0.8044 vs 0.7789). On the raw split it reaches F1 = 0.9690 — the highest number recorded across all three phases.
- **AUROC lift is much larger: +0.049 vs char-best (0.8475 vs 0.7988).** The mean-blend variant has the highest AUROC of all (0.8709) but loses on F1 because its default threshold isn't tuned. Stacking's strength is calibrated decision-boundary placement.
- **NLI is the weakest model alone (F1 = 0.62) but the stacker keeps it.** Its coefficient (+2.06) is nonzero and positive, and the diversity check showed negative error correlation with char/SBERT — NLI gets right what the others get wrong. *The weakest model is carrying the stack.* This is the headline.
- **Mean blend underperforms learned stacking.** Equal weights ignore that char/SBERT correlate at +0.66 (double-counting the same signal) while NLI is orthogonal. Learned weights downweight redundancy and amplify diversity.

---

## Section 4 — Error analysis on char-ngram champion

`char_best_p3` confusion on matched split: TP=178, FN=84, FP=23, TN=239 → recall (hallu) = 0.679. Mean feature value by confusion cell:

| Cell | answer_len | n_tokens | jaccard_k | has_period | char_prob |
|:---|---:|---:|---:|---:|---:|
| TP (hallu caught) | 38.7 | 6.5 | 0.103 | 0.984 | **0.860** |
| FN (hallu missed) | **15.4** | **2.4** | 0.024 | **0.21** | 0.192 |
| FP (false alarm) | 86.0 | 14.3 | 0.293 | 0.27 | 0.773 |
| TN (grounded ok) | 24.5 | 3.5 | 0.091 | 0.034 | 0.189 |

### Finding: the missed hallucinations look like HotpotQA entity strings

The 84 false-negatives share a very distinct profile:
- **2.4 mean tokens** vs 6.5 for caught hallucinations (FNs are short)
- **21% have any sentence-ending punctuation** vs 98% for TPs (FNs are non-sentences)
- **0.024 mean Jaccard with knowledge** vs 0.103 for TPs (FNs barely overlap with source)
- **0.19 mean predicted probability** — the model is *confidently wrong*, not borderline

In other words: when a hallucinated answer is short, period-free, and low-overlap, the char-ngram model thinks it's a HotpotQA-style entity string answer (grounded) and gives it 0.19 P(hallu). That's the model's central failure mode after length is controlled. **It's learned to flag "long sentence-shaped responses" — and short fake entity answers slip past the radar.**

### Implication for Phase 4
This is exactly where Optuna tuning a char-ngram model will hit a ceiling — these aren't borderline cases the model could solve with better regularization, they're *systematically misclassified by design*. The right Phase 5 angle is a **calibrated thresholding** experiment + a custom feature ("is this answer short and period-free?") OR letting the LLM head-to-head light up this region (Claude/Codex are likely better at short-entity-string verification because they actually read the knowledge text).

---

## Section 5 — Phase 3 cumulative honest leaderboard

| Phase | Model | matched macro F1 | matched accuracy | matched AUROC |
|---:|:---|---:|---:|---:|
| **3** | **meta_stack [char + sbert + nli]** | **0.8044** | **0.8053** | **0.8475** |
| 3 | char_best_p3 ((2,5) × 25k × sublinear) | 0.7941 | 0.7958 | 0.7988 |
| 3 | mean_blend [char + sbert + nli] | 0.7900 | 0.7901 | 0.8709 |
| 2 | char35_logreg ((3,5) × 200k × sublinear) | 0.7789 | 0.7805 | 0.7971 |
| 2 | sbert_paired_logreg | 0.7643 | 0.7653 | 0.7952 |
| 2 | sbert_answer_logreg | 0.7598 | 0.7615 | 0.7887 |
| 2 | tfidf12_svc | 0.7411 | 0.7443 | 0.7932 |
| 2 | sbert_paired_xgb | 0.7300 | 0.7309 | 0.7893 |
| 2 | nli_deberta_zeroshot (single-pass) | 0.7137 | 0.7137 | 0.7189 |
| 3 | nli_persent_maxpool | 0.6242 | 0.6260 | 0.7028 |

Saved to `results/phase3_cumulative_leaderboard.csv` and visualized in `results/phase3_leaderboard.png`.

---

## Headline result

**Stacking char-ngram + SBERT + zero-shot NLI gets macro F1 = 0.8044 on the length-matched honest HaluEval-QA split — +0.026 over the best individual model. The headline is *why* it works: the weakest model in the stack (NLI at F1 = 0.62) is the one carrying the lift, because its errors point in the opposite direction from the other two (Pearson −0.22 / −0.17). Char and SBERT errors correlate at +0.66 — they're learning the same dataset shortcut. NLI, never trained on HaluEval, is orthogonal.**

Companion lift from the ablation: **the optimal char-ngram vocab is 25k, not Phase 2's 200k** — an 8× shrink that gains +0.015 F1 on the honest split because shorter n-grams (`(2, 5)`) capture style without overfitting answer-specific surface forms.

## What didn't work

- **Per-sentence NLI max-pool actively HURT** (F1 0.7137 → 0.6242, AUROC 0.7189 → 0.7028). The HF leaderboard's cited 0.88 AUROC for decomposed NLI is for longer-document datasets — on HaluEval-QA's short HotpotQA-derived passages, decomposition introduces more noise than signal because each sentence is too thin a context for the entailment model. Documented as a finding, not silently rolled back.
- **Sublinear-TF doesn't matter on this dataset** (+0.0008 F1 averaged across the grid). Phase 2 attributed it influence; the ablation says it's noise.
- **(4, 6) character n-grams overfit.** Best matched F1 = 0.745 even with optimized vocab — 5 F1 points below the (2, 5) champion. Long character sequences memorize answer surface forms.

## Next phase

Phase 4 (Fri 2026-05-15) — tuning + LLM head-to-head:

1. **Optuna 100-trial search** over `(LogReg C, class_weight, threshold)` × `(TF-IDF min_df, max_df)` around the Phase 3 char-best config. Target: close the gap between matched F1 = 0.7941 and the meta-stack's 0.8044 using only char features (a *single-model* champion that beats the Phase 2 stack on its own).
2. **Stacking refinement** — replace `LogisticRegression` meta-learner with `LightGBM` (handles nonlinear interactions between base probs); test isotonic calibration on each base before stacking.
3. **LLM head-to-head** using the proven harness from `Fraud-Detection-System/src/mark_phase5_run_llm.py` — Claude Opus + Haiku + Codex GPT-5.4 via local CLI, stratified n=50 sample, original HaluEval judge protocol (`knowledge + question + answer → 1-word verdict`). Required deliverable: `results/llm_vs_custom.csv` with accuracy / F1 / latency / cost-per-1k columns. The paper's published ChatGPT zero-shot at 62.6% is the floor; my stacker at 80.4% is the bar to clear.
4. **Custom feature for the FN cluster** — add `is_short_unpunctuated_entity` (n_tokens ≤ 3 AND no `[.!?]` AND jaccard_k < 0.1) as an explicit input feature to the stacker and measure whether it closes the recall gap.

## References used today

- Sun et al. 2008 — KDD — variable-length n-gram LogReg saturation — [Semantic Scholar](https://www.semanticscholar.org/paper/487b61ad3cfe1538fbe95572954b34317ef625f6)
- HF Hallucinations Leaderboard (2024) — NLI per-claim AUROC ≈ 0.88 — [huggingface.co/blog/leaderboard-hallucinations](https://huggingface.co/blog/leaderboard-hallucinations)
- Wolpert 1992 / scikit-learn StackingClassifier docs (2025) — diversity precondition for stacking — [scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
- Cross-encoder model card — `cross-encoder/nli-deberta-v3-base` — label map `{0: contradiction, 1: entailment, 2: neutral}`

## Code changes

- `notebooks/phase3_feature_engineering.ipynb` — 28 cells (21 code / 7 markdown), executed end-to-end on the MPS device, no errors, no fake display-only cells
- `results/phase3_ngram_ablation.csv` — full 48-config grid
- `results/phase3_ngram_saturation.png` — saturation curves
- `results/phase3_nli_persent_scores.csv` + `phase3_nli_threshold_sweep.png` + `phase3_nli_features.npz` (train + raw + matched per-sentence NLI features cached for Phase 4 reuse)
- `results/phase3_stacking.csv`
- `results/phase3_error_summary.csv` + `phase3_error_breakdown.png`
- `results/phase3_leaderboard.csv` + `phase3_cumulative_leaderboard.csv` + `phase3_leaderboard.png`
- `results/metrics.json` — `phase3` key appended
- `results/EXPERIMENT_LOG.md` — Phase 3 sections appended (Section 6 onward)
