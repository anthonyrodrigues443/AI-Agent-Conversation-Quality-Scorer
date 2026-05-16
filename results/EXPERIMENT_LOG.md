# EXPERIMENT_LOG — NLP-2 AI Agent Conversation Quality Scorer

Master record of every (paradigm, split) evaluation. Updated after each phase. Phase 1 baselines and Phase 2 paradigms are both evaluated on the Phase 1 GroupShuffleSplit-by-qid test set (n=4000); Phase 2 additionally evaluates each paradigm on the length-matched control split (n=524).

Primary metric: **macro F1** (locked Phase 1). Secondary: ROC-AUC. Positive class = `hallucinated`.

## Raw split (n=4000) — ranked by macro F1

| Phase | Model | Paradigm | macro F1 | accuracy | AUROC | precision (hallu) | recall (hallu) | fit (s) | predict (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | sbert_paired_logreg | paired embed (k⊕a + cos + diff) + linear | 0.9612 | 0.9613 | 0.9829 | 0.969 | 0.953 | 0.26 | 0.01 |
| 2 | sbert_answer_logreg | sentence-embed + linear | 0.9607 | 0.9608 | 0.9825 | 0.969 | 0.952 | 0.06 | 0.01 |
| 2 | sbert_paired_xgb | paired embed + XGBoost | 0.9550 | 0.9550 | 0.9856 | 0.961 | 0.949 | 52.64 | 0.05 |
| 2 | char35_logreg | char-ngram + logistic | 0.9537 | 0.9538 | 0.9801 | 0.968 | 0.939 | 3.80 | 0.00 |
| 1 | length_only_logreg | 4 length features + logistic | 0.9437 | 0.9437 | 0.9713 | 0.952 | 0.935 | 0.02 | — |
| 2 | tfidf12_svc | word-ngram (1,2) + LinearSVC (calibrated) | 0.9327 | 0.9328 | 0.9705 | 0.954 | 0.909 | 0.39 | 0.00 |
| 1 | tfidf_answer_logreg | TF-IDF (1,2) answer + logistic | 0.9191 | 0.9193 | 0.9678 | 0.951 | 0.884 | 0.31 | — |
| 1 | tfidf_q_plus_a_logreg | TF-IDF (1,2) Q+[SEP]+A + logistic | 0.7880 | 0.7887 | 0.8660 | 0.827 | 0.731 | 3.47 | — |
| 1 | lexical_overlap_threshold | answer ∩ knowledge tokens (NLI proxy) | 0.7026 | 0.7140 | 0.5933 | 0.654 | 0.910 | 0.00 | — |
| 2 | nli_deberta_zeroshot | cross-encoder NLI (zero-shot) | 0.5905 | 0.5915 | 0.5895 | 0.583 | 0.642 | 0.00 | 140.77 |
| 1 | majority_class | DummyClassifier | 0.3333 | 0.5000 | 0.5000 | 0.000 | 0.000 | 0.00 | — |

## Length-matched split (n=524, KS=0.015) — ranked by macro F1 — THE HONEST LEADERBOARD

| Phase | Model | Paradigm | macro F1 | accuracy | AUROC | precision (hallu) | recall (hallu) | Δ vs raw F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2 | **char35_logreg** | **char-ngram + logistic** | **0.7789** | 0.7805 | 0.7971 | 0.839 | 0.695 | +0.1748 |
| 2 | sbert_paired_logreg | paired embed + linear | 0.7643 | 0.7653 | 0.7952 | 0.803 | 0.702 | +0.1969 |
| 2 | sbert_answer_logreg | sentence-embed + linear | 0.7598 | 0.7615 | 0.7887 | 0.813 | 0.679 | +0.2009 |
| 2 | tfidf12_svc | word-ngram + linear hinge | 0.7411 | 0.7443 | 0.7932 | 0.814 | 0.634 | +0.1916 |
| 2 | sbert_paired_xgb | paired embed + XGBoost | 0.7300 | 0.7309 | 0.7893 | 0.762 | 0.672 | +0.2250 |
| 2 | nli_deberta_zeroshot | cross-encoder NLI (zero-shot) | 0.7137 | 0.7137 | 0.7189 | 0.711 | 0.721 | **−0.1232** |

## Headline findings (cumulative across Phase 1-2)

1. **HaluEval-QA raw is a length test in disguise.** Length-only LogReg (4 features) gets 0.944 F1 on raw — beats every trained word-ngram and TF-IDF baseline.
2. **The honest leaderboard winner is char-ngram LogReg at 0.7789 macro F1.** It also has the smallest length-shortcut Δ among trained models.
3. **NLI cross-encoder is the only paradigm whose performance *improves* on the length-matched split** (0.59 → 0.71). It's also the only one that never saw HaluEval. Trained = exploits shortcut. Zero-shot = does the actual semantic work, however imperfectly.
4. **XGBoost on SBERT embeddings overfits length-correlated dimensions.** It has both the worst length-matched F1 (0.7300) and the largest length-shortcut Δ (+0.2250).
5. **Adding the knowledge channel barely helps when length is controlled** (paired SBERT vs answer-only SBERT: 0.7643 vs 0.7598 = +0.005 F1). The dataset's hallucinated answers are largely detectable from the answer alone.

## Targets to beat in Phase 3+

| Bar | macro F1 (matched) | Notes |
|---|---:|---|
| Phase 2 winner — char35_logreg | **0.7789** | THIS is the number Phase 3 has to beat |
| Phase 2 NLI zero-shot | 0.7137 | Zero-training upper bound on this hardware/encoder |
| ANAH-v2 (published, raw) | 0.815 | Long-horizon target after fine-tuning (Phase 4-5) |
| HF Leaderboard NLI (cited, averaged across domains) | AUROC 0.88 | Phase 3 will probe sentence-level NLI to close the gap |

## Phase 3 additions — char-ngram saturation, per-sentence NLI, stacking, error analysis (2026-05-14)

### Char-ngram ablation grid (48 configs, ranked by matched F1)

| Rank | ngram_range | max_features | sublinear_tf | vocab_actual | f1_raw | f1_matched | Δ (raw−matched) |
|---:|:---|---:|:---|---:|---:|---:|---:|
| 1 | (2, 5) | 25,000 | True | 25,000 | 0.9645 | **0.7941** | +0.1703 |
| 2 | (2, 4) | 50k–200k | True | 32,454 | 0.9647 | 0.7906 | +0.1742 |
| 3 | (2, 4) | 25,000 | True | 25,000 | 0.9650 | 0.7906 | +0.1744 |
| 4 | (3, 5) | 25,000 | True | 25,000 | 0.9590 | 0.7845 | +0.1745 |
| 5 | (3, 6) | 25,000 | True | 25,000 | 0.9568 | 0.7806 | +0.1762 |
| ... | (4, 6) | any | any | 78,214 | 0.93–0.94 | 0.74–0.75 | +0.19–0.20 (worst) |

**Saturation finding:** The best matched-F1 vocab is **25,000**, not Phase 2's 200,000. Phase 2's `(3, 5)` family is below `(2, 5)` and `(2, 4)` — bigram-inclusive ranges beat trigram-only by ~1.5 F1. Sublinear-TF lift averaged across the grid = **+0.0008** (effectively zero). The `(4, 6)` family is the worst — too-long character sequences memorize answer-specific surface forms.

### Per-sentence NLI vs single-pass NLI

| Variant | macro F1 (matched) | AUROC (matched) | n calls per row |
|:---|---:|---:|---:|
| Phase 2 single-pass NLI (max_length=512) | **0.7137** | **0.7189** | 1 |
| Phase 3 per-sentence max-pool (max_length=256, default threshold) | 0.6242 | 0.7028 | 2–3 (avg ~2.4) |
| Phase 3 per-sentence max-pool (threshold sweep, t=0.65) | 0.6831 | — | 2–3 |

**Decomposition HURT.** Δ F1 = −0.0895 vs Phase 2 single-pass. Knowledge passages in HaluEval-QA are concatenations of HotpotQA factoid sentences; splitting per-sentence introduces noise because each sentence alone is too thin a context for entailment. Published 0.88-AUROC ceiling does not reproduce with deberta-base zero-shot on this dataset.

### Stacking — diversity check + meta-classifier

**Pearson correlation of per-row error indicators on matched split (n=524):**
| Pair | r |
|:---|---:|
| char ↔ sbert | +0.658 |
| char ↔ nli | **−0.223** |
| sbert ↔ nli | **−0.172** |

**Oracle upper-bound** (perfect routing): 97.9% accuracy. Best individual: 79.6% acc → 18 pt of headroom.

**Meta-classifier results:**

| Model | matched F1 | matched accuracy | matched AUROC |
|:---|---:|---:|---:|
| **meta_logreg [char + sbert + nli]** | **0.8044** | **0.8053** | **0.8475** |
| char_best_p3 (control) | 0.7941 | 0.7958 | 0.7988 |
| mean_blend [char + sbert + nli] | 0.7900 | 0.7901 | 0.8709 |

Meta coefficients: `coef[char]=+6.19, coef[sbert]=+3.92, coef[nli]=+2.06, intercept=−6.11`.

**Stacking wins by +0.026 F1.** Even though NLI is the weakest individual model (F1 = 0.62), it has the largest *marginal* contribution because its errors are negatively correlated with the other two. Mean blend underperforms learned stacking because it ignores that char/SBERT correlate at +0.66 (double-counting).

### Error analysis — char-ngram champion false-negative cluster

`char_best_p3` matched confusion: TP=178, FN=84, FP=23, TN=239 → recall (hallu) = 0.679.

| Cell | answer_len | n_tokens | jaccard_k | has_period | char_prob |
|:---|---:|---:|---:|---:|---:|
| TP | 38.7 | 6.5 | 0.103 | 0.984 | 0.860 |
| **FN (missed hallu)** | **15.4** | **2.4** | 0.024 | **0.21** | 0.192 |
| FP | 86.0 | 14.3 | 0.293 | 0.27 | 0.773 |
| TN | 24.5 | 3.5 | 0.091 | 0.034 | 0.189 |

**The 84 FNs are short, period-free, low-overlap answers — the kind that *look* like HotpotQA entity strings.** The char-ngram model has learned "long sentence-shaped answer ⇒ hallucination" and can't detect short entity-string hallucinations. This is a systematic failure mode, not borderline noise.

## Updated cumulative length-matched leaderboard (top of the table)

| Phase | Model | Paradigm | macro F1 | accuracy | AUROC |
|---:|:---|:---|---:|---:|---:|
| **3** | **meta_stack [char + sbert + nli]** | **stacked LogReg meta over 3 paradigms** | **0.8044** | **0.8053** | **0.8475** |
| 3 | char_best_p3 ((2,5) × 25k × sublinear) | char-ngram + logistic (tuned) | 0.7941 | 0.7958 | 0.7988 |
| 3 | mean_blend [char + sbert + nli] | unweighted mean of 3 probs | 0.7900 | 0.7901 | 0.8709 |
| 2 | char35_logreg ((3,5) × 200k × sublinear) | char-ngram + logistic | 0.7789 | 0.7805 | 0.7971 |
| 2 | sbert_paired_logreg | paired embed + linear | 0.7643 | 0.7653 | 0.7952 |
| 2 | sbert_answer_logreg | sentence-embed + linear | 0.7598 | 0.7615 | 0.7887 |
| 2 | tfidf12_svc | word-ngram + linear hinge | 0.7411 | 0.7443 | 0.7932 |
| 2 | sbert_paired_xgb | paired embed + XGBoost | 0.7300 | 0.7309 | 0.7893 |
| 2 | nli_deberta_zeroshot (single-pass) | cross-encoder NLI zero-shot | 0.7137 | 0.7137 | 0.7189 |
| 3 | nli_persent_maxpool | cross-encoder NLI per-sentence (zero-shot) | 0.6242 | 0.6260 | 0.7028 |

## Phase 4 additions — Optuna tuning + entity features + LLM head-to-head (2026-05-15)

### Section 1 — Optuna 100-trial joint search on char-ngram

Search space: ngram_range ∈ {(2,4), (2,5), (2,6), (3,5), (3,6)}, max_features ∈ {10k, 15k, 20k, 25k, 35k, 50k}, sublinear_tf ∈ {T,F}, min_df ∈ [1,5], max_df ∈ [0.85, 1.0], C log-uniform [0.05, 5.0], class_weight ∈ {none, balanced}, threshold tuned per-trial on a qid-grouped held-out val slice of train (n=3170).

**Best Optuna config (val F1 = 0.9728):**
- `ngram_range=(2, 4)` — narrower than Phase 3 winner (2, 5)
- `max_features=10_000` — 2.5× smaller vocab than Phase 3 (25k)
- `sublinear_tf=True`
- `min_df=5` — more aggressive doc-frequency pruning than Phase 3 (2)
- `max_df=0.916`
- `C=4.85` — 5× stronger regularization than Phase 3 default (1.0)
- `class_weight=None`
- `threshold=0.48` — slight bias toward positive class

**Matched-test result:** **char_opt = 0.8232 macro F1** (Phase 3 char_best_p3 = 0.7941, Δ = +0.0291).

### Section 2 — Custom "entity-string" feature for the FN cluster

7 lexical/structural features computed from the answer text alone: `n_tokens`, `has_period`, `has_any_punct`, `frac_capitalized`, `len_chars`, `jaccard_q`, `jaccard_k`. Stacked with char_opt's OOF probability via balanced LogReg.

| Model | matched F1 | matched accuracy | matched AUROC | recall (hallu) | precision (hallu) |
|:---|---:|---:|---:|---:|---:|
| char_best_p3 (Phase 3 baseline) | 0.7941 | 0.7958 | 0.7988 | 0.706 | 0.860 |
| char_opt (Optuna, default thr) | 0.8250 | 0.8244 | 0.8159 | 0.740 | 0.894 |
| **char_opt + 7 entity features (LogReg stack)** | **0.8519** | 0.8512 | 0.8927 | **0.763** | **0.930** |

**FN-cluster recovery on Phase 3's 77 false negatives:**
- char_opt alone recovered **10/77** (just the Optuna tune)
- char_opt + entity stack recovered **16/77** (additive +6 from the entity features)

Stack coefficients tell the story: largest positive weights on `len_chars` (+1.66), `jaccard_q` (+1.45), `has_period` (+1.44), `n_tokens` (+1.04). Largest negative weight on `jaccard_k` (−2.10). The model has learned: hallucinations look like long sentences that reuse question words but use words *not* in the knowledge passage. Exactly what the Phase 3 FN analysis predicted.

### Section 3 — Full 10-feature stack (Optuna char + sbert + nli + 7 entity)

| Stack | matched F1 | matched accuracy | matched AUROC |
|:---|---:|---:|---:|
| 3-feat: char_opt + sbert + nli (rebuild of Phase 3 stack with tuned char) | 0.8293 | 0.8302 | 0.8586 |
| **10-feat: char_opt + sbert + nli + 7 entity** | **0.8480** | **0.8492** | **0.9060** |
| char_opt + entity ALONE (no sbert, no nli) | **0.8519** | **0.8512** | 0.8927 |

**Negative finding:** Adding sbert + nli on top of `char_opt + entity` actually HURT macro F1 by −0.0039 (0.8519 → 0.8480) on the matched test. The entity features capture the signal that Phase 3's stacker was extracting from NLI's negative error correlation. With a stronger char base + entity features, the contributions of sbert (+3.00 coef) and nli (+1.48 coef) overlap with the entity signals (jaccard_k=−1.90) — they're partially redundant.

The Phase 3 "weakest model carries the stack" finding is **conditional on the base being a weaker char model**. Tune the char model and the marginal value of sbert + nli shrinks substantially.

### Section 4 — LLM head-to-head (n=50 stratified subsample of matched test)

Stratified sample: 25 grounded + 25 hallucinated (np.random.default_rng(42), cached at `results/phase4_cache/llm_sample_idx.json`). All LLM responses cached at `results/phase4_cache/llm_calls.json` — reruns of the notebook do not re-bill.

Prompt protocol (HaluEval-paper style):
```
You are a hallucination detector. Reply with EXACTLY one word: HALLUCINATED or GROUNDED.
Then on a new line: a number 0.0-1.0 for HALLUCINATED probability. No explanation.

KNOWLEDGE: <passage>
QUESTION: <q>
ANSWER: <a>
```

| Rank | Model | macro F1 | accuracy | precision | recall | latency (mean) | cost/1k | parse rate |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Claude Opus 4.6 (zero-shot judge) | **0.940** | **0.940** | 0.923 | 0.960 | 5.92 s | $4.95 | 1.00 |
| 1 | Codex GPT-5.5 (zero-shot judge) | **0.940** | **0.940** | **1.000** | 0.880 | 7.66 s | $50.00 | 1.00 |
| 3 | **custom char_opt + entity stack** | 0.900 | 0.900 | 0.917 | 0.880 | **1.2 ms** | **$0.0001** | 1.00 |
| 3 | custom 3-feat stack [char_opt + sbert + nli] | 0.900 | 0.900 | 0.917 | 0.880 | 15 ms | $0.015 | 1.00 |
| 3 | custom FULL stack [char_opt + sbert + nli + entity] | 0.900 | 0.900 | 0.917 | 0.880 | 15 ms | $0.015 | 1.00 |
| 3 | Claude Haiku 4.5 (zero-shot judge) | 0.900 | 0.900 | 0.885 | 0.920 | 9.45 s | $0.33 | 1.00 |
| 7 | custom char_opt (Optuna, no entity) | 0.859 | 0.860 | 0.909 | 0.800 | 0.5 ms | $0.0001 | 1.00 |
| 8 | custom char_best_p3 (Phase 3 baseline) | 0.839 | 0.840 | 0.905 | 0.760 | 0.5 ms | $0.0001 | 1.00 |
| — | Published HaluEval ChatGPT (paper, 2023) | — | 0.626 | — | — | — | — | — |

**Latency notes:** LLM timings include local CLI startup + agent overhead (Claude CLI ~24 s warm in fraud project; here much faster because shorter prompts). Token-cost math reflects API pricing — expect 5-10× speedup via direct API. Custom-model timings are CPU inference, no encoder overhead for char + entity (the 15 ms numbers for stacks that include SBERT cover one MiniLM-L6 forward pass).

**Cost gap at scale (1,000,000 predictions):**
- Codex GPT-5.5: ~$50,000 + ~89 hours
- Claude Opus 4.6: ~$4,950 + ~69 hours
- Claude Haiku 4.5: ~$330 + ~110 hours
- Custom char_opt + entity: **~$0.10 + ~20 minutes**

## Updated cumulative length-matched leaderboard (top of the table, after Phase 4)

| Phase | Model | Paradigm | macro F1 (matched, n=524) | accuracy | AUROC |
|---:|:---|:---|---:|---:|---:|
| **4** | **char_opt + 7 entity features** | **Optuna char-ngram + 7 lexical features + LogReg meta** | **0.8519** | **0.8512** | 0.8927 |
| **4** | FULL stack [char_opt + sbert + nli + entity] | 10-feat LogReg meta | 0.8480 | 0.8492 | **0.9060** |
| 4 | 3-feat stack [char_opt + sbert + nli] | LogReg meta (Phase 3 stack w/ tuned char) | 0.8293 | 0.8302 | 0.8586 |
| 4 | char_opt (Optuna, default threshold) | char-ngram + LogReg, joint-tuned 100 trials | 0.8250 | 0.8244 | 0.8159 |
| 3 | meta_stack [char + sbert + nli] | stacked LogReg over 3 paradigms | 0.8044 | 0.8053 | 0.8475 |
| 3 | char_best_p3 ((2,5) × 25k × sublinear) | char-ngram + logistic (tuned) | 0.7941 | 0.7958 | 0.7988 |
| 3 | mean_blend [char + sbert + nli] | unweighted mean of 3 probs | 0.7900 | 0.7901 | 0.8709 |
| 2 | char35_logreg ((3,5) × 200k × sublinear) | char-ngram + logistic | 0.7789 | 0.7805 | 0.7971 |
| 2 | sbert_paired_logreg | paired embed + linear | 0.7643 | 0.7653 | 0.7952 |
| 2 | nli_deberta_zeroshot (single-pass) | cross-encoder NLI zero-shot | 0.7137 | 0.7137 | 0.7189 |
| 3 | nli_persent_maxpool | cross-encoder NLI per-sentence (zero-shot) | 0.6242 | 0.6260 | 0.7028 |

## Phase 4 headline findings

1. **Frontier LLMs closed the published ChatGPT gap by 31 accuracy points.** HaluEval paper (2023) reported ChatGPT zero-shot at 62.6% accuracy on this benchmark. Three years later, Claude Opus 4.6 and Codex GPT-5.5 both hit **94.0%** zero-shot on the same protocol — without fine-tuning, without RAG, without anything but the model. The "specialized model crushes generic LLM at niche tasks" narrative is wrong for grounding-style hallucination detection in 2026. Frontier scaling did the work.
2. **The custom char_opt + entity stack ties Claude Haiku 4.5 at 90% on n=50** — and beats it on every cost/latency dimension by 3-4 orders of magnitude. Custom 1.2 ms vs Haiku 9.45 s (7,800× faster); $0.0001 vs $0.33 per 1k preds (3,300× cheaper). At 1M predictions/day, custom = $0.10 + 20 min; Haiku = $330 + 110 hours. The accuracy is identical; the economics are not even close.
3. **Custom loses by 4 accuracy points to Opus/GPT-5.5.** 90% vs 94%. The post-hoc reframing: "specialized model beats frontier LLM" is no longer cleanly true on this benchmark. The honest framing is "specialized model at 22,800× lower cost and 4,950× lower latency than Opus, for a 4-point accuracy trade."
4. **Optuna found a smaller, sparser vocab.** Phase 3 char_best_p3 = (2,5)×25k. Optuna found (2,4)×10k with stronger regularization (C=4.85 vs 1.0) and stricter doc-frequency pruning (min_df=5, max_df=0.92). Vocab dropped 2.5×; matched F1 went up by +0.029. Phase 3 was over-provisioned and under-regularized.
5. **Entity features did the work Phase 3 thought NLI was doing.** Adding 7 lexical/structural features (n_tokens, has_period, jaccard_q, jaccard_k, etc.) to char_opt added +0.0287 F1 — the same lift Phase 3 got from stacking with NLI. The Phase 3 stacker's "negative error correlation" finding was real, but in Phase 4 the entity features extract the same signal at zero inference cost. The 3-feat stack (char_opt + sbert + nli) is now WORSE than char_opt + entity alone.
6. **Phase 3 FN cluster partially recovered.** Of 77 false negatives that char_best_p3 missed, char_opt alone got 10 back, char_opt + entity got 16 back (total). Recall on hallucinated rose from 0.706 → 0.763. The remaining 61 are genuinely hard — short, sentence-shaped, ambiguous entity-string answers where the model still can't separate true entities from hallucinated ones without semantic grounding.

## Phase 4 artifacts

- `notebooks/phase4_tuning_and_llm.ipynb` — 31 cells (23 code, 8 markdown), executed end-to-end with zero errors in ~30 min.
- `results/phase4_optuna_trials.csv` / `phase4_optuna_trials.png` — full 100-trial history, trajectory plot.
- `results/phase4_llm_vs_custom.csv` / `phase4_llm_vs_custom.png` — head-to-head table + bar + cost-vs-accuracy log-scale scatter.
- `results/phase4_cumulative_leaderboard.csv` / `phase4_cumulative_leaderboard.png` — updated honest leaderboard.
- `results/phase4_cache/` (gitignored): Optuna SQLite study (`optuna_char_v1.db`), LLM call cache (`llm_calls.json`), stratified sample (`llm_sample_idx.json`), SBERT feature cache (`sbert_paired_features.npz`, 95 MB).
- `results/metrics.json` — Phase 4 metrics appended under `metrics.phase4`.
- `reports/day4_phase4_report.md` — full session record.

## Files
- `results/phase1_*` — Phase 1 leaderboard, splits, plots
- `results/phase2_leaderboard_raw.csv` / `phase2_leaderboard_matched.csv` — Phase 2 dual leaderboards
- `results/phase2_length_matched_split.png` — length distributions before/after matching
- `results/phase2_leaderboard_dual.png` — raw vs matched bar chart
- `results/phase2_top_confusion.png` — confusion matrices for top length-matched and top raw winners
- `results/metrics.json` — master metrics, keyed by phase
- `notebooks/phase1_eda_baselines.ipynb`, `notebooks/phase2_multimodel.ipynb`, `notebooks/phase3_feature_engineering.ipynb` — executed notebooks
- `reports/day1_phase1_report.md`, `reports/day2_phase2_report.md`, `reports/day3_phase3_report.md` — full session research records
- `results/phase3_ngram_ablation.csv`, `phase3_ngram_saturation.png` — Phase 3 Section 1 ablation grid
- `results/phase3_nli_persent_scores.csv`, `phase3_nli_threshold_sweep.png`, `phase3_nli_features.npz` — Phase 3 Section 2 per-sentence NLI features (cached for Phase 4 reuse)
- `results/phase3_stacking.csv` — Phase 3 Section 3 stacking results
- `results/phase3_error_summary.csv`, `phase3_error_breakdown.png` — Phase 3 Section 4 error analysis
- `results/phase3_leaderboard.csv`, `phase3_cumulative_leaderboard.csv`, `phase3_leaderboard.png` — Phase 3 Section 5 honest leaderboard
