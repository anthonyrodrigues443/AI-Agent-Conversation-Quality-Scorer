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

## Phase 5 LLM judge head-to-head (placeholders — populated Phase 5)

| Model | n (matched=524) | macro F1 | accuracy | precision | recall | latency/row | cost/1k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom char35_logreg (Phase 2 winner) | 524 | 0.7789 | 0.7805 | 0.839 | 0.695 | 0 ms | ~$0.00001 |
| Claude Opus 4.6 (zero-shot judge) | 50 (subsample) | TBD | TBD | TBD | TBD | ~24 s | ~$0.0045 |
| Claude Haiku 4.5 (zero-shot judge) | 50 | TBD | TBD | TBD | TBD | ~3 s | ~$0.0003 |
| Codex GPT-5.4 (zero-shot judge) | 50 | TBD | TBD | TBD | TBD | ~30 s | ~$0.05 |
| nli_deberta_zeroshot (Phase 2 zero-shot baseline) | 524 | 0.7137 | 0.7137 | 0.711 | 0.721 | ~40 ms | ~$0 |

## Files
- `results/phase1_*` — Phase 1 leaderboard, splits, plots
- `results/phase2_leaderboard_raw.csv` / `phase2_leaderboard_matched.csv` — Phase 2 dual leaderboards
- `results/phase2_length_matched_split.png` — length distributions before/after matching
- `results/phase2_leaderboard_dual.png` — raw vs matched bar chart
- `results/phase2_top_confusion.png` — confusion matrices for top length-matched and top raw winners
- `results/metrics.json` — master metrics, keyed by phase
- `notebooks/phase1_eda_baselines.ipynb`, `notebooks/phase2_multimodel.ipynb` — executed notebooks
- `reports/day1_phase1_report.md`, `reports/day2_phase2_report.md` — full session research records
