# Phase 2 — Six Paradigms × Two Splits (Raw vs Length-Matched)

**Project:** NLP-2 — AI Agent Conversation Quality Scorer
**Date:** 2026-05-13 (Wed; one day late after a missed Tue run)
**Session:** 2 of 7
**Notebook:** `notebooks/phase2_multimodel.ipynb` (31 cells, 16 code/15 markdown, executed end-to-end, no errors)

## Objective
Phase 1 exposed a length artifact: a 4-feature length-only LogReg hits macro F1 = 0.944 on HaluEval-QA, a benchmark whose published ChatGPT zero-shot judge gets 62.6% accuracy. Today's Phase 2 had to answer two distinct questions, kept separate:

1. **Raw split** — On the official HaluEval-QA test set (n=4000), which of 6 paradigms wins? This is the apples-to-apples version of the published leaderboard.
2. **Length-matched split** — On a derived split where the marginal char-length distribution is identical across classes (n=524), which paradigm wins? This is the **honest** measurement — by construction it cannot exploit length.

The gap between (1) and (2) for each paradigm — call it Δ — quantifies how much of its score was length pattern-matching vs. genuine grounding signal.

## Research & References

1. **Honovich et al., 2022 — TRUE: Re-evaluating Factual Consistency Evaluation** (NAACL). [arXiv:2204.04991](https://arxiv.org/abs/2204.04991). Shows NLI-based scoring is the strongest single signal for grounded factuality. → Motivates Experiment 2.6.
2. **HF Hallucinations Leaderboard** (2024) — [huggingface.co/blog/leaderboard-hallucinations](https://huggingface.co/blog/leaderboard-hallucinations). Reports cross-encoder NLI verifiers at **AUROC ≈ 0.88** averaged across domains. → The bar Experiment 2.6 had to clear; mine reached only 0.59 raw / 0.72 matched, which itself is a finding (below).
3. **Reimers & Gurevych, 2019 — Sentence-BERT** (EMNLP). [arXiv:1908.10084](https://arxiv.org/abs/1908.10084). → `all-MiniLM-L6-v2` is the encoder for Experiments 2.3 / 2.4 / 2.5.
4. **Chen et al., 2025 — The Mirage of Hallucination Detection** (EMNLP Findings). [aclanthology.org/2025.findings-emnlp.1035.pdf](https://aclanthology.org/2025.findings-emnlp.1035.pdf). Argues most HaluEval gains are shortcut learning on dataset artifacts. → Directly motivates the length-matched control split.
5. **Liu et al., 2025 — ANAH-v2: Iterative Self-Training for Hallucination Detection** ([arXiv:2407.04693](https://arxiv.org/abs/2407.04693)). Best reported HaluEval-QA zero-shot open-model result at **81.5% accuracy**. → Long-horizon target for the fine-tuned classifier I'll build in Phase 4-5.

### How research shaped today's experiments
- Six paradigms, not six models: word-ngram lexical, char-ngram lexical, sentence-embed linear, paired sentence-embed linear, paired sentence-embed tree, cross-encoder NLI.
- Two splits reported for every model (raw + length-matched). A finding that holds on both is real; one that only holds on raw is a length shortcut.
- Cross-encoder NLI evaluated **zero-shot** — it has never seen HaluEval. This makes its raw-vs-matched delta the cleanest test of whether the dataset rewards real semantic understanding.

## Dataset & splits

| | Phase 1 raw | Phase 2 length-matched |
|---|---:|---:|
| Construction | GroupShuffleSplit by qid, test_size=0.2, seed=42 | 10-char bin-by-bin equalize on answer char-length, max_len=200 |
| n_train | 16,000 | 16,000 (shared) |
| n_test | 4,000 | 524 (262 grounded / 262 hallucinated) |
| KS distance between class length distributions | **D = 0.886, p ≈ 0** | **D = 0.0152, p ≈ 1.0** |

Interpretation of KS: raw test has near-total separation between class length distributions (D close to 1); length-matched has none (D close to 0). The matched split is the apples-to-apples test for *non-length* signal.

## Experiments

Every model trains on the same 16K examples (Phase 1 train), evaluates on both 4,000-row raw and 524-row length-matched test sets, and reports macro F1, accuracy, balanced accuracy, precision/recall (positive=hallucinated), ROC-AUC, fit/predict time.

### 2.1 — TF-IDF (1,2)-gram on answer + calibrated LinearSVC
**Hypothesis:** Hinge loss recovers a different operating point than Phase 1's log loss on the same word-ngram features.
**Result:** raw 0.9327 / matched 0.7411 (Δ +0.1916). Underperforms char-ngram and SBERT variants on both splits — the calibration step doesn't help when the base lexical features are saturated by length.

### 2.2 — Char-ngram TF-IDF (3-5) + LogReg
**Hypothesis:** Sub-word features capture style (punctuation, sentence connectives, casing) that ChatGPT-generated answers have but HotpotQA entity strings don't. These are *correlated* with length but not *equivalent* to length.
**Result:** **raw 0.9537 / matched 0.7789 (Δ +0.1748).** Wins the length-matched split. The matched-F1 of 0.7789 is the new bar Phase 3 must beat.

### 2.3 — `all-MiniLM-L6-v2` answer-only embeddings + LogReg
**Hypothesis:** A general-purpose sentence encoder, even of the answer alone, encodes stylistic and content information that a bag-of-words cannot.
**Result:** raw 0.9607 / matched 0.7598 (Δ +0.2009). Wins raw by a hair, but loses matched to char35_logreg — and has a *larger* length-shortcut Δ. The SBERT encoder's representation of the answer encodes length more than I expected.

### 2.4 — Paired SBERT (answer ⊕ knowledge + cos + |diff|, 770-d) + LogReg
**Hypothesis:** Giving the classifier the explicit relationship between the answer and the source passage recovers grounding signal that the answer-only models can't see.
**Result:** raw 0.9612 / matched 0.7643 (Δ +0.1969). Marginally better than answer-only on both splits, but the **knowledge channel adds only ~0.005 F1 on length-matched** — a much smaller bump than I expected. Surprising finding: the dataset's hallucinated answers are detectable largely *without consulting the knowledge*.

### 2.5 — Paired SBERT + XGBoost (500 trees, depth 6)
**Hypothesis:** A non-linear classifier on the same 770-d paired features catches dimension interactions LogReg misses.
**Result:** raw 0.9550 / matched 0.7300 (Δ +0.2250). **XGBoost is WORSE than LogReg on both splits, and has the largest length-shortcut Δ of all trained models.** Mirrors Keeper's Phase-7 finding that overparameterized classifiers overfit length-correlated noise. The encoder's embedding space is approximately linearly separable for this task; the trees latched onto length signals.

### 2.6 — Cross-encoder NLI (`nli-deberta-v3-base`) zero-shot
**Hypothesis:** Use `P(contradiction) − P(entailment)` over `(knowledge, answer)` pairs as the hallucination score. The HF Leaderboard cites this paradigm at AUROC ≈ 0.88.
**Result:** **raw 0.5905 / matched 0.7137 (Δ = −0.1232).** The ONLY model whose length-matched score exceeds its raw score. Its AUROC of 0.59 on raw is well below the 0.88 leaderboard number — likely because the leaderboard averages across QA/dialogue/summarization, and HaluEval-QA is particularly hostile to off-the-shelf NLI (the knowledge passages are long HotpotQA paragraphs that overflow the 512-token window for many examples).

The headline observation: **NLI's score *improves* when length is controlled.** Every trained classifier loses 17-23 F1 points going raw → matched; NLI gains 12. The model with the most *honest* approach to the task is also the only one whose performance increases when the shortcut is removed.

## Combined leaderboard

### Raw split (n=4000) — ranked by macro F1

| Rank | Model | Paradigm | macro F1 | accuracy | AUROC | fit (s) |
|---:|---|---|---:|---:|---:|---:|
| 1 | sbert_paired_logreg | paired embed + linear | **0.9612** | 0.9613 | 0.9829 | 0.26 |
| 2 | sbert_answer_logreg | sentence-embed + linear | 0.9607 | 0.9608 | 0.9825 | 0.06 |
| 3 | sbert_paired_xgb | paired embed + XGBoost | 0.9550 | 0.9550 | 0.9856 | 52.64 |
| 4 | char35_logreg | char-ngram + logistic | 0.9537 | 0.9538 | 0.9801 | 3.80 |
| 5 | PH1 length_only_logreg | phase-1 baseline | 0.9437 | 0.9437 | 0.9713 | 0.02 |
| 6 | tfidf12_svc | word-ngram + linear hinge | 0.9327 | 0.9328 | 0.9705 | 0.39 |
| 7 | PH1 tfidf_answer_logreg | phase-1 baseline | 0.9191 | 0.9193 | 0.9678 | 0.31 |
| 8 | PH1 tfidf_q_plus_a_logreg | phase-1 baseline | 0.7880 | 0.7887 | 0.8660 | 3.47 |
| 9 | PH1 lexical_overlap | phase-1 baseline | 0.7026 | 0.7140 | 0.5933 | 0.00 |
| 10 | nli_deberta_zeroshot | cross-encoder NLI (zero-shot) | 0.5905 | 0.5915 | 0.5895 | 0.00 |
| 11 | PH1 majority_class | phase-1 baseline | 0.3333 | 0.5000 | 0.5000 | 0.00 |

### Length-matched split (n=524) — ranked by macro F1 — **THIS IS THE HONEST LEADERBOARD**

| Rank | Model | Paradigm | macro F1 | accuracy | AUROC | precision (hallu) | recall (hallu) |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | **char35_logreg** | **char-ngram + logistic** | **0.7789** | 0.7805 | 0.7971 | 0.839 | 0.695 |
| 2 | sbert_paired_logreg | paired embed + linear | 0.7643 | 0.7653 | 0.7952 | 0.803 | 0.702 |
| 3 | sbert_answer_logreg | sentence-embed + linear | 0.7598 | 0.7615 | 0.7887 | 0.813 | 0.679 |
| 4 | tfidf12_svc | word-ngram + linear hinge | 0.7411 | 0.7443 | 0.7932 | 0.814 | 0.634 |
| 5 | sbert_paired_xgb | paired embed + XGBoost | 0.7300 | 0.7309 | 0.7893 | 0.762 | 0.672 |
| 6 | nli_deberta_zeroshot | cross-encoder NLI (zero-shot) | 0.7137 | 0.7137 | 0.7189 | 0.711 | 0.721 |

### Length-shortcut reliance — Δ(raw − matched) macro F1

| Model | raw F1 | matched F1 | Δ | Interpretation |
|---|---:|---:|---:|---|
| nli_deberta_zeroshot | 0.5905 | 0.7137 | **−0.1232** | Improves when length removed. Doing real semantic work. |
| char35_logreg | 0.9537 | 0.7789 | +0.1748 | Smallest positive Δ. Style features carry signal beyond length. |
| tfidf12_svc | 0.9327 | 0.7411 | +0.1916 | Word-ngram bag is mostly length-correlated. |
| sbert_paired_logreg | 0.9612 | 0.7643 | +0.1969 | SBERT representations bake length in. |
| sbert_answer_logreg | 0.9607 | 0.7598 | +0.2009 | Same — paired channel barely helps on matched. |
| sbert_paired_xgb | 0.9550 | 0.7300 | **+0.2250** | Largest shortcut reliance among trained models. Trees overfit length-correlated dimensions. |

## Key findings (post-worthy material)

1. **The "honest" winner on HaluEval-QA is a char-ngram logistic regression, not a deep model.** Char 3-5 grams + LogReg gets macro F1 = **0.7789** on the length-matched test, beating paired SBERT (0.7643), answer-only SBERT (0.7598), and the entire SBERT-XGB stack (0.7300). The win is small (+1.5 F1 points over the next best) but consistent — char n-grams capture stylistic differences (punctuation, casing, sentence connectives) that survive when length is controlled out.

2. **Adding non-linearity (XGBoost) actively HURT.** Paired SBERT + XGBoost was the worst trained paradigm on the length-matched split (0.7300 F1, Δ = +0.2250). With 500 trees at depth 6 on 770 dimensions, XGBoost found *more* length-correlated noise than the linear classifier ignored. This is the same shape as Keeper's "regularization matters more than capacity" finding — for a 16K-row task with binary grounding labels, linear separation in the embedding space is already maximal.

3. **The cross-encoder NLI is the only model whose score IMPROVES when length is controlled** (raw 0.5905 → matched 0.7137, Δ = −0.1232). Every trained classifier loses 17-23 F1 points on the honest split; pretrained NLI *gains* 12. Translation: **the trained models are partially solving the length-classification task in disguise; the zero-shot NLI model is the only one whose 0.71 is what it "thinks it deserves" on this dataset**. It's also the only model that has never seen HaluEval.

4. **The knowledge channel barely helps on the honest split.** Paired SBERT (which sees `[answer, knowledge, cos, |diff|]`) beats answer-only SBERT by 0.005 F1 on length-matched. The dataset's hallucinated answers are largely detectable from style/content of the answer alone, without consulting the source passage — meaning HaluEval-QA, even after length control, is still measuring "is this a ChatGPT-style sentence" more than "is this contradicted by the knowledge."

## What didn't work / what surprised me
- I expected NLI cross-encoder to win the length-matched split outright (HF leaderboard's 0.88 AUROC). It got 0.72 AUROC. The likely cause: HaluEval-QA's knowledge passages are HotpotQA paragraphs that often exceed 512 tokens, truncating away the part that contains the answer's grounding evidence. **Phase 3 will probe per-sentence aggregation of NLI scores** — score each (knowledge_sentence, answer) pair, max-pool the contradiction signal. That's a documented technique from the FactCC line of work.
- I expected paired SBERT + XGBoost to win raw. Instead it had the *worst* length-shortcut reliance of any trained model. Trees + length-correlated dimensions = overfitting machine.
- Phase-1's length-only LogReg (raw F1 0.9437) still beats word-ngram TF-IDF + SVC (0.9327) on raw. The simplest possible 4-feature model beat a more complex word-bag classifier — a clean restatement of Phase 1's headline.

## Frontier-model preview (deferred to Phase 5)
NLI cross-encoder zero-shot on length-matched: 0.7137 F1. Phase 5 will run `claude --print` (Opus + Haiku) and `codex exec` (GPT-5.4) on the same 524-row length-matched split under the original HaluEval judge protocol (knowledge + question + answer → predict hallucinated/grounded). The frontier comparison number is then:

| System | length-matched F1 (target) |
|---|---:|
| char35_logreg (Phase 2 winner) | 0.7789 |
| nli_deberta_zeroshot | 0.7137 |
| Claude Opus 4.6 zero-shot judge | TBD Phase 5 |
| Claude Haiku 4.5 zero-shot judge | TBD Phase 5 |
| Codex GPT-5.4 zero-shot judge | TBD Phase 5 |

Per-row CLI latency from the fraud-detection harness was ~24 s/call for Claude, ~30 s/call for Codex — so this will take ~1 hour wall time on Phase 5, manageable.

## Error analysis (length-matched winner: char35_logreg)
On the 524-row matched test, char35_logreg has precision 0.839 / recall 0.695 (positive = hallucinated). It's biased toward false negatives — calling a hallucinated answer "grounded." 80 of 262 hallucinated answers are missed; 30 of 262 grounded answers are flagged.

This recall ceiling is the natural target for Phase 3's feature-engineering work:
- Atomic-claim decomposition of the answer + per-claim entailment against the knowledge
- Named-entity overlap between answer and knowledge passage (HotpotQA-specific signal)
- Style features: punctuation density, sentence count, presence of hedge phrases

## Next steps (Phase 3, Thu 2026-05-14)
1. **Deep-dive on char35_logreg**: ablate the n-gram range (try (2,4), (3,6), (4,7)) and `max_features` (try 100k, 200k). Find the saturation point on length-matched.
2. **Hybrid: per-sentence NLI max-pool.** Split each knowledge passage into sentences, score every (sent_i, answer) with the NLI cross-encoder, take max contradiction probability. Engineering hypothesis: this recovers NLI's missing signal vs. the 0.72 AUROC at 512-token truncation.
3. **Stacking: char35 features + paired-SBERT features + sentence-level NLI scores → final linear meta-classifier.** First test of whether the three Phase-2 paradigms are *complementary* (errors uncorrelated) or *redundant* (same examples wrong).
4. **Error-pattern analysis** of the char35_logreg false-negatives on the length-matched split — looking for short hallucinated answers that pattern-match grounded entity strings.

## References used today
- Honovich, O. et al. (2022). TRUE: Re-evaluating Factual Consistency Evaluation. NAACL. https://arxiv.org/abs/2204.04991
- Hugging Face. The Hallucinations Leaderboard (2024). https://huggingface.co/blog/leaderboard-hallucinations
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT. EMNLP. https://arxiv.org/abs/1908.10084
- Chen et al. (2025). The Mirage of Hallucination Detection. EMNLP Findings. https://aclanthology.org/2025.findings-emnlp.1035.pdf
- Liu et al. (2025). ANAH-v2: Iterative Self-Training. https://arxiv.org/abs/2407.04693
- cross-encoder/nli-deberta-v3-base (Reimers). https://huggingface.co/cross-encoder/nli-deberta-v3-base
- sentence-transformers/all-MiniLM-L6-v2 (Reimers). https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Code changes
- `notebooks/phase2_multimodel.ipynb` (31 cells, executed end-to-end on python3 kernel with HF stack at user-site)
- `requirements.txt` — added `sentence-transformers`, `xgboost`, `torch`, `transformers`, `scipy` for Phase 2
- `results/phase2_leaderboard_raw.csv`, `phase2_leaderboard_matched.csv` (11-row and 6-row dual leaderboard)
- `results/phase2_length_matched_split.png` (length distributions before/after matching, with KS distance annotation)
- `results/phase2_leaderboard_dual.png` (raw vs matched bar chart, sorted by matched F1)
- `results/phase2_top_confusion.png` (confusion matrices for top length-matched and top raw winners)
- `results/metrics.json` — appended `phase2` key with all 12 (model, split) rows + summary pivot
- `reports/day2_phase2_report.md` (this file)
