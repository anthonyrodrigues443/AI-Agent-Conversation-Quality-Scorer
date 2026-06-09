# Experiment Log — AI Agent Conversation Quality Scorer

Cumulative record of every experiment. Primary metric: **macro-F1**. All leaderboards report both the
raw split and a **length-matched control** (HaluEval-QA leaks a strong length shortcut).

---

## Phase 1 — Domain research + dataset + EDA + baselines (2026-06-08)

**Dataset:** HaluEval-QA (Li et al., EMNLP 2023). 10k items → 20k balanced samples
(0 = grounded, 1 = hallucinated). Split: GroupShuffleSplit by `qid`, 80/20, seed 42, 0 item leakage.
Reference: published ChatGPT zero-shot = 62.6% accuracy.

### Baselines — raw split
| Model | macro-F1 | accuracy | ROC-AUC |
|-------|---:|---:|---:|
| length_only_logreg | 0.9437 | 0.9437 | 0.9713 |
| answer_length_only_logreg | 0.9435 | 0.9435 | 0.9717 |
| grounding_overlap_threshold | 0.9252 | 0.9253 | 0.8989 |
| tfidf_answer_logreg | 0.9194 | 0.9195 | 0.9674 |
| tfidf_q_plus_a_logreg | 0.8035 | 0.8037 | 0.8809 |
| majority_class | 0.3333 | 0.5000 | — |

### Length-matched control (KS 0.874 → 0.123, n=522)
| Model | raw F1 | matched F1 | Δ drop | verdict |
|-------|---:|---:|---:|---|
| grounding_overlap_threshold | 0.9252 | **0.9192** | −0.006 | real grounding signal (honest floor) |
| tfidf_answer_logreg | 0.9194 | 0.7131 | −0.206 | part style, part shortcut |
| tfidf_q_plus_a_logreg | 0.8035 | 0.6915 | −0.112 | partial |
| length_only_logreg | 0.9437 | 0.6150 | −0.329 | pure length shortcut |
| answer_length_only_logreg | 0.9435 | 0.6150 | −0.329 | pure length shortcut |
| majority_class | 0.3333 | 0.3333 | 0.000 | floor |

**Findings:**
1. Length leak: length-only 0.944 raw → 0.615 matched. The −0.33 drop is the shortcut.
2. Only answer length matters (2-feat ≈ 4-feat).
3. Grounding-overlap is genuine, not length-in-disguise (0.925 → 0.919 under matching), despite
   Spearman(overlap, length) = −0.52. **Honest floor = 0.919.** Caveat: leans on grounded-answers-are-
   verbatim-spans construction; OOD-generalization unknown.
4. Adding the question hurts (TF-IDF 0.919 → 0.804).

**Honest bar carried to Phase 2:** beat **0.919 matched macro-F1**.

---

## Phase 2 — Six paradigms across the lexical→semantic spectrum (2026-06-09)

**Same frozen split.** Rigorous length-matched control: nearest-length pairing (caliper 8), KS 0.874 → **0.122**,
n=572. Models trained on raw train, scored on raw test **and** the matched control.

### Dual leaderboard — ranked by length-matched macro-F1 (the honest metric)
| Rank | Paradigm | raw F1 | **matched F1** | drop | reads meaning? |
|---|---|---:|---:|---:|:--:|
| 1 | **grounding_overlap_threshold** | 0.9252 | **0.9244** | 0.0008 | ❌ lexical |
| 2 | embed_xgb_semantic (overlap + 2 emb feats) | 0.9582 | 0.8890 | 0.069 | ◑ |
| 3 | embed_xgb_plus_length | 0.9742 | 0.8671 | 0.107 | ◑+len |
| 4 | char_ngram_logreg | 0.9515 | 0.7636 | 0.188 | ❌ style |
| 5 | nli_zeroshot_qa (DeBERTa-v3) | 0.6427 | 0.6866 | −0.044 | ✅ |
| 6 | word_tfidf_ans⊕know | 0.6951 | 0.6311 | 0.064 | ❌ |
| 7 | length_only_logreg | 0.9437 | 0.6174 | 0.326 | ❌ shortcut |
| 8 | nli_zeroshot_ans (DeBERTa-v3) | 0.5197 | 0.5885 | −0.069 | ✅ |
| 9 | embed_cosine_e5-base | 0.3359 | 0.3411 | −0.005 | ✅ |

### Embedding encoder mini-leaderboard (cosine-to-passage)
e5-base 0.341 / MiniLM-L6 0.333 / bge-small 0.333 matched macro-F1 — **all at the 0.333 floor**.

### Entity-reuse stress test
Of 1,054 high-overlap hallucinations, overlap misclassifies 175 (near-verbatim, overlap ≥ 0.96). Rescued by:
length 0.90, char-ngram 0.89, xgb+len 0.68, word-tfidf 0.57, xgb-sem 0.30, **NLI-qa 0.21, NLI-ans 0.20, cosine 0.00**.

**Findings:**
1. The 1-line lexical-overlap rule (0.9244) **beats the 0.919 bar** and tops the matched leaderboard, drop 0.0008.
   No paradigm beats it.
2. **Hypothesis inverted:** zero-shot NLI is among the *worst*. Mean P(entailment) is *higher* for hallucinations
   (0.258) than grounded answers (0.208) — grounded bare-entity spans read as non-entailed, fluent entity-reusing
   hallucinations read as entailed. NLI is confounded by **answer form**, not grounding. Q+A framing partially
   repairs it (dev 0.515 → 0.674) but stays far below overlap.
3. Dense embedding cosine ≈ chance (0.34). Topical similarity ≠ factual grounding.
4. XGBoost importance: ground_overlap 0.942 vs emb_cos 0.041, emb_maxsent 0.017 — embeddings discarded.
5. **No free lunch:** the cases overlap misses are rescued only by surface/length cues, not meaning. A real meaning
   model must be *trained* on the claim↔passage relation (Phase 3–5 target).

**Bar carried to Phase 3:** beat **0.9244 matched macro-F1**, and specifically catch the 175 near-verbatim
entity-reuse hallucinations overlap misses.

---
*(Phase 3+ appended on subsequent sessions.)*
