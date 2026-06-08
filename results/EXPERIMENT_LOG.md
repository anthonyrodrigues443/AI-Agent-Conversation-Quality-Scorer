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
*(Phase 2+ appended on subsequent sessions.)*
