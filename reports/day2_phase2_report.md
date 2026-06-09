# Phase 2: Multi-Paradigm Experiment — AI Agent Conversation Quality Scorer
**Date:** 2026-06-09
**Session:** 2 of 7

## Objective
Phase 1 proved HaluEval-QA hides a brutal **length shortcut** (answer-character-count LogReg: 0.944 raw →
0.615 length-matched macro-F1) and that the only cheap feature surviving a length-matched control was
**grounding-overlap** at **0.919 matched macro-F1** — the honest floor.

Phase 2 asks one question: **when the length crutch is removed, does a model that reads *meaning* beat one that
counts *lexical overlap* — especially on entity-reuse hallucinations (high overlap, wrong claim) that overlap
misses?** To answer it I ran **six paradigms across the lexical→semantic spectrum** on the frozen Phase-1 split,
each scored on the raw test set **and** a rigorous length-matched control.

## Research & References
1. **Li et al., *HaluEval* (EMNLP 2023)** — dataset + ChatGPT zero-shot 62.6% acc reference. The grounded answers
   are near-verbatim HotpotQA entity spans; the hallucinated answers are ChatGPT-written full sentences. That
   construction turns out to drive every result below. <https://github.com/RUCAIBox/HaluEval>
2. **NLI-for-factual-consistency lineage (FactCC, SummaC; Laban et al. 2022)** — treating source as premise and a
   claim as hypothesis is the textbook zero-shot grounding check. This motivated paradigms 5–6 and made their
   failure here the headline.
3. **"The Illusion of Progress" (arXiv:2508.08285, 2025)** — detectors silently exploit length/surface cues;
   every leaderboard reports raw **and** length-matched.
4. **"Representation-based detectors fail OOD" (arXiv:2509.19372, 2025)** — a detector that *never trained on
   HaluEval* (zero-shot NLI, off-the-shelf embeddings) is the cleanest test of transferable grounding signal.

**How research shaped the experiments:** I deliberately included two *never-trained-on-HaluEval* meaning models
(zero-shot NLI cross-encoder, sentence-embedding cosine) expecting them to expose the lexical baselines as
shortcut-driven. The result inverted that expectation — documented honestly below.

## Dataset
| Metric | Value |
|--------|-------|
| Dataset | HaluEval-QA (Li et al., EMNLP 2023) |
| Samples | 20,000 (10,000 items × {grounded, hallucinated}) |
| Target | `label` ∈ {0 grounded, 1 hallucinated}, 50/50 |
| Split | frozen Phase-1 GroupShuffleSplit by `qid`, 16,000 train / 4,000 test, 0 leakage |
| Primary metric | **macro-F1** (ranks every table) |
| Honest bar to beat | grounding-overlap **0.919** length-matched macro-F1 |

**Rigorous length-matched control (this phase):** greedy nearest-length pairing (each hallucinated test answer
matched to the nearest-length grounded answer, caliper 8 chars). Answer-length KS between classes **0.874 → 0.122**,
n = 572 (286/286). Comparable KS to Phase-1's 10-char-bin control (0.123) but deterministic and slightly larger.

## Experiments

### Dual leaderboard — raw vs. length-matched (ranked by matched macro-F1)
| Rank | Paradigm | reads meaning? | raw F1 | **matched F1** | drop | matched recall (hallu) |
|---|---|:--:|---:|---:|---:|---:|
| 1 | **grounding_overlap_threshold** | ❌ lexical | 0.9252 | **0.9244** | **0.0008** | 0.853 |
| 2 | embed_xgb_semantic (overlap+2 emb feats) | ◑ | 0.9582 | 0.8890 | 0.069 | 0.801 |
| 3 | embed_xgb_plus_length | ◑+len | 0.9742 | 0.8671 | 0.107 | 0.874 |
| 4 | char_ngram_logreg (surface style) | ❌ | 0.9515 | 0.7636 | 0.188 | 0.671 |
| 5 | nli_zeroshot_qa (DeBERTa-v3) | ✅ | 0.6427 | 0.6866 | −0.044 | 0.647 |
| 6 | word_tfidf_ans⊕know | ❌ | 0.6951 | 0.6311 | 0.064 | 0.619 |
| 7 | length_only_logreg (shortcut) | ❌ | 0.9437 | 0.6174 | 0.326 | 0.556 |
| 8 | nli_zeroshot_ans (DeBERTa-v3) | ✅ | 0.5197 | 0.5885 | −0.069 | 0.629 |
| 9 | embed_cosine_e5-base (dense sim) | ✅ | 0.3359 | 0.3411 | −0.005 | 0.007 |

### Embedding encoder mini-leaderboard (cosine-to-passage, threshold-tuned on train)
| Encoder | dim | raw F1 | matched F1 |
|---|---:|---:|---:|
| e5-base-v2 | 768 | 0.3359 | 0.3411 |
| all-MiniLM-L6 | 384 | 0.3333 | 0.3333 |
| bge-small-en-v1.5 | 384 | 0.3332 | 0.3333 |

All three sit on the 0.333 majority floor → whole-passage cosine carries **no** grounding signal here.

### Entity-reuse stress test (the headline test)
- Hallucinations with above-median overlap (**entity-reuse**, n=1,054) vs. below (n=946). Recall on entity-reuse:
  length 0.97, char-ngram 0.96, xgb+len 0.95, xgb-sem 0.88, overlap 0.83, word-tfidf 0.72,
  **NLI-ans 0.49, NLI-qa 0.48, embed-cosine 0.00**.
- Of the 175 entity-reuse hallucinations **overlap got wrong** (near-verbatim, overlap ≥ 0.96), fraction rescued:
  length **0.90**, char-ngram **0.89**, xgb+len 0.68, word-tfidf 0.57, xgb-sem 0.30, **NLI-qa 0.21, NLI-ans 0.20,
  embed-cosine 0.00**.

## Key Findings
1. **The 1-line lexical-overlap rule is undefeated and almost perfectly length-invariant** — 0.9244 matched, drop
   0.0008. Six richer paradigms (3 embedding models + a 184M-param zero-shot NLI cross-encoder) all fail to beat it.
2. **My hypothesis was wrong: zero-shot NLI is among the worst.** NLI's mean P(entailment) is *higher* for
   hallucinations (0.258) than grounded answers (0.208) — the meaning signal is **inverted**. HaluEval's grounded
   answers are bare entity spans NLI reads as non-entailed (a fragment isn't a proposition); the fluent
   hallucinations reuse passage entities and read as entailed. NLI entailment is confounded by **answer form**, not
   grounding. Q+A framing partially repairs it (dev macro-F1 0.515 → 0.674) but it stays far below overlap.
3. **Dense embedding cosine ≈ chance (matched F1 ≈ 0.34).** Topical similarity ≠ factual grounding — both answers
   are *about* the same passage.
4. **XGBoost confirms overlap is the signal:** importance ground_overlap 0.942 vs emb_cos 0.041, emb_maxsent 0.017.
   The tree throws the embeddings away. Its small raw edge (0.958) does not survive length matching (0.889).
5. **No free lunch:** the cases overlap misses are rescued only by *surface/length* cues (which the matched control
   invalidates), not by meaning. A genuine meaning model for this task must be **trained** on the claim↔passage
   relation, not used zero-shot.

## What Didn't Work (and why)
- **Zero-shot NLI / embeddings** — failed because HaluEval hallucinations are *built to be topically grounded*
  (entity reuse), the exact regime where off-the-shelf entailment/similarity cannot distinguish a true claim from a
  plausible false one; and because the grounded-class bare-entity form depresses entailment.
- **word TF-IDF on answer⊕knowledge (0.695 raw)** — concatenating the knowledge floods the linear model with
  shared, within-pair-constant passage tokens (the same "shared context hurts" pattern Phase 1 found for the
  question). The answer signal is diluted.
- **char-ngram (0.952 raw → 0.764 matched)** — a *partial* shortcut: surface style correlates with length/fluency,
  so ~40% of its raw edge evaporates under length matching.

## Frontier Model Comparison
Deferred to Phase 4/5 (local `claude` / `codex` CLIs vs. the champion under the HaluEval judge protocol). Phase-1
reference point only: published ChatGPT zero-shot = 62.6% accuracy. *Note:* the off-the-shelf DeBERTa-v3 NLI model
tested here is a transformer baseline, not a generative frontier LLM — its weak result foreshadows that zero-shot
"read the meaning" is not free even for large models on adversarially-grounded hallucinations.

## Error Analysis
- **NLI** systematically mislabels short grounded entity spans as "not entailed" (high hallucination score) and
  fluent entity-reusing hallucinations as "entailed" — a form artifact, the mirror image of the length shortcut.
- **grounding-overlap's** only real misses are the 175/1,054 near-verbatim entity-reuse hallucinations (overlap ≥
  0.96). These are the Phase-3/5 target.

## Next Steps (Phase 3 — feature engineering + deep dive)
- The bottleneck is **features, not models**: every learned model collapses to overlap. Engineer features that
  capture the *claim↔passage relation* overlap misses — e.g. answer-vs-knowledge **token-level NLI on the specific
  claim**, numeric/date/negation consistency checks, named-entity *role* matching (does the answer assign the right
  attribute to the right entity?), and a **HaluEval-fine-tuned cross-encoder** (the honest way to "read meaning").
- Re-test specifically on the 175 near-verbatim hallucinations — that subset is the real frontier.
- Carry the rigorous nearest-length matched control into every Phase-3 table.

## References Used Today
- [1] Li et al. *HaluEval.* EMNLP 2023. <https://github.com/RUCAIBox/HaluEval>
- [2] Laban et al. *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection.* TACL 2022.
- [3] *The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs.* arXiv:2508.08285 (2025).
- [4] *Representation-based Broad Hallucination Detectors Fail to Generalize Out of Distribution.* arXiv:2509.19372 (2025).

## Code Changes
- `notebooks/phase2_multimodel.ipynb` — 33 cells (16 code / 17 markdown), executed end-to-end on `convo-quality-venv`,
  0 errors. Embedding + NLI scores cached under `results/phase2_cache/` for instant reruns.
- `results/` — `phase2_model_comparison.csv`, `phase2_embedding_leaderboard.csv`, `phase2_entity_reuse.csv`,
  `metrics.json` (`phase2` block), and figures `phase2_dual_leaderboard.png`, `phase2_entity_reuse_recall.png`,
  `phase2_nli_separation.png`.
