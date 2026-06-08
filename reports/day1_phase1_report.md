# Phase 1: Domain Research + Dataset + EDA + Baselines — AI Agent Conversation Quality Scorer
**Date:** 2026-06-08
**Session:** 1 of 7

## Objective
Establish the foundation for an AI-agent conversation **quality scorer**, starting with its most
load-bearing sub-problem: detecting when an agent's answer is **ungrounded / hallucinated** vs.
faithful to the source it was given. Before building anything sophisticated, answer one question:
**what is the honest performance floor, and is the benchmark actually testing grounding — or leaking
a cheap surface shortcut?**

## Research & References
1. **Li et al., *HaluEval*, EMNLP 2023 (RUCAIBox)** — the canonical hallucination benchmark. Provides
   the dataset (10k HotpotQA-derived QA items, each with a grounded and a hallucinated answer) and the
   headline number: **ChatGPT zero-shot = 62.59% accuracy** at spotting hallucinations on the QA task.
   <https://github.com/RUCAIBox/HaluEval>
2. **"The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs" (arXiv 2508.08285, 2025)**
   — warns that hallucinated text is systematically **longer**, and that detectors silently exploit
   length and other spurious cues rather than meaning. This directly motivated the length-artifact
   probe and the length-matched control built here.
3. **"Representation-based Broad Hallucination Detectors Fail to Generalize Out of Distribution"
   (arXiv 2509.19372, 2025)** — detectors latch onto spurious phenomena (e.g. negation, surface form)
   and break under minor distribution shift. Reinforces that a "win" on the raw split is not credible
   without a controlled split.

**How research shaped today's experiments:** rather than chase a high raw-split number, I treated the
benchmark itself as the object of study — quantified the length asymmetry, built a length-only model
to *measure* the shortcut, and constructed a length-matched control that re-scores every baseline so
the genuine grounding signal is separated from length.

## Dataset
| Metric | Value |
|--------|-------|
| Dataset | HaluEval-QA (Li et al., EMNLP 2023) |
| Total items | 10,000 |
| Total samples | 20,000 (each item → grounded + hallucinated) |
| Features used | `knowledge`, `question`, `answer` (+ engineered: lengths, token counts, grounding overlap) |
| Target | `label` ∈ {0 = grounded, 1 = hallucinated} |
| Class distribution | 50.0% / 50.0% (perfectly balanced) |
| Split | GroupShuffleSplit by `qid`, 80/20, seed 42 → 16,000 train / 4,000 test, **0 item leakage** |
| Primary metric | **macro-F1** (balanced task; we weight both classes equally) |

**Data quality:** 0 missing answers, 0 empty answers, 0 missing knowledge, 0 grounded==hallucinated
collisions. Clean.

## Experiments

### Experiment 1.1: Length-asymmetry probe
**Hypothesis:** Grounded answers (HotpotQA entity spans) are far shorter than ChatGPT-written
hallucinated sentences, leaking a length shortcut.
**Method:** Compare answer char/token length and trailing-punctuation rate by class.
**Result:** Hallucinated answers average **4.85× more characters** than grounded; **95.2%** end in
punctuation vs **1.4%** of grounded.
**Interpretation:** A massive, trivially-learnable surface gap. Confirmed — proceed to weaponize it.

### Experiment 1.2: Baseline leaderboard (raw split)
**Method:** Five baselines on the leak-free 4,000-row test set.

| Model | macro-F1 | accuracy | ROC-AUC | note |
|-------|---:|---:|---:|---|
| length_only_logreg (4 feat) | **0.9437** | 0.9437 | 0.9713 | the shortcut, made explicit |
| answer_length_only_logreg (2 feat) | 0.9435 | 0.9435 | 0.9717 | ≈ identical → only answer length matters |
| grounding_overlap_threshold | 0.9252 | 0.9253 | 0.8989 | the only baseline that reads the source |
| tfidf_answer_logreg | 0.9194 | 0.9195 | 0.9674 | strong, largely redundant with length |
| tfidf_q_plus_a_logreg | 0.8035 | 0.8037 | 0.8809 | adding the question HURT by −0.116 F1 |
| majority_class | 0.3333 | 0.5000 | — | floor |

**Interpretation:** Every trained baseline crushes the published ChatGPT 0.626 accuracy (a fair
comparison despite the differing metric labels: on this perfectly balanced task macro-F1 ≈ accuracy —
length-only is 0.944 on both, per the table above) — but that is *because*
of the length leak, not despite it. The question-text experiment (−0.116 F1) is a clean "more features
≠ better": the question is shared across an item's two answers, so its tokens are non-discriminative
noise.

### Experiment 1.3: Length-matched control (the truth serum)
**Hypothesis:** Once answer length is matched between classes, the length models collapse toward
chance; whatever survives is real signal.
**Method:** Bin test answers into 10-char bins; within each bin downsample to equal grounded/
hallucinated counts → 522-sample balanced control. Answer-length KS between classes drops **0.874 →
0.123**. Re-score every baseline.

| Model | raw F1 | matched F1 | Δ (drop) | verdict |
|-------|---:|---:|---:|---|
| grounding_overlap_threshold | 0.9252 | **0.9192** | **−0.006** | **real grounding signal — the honest floor** |
| tfidf_answer_logreg | 0.9194 | 0.7131 | −0.206 | part style, part shortcut |
| tfidf_q_plus_a_logreg | 0.8035 | 0.6915 | −0.112 | partial |
| length_only_logreg | 0.9437 | 0.6150 | −0.329 | **pure length shortcut** |
| answer_length_only_logreg | 0.9435 | 0.6150 | −0.329 | pure length shortcut |
| majority_class | 0.3333 | 0.3333 | 0.000 | floor |

**Interpretation — and a correction.** I went in assuming grounding-overlap was *length in disguise*
(overlap is a fraction of answer tokens, and Spearman(overlap, length) = −0.52). The control proved me
wrong: overlap barely moves (0.925 → 0.919) while the length models fall off a cliff to 0.615. At
*matched* lengths, grounded answers still overlap the knowledge far more than hallucinations do — so
overlap reads genuine **content grounding**, not length. The length-matched control cleanly sorts the
cheap features into shortcut (length: → 0.615 ≈ chance) and real signal (overlap: → 0.919).

## Head-to-Head Comparison (Phase 1 honest leaderboard, ranked by length-matched macro-F1)
| Rank | Model | matched F1 | raw F1 | reads the source? |
|---:|---|---:|---:|:--:|
| 1 | grounding_overlap_threshold | **0.9192** | 0.9252 | ✅ |
| 2 | tfidf_answer_logreg | 0.7131 | 0.9194 | ❌ |
| 3 | tfidf_q_plus_a_logreg | 0.6915 | 0.8035 | ❌ |
| 4 | length_only_logreg | 0.6150 | 0.9437 | ❌ |
| 5 | answer_length_only_logreg | 0.6150 | 0.9435 | ❌ |
| 6 | majority_class | 0.3333 | 0.3333 | — |

## Key Findings
1. **HaluEval-QA is largely solvable by counting characters — and a matched control proves it.**
   length-only LogReg: 0.944 raw → **0.615 matched**. The −0.33 F1 drop *is* the shortcut, made visible.
2. **Only the answer's length matters** (answer-length-only ≈ full length model; question/knowledge
   lengths are constant within a pair).
3. **Grounding-overlap is the genuine cheap signal — I was wrong to dismiss it.** 0.925 → 0.919 under
   length matching. It is the honest floor at **0.919 macro-F1** and the concrete bar Phases 2–5 must
   beat. *Caveat:* it leans on HaluEval-QA's construction (grounded answers are near-verbatim knowledge
   spans), so its robustness on free-form agent responses is an open OOD question.
4. **Adding the question HURTS** (0.919 → 0.804, raw): shared tokens = non-discriminative noise.

## Frontier Model Comparison
Deferred to Phase 4/5 (the LLM head-to-head uses the original HaluEval judge protocol via the local
`claude`/`codex` CLIs). Phase-1 reference point only: published **ChatGPT zero-shot = 62.6% accuracy**.

## Error Analysis
- The length shortcut's residual errors on the matched control are the interesting cases: grounded
  answers that happen to be long, and hallucinations that happen to be short entity strings. Those are
  exactly where a length detector fails and where Phase 2's semantic models must earn their keep.
- grounding-overlap's misses are hallucinations that reuse the passage's entities (high overlap but
  wrong claim) — these need *meaning*, not lexical overlap. Prime target for the zero-shot NLI paradigm.

## Next Steps (Phase 2)
- Promote the quick per-bin control to a **rigorous length-matched split** (nearest-length pairing,
  report KS), so every leaderboard reports raw + matched.
- Run 4–6 paradigms: word-ngram + linear, char-ngram + LogReg, sentence-transformer embeddings (±
  tree head), and a **zero-shot cross-encoder NLI** that has never seen HaluEval.
- **Concrete bar:** beat grounding-overlap's **0.919 matched macro-F1**, and test whether a model that
  reads meaning closes the gap on entity-reuse hallucinations that overlap misses.

## References Used Today
- [1] Li et al. *HaluEval.* EMNLP 2023. <https://github.com/RUCAIBox/HaluEval>
- [2] *The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs.* arXiv:2508.08285 (2025).
- [3] *Representation-based Broad Hallucination Detectors Fail to Generalize Out of Distribution.*
      arXiv:2509.19372 (2025).

## Code Changes
- `notebooks/phase1_eda_baselines.ipynb` — 40 cells (20 code / 20 markdown), executed end-to-end on the
  `convo-quality-venv` (Python 3.14) kernel, 0 errors.
- `results/` — `phase1_baselines.csv`, `phase1_baselines_matched.csv`, `phase1_raw_vs_matched.csv`,
  `phase1_baselines.json`, `phase1_split_qids.json`, `metrics.json`, and figures
  `phase1_length_and_overlap.png`, `phase1_baseline_comparison.png`, `phase1_confusion_raw_vs_matched.png`.
- `requirements.txt`, `config/config.yaml`, `data/README.md`, `src/__init__.py`, project scaffolding.
