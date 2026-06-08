# AI Agent Conversation Quality Scorer

Detecting when an AI agent's answer is **ungrounded / hallucinated** vs. faithful to the source it was
given — the load-bearing sub-problem of any agent **conversation-quality** scorer. Built as a research
log: multiple approaches compared, the benchmark itself interrogated, and findings corrected when the
data disagreed with my assumptions.

**Dataset:** [HaluEval-QA](https://github.com/RUCAIBox/HaluEval) (Li et al., EMNLP 2023) — 10k
HotpotQA items, each with a grounded and a ChatGPT-hallucinated answer → a balanced 20k-sample binary
task. **Primary metric:** macro-F1. **Reference point:** the HaluEval paper's ChatGPT zero-shot =
**62.6% accuracy**.

---

## Headline (Phase 1)

> **HaluEval-QA is mostly solvable by counting characters — so I built a length-matched control to find out what's real.**
> A length-only classifier scores **0.944 macro-F1** on the raw split (vs the paper's 0.626 accuracy for ChatGPT — a fair beat: on this balanced task macro-F1≈accuracy, so the length model is 0.944 on both).
> But on a length-matched control it **collapses to 0.615** — basically chance. The drop *is* the shortcut.
> The twist: token-overlap with the source, which I assumed was just length in disguise (it correlates at
> ρ=−0.52), **holds at 0.919** when length is matched. The control is a truth serum — it cleanly separates
> the shortcut features from the one cheap baseline that actually reads the source.

![raw vs length-matched leaderboard](results/phase1_baseline_comparison.png)

## Phase 1 — honest leaderboard (ranked by length-matched macro-F1)

| Rank | Model | matched F1 | raw F1 | Δ drop | reads source? |
|---:|---|---:|---:|---:|:--:|
| 1 | grounding_overlap_threshold | **0.9192** | 0.9252 | −0.006 | ✅ |
| 2 | tfidf_answer_logreg | 0.7131 | 0.9194 | −0.206 | ❌ |
| 3 | tfidf_q_plus_a_logreg | 0.6915 | 0.8035 | −0.112 | ❌ |
| 4 | length_only_logreg | 0.6150 | 0.9437 | −0.329 | ❌ |
| 5 | answer_length_only_logreg | 0.6150 | 0.9435 | −0.329 | ❌ |
| 6 | majority_class | 0.3333 | 0.3333 | 0.000 | — |

**The bar Phase 2+ must beat is 0.919 (matched), not the inflated 0.944 (raw).**

## Key findings
1. **The benchmark leaks length.** Hallucinated answers are ~4.8× longer (95% end in punctuation vs 1%
   of grounded). A length-only model rides that to 0.944 raw, 0.615 matched.
2. **Only the answer's length matters** — question/knowledge lengths are constant within a matched pair,
   so the 2-feature answer-length model equals the 4-feature one.
3. **Grounding-overlap is genuine signal, not a length artifact** — I was wrong to assume otherwise; the
   matched control vindicated it (0.925 → 0.919). It's the honest floor.
4. **Adding the question hurts** (0.919 → 0.804): shared tokens are non-discriminative noise.

## Architecture

```mermaid
flowchart LR
    A[HaluEval-QA<br/>10k items] --> B[expand: grounded 0 / hallucinated 1<br/>20k balanced]
    B --> C[GroupShuffleSplit by qid<br/>no item leakage]
    C --> D[Baselines<br/>length · tfidf · grounding-overlap]
    D --> E[Length-matched control<br/>KS 0.87 → 0.12]
    E --> F[Honest leaderboard<br/>raw vs matched]
```

## Repo layout
```
notebooks/phase1_eda_baselines.ipynb   # the Phase 1 experiment (executed, 40 cells)
config/config.yaml                     # dataset, split, metric config
data/README.md                         # HaluEval-QA download + license
results/                               # leaderboards (csv/json), metrics.json, figures
reports/day1_phase1_report.md          # full Phase 1 research report
results/EXPERIMENT_LOG.md              # cumulative master log
requirements.txt
```

## Reproduce
```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw && curl -s -o data/raw/qa_data.json \
  https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json
jupyter nbconvert --to notebook --execute --inplace notebooks/phase1_eda_baselines.ipynb
```

## Roadmap
| Phase | Focus | Status |
|------:|-------|:------:|
| 1 | Domain research + dataset + EDA + baselines + length-matched control | ✅ |
| 2 | 4–6 paradigms (n-grams, SBERT, zero-shot NLI) × raw vs matched | ⏳ |
| 3 | Feature engineering + top-model deep dive | ⏳ |
| 4 | Hyperparameter tuning + error analysis | ⏳ |
| 5 | Advanced techniques + ablation + **LLM head-to-head** (Claude/Codex) | ⏳ |
| 6 | Production pipeline + Streamlit UI | ⏳ |
| 7 | Tests + README polish + consolidation | ⏳ |

## References
- Li et al. *HaluEval.* EMNLP 2023.
- *The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs.* arXiv:2508.08285 (2025).
- *Representation-based Broad Hallucination Detectors Fail to Generalize OOD.* arXiv:2509.19372 (2025).
