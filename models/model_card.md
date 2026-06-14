# Model Card — AI Agent Conversation Quality Scorer (champion tree + LLM router)

## Overview
Detects hallucinated / ungrounded agent answers against a source passage, framed as
binary classification on **HaluEval-QA** (Li et al., EMNLP 2023): `0 = grounded`,
`1 = hallucinated`. Ships as a two-head system:

- **Champion tree** — XGBoost on 13 claim-relation features. ~0.3 MB, ~0.02 ms/row on CPU.
- **LLM relevance router** — escalates *verbatim suspects* (answers copied verbatim from the
  source that the tree calls grounded) to a frontier-LLM relevance check. Optional, on-demand.

## Intended use
QA / observability gate for retrieval-grounded agent answers: flag responses that are not
supported by the provided context. Built for the case where the agent's answer should be
*entailed by a known source*. Not a general open-domain fact checker.

## Training data
| | |
|---|---|
| Dataset | HaluEval-QA, 10,000 items → 20,000 rows (grounded + hallucinated answer per item) |
| Split | GroupShuffleSplit by item id (an item's two answers never straddle the split), 80/20, seed 42, frozen in `results/phase1_split_qids.json` |
| Train / test | 16,000 / 4,000 rows |
| Primary metric | macro-F1 (balanced classes; chosen so neither class dominates) |

## Features (13)
`ground_overlap` (Phase-2 lexical overlap) + 12 engineered claim-relation features:
`is_substr, lcs_char_ratio, lcs_token_ratio, num_frac_in_know, n_num_missing, year_mismatch,
sent_overlap_max, overlap_spread, novel_overlap, idf_overlap, ans_has_neg, neg_mismatch`.
Only `idf_overlap` is train-dependent (document frequencies), persisted in the bundle.

**The model is a 1-feature model in disguise.** Phase-5 leave-one-out: dropping `lcs_char_ratio`
costs **−0.074** macro-F1; dropping *any* of the other 12 costs exactly **0.000**. Verbatim
character contiguity between answer and source *is* the task on this dataset.

## Performance (decision threshold 0.5)
| Metric | Raw test | Length-matched |
|---|---|---|
| macro-F1 | **0.9967** | **0.9808** |
| recall (hallucinations) | 0.994 | — |
| confusion (raw) | TN 1999 / FP 1 / FN 12 / TP 1988 | — |

Length-matched control downsamples the test set to equal answer-length distributions (HaluEval
hallucinations are ~4.8× longer than grounded entity strings) — it isolates real grounding signal
from the length shortcut.

### Frontier head-to-head (Phase 5, zero-shot, n=50)
| Model | macro-F1 | cost/1k |
|---|---|---|
| **Champion tree (CPU)** | **1.000** | ~$0.0001 |
| Claude Haiku 4.5 | 0.900 | ~$0.35 |
| Codex GPT-5.5 | 0.899 | ~$50 |
| Claude Opus 4.8 | 0.816 | ~$5.25 |

## Limitations — the verbatim blind spot
On the full 4,000-row test split the tree misses **10 verbatim hallucinations**: answers copied
verbatim from the source that are the *wrong answer to the question* (multi-hop relevance errors,
e.g. quoting the "Leading Minister" when asked who succeeded Hitler). These are a **relevance**
failure, not a grounding failure — lexical features cannot see them.

**Router caveat (Phase 6):** escalating verbatim suspects to an LLM does *not* free-lunch this
away. The verbatim-suspect pool is ~48% of traffic and is **99.5% verbatim-correct** answers, so
any LLM with a non-trivial false-positive rate creates more new false positives than the 10
hallucinations it rescues. The router is a deliberate cost/precision trade, documented in the
README and `results/phase6_router_policies.csv`; **tree-only is the default**.

## Ethical considerations
- Trained on Wikipedia-derived QA; coverage and biases follow HaluEval-QA. Not validated on
  medical/legal/financial grounding.
- A "grounded" verdict means *supported by the supplied source*, not *true in the world*.
- Use as an assistive signal, not an autonomous gate, for high-stakes content.

## Reproduce
```bash
python -m src.train          # -> models/champion.joblib (reproduces 0.9808 matched)
python -m src.evaluate       # raw + length-matched metrics
python -m src.router_eval    # full-test router economics (cached LLM calls)
```
