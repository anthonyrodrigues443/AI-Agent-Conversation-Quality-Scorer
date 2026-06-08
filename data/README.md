# Data — HaluEval-QA

**Source:** [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval) — `data/qa_data.json`
**Citation:** Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, Ji-Rong Wen.
*HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models.* EMNLP 2023.
**License:** MIT (per the HaluEval repository).

## What it is
10,000 HotpotQA-derived items. Each line of `qa_data.json` is a JSON object:

| field | description |
|-------|-------------|
| `knowledge` | the source passage the answer should be grounded in |
| `question` | the question |
| `right_answer` | the grounded gold answer (usually a HotpotQA entity span) |
| `hallucinated_answer` | a ChatGPT-generated plausible-but-wrong answer |

We expand each item into **two labeled rows** — `right_answer` → label 0 (grounded),
`hallucinated_answer` → label 1 (hallucinated) — giving a perfectly balanced **20,000-sample**
binary classification set. Both rows of an item share a `qid` so the train/test split can group by
item and never leak an item's grounded twin into the other side.

## Download
```bash
mkdir -p data/raw
curl -s -o data/raw/qa_data.json \
  https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json
# 10,000 lines, ~6.2 MB
```
`data/` is git-ignored; download locally before running the notebooks.

## Published benchmark
The HaluEval paper reports **ChatGPT zero-shot at 62.59% accuracy** on this QA hallucination-
classification task — the contextual number our models are measured against.

## Known artifact (Phase 1 finding)
Grounded answers (entity strings, mean ~14 chars) are ~4.8× shorter than hallucinated answers
(full sentences, mean ~66 chars). A length-only classifier scores 0.944 macro-F1 on the raw split —
i.e. the benchmark is largely solvable by counting characters. All leaderboards in this project
therefore report both raw and **length-matched** macro-F1. See `notebooks/phase1_eda_baselines.ipynb`.
