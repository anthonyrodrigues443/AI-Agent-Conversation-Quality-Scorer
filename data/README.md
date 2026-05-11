# Data

## HaluEval-QA (primary dataset for Phase 1+)

- **Source:** [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval) — EMNLP 2023 ([Li et al. 2023](https://arxiv.org/abs/2305.11747))
- **License:** MIT (per repo)
- **File:** `qa_data.json` — JSONL, 10,000 rows
- **Schema:** `{knowledge, question, right_answer, hallucinated_answer}` — each row supplies one grounded + one hallucinated answer, so the binary classification corpus is 20,000 balanced rows.
- **Construction:** Seed data is HotpotQA. ChatGPT is prompted to generate plausible hallucinated alternatives to the gold answer. Final samples are human-validated.

### Download
The raw file is NOT committed (gitignored). To reproduce:

```bash
python -c "import urllib.request, os; \
  os.makedirs('data/raw', exist_ok=True); \
  urllib.request.urlretrieve( \
    'https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json', \
    'data/raw/halueval_qa.json')"
```

### Known caveats (surfaced in Phase 1 EDA)

1. **Severe answer-length leakage** — grounded `right_answer` averages 14 chars (just the entity name from HotpotQA); `hallucinated_answer` averages 66 chars (a full fabricated sentence). Naive classifiers exploit length, not hallucination signals.
2. **Knowledge field is the HotpotQA passage** — already paired with the gold answer, so any model with access to `knowledge` has a (slightly) leaked feature for the grounded class.
3. **Single-source generator** — all hallucinations come from one model (ChatGPT). Detectors may not transfer to other generators.
