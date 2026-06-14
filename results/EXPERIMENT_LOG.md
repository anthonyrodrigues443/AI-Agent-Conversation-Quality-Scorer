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

## Phase 3 — Engineered claim-relation features + a fine-tuned cross-encoder (2026-06-10)

**Same frozen split + same nearest-length matched control (n=572, KS 0.123).** Built 12 engineered features
(verbatim grounding, numeric/date, sentence-concentration, novelty/IDF, polarity); fine-tuned a 22M
cross-encoder on HaluEval's own (knowledge, answer) supervision. Bar to beat: **0.9244 matched**.

### Master leaderboard — ranked by length-matched macro-F1
| Rank | Model | raw F1 | **matched F1** | drop | beats 0.9244 |
|---|---|---:|---:|---:|:--:|
| 1 | **eng_xgboost** (overlap ⊕ 12 eng) | 0.9967 | **0.9808** | 0.016 | ✅ +0.056 |
| 1 | **eng_only_xgboost** (12 eng, no overlap) | 0.9967 | **0.9808** | 0.016 | ✅ +0.056 |
| 3 | hybrid_ce_plus_eng | 0.9960 | 0.9790 | 0.017 | ✅ |
| 4 | eng_logreg | 0.9917 | 0.9720 | 0.020 | ✅ |
| 5 | ce_minilm_l6_finetuned (22M) | 0.9935 | 0.9633 | 0.030 | ✅ |
| 6 | grounding_overlap_threshold (Phase-2 bar) | 0.9252 | 0.9244 | 0.001 | — |

### Single-feature probe (each feature thresholded alone, matched control)
lcs_token_ratio **0.9843** · is_substr 0.9825 · lcs_char_ratio 0.9825 · sent_overlap_max 0.9247 ·
ground_overlap 0.9244 · idf_overlap 0.9244 · novel_overlap 0.852 · overlap_spread/numeric/negation 0.38–0.43 (≈chance).
A **single** verbatim-contiguity feature beats the bar.

### Entity-reuse rescue (overlap missed 175/1054; recall 0.834)
hybrid **0.994 recall, rescues 96.6%** of overlap's 175 misses · eng_xgboost 0.991 / 0.949 · CE 0.989 / 0.949 · overlap 0.834 / 0.000.

### Honesty check — form-ambiguous slice (grounded answer not a verbatim span; n=190, balanced)
| Model | form-ambiguous F1 | full matched F1 |
|---|---:|---:|
| ce_minilm_l6_finetuned | 0.9947 | 0.9633 |
| eng_xgboost | 0.9947 | 0.9808 |
| hybrid_ce_plus_eng | 0.9895 | 0.9790 |
| **grounding_overlap_threshold** | **0.3286** | 0.9244 |

**Findings:**
1. **Bottleneck = features, not models.** `eng_only_xgboost` (no overlap feature) **ties** `eng_xgboost` at
   0.9808 — the engineered features fully *subsume* the Phase-2 champion. XGB importance: is_substr 0.795,
   lcs_token_ratio 0.116, lcs_char_ratio 0.060, all else < 0.03.
2. **A single feature beats the bar** — longest common token run (0.9843). Verbatim contiguity, not semantics.
3. **Fine-tuning rescued the meaning model:** zero-shot NLI 0.687 → fine-tuned MiniLM-L6 CE **0.963** (+0.276),
   clearing the lexical bar (Luna / "1-to-5" thesis holds in-domain).
4. **Entity-reuse target crushed:** overlap 0.834 → hybrid 0.994 recall; 96.6% of overlap's misses rescued.
5. **(Self-correction) The verbatim win is partly answer-form.** On the form-ambiguous slice the overlap
   champion collapses to **0.33** (chance); only the fine-tuned CE / engineered ensemble hold (≈0.99) — they
   are what actually read grounding when form is stripped.

**What didn't work:** numeric/date/negation features in isolation (≈chance, <2% XGB importance — HaluEval
hallucinations reformulate rather than swap a checkable number); `overlap_spread` alone (0.43, below chance);
fine-tuning deberta-v3-base (MPS CPU-fallback, >45 min/epoch — the 22M model sufficed).

**Bar carried to Phase 4:** champion `eng_xgboost`/hybrid at **0.9808 matched**; focus shifts from raw score
(near-saturated) to calibration, operating point for the entity-reuse minority, and a standing **form-ambiguous** control.

---

## Phase 4 — Tuning, calibration, operating points & error analysis (2026-06-11)

**Same frozen split + matched control.** CV = **StratifiedGroupKFold(4) by `qid`** (group-disjoint, leakage-free).
The question was *not* "raise the score" (it's saturated) but "is anything left to move it, and where does it
still fail." Answer: nothing moves it; one residual failure mode remains.

### 4.1 Optuna tuning of the XGBoost champion (9 knobs, grouped-CV objective)
| | CV F1 | raw F1 | matched F1 |
|---|---:|---:|---:|
| DEFAULT (Phase-3) | 0.9951 | 0.9967 | 0.9808 |
| TUNED (best of 11 trials) | 0.9949 | 0.9967 | 0.9808 |
| **Δ** | **−0.0002** | **+0.0000** | **+0.0000** |
Top-10 trials all sit at CV ≈ 0.9949 across depth 3→10, lr 0.019→0.29, 200→800 trees — a **flat ridge**.

### 4.2 Gradient-boosting family head-to-head (each Optuna-tuned)
| Rank | Model | CV F1 | raw F1 | matched F1 |
|---|---|---:|---:|---:|
| 1 | eng_xgboost (tuned) | 0.9949 | 0.9967 | **0.9808** |
| 2 | eng_lightgbm (tuned) | 0.9949 | 0.9967 | **0.9808** |
| 3 | eng_catboost (tuned) | 0.9951 | 0.9967 | **0.9808** |
| 4 | eng_logreg (C=0.03) | 0.9910 | 0.9927 | 0.9790 |
Three GB implementations land on the **exact same** numbers; a linear head is within 0.0018 matched.

### 4.3 Calibration (uncalibrated vs Platt vs isotonic; group-disjoint calibration split)
| Calibration | ECE (15-bin) | Brier | macro-F1@0.5 |
|---|---:|---:|---:|
| **uncalibrated** | **0.0041** | 0.0034 | 0.9967 |
| platt | 0.0101 | 0.0033 | 0.9967 |
| isotonic | 0.0041 | 0.0032 | 0.9967 |
Already calibrated; isotonic = no change, **Platt 2.5× worse**. Textbook "boosting needs calibration" does not transfer.

### 4.4 Operating points (full-train champion; AUPRC 0.9976, AUROC 0.9975)
F1-optimal threshold = **0.163, not 0.5**, but bimodal scores collapse **all four** operating points
(max-F1 / high-recall / high-precision / cost-optimal FN:FP=5:1) to one point: precision 0.9995, recall 0.994,
**entity-reuse-minority recall 0.9915**. No precision–recall tradeoff to navigate.

### 4.5 Error analysis (raw test: 1 FP, 12 FN of 4,000)
**10 of 12 FN are verbatim substrings of the knowledge** (9 at overlap=1.0), scored P≈0.01. They are
grounded-but-*irrelevant* — a verbatim span that is the wrong answer to the question (e.g. *"From Eden" →
"Take Me To Church"*). Error rate flat across overlap/length quartiles; the only live axis is `grounded_non_verbatim`
(form-ambiguous) at 1.05% vs `grounded_verbatim` 0.00%.

### 4.6 Learning curve — matched-F1 within 0.005 of max by **800 rows (5% of train)**. Data-saturated.

**Findings:**
1. **Tuning is a no-op** (+0.0000 matched; flat objective ridge) — the model is feature-bound, proven by exhaustion.
2. **The learner is interchangeable** — XGB = LGBM = CatBoost = 0.9967/0.9808; logreg within 0.0018.
3. **Already calibrated; Platt hurts** — ECE 0.0041, isotonic no change, Platt 0.0101.
4. **No operating-point tradeoff** — bimodal scores, one dominating threshold, entity-reuse recall 0.9915.
5. **The ceiling is question-relevance, not grounding** — 10/12 misses are verbatim-but-wrong; no grounding
   feature can catch them by construction.
6. **Data-saturated at 5%** — 20× more data does not help.

**What didn't work:** Optuna (flat ridge), Platt scaling (worsened ECE), threshold-tuning as a lever (correct
optimum 0.163 but inert — scores are separated).

**Bar carried to Phase 5:** still **0.9808 matched**. The only component that could move it is a *question→answer
relevance* signal aimed squarely at the 10 verbatim-but-wrong misses — plus the frontier-LLM head-to-head and a
one-feature-out ablation.

---

## Phase 5 — Advanced techniques + ablation + frontier-LLM head-to-head (2026-06-12)

**Question:** can a question-aware signal break the saturated 0.9808 ceiling and rescue the grounded-but-irrelevant residual; which features actually carry the signal; and how does the champion fare against zero-shot frontier LLMs — overall and on that residual?

### 5.1 Question-aware cross-encoder + QA-relevance hybrid
Fine-tuned `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M) with identical Phase-3 hyper-parameters, input changed *only* to put the question in segment A (`"question: {q} context: {k}"`). Threshold-free argmax.

| Model | matched F1 | raw F1 | residual (of 10) | ctrl FPR |
|---|---:|---:|:--:|---:|
| qa_relevance_hybrid (13f + ce_qa logit) | **0.9860** | 0.9972 | 2/10 | — |
| eng_xgboost (Phase-4 champion) | 0.9808 | 0.9967 | 0/10 | — |
| ce_grounding (k,a) — Phase 3 | 0.9457 | 0.9887 | 2/10 | 0.0031 |
| ce_qa_aware (q+k,a) — Phase 5 | 0.9211 | 0.9867 | 2/10 | 0.0037 |

**Adding the question HURT the standalone classifier** (0.9457 → 0.9211) but the QA-CE *probability* is the hybrid's #2 feature (importance 0.303) and nudges the ceiling to **0.9860 (+0.0052)**. Soft signal > hard argmax. Residual barely moved (0 → 2/10).

### 5.2 Leave-one-feature-out ablation
| Dropped | matched F1 | Δ |
|---|---:|---:|
| (none) full 13f | 0.9808 | 0.0000 |
| **lcs_char_ratio** | 0.9065 | **−0.0742** |
| every other feature (×12) | 0.9808 | 0.0000 |

**The 13-feature champion is a 1-feature model in disguise** — only `lcs_char_ratio` (longest common character run) carries unique signal; the other twelve are exactly redundant. No feature hurts. Removal beats split-importance: `is_substr` (Phase-3 "top") is fully substituted by its continuous cousin.

### 5.3 Frontier-LLM head-to-head (zero-shot, 210 cached calls, 100% parse-success)
**Representative n=50 (stratified 25/25), ranked by macro-F1:**
| Rank | Model | Acc | macro-F1 | Prec | Recall | Latency/row | Cost/1k |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **eng_xgboost (champion, CPU)** | 1.00 | **1.000** | 1.000 | 1.00 | 0.031 ms | $0.0001 |
| 2 | qa_relevance_hybrid (CPU+CE) | 1.00 | 1.000 | 1.000 | 1.00 | ~15 ms | $0.012 |
| 3 | Claude Haiku 4.5 (zero-shot) | 0.90 | 0.900 | 0.917 | 0.88 | 9.7 s | $0.35 |
| 4 | Codex GPT-5.5 (zero-shot) | 0.90 | 0.899 | 1.000 | 0.80 | 18.6 s | $50* |
| 5 | Claude Opus 4.8 (zero-shot) | 0.82 | 0.816 | 0.944 | 0.68 | 7.5 s | $5.25 |

**Grounded-but-irrelevant probe (10 verbatim hallucinations + 10 grounded controls):**
| Model | residual recall | ctrl FPR | bal-acc |
|---|:--:|---:|---:|
| Codex GPT-5.5 | **10/10** | 0.10 | 0.95 |
| Claude Haiku 4.5 | 9/10 | 0.20 | 0.85 |
| Claude Opus 4.8 | 5/10 | 0.00 | 0.75 |
| ce_qa_aware (fine-tuned) | 2/10 | 0.00 | 0.60 |
| qa_relevance_hybrid | 2/10 | 0.00 | 0.60 |
| **eng_xgboost (champion)** | **0/10** | 0.00 | 0.50 |

\* Codex cost is the CLI-realistic figure (~12k agent-loop tokens/call); direct GPT-5.5 API at the task's I/O ≈ $0.5/1k.

**Findings:**
1. **The champion is a 1-feature model** — only `lcs_char_ratio` matters (−0.074 when removed); all other 12 features Δ=0.0000.
2. **Question-awareness hurts the classifier, helps the feature** — standalone −0.025; as a soft input, hybrid +0.0052.
3. **The tiny tree beats Opus 4.8 / Haiku 4.5 / Codex GPT-5.5** on the representative distribution (1.000 vs 0.82–0.90) at 3,500×–500,000× lower cost.
4. **…but only frontier reasoning catches grounded-but-irrelevant hallucinations** — Codex 10/10, Haiku 9/10, Opus 5/10 vs champion 0/10, fine-tuned QA-CE 2/10.
5. **Opus is the conservative outlier** — highest precision, lowest recall; never false-alarms but catches half.

**Production recommendation (corrected after Codex review #7):** trigger router — tree on 100% of traffic (~free, sub-ms), route the `is_substr==1` "looks-grounded" suspects to a frontier LLM. **But on HaluEval that trigger is ~48% of traffic (1,915/4,000)** — verbatim-correct answers are ~half the data — and at the LLM's probe FPR (~0.10) routing them risks ~190 new false positives to rescue 10 hallucinations. So the naive router is *not* cheap; grounded-but-irrelevant detection is genuinely expensive. Tightening the trigger + a high-precision LLM threshold (blended cost/recall on full test) is Phase-6 work.

---

## Phase 6 — Production pipeline + Streamlit UI + the blended router economics (2026-06-14)

Productionised the champion as an importable pipeline (`src/{data_pipeline,feature_engineering,train,predict,evaluate,router,llm_judge,router_eval}.py`, `models/champion.joblib` 0.32 MB reproducing matched-F1 **0.9808** exactly) and answered the question Phase 5 deferred: **does escalating verbatim suspects to a frontier LLM actually pay, once you account for its false-positive rate on the verbatim-correct majority?** Measured on the full 4,000-row test split.

### 6.1 Router trigger sizing (full test, n=4,000)
| quantity | value |
|---|---:|
| verbatim answers (`is_substr==1`) | 1,915 (47.9%) |
| naive trigger (verbatim **and** tree says grounded) | 1,915 (47.9% of traffic) |
| — of which truly hallucinated (the residual the tree misses) | **10** |
| — of which verbatim-**correct** grounded answers | 1,905 (99.5%) |
| Phase-5's "multi-candidate only" trigger | 165 rows, **0/10** residual caught |

**Discovery — the Phase-5 heuristic is exactly wrong.** The 10 residual hallucinations are *single-candidate, multi-hop* questions (the verbatim answer is the wrong **hop**, e.g. quoting the "Leading Minister" when asked who succeeded Hitler), not comparative "X or Y" questions. The 165 multi-candidate verbatim suspects contain **zero** hallucinations — pure false-positive risk.

### 6.2 Blended router policies (expected-value over measured LLM rates)
Measured rates: **Haiku** residual-recall 0.90 / FPR 0.15 (n=40 controls); **Codex GPT-5.5** residual-recall 1.00 / FPR 0.167 (n=18 controls).

**Judge = Claude Haiku 4.5 (τ=0.5):**
| Policy | macro-F1 | prec | recall | new FP | residual caught | routed | cost/1k |
|---|--:|--:|--:|--:|:--:|--:|--:|
| **tree only** | **0.9967** | 0.9995 | 0.994 | 0 | 0/10 | 0% | $0.0001 |
| multihop (both/also/share) | 0.9908 | 0.985 | 0.997 | 30 | 5/10 | 4.9% | $0.015 |
| multi-candidate (Phase-5 idea) | 0.9906 | 0.987 | 0.994 | 26 | 0/10 | 4.1% | $0.013 |
| complexity (q_words≥20) | 0.9759 | 0.957 | 0.997 | 90 | 5/10 | 14.9% | $0.045 |
| naive (all verbatim suspects) | 0.9272 | 0.874 | 0.999 | 287 | 9/10 | 47.9% | $0.144 |

**Judge = Codex GPT-5.5 (τ=0.5), the best reasoner:**
| Policy | macro-F1 | new FP | residual caught | cost/1k |
|---|--:|--:|:--:|--:|
| **tree only** | **0.9967** | 0 | 0/10 | $0.0001 |
| naive (all verbatim suspects) | 0.9194 | 318 | **10/10** | $24.0 |

### Findings
1. **tree-only is Pareto-optimal — every LLM router LOWERS macro-F1.** The residual is 10/4,000 = 0.25% of test; the verbatim-suspect pool it hides in is 47.9% of traffic and 99.5% correct, so any LLM FPR > ~0.5% creates more new false positives than the hallucinations it rescues.
2. **Even Codex GPT-5.5, catching 10/10 of the residual, drops macro-F1 to 0.9194** by manufacturing ~318 false positives at ~$24/1k. Perfect recall on the slice is not enough.
3. **This CORRECTS the Phase-5 recommendation.** Phase 5 said "ship both — tree at scale, LLM on the suspects." Phase 6 shows that on HaluEval-QA you should **ship tree-only**: the suspect pool is too majority-correct for escalation to pay. The right escalation target would be a tiny, high-precision sub-slice that no cheap lexical rule isolates — which is the whole reason the residual is a *reasoning* problem.

**Deliverables:** `src/` production pipeline (8 modules), `models/{champion.joblib,model_card.md}`, `app.py` (Streamlit two-head demo with live scoring + on-demand LLM relevance check), `tests/` (19 pytest, all green), `results/{phase6_router_policies.csv,phase6_router_sizing.json,phase6_router_tradeoff.png,ui_screenshot.png}`, append-only LLM cache `results/phase6_cache/`.
