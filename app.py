"""AI Agent Conversation Quality Scorer -- Streamlit UI.

    streamlit run app.py

Exposes the two-head router: a 0.3 MB CPU tree scores grounding/hallucination in
~0.02 ms, and verbatim suspects (answers copied from the source that the tree
calls grounded) can be escalated to a frontier-LLM relevance check on demand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.predict import ConversationQualityScorer  # noqa: E402
from src.utils import RESULTS_DIR  # noqa: E402

st.set_page_config(page_title="Agent Conversation Quality Scorer", page_icon="🔎", layout="wide")

EXAMPLES = {
    "— pick an example —": ("", "", ""),
    "Clean grounded answer": (
        "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia. "
        "First for Women is a woman's magazine published by Bauer Media Group in the USA.",
        "Which magazine was started first, Arthur's Magazine or First for Women?",
        "Arthur's Magazine",
    ),
    "Obvious hallucination (contradicts source)": (
        "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia. "
        "First for Women is a woman's magazine published by Bauer Media Group in the USA.",
        "Which magazine was started first, Arthur's Magazine or First for Women?",
        "First for Women was started first, in 1820.",
    ),
    "⚠️ Verbatim trap (right quote, wrong hop)": (
        "It was headed by Grand Admiral Karl Donitz as the Reichsprasident and Lutz Graf Schwerin von "
        "Krosigk as the Leading Minister. Donitz briefly succeeded Adolf Hitler as the head of state of Germany.",
        "Who was part of the Flensburg Government and succeeded Adolf Hitler?",
        "Lutz Graf Schwerin von Krosigk",
    ),
    "⚠️ Verbatim trap (Savile / Top of the Pops)": (
        "\"Clunk Click Every Trip\" is the slogan of a series of British public information films, "
        "commencing in January 1971 and starring the now disgraced entertainer Jimmy Savile.",
        "Clunk Click Every Trip starred the DJ and fundraiser who hosted what BBC television show?",
        "Top of the Pops",
    ),
}


@st.cache_resource
def get_scorer():
    return ConversationQualityScorer.load()


@st.cache_data
def get_headtohead():
    fp = RESULTS_DIR / "llm_vs_custom.csv"
    return pd.read_csv(fp) if fp.exists() else None


def gauge(prob: float) -> str:
    pct = int(round(prob * 100))
    color = "#2ca25f" if prob < 0.5 else "#de2d26"
    return (f"<div style='background:#eee;border-radius:8px;height:26px;width:100%'>"
            f"<div style='background:{color};width:{pct}%;height:26px;border-radius:8px;"
            f"text-align:center;color:white;font-weight:600;line-height:26px'>{pct}%</div></div>")


st.title("🔎 AI Agent Conversation Quality Scorer")
st.caption("Detect hallucinated / ungrounded agent answers against a source passage. "
           "13-feature CPU tree at scale + frontier-LLM relevance check on the verbatim suspects.")

try:
    scorer = get_scorer()
except FileNotFoundError:
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

with st.sidebar:
    st.header("Model")
    m = scorer.meta.get("metrics", {})
    raw = m.get("raw_test", {})
    mat = m.get("length_matched", {})
    st.metric("macro-F1 (raw test)", raw.get("macro_f1", "—"))
    st.metric("macro-F1 (length-matched)", mat.get("macro_f1", "—"))
    st.metric("recall on hallucinations", raw.get("recall_hallu", "—"))
    st.caption("Dataset: HaluEval-QA (Li et al., EMNLP 2023). Primary metric: macro-F1.")
    thr = st.slider("Decision threshold P(hallucinated)", 0.0, 1.0,
                    float(scorer.default_threshold), 0.01)
    st.divider()
    st.subheader("Phase 5 head-to-head (n=50)")
    h2h = get_headtohead()
    if h2h is not None:
        st.dataframe(h2h[["model", "macro_F1", "cost/1k($)"]], hide_index=True,
                     use_container_width=True)
    st.caption("The tree gets a perfect score on the representative distribution at "
               "~$0.0001/1k — but is blind to the verbatim-trap slice (next column).")

col_in, col_out = st.columns([1, 1])
with col_in:
    st.subheader("Input")
    _ex_keys = list(EXAMPLES.keys())
    # default to the verbatim-trap example -- the case that motivates the whole project
    ex = st.selectbox("Example", _ex_keys,
                      index=_ex_keys.index("⚠️ Verbatim trap (right quote, wrong hop)"))
    ek, eq, ea = EXAMPLES[ex]
    knowledge = st.text_area("SOURCE (knowledge the agent was given)", value=ek, height=150)
    question = st.text_area("QUESTION", value=eq, height=70)
    answer = st.text_area("AGENT ANSWER", value=ea, height=70)
    go = st.button("Score", type="primary", use_container_width=True)

with col_out:
    st.subheader("Result")
    # live-score whenever a source + answer are present (the Score button is optional);
    # this lets the selected example render its verdict immediately
    if (go or (knowledge.strip() and answer.strip())) and knowledge.strip() and answer.strip():
        res = scorer.score(knowledge, question, answer, threshold=thr)
        verdict = res["verdict"]
        st.markdown(f"### {'🔴' if verdict=='HALLUCINATED' else '🟢'} {verdict}")
        st.markdown("**P(hallucinated)**", help="Champion tree probability")
        st.markdown(gauge(res["prob_hallucinated"]), unsafe_allow_html=True)

        if res["verbatim_suspect"]:
            st.warning("**Verbatim suspect.** The answer is copied verbatim from the source and the "
                       "tree calls it grounded. This is exactly the slice where a right-looking quote "
                       "can be the *wrong* answer to the question — the tree is structurally blind here.")
            if st.button("🤖 Escalate to LLM relevance check (Claude Haiku)"):
                with st.spinner("Asking the frontier model... (CLI ~15-20s)"):
                    from src.llm_judge import judge
                    v = judge(knowledge, question, answer, backend="claude/haiku")
                if v["ok"]:
                    lab = "HALLUCINATED" if v["label"] == 1 else "GROUNDED"
                    st.markdown(f"**LLM verdict:** {'🔴' if v['label']==1 else '🟢'} {lab} "
                                f"(P={v['prob_hallucinated']:.2f}, {v['latency_s']}s)")
                    if v["label"] == 1:
                        st.error("The LLM caught a hallucination the tree missed — relevance is reasoning.")
                else:
                    st.info("LLM judge unavailable (CLI error / timeout) — tree verdict stands.")

        st.markdown("**Feature breakdown** (the tree leans on `lcs_char_ratio` — verbatim contiguity)")
        feats = res["features"]
        fdf = pd.DataFrame({"feature": list(feats.keys()), "value": list(feats.values())})
        fdf["★"] = fdf.feature.map(lambda x: "★" if x in ("lcs_char_ratio", "is_substr") else "")
        st.dataframe(fdf, hide_index=True, use_container_width=True, height=300)
    elif go:
        st.info("Enter a source passage and an answer.")

with st.expander("How it works — the two-head router"):
    st.markdown(
        "- **Tree (always on):** XGBoost on 13 cheap claim-relation features (regex + set ops + one "
        "longest-common-substring). ~0.02 ms/row on CPU, 0.3 MB. Beats Claude Opus 4.8 / GPT-5.5 / "
        "Haiku 4.5 on the representative HaluEval-QA distribution (macro-F1 1.000 vs 0.82–0.90, n=50).\n"
        "- **The blind spot:** on the full 4,000-row test split the tree misses **10 verbatim "
        "hallucinations** — answers copied verbatim from the source that are the *wrong answer to the "
        "question* (a relevance error, not a grounding error).\n"
        "- **LLM relevance check (on demand):** only verbatim suspects are escalated. Phase 6 found that "
        "no cheap rule cleanly isolates those 10 from the ~1,900 verbatim-*correct* answers, so escalation "
        "is a deliberate cost/precision trade — see the README for the full router economics.")
