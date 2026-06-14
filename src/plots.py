"""Phase 6 figures: the router cost/benefit story.

    python -m src.plots      # reads results/phase6_router_policies.csv -> two PNGs
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .utils import RESULTS_DIR

SHORT = {
    "tree_only": "tree only",
    "naive_router (all verbatim suspects)": "naive router\n(all suspects)",
    "multi_candidate_router (Phase-5 suggestion)": "multi-candidate\n(Phase-5 idea)",
    "complexity_router (q_words>=20)": "complexity\n(q_words>=20)",
    "multihop_router (both/also/share)": "multihop\n(both/also/share)",
}


def make_plots(backend: str = "claude/haiku") -> None:
    df = pd.read_csv(RESULTS_DIR / "phase6_router_policies.csv")
    d = df[df.backend == backend].copy()
    d["short"] = d.policy.map(SHORT).fillna(d.policy)
    tree_f1 = float(d[d.policy == "tree_only"].macro_f1.iloc[0])

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- panel 1: macro-F1 by policy; every router is below tree-only ---
    order = d.sort_values("macro_f1")
    colors = ["#2ca25f" if p == "tree_only" else "#de2d26" for p in order.policy]
    ax[0].barh(range(len(order)), order.macro_f1, color=colors)
    ax[0].set_yticks(range(len(order)))
    ax[0].set_yticklabels(order.short, fontsize=9)
    ax[0].axvline(tree_f1, ls="--", c="#2ca25f", lw=1.2)
    ax[0].set_xlim(0.90, 1.0)
    ax[0].set_xlabel("macro-F1 (full test, n=4000)")
    ax[0].set_title(f"Every LLM router LOWERS macro-F1\n(judge: {backend}, tau=0.5)")
    for i, (_, r) in enumerate(order.iterrows()):
        ax[0].text(r.macro_f1 + 0.001, i, f"{r.macro_f1:.4f}", va="center", fontsize=8)

    # --- panel 2: cure vs disease -- hallucinations rescued vs new false positives ---
    routers = d[d.policy != "tree_only"]
    rescued = routers.residual_caught.str.split("/").str[0].astype(int)
    new_fp = routers.new_false_positives
    ax[1].scatter(rescued, new_fp, s=110, c="#de2d26", zorder=3)
    for x, y, lab in zip(rescued, new_fp, routers.short):
        ax[1].annotate(lab.replace("\n", " "), (x, y), fontsize=8,
                       xytext=(7, -2), textcoords="offset points")
    xmax = 12
    ax[1].plot([0, xmax], [0, xmax], ls=":", c="gray", label="break-even (1 FP per rescue)")
    ax[1].set_xlabel("residual hallucinations rescued (of 10)")
    ax[1].set_ylabel("NEW false positives created")
    ax[1].set_title("The cure is worse than the disease\nrouters sit far above break-even")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].set_xlim(0, xmax)
    ax[1].set_ylim(0, new_fp.max() * 1.15)

    plt.tight_layout()
    out = RESULTS_DIR / "phase6_router_tradeoff.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    make_plots()
