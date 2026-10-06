"""Stacked bar of % alive / senescent / dead per condition (cell line x treatment x time point)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOURS = {"alive": "#2a9d8f", "senescent": "#e9c46a", "dead": "#e76f51"}


def plot_percentages(T: pd.DataFrame, path) -> None:
    pc = [c for c in T.columns if c.startswith("pct_")]
    T = T.copy()
    T["condition"] = T["cell_line"].astype(str) + "  " + T["treatment"].astype(str)
    for c in [c for c in T.columns if c.lower().startswith("dose")]:
        unit = c[5:].replace("_", " ")  # dose_uM -> "uM"
        T["condition"] += pd.to_numeric(T[c], errors="coerce").map(lambda v: f" {v:.3g} {unit}".rstrip() if v == v else "")
    if T["elapsed_h"].nunique() > 1:
        T["condition"] += "  " + T["elapsed_h"].map(lambda h: f"{h:g} h")
    S = T.groupby("condition", sort=False)[pc + ["cells_vs_control_pct"]].mean()
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(S) + 1.5))
    left = np.zeros(len(S))
    for c in pc:
        k = c.removeprefix("pct_")
        ax.barh(S.index, S[c], left=left, color=COLOURS.get(k), label=k)
        left += S[c].to_numpy()
    for i, v in enumerate(S["cells_vs_control_pct"]):
        ax.text(101, i, f"{v:.0f}% cells" if np.isfinite(v) else "", va="center", fontsize=7)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.set_xlabel("% of segmented cells   (right: cells per field vs untreated control)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
