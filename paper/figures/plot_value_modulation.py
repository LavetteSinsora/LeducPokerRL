"""
figures/plot_value_modulation.py
=================================
Single-panel figure: modulation Δ per info set for 3 stylistically distinct opponents.

X-axis : decision states sorted by GT EV spread (max − min across 3 opponents), ascending.
         Left  = opponent type barely matters for this state.
         Right = opponent type is highly consequential.
Y-axis : Modulation Δ = V_mod(s, stats_j) − V_base(s), chips/hand.
Lines  : Tight-Passive, Loose-Aggressive, Maniac (±1 std across seeds).

Output : figures/value_modulation.pdf / .png
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif']  = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

HERE = os.path.dirname(os.path.abspath(__file__))

OPPONENTS = ["tight_passive", "loose_aggressive", "maniac"]
OPP_LABELS = {
    "tight_passive":    "Tight-Passive",
    "loose_aggressive": "Loose-Aggressive",
    "maniac":           "Maniac",
}
# Colors consistent with Figure 1 palette
OPP_COLORS = {
    "tight_passive":    "#B4D3D9",
    "loose_aggressive": "#9B8EC7",
    "maniac":           "#D4B896",
}


def main():
    with open(os.path.join(HERE, "modulation_deltas.json")) as f:
        data = json.load(f)
    records = data["records"]

    # ── build per-state dict: state_id → {opp → {delta_mean, delta_std, gt_ev_spread}} ──
    states = {}
    for r in records:
        sid = r["state_id"]
        if sid not in states:
            states[sid] = {"gt_ev_spread": r["gt_ev_spread"]}
        states[sid][r["opponent"]] = {
            "delta_mean": r["delta_mean"],
            "delta_std":  r["delta_std"],
        }

    # Keep only states with all 3 opponents present
    complete = {sid: v for sid, v in states.items()
                if all(opp in v for opp in OPPONENTS)}

    # Sort by GT EV spread ascending
    sorted_ids = sorted(complete.keys(), key=lambda sid: complete[sid]["gt_ev_spread"])
    x = np.arange(len(sorted_ids))

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def smooth(arr, w=20):
        """Simple centred rolling mean, edge-padded."""
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")

    for opp in OPPONENTS:
        d_mean = np.array([complete[sid][opp]["delta_mean"] for sid in sorted_ids])
        d_std  = np.array([complete[sid][opp]["delta_std"]  for sid in sorted_ids])
        color  = OPP_COLORS[opp]
        label  = OPP_LABELS[opp]

        s_mean = smooth(d_mean)
        s_std  = smooth(d_std)

        ax.plot(x, s_mean, color=color, linewidth=1.8, zorder=3, label=label)
        ax.fill_between(x, s_mean - s_std, s_mean + s_std,
                        color=color, alpha=0.18, zorder=1)

    # y = 0 reference
    ax.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=2)


    # ── axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(-1, len(x))
    ax.set_ylabel("Smoothed Modulation Δ", fontsize=10.5)
    ax.set_xlabel("Information Sets", fontsize=10)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks([])   # states too dense to label individually

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")

    # ── legend ────────────────────────────────────────────────────────────────
    handles = [
        Line2D([0],[0], color=OPP_COLORS[opp], lw=1.6, label=OPP_LABELS[opp])
        for opp in OPPONENTS
    ]
    legend = ax.legend(handles=handles, loc="upper left", fontsize=9,
                       framealpha=0.92, edgecolor="#DDDDDD",
                       frameon=True, borderpad=0.8, labelspacing=0.45)
    legend.get_frame().set_facecolor("white")

    plt.tight_layout()

    out_pdf = os.path.join(HERE, "value_modulation.pdf")
    out_png = os.path.join(HERE, "value_modulation.png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
