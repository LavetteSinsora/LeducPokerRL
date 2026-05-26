"""
figures/plot_performance_profile.py
=====================================
Performance profile: avg chips per session relative to CFR Nash baseline.

Y = 0 is CFR's earnings.  Maniac values exceed the y-limit and are shown as
dashed upward extensions with annotated values.

Output : figures/performance_profile.pdf  (and .png at 300 dpi)
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif']  = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "paper", "evaluation", "results")

# ── opponent axis ──────────────────────────────────────────────────────────────
ALL_OPPS = [
    ("cfr",              "CFR"),
    ("heuristic",        "Heuristic"),
    ("tight_passive",    "Tight-Passive"),
    ("tight_aggressive", "Tight-Agg."),
    ("loose_passive",    "Loose-Passive"),
    ("loose_aggressive", "Loose-Agg."),
    ("random",           "Random"),
    ("opp_encoder_v1",  "Opp. Encoder"),
    ("reinforce_v3",    "REINFORCE"),
    ("actor_critic_v3", "Actor-Critic"),
    ("dqn_v3",          "DQN"),
]
OPP_KEYS   = [k for k, _ in ALL_OPPS]
OPP_LABELS = [l for _, l in ALL_OPPS]
N_OPPS     = len(ALL_OPPS)

# ── agents ─────────────────────────────────────────────────────────────────────
# (result_dir_prefix, seeds, display_name, color, linestyle, lw, ms)
AGENTS = [
    ("full_modulation", [0,1,2], "Modulated Value Net. (Ours)", "#9B8EC7", "solid",   2.4, 5.5),
    ("reinforce_v3",    [0],     "REINFORCE",                   "#B4D3D9", "solid",   1.7, 4.5),
    ("actor_critic_v3", [0],     "Actor-Critic",                "#BDA6CE", "solid",   1.7, 4.5),
    ("dqn_v3",          [0],     "DQN",                         "#D4B896", "solid",   1.7, 4.5),
]

# ── data loading ───────────────────────────────────────────────────────────────

def load_scores(dir_prefix, seeds, opp_keys):
    means, stds = [], []
    for opp_key in opp_keys:
        # Self-matchup: agent playing against itself → expected value is 0
        if opp_key == dir_prefix:
            means.append(0.0)
            stds.append(0.0)
            continue
        seed_vals = []
        for seed in seeds:
            dir_name = dir_prefix if seed is None else f"{dir_prefix}_seed{seed}"
            fpath = os.path.join(RESULTS, dir_name, f"vs_{opp_key}.json")
            if os.path.isfile(fpath):
                with open(fpath) as f:
                    seed_vals.append(json.load(f)["chips_per_round"] * 100)
        if seed_vals:
            means.append(np.mean(seed_vals))
            stds.append(np.std(seed_vals, ddof=0))
        else:
            means.append(np.nan)
            stds.append(0.0)
    return np.array(means), np.array(stds)


def main():
    cfr_baseline, _ = load_scores("cfr", [None], OPP_KEYS)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(N_OPPS)

    agent_data = []
    for dir_prefix, seeds, label, color, lstyle, lw, ms in AGENTS:
        means, stds = load_scores(dir_prefix, seeds, OPP_KEYS)
        rel = means - cfr_baseline
        agent_data.append((rel, stds, label, color, lstyle, lw, ms, len(seeds) > 1))

    for rel, stds, label, color, lstyle, lw, ms, multi_seed in agent_data:
        # Plot main line with clipping on
        ax.plot(x, rel, color=color, linewidth=lw, linestyle=lstyle,
                marker="o", markersize=ms, zorder=3, label=label, clip_on=True)
        if multi_seed:
            valid = ~np.isnan(rel)
            ax.fill_between(x[valid],
                            np.clip((rel - stds)[valid], -200, 80),
                            np.clip((rel + stds)[valid], -200, 80),
                            color=color, alpha=0.14, zorder=1, clip_on=True)

    # ── CFR reference at y = 0 ────────────────────────────────────────────────
    ax.axhline(0, color="#888888", linewidth=0.9, linestyle="--", zorder=2)

    # ── axes limits & ticks ───────────────────────────────────────────────────
    ax.set_ylim(-62, 80)
    ax.set_xticks(x)
    ax.set_xticklabels(OPP_LABELS, rotation=35, ha="right", fontsize=9.5)
    ax.set_xlim(-0.6, N_OPPS - 0.4)
    ax.set_ylabel("Chips per Session (relative to CFR Nash)", fontsize=10.5)

    # ── grid & spines ─────────────────────────────────────────────────────────
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, color="#DDDDDD", alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")

    # ── in-dist / OOD separator + shading ────────────────────────────────────
    N_IN_DIST = 7   # CFR … Random
    sep_x = N_IN_DIST - 0.5
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.axvspan(xlim[0], sep_x, color="#91D06C", alpha=0.08, zorder=0)
    ax.axvspan(sep_x, xlim[1], color="#89D4FF", alpha=0.08, zorder=0)
    ax.axvline(sep_x, color="#AAAAAA", linewidth=1.0, linestyle=":", zorder=2)

    # Labels at the top of each region (in data coordinates)
    label_y = ylim[1] * 0.97
    ax.text((xlim[0] + sep_x) / 2, label_y, "In-Distribution",
            ha="center", va="top", fontsize=10, color="#3A7A30", fontweight="bold")
    ax.text(sep_x + 0.7, label_y, "OOD",
            ha="left", va="top", fontsize=10, color="#1A6A9A", fontweight="bold")

    # ── legend — top-left inside ──────────────────────────────────────────────
    cfr_handle = Line2D([0],[0], color="#888888", lw=0.9, linestyle="dashed",
                        label="CFR Nash (y = 0)")
    agent_handles = [
        Line2D([0],[0], color=color, lw=lw, linestyle=lstyle,
               marker="o", markersize=ms, label=label)
        for _, _, label, color, lstyle, lw, ms in AGENTS
    ]
    legend = ax.legend(
        handles=[cfr_handle] + agent_handles,
        loc="upper right",
        fontsize=8.5,
        framealpha=0.92,
        edgecolor="#DDDDDD",
        frameon=True,
        borderpad=0.8,
        labelspacing=0.45,
    )
    legend.get_frame().set_facecolor("white")

    plt.tight_layout()

    # ── save ──────────────────────────────────────────────────────────────────
    out_pdf = os.path.join(HERE, "performance_profile.pdf")
    out_png = os.path.join(HERE, "performance_profile.png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
