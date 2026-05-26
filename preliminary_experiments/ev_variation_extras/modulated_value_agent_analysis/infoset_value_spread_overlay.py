"""
infoset_value_spread_overlay.py
=================================
Reproduces the infoset_value_spread figure and overlays ModulatedValueAgent
predictions conditioned on each opponent's collected stats.

Original figure elements (preserved):
  - Gray band:  min–max range of 6 opponent-specific best-response value nets
  - Black dots: self-play value_based prediction
  - Gray ×:     exact CFR/Nash value (same collapsed obs)
  - Bottom bar: spread across BR nets

New overlay added:
  - Teal band:  min–max range of ModulatedValue predictions
                conditioned on 6 opponents' 4-dim stats
  - Colored ×:  per-opponent ModulatedValue predictions (one per opponent color)

The modulated predictions are evaluated by calling:
    mod_agent._get_value(obs_with_stats, viewer_id=cp)
for each of the 6 opponents' stats vectors collected by collect_stats.py.

This reveals whether the modulated network differentiates its value estimates
across opponent types and whether that differentiation tracks the BR-net range.

Data sources:
  - OpponentModeling/best_response_agents/analysis/outputs/infoset_value_spread.csv
    (pre-computed BR values + infoset metadata, sorted by spread)
  - modulated_value_agent_analysis/opponent_stats.json
    (4-dim OpponentStats from 10k-hand rollouts — from collect_stats.py)

Output:
  - modulated_value_agent_analysis/infoset_value_spread_with_modulated.png

Run from project root:
    python -m preliminary_experiments.ev_variation_extras.modulated_value_agent_analysis.infoset_value_spread_overlay
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from engine.observation import Observation
from preliminary_experiments.promoted_registry.modulated_value.agent import ModulatedValueAgent

CSV_PATH    = os.path.join(ROOT, "preliminary_experiments", "best_response_agents",
                           "analysis", "outputs", "infoset_value_spread.csv")
STATS_PATH  = os.path.join(HERE, "opponent_stats.json")
MOD_CKPT    = os.path.join(ROOT, "agents", "modulated_value", "checkpoint.pt")
OUT_PATH    = os.path.join(HERE, "infoset_value_spread_with_modulated.png")

MODEL_ORDER = [
    "tight_passive",
    "tight_aggressive",
    "loose_passive",
    "loose_aggressive",
    "maniac",
    "random",
]

MODEL_LABELS = {
    "tight_passive":    "Tight Passive",
    "tight_aggressive": "Tight Aggressive",
    "loose_passive":    "Loose Passive",
    "loose_aggressive": "Loose Aggressive",
    "maniac":           "Maniac",
    "random":           "Random",
}

MODEL_COLORS = {
    "tight_passive":    "#1f77b4",
    "tight_aggressive": "#d62728",
    "loose_passive":    "#2ca02c",
    "loose_aggressive": "#ff7f0e",
    "maniac":           "#9467bd",
    "random":           "#8c564b",
}

ROUND_SHADE = {
    0: "#DDEAF7",   # Pre-flop — light blue
    1: "#DFF1E2",   # Flop — light green
}

COLOR_MOD_BAND = "#17becf"   # teal for modulated prediction band


# ---------------------------------------------------------------------------
# Helper: FrozenStats wrapper
# ---------------------------------------------------------------------------

class FrozenStats:
    """Minimal stand-in for OpponentStats with a pre-computed 4-dim vector."""
    def __init__(self, vec: list):
        self._vec = list(vec)

    def to_feature_vector(self) -> list:
        return self._vec


# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list:
    """Load infoset_value_spread.csv preserving the sorted order (by spread)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "rank":             int(row["rank"]),
                "infoset_id":       row["infoset_id"],
                "label":            row["label"],
                "my_hand":          row["my_hand"],
                "board":            row["board"] if row["board"] else None,
                "pot":              [int(row["pot0"]), int(row["pot1"])],
                "round":            int(row["round"]),
                "raises":           int(row["raises"]),
                "current_player":   int(row["current_player"]),
                "spread":           float(row["spread"]),
                "br_min":           float(row["br_min"]),
                "br_max":           float(row["br_max"]),
                "baseline_value":   float(row["baseline_value"]),
                "nash_value":       float(row["nash_value"]) if row["nash_value"] else float("nan"),
                # Per-opponent BR net values
                "br_values": {k: float(row[k]) for k in MODEL_ORDER},
            })
    rows.sort(key=lambda r: r["rank"])
    return rows


# ---------------------------------------------------------------------------
# Build mock Observation for each infoset
# ---------------------------------------------------------------------------

def build_obs(row: dict, stats_obj=None) -> Observation:
    """
    Reconstruct an Observation from CSV row fields.
    legal_actions=[] is safe because encode_observation and _get_value
    only read hand/board/pot/round/raises/current_player fields.
    """
    return Observation(
        player_hand=row["my_hand"],
        board=row["board"],
        pot=list(row["pot"]),
        current_player=row["current_player"],
        current_round=row["round"],
        legal_actions=[],
        is_finished=False,
        raises_this_round=row["raises"],
        opponent_stats=stats_obj,
    )


# ---------------------------------------------------------------------------
# Compute modulated predictions
# ---------------------------------------------------------------------------

def compute_modulated_preds(rows: list, mod_agent, opp_stats_raw: dict) -> list:
    """
    For each infoset × each opponent, compute mod_agent._get_value(obs_with_stats, cp).
    Returns list of dicts parallel to rows, each with:
      mod_preds[opp_key] = float
      mod_min, mod_max, mod_spread
    """
    results = []
    for row in rows:
        cp    = row["current_player"]
        preds = {}
        for opp_key in MODEL_ORDER:
            stats_vec = opp_stats_raw[opp_key]
            frozen    = FrozenStats(stats_vec)
            obs       = build_obs(row, stats_obj=frozen)
            preds[opp_key] = mod_agent._get_value(obs, viewer_id=cp)

        vals = list(preds.values())
        results.append({
            "mod_preds":  preds,
            "mod_min":    float(min(vals)),
            "mod_max":    float(max(vals)),
            "mod_spread": float(max(vals) - min(vals)),
        })
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(rows: list, mod_results: list, out_path: str):
    n = len(rows)
    x = np.arange(n)

    spreads      = np.array([r["spread"]         for r in rows])
    br_min       = np.array([r["br_min"]          for r in rows])
    br_max       = np.array([r["br_max"]          for r in rows])
    baseline_vals = np.array([r["baseline_value"] for r in rows])
    nash_vals    = np.array([r["nash_value"]       for r in rows])
    round_ids    = np.array([r["round"]            for r in rows])

    mod_min    = np.array([mr["mod_min"]    for mr in mod_results])
    mod_max    = np.array([mr["mod_max"]    for mr in mod_results])
    mod_spreads = np.array([mr["mod_spread"] for mr in mod_results])

    # Per-opponent modulated predictions arrays
    mod_by_opp = {
        k: np.array([mr["mod_preds"][k] for mr in mod_results])
        for k in MODEL_ORDER
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(21, 10),
        gridspec_kw={"height_ratios": [3.4, 1.2]},
    )

    # ---- Background shading by round ----------------------------------------
    for i in range(n):
        shade = ROUND_SHADE[int(round_ids[i])]
        ax1.axvspan(i - 0.5, i + 0.5, color=shade, alpha=0.24, linewidth=0)

    # ---- BR net range band (gray, original) ----------------------------------
    ax1.fill_between(x, br_min, br_max, color="#9E9E9E", alpha=0.22, zorder=1)
    ax1.plot(x, br_min, color="#A9A9A9", linewidth=0.7, alpha=0.8, zorder=2)
    ax1.plot(x, br_max, color="#A9A9A9", linewidth=0.7, alpha=0.8, zorder=2)

    # ---- Modulated prediction band (teal) ------------------------------------
    ax1.fill_between(x, mod_min, mod_max,
                     color=COLOR_MOD_BAND, alpha=0.28, zorder=3,
                     label="_nolegend_")
    ax1.plot(x, mod_min, color=COLOR_MOD_BAND, linewidth=0.8, alpha=0.7, zorder=4)
    ax1.plot(x, mod_max, color=COLOR_MOD_BAND, linewidth=0.8, alpha=0.7, zorder=4)

    # ---- Per-opponent modulated predictions (colored × markers) --------------
    for opp_key in MODEL_ORDER:
        ax1.scatter(
            x, mod_by_opp[opp_key],
            s=10,
            marker="x",
            linewidths=0.7,
            color=MODEL_COLORS[opp_key],
            alpha=0.75,
            zorder=5,
        )

    # ---- Self-play baseline dots (black, original) ---------------------------
    ax1.scatter(x, baseline_vals, s=16, color="black", zorder=6)

    # ---- Nash × marks (gray, original) ---------------------------------------
    has_nash = not np.isnan(nash_vals).all()
    if has_nash:
        ax1.scatter(
            x, nash_vals,
            s=22,
            marker="x",
            linewidths=0.9,
            color="#5F6368",
            zorder=7,
        )

    # ---- Top-8 spread labels (original) --------------------------------------
    top_rows = rows[-8:]
    for row in top_rows:
        idx = row["rank"]
        ax1.text(
            idx,
            rows[idx]["br_max"] + 0.12,
            row["infoset_id"],
            fontsize=7,
            rotation=45,
            ha="left",
            va="bottom",
            color="#444444",
        )

    ax1.axhline(0, color="#888888", linewidth=0.6, linestyle="--", alpha=0.5)
    ax1.set_xlim(-1, n)
    ax1.set_xticks([])
    ax1.set_ylabel("Direct value network output V(obs) [chips]")

    # ---- Legend --------------------------------------------------------------
    legend_handles = [
        mpatches.Patch(facecolor="#9E9E9E", alpha=0.3,
                       label="Range across 6 opponent-specific BR value nets"),
        Line2D([0], [0], color="#A9A9A9", linewidth=1.0,
               label="BR range boundary"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor="black", markeredgewidth=0,
               label="Self-play value_based"),
    ]
    if has_nash:
        legend_handles.append(
            Line2D([0], [0], marker="x", linestyle="none", markersize=6,
                   color="#5F6368", linewidth=0.9,
                   label="Exact CFR/Nash value (same collapsed obs)")
        )
    legend_handles.extend([
        mpatches.Patch(facecolor=ROUND_SHADE[0], alpha=0.8, label="Pre-flop"),
        mpatches.Patch(facecolor=ROUND_SHADE[1], alpha=0.8, label="Flop"),
        # Modulated entries
        mpatches.Patch(facecolor=COLOR_MOD_BAND, alpha=0.35,
                       label="Modulated pred range (min–max across 6 opp stats)"),
        Line2D([0], [0], color=COLOR_MOD_BAND, linewidth=1.0,
               label="Modulated range boundary"),
    ])
    # Per-opponent modulated markers
    for opp_key in MODEL_ORDER:
        legend_handles.append(
            Line2D([0], [0], marker="x", linestyle="none", markersize=5,
                   color=MODEL_COLORS[opp_key], linewidth=0.7, alpha=0.9,
                   label=f"Mod pred — {MODEL_LABELS[opp_key]}")
        )
    ax1.legend(handles=legend_handles,
               loc="upper left", fontsize=7.5, ncol=3,
               framealpha=0.95, edgecolor="#cccccc")

    # ---- Lower panel: spread bars + modulated spread overlay -----------------
    spread_colors = ["#4E79A7" if rnd == 0 else "#59A14F" for rnd in round_ids]
    ax2.bar(x, spreads, width=0.9, color=spread_colors, alpha=0.75,
            label="BR net spread")
    ax2.bar(x, mod_spreads, width=0.5, color=COLOR_MOD_BAND, alpha=0.65,
            label="Modulated pred spread")

    ax2.set_xlim(-1, n)
    ax2.set_xlabel("Infosets ordered by spread across fixed-opponent value networks")
    ax2.set_ylabel("Spread")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # ---- Title ---------------------------------------------------------------
    fig.suptitle(
        "Observation-Level Value Spread: Best-Response Nets vs Modulated Value Agent\n"
        "180 reachable Leduc observation classes, sorted by BR-net spread. "
        "Teal band = ModulatedValue prediction range across 6 opponents' stats.",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading CSV from {CSV_PATH}")
    rows = load_csv(CSV_PATH)
    print(f"  {len(rows)} infosets loaded.")

    print(f"Loading opponent stats from {STATS_PATH}")
    with open(STATS_PATH) as f:
        opp_stats_raw = json.load(f)

    print(f"Loading ModulatedValueAgent from {MOD_CKPT}")
    mod_agent = ModulatedValueAgent(model_path=MOD_CKPT)
    mod_agent.set_train_mode(False)

    print("Computing modulated predictions ...")
    mod_results = compute_modulated_preds(rows, mod_agent, opp_stats_raw)

    # Print summary stats
    mod_spreads = [mr["mod_spread"] for mr in mod_results]
    br_spreads  = [r["spread"]      for r in rows]
    print(f"\n  BR-net spread:        mean={np.mean(br_spreads):.4f}  "
          f"median={np.median(br_spreads):.4f}  max={np.max(br_spreads):.4f}")
    print(f"  Modulated pred spread: mean={np.mean(mod_spreads):.4f}  "
          f"median={np.median(mod_spreads):.4f}  max={np.max(mod_spreads):.4f}")

    print("\nGenerating figure ...")
    plot(rows, mod_results, OUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
