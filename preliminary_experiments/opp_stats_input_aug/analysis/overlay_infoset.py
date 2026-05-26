"""
overlay_infoset.py — Infoset-level EV Range vs Model Value Range Overlay

For each of the ~180 unique infosets (current player's perspective: their hand,
board, pot, round, raises), plots three overlaid elements sorted by real EV spread:

  Blue floating bar  : [min_EV, max_EV] across all (opponent_hand × 6 archetypes)
                       — real outcome range from EV_variation_analysis ground truth
  Orange floating bar: [min_V, max_V] of max_a V(post_state, archetype) across 6 archetypes
                       — 22-dim model's predicted value range
  Black dot          : max_a V_base(post_state) from baseline ValueBasedAgent (no opp stats)
                       — single scalar, same for all archetypes

X-axis: infosets sorted by real EV spread (ascending left → right).

Run from project root:
    python -m preliminary_experiments.opp_stats_input_aug.analysis.overlay_infoset
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from engine.leduc_game import LeducGame
from preliminary_experiments.ev_variation_extras.code.sim_engine import FixedStateSimulator
from preliminary_experiments.opp_stats_input_aug.agent import StatAugValueAgent
from agents.value_based.agent import ValueBasedAgent

# ── Config ─────────────────────────────────────────────────────────────────────
CKPT_AUG      = os.path.join(HERE, "..", "outputs", "pool_random", "checkpoint_best.pt")
CKPT_BASELINE = os.path.join(ROOT, "agents", "value_based", "checkpoint.pt")
EV_DATA_PATH  = os.path.join(ROOT, "paper", "ev_analysis", "data.json")
PROTO_PATH    = os.path.join(ROOT, "paper", "evaluation", "shared", "data", "opponent_prototype_stats.json")
OUT_PATH      = os.path.join(HERE, "outputs", "overlay_infoset.png")

ARCHETYPES = ["tight_passive", "tight_aggressive", "loose_passive",
              "loose_aggressive", "maniac", "random"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _infoset_key(record):
    cp   = record["current_player"]
    hand = record["hand0"] if cp == 0 else record["hand1"]
    return (hand, record["board"], tuple(record["pot"]),
            record["round"], record["raises"], cp)


def _get_obs(record):
    game = LeducGame()
    sim  = FixedStateSimulator(
        hand0=record["hand0"], hand1=record["hand1"],
        pot=record["pot"],     current_player=record["current_player"],
        rnd=record["round"],   raises=record["raises"],
        board=record["board"],
    )
    sim._inject(game)
    return game.get_observation(viewer_id=record["current_player"])


def _aug_best_value(agent, obs, stats_vec):
    evals = agent.get_action_evaluations(obs, np.array(stats_vec, dtype=np.float32))
    return max(e["value"] for e in evals) if evals else 0.0


def _base_best_value(baseline, obs):
    evals = baseline.get_action_evaluations(obs)
    return max(e["value"] for e in evals) if evals else 0.0


# ── Build infosets ─────────────────────────────────────────────────────────────

def build_infosets(records, proto_stats, aug_agent, baseline_agent):
    """
    Groups records by infoset key; computes per-infoset:
      real_min / real_max : EV range across (opponent_hand × archetype) combos
      aug_min  / aug_max  : model value range across 6 archetypes
      base_val            : baseline model single scalar
    """
    groups = {}   # key → list of records (all archetype records for all opponent hands)
    for r in records:
        if r["opponent"] not in ARCHETYPES:
            continue
        k = _infoset_key(r)
        groups.setdefault(k, []).append(r)

    print(f"  Unique infosets: {len(groups)}")

    infosets = []
    n = len(groups)
    for i, (key, recs) in enumerate(groups.items()):
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{n} processed...")

        # Real EV range
        evs = [r["ev"] for r in recs]
        real_min, real_max = min(evs), max(evs)

        # Use first record's full state (any opponent hand gives same cp obs)
        ref_rec = recs[0]
        obs = _get_obs(ref_rec)

        # Aug model: max_a V across 6 archetypes
        aug_vals = [_aug_best_value(aug_agent, obs, proto_stats[arch])
                    for arch in ARCHETYPES]
        aug_min, aug_max = min(aug_vals), max(aug_vals)

        # Baseline: single scalar
        base_val = _base_best_value(baseline_agent, obs)

        cp   = ref_rec["current_player"]
        hand = ref_rec["hand0"] if cp == 0 else ref_rec["hand1"]
        infosets.append({
            "key":      key,
            "hand":     hand,
            "board":    ref_rec["board"],
            "pot":      ref_rec["pot"],
            "round":    ref_rec["round"],
            "raises":   ref_rec["raises"],
            "cp":       cp,
            "real_min": real_min, "real_max": real_max,
            "real_spread": real_max - real_min,
            "aug_min":  aug_min,  "aug_max":  aug_max,
            "aug_spread": aug_max - aug_min,
            "base_val": base_val,
        })

    # Sort by real EV spread ascending
    infosets.sort(key=lambda d: d["real_spread"])
    return infosets


# ── Figure ─────────────────────────────────────────────────────────────────────

def plot_overlay(infosets, out_path):
    # ── Filter: drop zero-real-spread states (outcome fixed regardless of opponent)
    shown = [d for d in infosets if d["real_spread"] > 0.0]
    n = len(shown)
    x = np.arange(n)

    real_min  = np.array([d["real_min"]    for d in shown])
    real_max  = np.array([d["real_max"]    for d in shown])
    real_sp   = np.array([d["real_spread"] for d in shown])
    aug_min   = np.array([d["aug_min"]     for d in shown])
    aug_max   = np.array([d["aug_max"]     for d in shown])
    aug_sp    = np.array([d["aug_spread"]  for d in shown])
    base_vals = np.array([d["base_val"]    for d in shown])
    rounds    = np.array([d["round"]       for d in shown])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 9),
        gridspec_kw={"height_ratios": [3, 1.4]},
    )
    fig.suptitle(
        "Per-Infoset EV Range Overlay  (zero-spread infosets excluded)\n"
        "Sorted by real EV spread  ←  narrow                                    wide  →",
        fontsize=12, fontweight="bold",
    )

    # ── Background shading by round ────────────────────────────────────────────
    for i, r in enumerate(rounds):
        for ax in (ax1, ax2):
            ax.axvspan(i - 0.5, i + 0.5,
                       color="#BBDEFB" if r == 0 else "#C8E6C9",
                       alpha=0.18, linewidth=0)

    # ── Panel 1: floating candlestick-style ranges ─────────────────────────────
    # Draw thin shaded bands + top/bottom ticks (less clutter than filled bars)
    lw = max(0.6, 14 / n)   # thinner lines when more infosets

    for i in range(n):
        # Real EV range: blue shaded bar
        ax1.fill_betweenx([real_min[i], real_max[i]], i - 0.35, i + 0.35,
                          color="#2196F3", alpha=0.30, linewidth=0)
        ax1.vlines(i, real_min[i], real_max[i],
                   color="#1565C0", linewidth=lw * 1.2, zorder=3)

        # Model range: orange outline only (so overlap with blue is readable)
        ax1.fill_betweenx([aug_min[i], aug_max[i]], i - 0.25, i + 0.25,
                          color="#FF9800", alpha=0.22, linewidth=0)
        ax1.vlines(i, aug_min[i], aug_max[i],
                   color="#E65100", linewidth=lw, zorder=4)

    # Baseline dots
    ax1.scatter(x, base_vals, color="black", s=12, zorder=6, linewidths=0)

    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.set_xticks([])
    ax1.set_ylabel("EV / Value  (chips)", fontsize=10)
    ax1.grid(axis="y", alpha=0.25)

    # Legend
    blue_patch  = mpatches.Patch(color="#2196F3", alpha=0.7,
                                 label="Real EV range (ground truth: min–max across opp hand × archetype)")
    org_patch   = mpatches.Patch(color="#FF9800", alpha=0.7,
                                 label="22-dim model value range (min–max across 6 archetypes)")
    dot_handle, = ax1.plot([], [], "ko", markersize=5,
                           label="Baseline (15-dim) single scalar")
    pf_patch = mpatches.Patch(color="#BBDEFB", alpha=0.6, label="Pre-flop infosets")
    fl_patch = mpatches.Patch(color="#C8E6C9", alpha=0.6, label="Flop infosets")
    ax1.legend(handles=[blue_patch, org_patch, dot_handle, pf_patch, fl_patch],
               fontsize=8, loc="upper left", framealpha=0.9)

    # ── Panel 2: spread magnitude comparison (simple bar chart) ───────────────
    bw = 0.38
    ax2.bar(x - bw / 2, real_sp, bw, color="#2196F3", alpha=0.75,
            label="Real EV spread")
    ax2.bar(x + bw / 2, aug_sp,  bw, color="#FF9800", alpha=0.75,
            label="Model value spread")

    ax2.set_xticks([])
    ax2.set_ylabel("Spread  (chips)", fontsize=9)
    ax2.set_xlabel(
        f"Infosets sorted by real EV spread  (n = {n}, {sum(rounds==0)} pre-flop, "
        f"{sum(rounds==1)} flop)",
        fontsize=9,
    )
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  ({n} infosets shown, {len(infosets)-n} zero-spread dropped)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading models and data...")

    aug_agent = StatAugValueAgent(model_path=CKPT_AUG)
    aug_agent.set_train_mode(False)

    baseline = ValueBasedAgent(model_path=CKPT_BASELINE)
    baseline.set_train_mode(False)

    with open(EV_DATA_PATH) as f:
        ev_data = json.load(f)
    with open(PROTO_PATH) as f:
        proto_raw = json.load(f)

    proto_stats = {k: v for k, v in proto_raw.items() if k in ARCHETYPES}

    print(f"\nBuilding {len(ev_data['records'])} records into infosets...")
    infosets = build_infosets(
        ev_data["records"], proto_stats, aug_agent, baseline)

    # Summary
    real_spreads = [d["real_spread"] for d in infosets]
    aug_spreads  = [d["aug_spread"]  for d in infosets]
    print(f"\n  {'Metric':<30} {'Real':>8}  {'Model (aug)':>12}")
    print(f"  {'─'*52}")
    print(f"  {'Mean spread':<30} {np.mean(real_spreads):>8.3f}  {np.mean(aug_spreads):>12.3f}")
    print(f"  {'Max spread':<30} {np.max(real_spreads):>8.3f}  {np.max(aug_spreads):>12.3f}")
    print(f"  {'Correlation (real vs aug)':<30} {np.corrcoef(real_spreads, aug_spreads)[0,1]:>8.3f}")
    print()

    plot_overlay(infosets, OUT_PATH)


if __name__ == "__main__":
    main()
