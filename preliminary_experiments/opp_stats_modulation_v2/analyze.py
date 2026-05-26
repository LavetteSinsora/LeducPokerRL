"""
opp_stats_modulation_v2 — Comprehensive Analysis Script
=========================================================
Produces four outputs:

  1. Tournament Evaluation  (10K rounds/opponent, both seats)
     Statistical comparison of v2a and v2b vs baseline_value_v1 and v1_a_td.
     Includes 95% CI estimate from game-variance assumption (σ ≈ 1.5 chips/round).

  2. EV Spread Overlay  (fig1_ev_spread_overlay.png)
     Mirrors the existing infoset_value_spread_overlay.py structure:
       - Gray band:  BR-net min–max range (from infoset_value_spread.csv)
       - Black dots: self-play value_based prediction
       - Orange band: v2a ungated prediction range across 6 opponent stats
       - Teal band:  v2b gated prediction range across 6 opponent stats
       - Lower panel: BR spread + modulated spread bars
     X-axis = infosets sorted by BR spread (ascending, matching original figure)

  3. Gate Analysis  (fig2_gate_analysis.png)
     v2b only. For each infoset (sorted by EV spread), shows:
       - Mean gate value across 6 opponents' prototype stats
       - Per-opponent gate scatter
       - Gate std per infoset (bottom panel)
     Reveals whether the gate has learned state-selective gating.

  4. Modulation Magnitude  (fig3_modulation_magnitude.png)
     For both v2a and v2b, per infoset per opponent:
       Modulation = V(s, opp_stats) − V_base(s)
     Shows: how much the modulation head adjusts the frozen base, and whether
     it tracks the ground-truth EV residual (GT EV vs base prediction).

Run from project root:
  python -m preliminary_experiments.opp_stats_modulation_v2.analyze
  python -m preliminary_experiments.opp_stats_modulation_v2.analyze --no-tournament   # skip 10K eval
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from engine.observation import Observation
from paper.evaluation.comparison_protocol import (
    STANDARD_OPPONENT_KEYS,
    build_standard_opponents,
    evaluate_stat_aware_pool,
    format_pool_summary,
    compute_pool_summary,
)
from paper.evaluation.shared.training_recipe import play_hand_v2
from preliminary_experiments.opp_stats_modulation_v2.variant_a_ungated.agent import UngatedModAgent
from preliminary_experiments.opp_stats_modulation_v2.variant_b_state_gated.agent import StateGatedModAgent

# ── paths ──────────────────────────────────────────────────────────────────────

CSV_PATH      = os.path.join(ROOT, "preliminary_experiments", "best_response_agents",
                             "analysis", "outputs", "infoset_value_spread.csv")
PROTO_PATH    = os.path.join(ROOT, "paper", "evaluation", "shared", "data",
                             "opponent_prototype_stats.json")
PRIORS_PATH   = os.path.join(HERE, "outputs", "variant_a_ungated", "pool_priors.json")
V2A_CKPT      = os.path.join(HERE, "outputs", "variant_a_ungated", "checkpoint_best_robust.pt")
V2B_CKPT      = os.path.join(HERE, "outputs", "variant_b_state_gated", "checkpoint_best_robust.pt")
OUT_DIR       = os.path.join(HERE, "outputs", "analysis")

SESSION_LENGTH = 100
PRIOR_STRENGTH = 20.0
TOURNAMENT_ROUNDS = 10_000     # per opponent (both seats combined)

# ── consistent look-and-feel ──────────────────────────────────────────────────

MODEL_ORDER = [
    "tight_passive", "tight_aggressive",
    "loose_passive", "loose_aggressive",
    "maniac", "random",
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
ROUND_SHADE  = {0: "#DDEAF7", 1: "#DFF1E2"}
COLOR_BR     = "#9E9E9E"
COLOR_VB     = "#e377c2"
COLOR_V2A    = "#ff7f0e"   # orange for ungated
COLOR_V2B    = "#17becf"   # teal for gated
DPI          = 160

BASELINES_KNOWN = {
    "baseline_value_v1": {"heuristic": 0.295, "cfr": -0.070,
                          "tight_passive": 0.745, "tight_aggressive": 0.703,
                          "loose_passive": 0.841, "loose_aggressive": 0.623,
                          "maniac": 1.165, "random": 1.353, "_avg": 0.580},
    "v1_a_td":           {"heuristic": 0.008, "cfr": -0.194,
                          "tight_passive": 0.704, "tight_aggressive": 0.581,
                          "loose_passive": 0.797, "loose_aggressive": 0.560,
                          "maniac": 1.533, "random": 1.644, "_avg": 0.539},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(path: str, fig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "rank":           int(row["rank"]),
                "infoset_id":     row["infoset_id"],
                "label":          row["label"],
                "my_hand":        row["my_hand"],
                "board":          row["board"] if row["board"] else None,
                "pot":            [int(row["pot0"]), int(row["pot1"])],
                "round":          int(row["round"]),
                "raises":         int(row["raises"]),
                "current_player": int(row["current_player"]),
                "spread":         float(row["spread"]),
                "br_min":         float(row["br_min"]),
                "br_max":         float(row["br_max"]),
                "baseline_value": float(row["baseline_value"]),
                "nash_value":     (float(row["nash_value"])
                                   if row.get("nash_value") else float("nan")),
                "br_values":      {k: float(row[k]) for k in MODEL_ORDER
                                   if k in row},
            })
    rows.sort(key=lambda r: r["rank"])
    return rows


def build_obs(row: dict) -> Observation:
    return Observation(
        player_hand=row["my_hand"],
        board=row["board"],
        pot=list(row["pot"]),
        current_player=row["current_player"],
        current_round=row["round"],
        legal_actions=[],
        is_finished=False,
        raises_this_round=row["raises"],
        opponent_stats=None,
    )


def state_label(row: dict) -> str:
    rnd   = "Pre" if row["round"] == 0 else "Flop"
    board = row["board"] if row["board"] else "—"
    return f"{rnd} {row['my_hand']}\n{board} [{row['pot'][0]},{row['pot'][1]}]"


# ── per-infoset predictions ───────────────────────────────────────────────────

def compute_predictions(rows: list[dict],
                        v2a_agent: UngatedModAgent,
                        v2b_agent: StateGatedModAgent,
                        proto_stats: dict) -> list[dict]:
    """
    For each infoset × each rule-based opponent prototype, compute:
      - v2a_pred[opp] : V = V_base + Δ(s, stats)
      - v2b_pred[opp] : V = V_base + g(s,stats)*Δ(s,stats)
      - v2a_mod[opp]  : modulation = V - V_base
      - v2b_mod[opp]  : modulation = V - V_base
      - gate[opp]     : gate activation (v2b only)
      - v_base        : frozen base value
    Returns list parallel to rows.
    """
    import torch
    results = []
    for row in rows:
        obs      = build_obs(row)
        cp       = row["current_player"]
        game_enc = v2a_agent._encode_game(obs, cp).unsqueeze(0)  # (1, 15)

        with torch.no_grad():
            v_base = v2a_agent.base(game_enc).item()

        v2a_pred, v2b_pred = {}, {}
        v2a_mod,  v2b_mod  = {}, {}
        gate_vals           = {}

        for opp in MODEL_ORDER:
            stats = proto_stats[opp]                          # list[7]
            stats_t = torch.tensor(stats, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                va = v2a_agent.compute_value(game_enc, stats_t).item()
                vb = v2b_agent.compute_value(game_enc, stats_t).item()
                g  = v2b_agent.get_gate_value(game_enc, stats_t)

            v2a_pred[opp] = va
            v2b_pred[opp] = vb
            v2a_mod[opp]  = va - v_base
            v2b_mod[opp]  = vb - v_base
            gate_vals[opp] = g

        results.append({
            "v_base":   v_base,
            "v2a_pred": v2a_pred,
            "v2b_pred": v2b_pred,
            "v2a_mod":  v2a_mod,
            "v2b_mod":  v2b_mod,
            "gate":     gate_vals,
            # derived spread metrics
            "v2a_spread": max(v2a_pred.values()) - min(v2a_pred.values()),
            "v2b_spread": max(v2b_pred.values()) - min(v2b_pred.values()),
            "gate_mean":  float(np.mean(list(gate_vals.values()))),
            "gate_std":   float(np.std(list(gate_vals.values()))),
        })
    return results


# ── section 1: tournament evaluation ─────────────────────────────────────────

def run_tournament(v2a_agent, v2b_agent, pool_means: dict) -> dict:
    opponents = build_standard_opponents(ROOT)
    all_keys  = list(STANDARD_OPPONENT_KEYS)

    results = {}
    for variant_name, agent in [("v2a_ungated", v2a_agent),
                                 ("v2b_state_gated", v2b_agent)]:
        print(f"\n  [{variant_name}] {TOURNAMENT_ROUNDS} rounds/opponent ...")
        t0     = time.time()
        result = evaluate_stat_aware_pool(
            agent=agent,
            opponents=opponents,
            play_hand_fn=play_hand_v2,
            pool_means=pool_means,
            num_rounds=TOURNAMENT_ROUNDS,
            session_length=SESSION_LENGTH,
            prior_strength=PRIOR_STRENGTH,
            opponent_keys=all_keys,
            alternate_positions=True,
        )
        elapsed = time.time() - t0
        scores  = result["scores"]
        summary = result["summary"]

        # 95% CI: game std ≈ 1.5 chips/round (empirical), n = TOURNAMENT_ROUNDS * 2 seats
        n_total  = TOURNAMENT_ROUNDS
        game_std = 1.5   # chips/round (typical Leduc variance)
        ci_95    = 1.96 * game_std / np.sqrt(n_total)

        results[variant_name] = {
            "scores":    scores,
            "summary":   {k: round(v, 4) for k, v in summary.items()
                          if k != "metric_values"},
            "ci_95_approx": round(ci_95, 4),
            "elapsed_s": round(elapsed, 1),
        }
        print(f"    Done in {elapsed:.0f}s | {format_pool_summary(summary)} "
              f"| ±{ci_95:.3f} (95% CI approx.)")

    return results


def plot_tournament(tournament: dict, out_path: str):
    """Bar chart comparing all variants across all opponents."""
    all_keys = list(STANDARD_OPPONENT_KEYS)
    x = np.arange(len(all_keys))
    width = 0.18

    fig, ax = plt.subplots(figsize=(15, 6))

    ref_data = [
        ("baseline_v1", BASELINES_KNOWN["baseline_value_v1"], "#b0b0b0", "//"),
        ("v1_a_td",     BASELINES_KNOWN["v1_a_td"],           "#808080", "\\\\"),
    ]
    for i, (lbl, scores, color, hatch) in enumerate(ref_data):
        vals = [scores.get(k, 0) for k in all_keys]
        ax.bar(x + (i - 1.5) * width, vals, width * 0.9,
               label=lbl, color=color, alpha=0.5, hatch=hatch, edgecolor="white")

    variant_data = [
        ("v2a_ungated",      COLOR_V2A),
        ("v2b_state_gated",  COLOR_V2B),
    ]
    for i, (key, color) in enumerate(variant_data):
        scores = tournament[key]["scores"]
        ci     = tournament[key]["ci_95_approx"]
        vals   = [scores.get(k, 0) for k in all_keys]
        yerr   = [ci] * len(all_keys)
        ax.bar(x + (i + 0.5) * width, vals, width * 0.9,
               label=key, color=color, alpha=0.80, edgecolor="white")
        ax.errorbar(x + (i + 0.5) * width, vals, yerr=yerr,
                    fmt="none", color="black", capsize=3, linewidth=1.2)

    ax.axhline(0, color="black", lw=0.7, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Avg chips / round")
    ax.set_title(
        f"Tournament Evaluation — opp_stats_modulation_v2\n"
        f"({TOURNAMENT_ROUNDS:,} rounds per opponent, ±95% CI bars on v2 variants)",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, framealpha=0.9)

    # Annotate avg scores
    for key, color in variant_data:
        avg = tournament[key]["summary"]["avg"]
        rob = tournament[key]["summary"]["robustness"]
        ci  = tournament[key]["ci_95_approx"]
        ax.text(0.01, 0.98 - (0.08 * variant_data.index((key, color))),
                f"{key}: avg={avg:+.3f}  rob={rob:+.3f}  ±{ci:.3f}",
                transform=ax.transAxes, fontsize=8.5, color=color,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, alpha=0.85))

    fig.tight_layout()
    _save(out_path, fig)


# ── section 2: EV spread overlay ─────────────────────────────────────────────

def plot_ev_spread_overlay(rows: list[dict], preds: list[dict], out_path: str):
    """
    Mirrors infoset_value_spread_overlay.py, extended with v2a and v2b bands.

    Top panel: value predictions vs BR net range
      - Gray band:   BR min–max
      - Black dots:  value_based (frozen base) prediction
      - Orange band: v2a prediction range across 6 prototype stats
      - Teal band:   v2b prediction range across 6 prototype stats
    Bottom panel: spread bars
      - Blue/green:  BR spread (by round)
      - Orange bars: v2a prediction spread
      - Teal bars:   v2b prediction spread
    """
    n           = len(rows)
    x           = np.arange(n)
    br_min      = np.array([r["br_min"]          for r in rows])
    br_max      = np.array([r["br_max"]          for r in rows])
    br_spread   = np.array([r["spread"]          for r in rows])
    baseline    = np.array([r["baseline_value"]  for r in rows])
    round_ids   = np.array([r["round"]           for r in rows])

    v2a_min = np.array([min(p["v2a_pred"].values()) for p in preds])
    v2a_max = np.array([max(p["v2a_pred"].values()) for p in preds])
    v2b_min = np.array([min(p["v2b_pred"].values()) for p in preds])
    v2b_max = np.array([max(p["v2b_pred"].values()) for p in preds])
    v2a_sp  = np.array([p["v2a_spread"] for p in preds])
    v2b_sp  = np.array([p["v2b_spread"] for p in preds])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(22, 11),
        gridspec_kw={"height_ratios": [3.2, 1.2]},
    )

    # ---- Background shading by round ----------------------------------------
    for i in range(n):
        shade = ROUND_SHADE[int(round_ids[i])]
        ax1.axvspan(i - 0.5, i + 0.5, color=shade, alpha=0.22, linewidth=0)

    # ---- BR net range band ---------------------------------------------------
    ax1.fill_between(x, br_min, br_max, color=COLOR_BR, alpha=0.18, zorder=1)
    ax1.plot(x, br_min, color=COLOR_BR, linewidth=0.7, alpha=0.7, zorder=2)
    ax1.plot(x, br_max, color=COLOR_BR, linewidth=0.7, alpha=0.7, zorder=2)

    # ---- v2a band (orange) ---------------------------------------------------
    ax1.fill_between(x, v2a_min, v2a_max,
                     color=COLOR_V2A, alpha=0.22, zorder=3)
    ax1.plot(x, v2a_min, color=COLOR_V2A, linewidth=0.8, alpha=0.6, zorder=4)
    ax1.plot(x, v2a_max, color=COLOR_V2A, linewidth=0.8, alpha=0.6, zorder=4)

    # ---- v2b band (teal) -----------------------------------------------------
    ax1.fill_between(x, v2b_min, v2b_max,
                     color=COLOR_V2B, alpha=0.22, zorder=5)
    ax1.plot(x, v2b_min, color=COLOR_V2B, linewidth=0.8, alpha=0.6, zorder=6)
    ax1.plot(x, v2b_max, color=COLOR_V2B, linewidth=0.8, alpha=0.6, zorder=6)

    # ---- Per-opponent prediction markers ------------------------------------
    for opp in MODEL_ORDER:
        v2a_vals = np.array([p["v2a_pred"][opp] for p in preds])
        v2b_vals = np.array([p["v2b_pred"][opp] for p in preds])
        ax1.scatter(x, v2a_vals, s=6, marker="o", color=MODEL_COLORS[opp],
                    alpha=0.55, zorder=7)
        ax1.scatter(x, v2b_vals, s=6, marker="x", linewidths=0.7,
                    color=MODEL_COLORS[opp], alpha=0.55, zorder=8)

    # ---- Frozen base (black dots) -------------------------------------------
    ax1.scatter(x, baseline, s=18, color="black", zorder=9, label="Frozen base (V_base)")

    ax1.axhline(0, color="#888888", linewidth=0.6, linestyle="--", alpha=0.5)
    ax1.set_xlim(-1, n)
    ax1.set_xticks([])
    ax1.set_ylabel("Value prediction V(obs) [chips]")

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_BR, alpha=0.3,
                       label="BR net range (min–max across 6 opp-specific nets)"),
        mpatches.Patch(facecolor=COLOR_V2A, alpha=0.35,
                       label="v2a ungated pred range (across 6 prototype stats)"),
        mpatches.Patch(facecolor=COLOR_V2B, alpha=0.35,
                       label="v2b gated pred range (across 6 prototype stats)"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor="black", label="Frozen base (V_base)"),
        mpatches.Patch(facecolor=ROUND_SHADE[0], alpha=0.8, label="Pre-flop"),
        mpatches.Patch(facecolor=ROUND_SHADE[1], alpha=0.8, label="Flop"),
    ]
    for opp in MODEL_ORDER:
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                   color=MODEL_COLORS[opp], label=f"v2a — {MODEL_LABELS[opp]}"))
    ax1.legend(handles=legend_handles, loc="upper left",
               fontsize=7, ncol=3, framealpha=0.95, edgecolor="#cccccc")

    # ---- Lower panel: spread bars -------------------------------------------
    spread_colors = ["#4E79A7" if r == 0 else "#59A14F" for r in round_ids]
    ax2.bar(x, br_spread, width=0.9, color=spread_colors, alpha=0.60, label="BR net spread")
    ax2.bar(x, v2a_sp,   width=0.55, color=COLOR_V2A, alpha=0.70, label="v2a pred spread")
    ax2.bar(x, v2b_sp,   width=0.30, color=COLOR_V2B, alpha=0.80, label="v2b pred spread")

    ax2.set_xlim(-1, n)
    ax2.set_xlabel("Infosets ordered by BR-net spread (ascending)")
    ax2.set_ylabel("Spread [chips]")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Value Prediction Spread: BR Nets vs v2a (Ungated) vs v2b (State-Gated)\n"
        f"{n} reachable Leduc infosets, sorted by BR-net spread. "
        "Bands = prediction range across 6 opponent prototype stats.",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(out_path, fig)

    # Print spread comparison stats
    print(f"\n  Spread comparison (mean across {n} infosets):")
    print(f"    BR net spread:  {br_spread.mean():.4f}  median={np.median(br_spread):.4f}")
    print(f"    v2a pred spread: {v2a_sp.mean():.4f}  median={np.median(v2a_sp):.4f}  "
          f"(ratio to BR: {v2a_sp.mean()/br_spread.mean():.2%})")
    print(f"    v2b pred spread: {v2b_sp.mean():.4f}  median={np.median(v2b_sp):.4f}  "
          f"(ratio to BR: {v2b_sp.mean()/br_spread.mean():.2%})")


# ── section 3: gate analysis ─────────────────────────────────────────────────

def plot_gate_analysis(rows: list[dict], preds: list[dict], out_path: str):
    """
    v2b gate activations per infoset, sorted by BR spread.
    Top panel: per-opponent gate scatter + mean line
    Bottom panel: gate std per infoset
    """
    n        = len(rows)
    x        = np.arange(n)
    round_ids = np.array([r["round"] for r in rows])

    gate_mean = np.array([p["gate_mean"] for p in preds])
    gate_std  = np.array([p["gate_std"]  for p in preds])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(22, 9),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ---- Background shading -------------------------------------------------
    for i in range(n):
        shade = ROUND_SHADE[int(round_ids[i])]
        ax1.axvspan(i - 0.5, i + 0.5, color=shade, alpha=0.22, linewidth=0)

    # ---- Per-opponent gate scatter -------------------------------------------
    for opp in MODEL_ORDER:
        gate_vals = np.array([p["gate"][opp] for p in preds])
        ax1.scatter(x, gate_vals, s=7, alpha=0.55,
                    color=MODEL_COLORS[opp], label=MODEL_LABELS[opp], zorder=3)

    # ---- Mean gate line ------------------------------------------------------
    ax1.plot(x, gate_mean, color="black", lw=1.4, zorder=5, label="Mean gate (across 6 opps)")
    ax1.fill_between(x, gate_mean - gate_std, gate_mean + gate_std,
                     color="black", alpha=0.12, zorder=4, label="±1 std")

    ax1.axhline(0.5, color="gray", lw=0.9, linestyle="--", alpha=0.6,
                label="Neutral gate = 0.5")
    ax1.set_ylim(0.35, 0.75)
    ax1.set_xlim(-1, n)
    ax1.set_xticks([])
    ax1.set_ylabel("Gate activation g(s, opp_stats) ∈ [0, 1]")

    legend_elems = [Line2D([0], [0], marker="o", linestyle="none",
                           markersize=5, color=MODEL_COLORS[opp],
                           label=MODEL_LABELS[opp]) for opp in MODEL_ORDER]
    legend_elems += [
        Line2D([0], [0], color="black", lw=1.4, label="Mean gate"),
        mpatches.Patch(facecolor="black", alpha=0.15, label="±1 std"),
        Line2D([0], [0], color="gray", linestyle="--", lw=0.9, label="Neutral (0.5)"),
        mpatches.Patch(facecolor=ROUND_SHADE[0], alpha=0.8, label="Pre-flop"),
        mpatches.Patch(facecolor=ROUND_SHADE[1], alpha=0.8, label="Flop"),
    ]
    ax1.legend(handles=legend_elems, loc="upper left",
               fontsize=7.5, ncol=4, framealpha=0.95, edgecolor="#cccccc")

    # ---- Lower panel: gate std per infoset -----------------------------------
    bar_colors = ["#4E79A7" if r == 0 else "#59A14F" for r in round_ids]
    ax2.bar(x, gate_std, width=0.85, color=bar_colors, alpha=0.7)
    ax2.set_xlim(-1, n)
    ax2.set_xlabel("Infosets ordered by BR-net spread (ascending)")
    ax2.set_ylabel("Gate std across 6 opponents")
    ax2.set_title("Per-Infoset Gate Standard Deviation", fontsize=9)

    # Print summary stats
    print(f"\n  Gate analysis ({n} infosets):")
    print(f"    Gate mean:  {gate_mean.mean():.4f}  ± {gate_mean.std():.4f}  "
          f"[{gate_mean.min():.4f}, {gate_mean.max():.4f}]")
    print(f"    Gate std:   {gate_std.mean():.4f}  (avg cross-opponent variation per infoset)")
    print(f"    Infosets with gate > 0.55:  {(gate_mean > 0.55).sum()} / {n}")
    print(f"    Infosets with gate < 0.45:  {(gate_mean < 0.45).sum()} / {n}")

    fig.suptitle(
        "v2b State-Conditioned Gate Analysis\n"
        f"Gate g(s, opp_stats) per infoset using 6 opponent prototype stats — "
        f"sorted by BR-net spread. Mean near 0.5 = gate learned mild uniform modulation.",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(out_path, fig)


# ── section 4: modulation magnitude ─────────────────────────────────────────

def plot_modulation_magnitude(rows: list[dict], preds: list[dict],
                              out_path: str):
    """
    Per-infoset modulation = V(s, opp_stats) − V_base(s).
    Three panels:
      Top:    v2a modulation per opponent + mean (sorted by BR spread)
      Middle: v2b modulation per opponent + mean
      Bottom: GT EV residual = BR net value − V_base (ground truth for comparison)
    """
    n        = len(rows)
    x        = np.arange(n)
    round_ids = np.array([r["round"] for r in rows])
    v_base    = np.array([p["v_base"] for p in preds])

    # Ground-truth residual: each BR net value minus frozen base
    gt_residual_by_opp = {}
    for opp in MODEL_ORDER:
        gt_vals = np.array([r["br_values"].get(opp, float("nan")) for r in rows])
        gt_residual_by_opp[opp] = gt_vals - v_base
    gt_res_mean = np.nanmean(np.stack(list(gt_residual_by_opp.values())), axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(22, 14),
                             gridspec_kw={"height_ratios": [2.5, 2.5, 2.0]})

    for panel_idx, (ax, variant, mod_key, color, title_str) in enumerate([
        (axes[0], "v2a", "v2a_mod", COLOR_V2A, "v2a Ungated — Modulation Δ(s, opp_stats)"),
        (axes[1], "v2b", "v2b_mod", COLOR_V2B, "v2b Gated — Modulation g(s,stats)×Δ(s,opp_stats)"),
    ]):
        # Background shading
        for i in range(n):
            shade = ROUND_SHADE[int(round_ids[i])]
            ax.axvspan(i - 0.5, i + 0.5, color=shade, alpha=0.22, linewidth=0)

        mod_arrays = np.array([[p[mod_key][opp] for opp in MODEL_ORDER] for p in preds])
        mod_mean = mod_arrays.mean(axis=1)
        mod_min  = mod_arrays.min(axis=1)
        mod_max  = mod_arrays.max(axis=1)

        ax.fill_between(x, mod_min, mod_max, color=color, alpha=0.18, zorder=2)
        for i, opp in enumerate(MODEL_ORDER):
            ax.scatter(x, mod_arrays[:, i], s=7, alpha=0.55,
                       color=MODEL_COLORS[opp], zorder=3)
        ax.plot(x, mod_mean, color=color, lw=1.5, zorder=4,
                label=f"{variant} mean modulation")
        ax.axhline(0, color="black", lw=0.6, linestyle="--", alpha=0.4)
        ax.set_xlim(-1, n)
        ax.set_xticks([])
        ax.set_ylabel("Modulation [chips]")
        ax.set_title(title_str, fontsize=9)

        # Print per-variant stats
        abs_mean = np.abs(mod_mean).mean()
        print(f"\n  {variant} modulation magnitude:")
        print(f"    Mean |modulation| (avg over infosets): {abs_mean:.4f}")
        print(f"    Cross-opp spread: {(mod_max - mod_min).mean():.4f}")
        print(f"    Modulation range: [{mod_min.min():.3f}, {mod_max.max():.3f}]")

        per_opp_means = {opp: float(np.mean(mod_arrays[:, i]))
                         for i, opp in enumerate(MODEL_ORDER)}
        print(f"    Per-opponent mean modulation:")
        for opp, val in per_opp_means.items():
            print(f"      {opp:<22} {val:+.4f}")

        # Legend
        handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                          color=MODEL_COLORS[opp], label=MODEL_LABELS[opp])
                   for opp in MODEL_ORDER]
        handles += [
            Line2D([0], [0], color=color, lw=1.5, label=f"{variant} mean modulation"),
            mpatches.Patch(facecolor=color, alpha=0.25, label="min–max range"),
        ]
        ax.legend(handles=handles, loc="upper left",
                  fontsize=7.5, ncol=4, framealpha=0.95)

    # Ground-truth residual panel
    ax3 = axes[2]
    for i in range(n):
        ax3.axvspan(i - 0.5, i + 0.5, color=ROUND_SHADE[round_ids[i]],
                    alpha=0.22, linewidth=0)
    for i, opp in enumerate(MODEL_ORDER):
        ax3.scatter(x, gt_residual_by_opp[opp], s=7, alpha=0.55,
                    color=MODEL_COLORS[opp], zorder=3)
    ax3.plot(x, gt_res_mean, color="#444444", lw=1.5, zorder=4,
             label="GT mean residual (BR value − V_base)")
    ax3.fill_between(x,
                     np.nanmin(np.stack(list(gt_residual_by_opp.values())), axis=0),
                     np.nanmax(np.stack(list(gt_residual_by_opp.values())), axis=0),
                     color="#444444", alpha=0.12, zorder=2)
    ax3.axhline(0, color="black", lw=0.6, linestyle="--", alpha=0.4)
    ax3.set_xlim(-1, n)
    ax3.set_xlabel("Infosets ordered by BR-net spread (ascending)")
    ax3.set_ylabel("EV residual [chips]")
    ax3.set_title("Ground Truth: EV Residual = BR Net Value − V_base (ideal modulation target)",
                  fontsize=9)

    gt_res_abs = np.abs(gt_res_mean).mean()
    print(f"\n  Ground-truth residual (target for modulation head):")
    print(f"    Mean |GT residual|: {gt_res_abs:.4f}")
    print(f"    (v2a captures {np.abs(np.array([p['v2a_mod']['tight_passive'] for p in preds])).mean()/gt_res_abs:.1%} of GT residual magnitude)")

    handles3 = [Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                       color=MODEL_COLORS[opp], label=MODEL_LABELS[opp])
                for opp in MODEL_ORDER]
    handles3.append(Line2D([0], [0], color="#444444", lw=1.5,
                            label="GT mean residual"))
    ax3.legend(handles=handles3, loc="upper left",
               fontsize=7.5, ncol=4, framealpha=0.95)

    fig.suptitle(
        "Modulation Magnitude Analysis — v2a vs v2b\n"
        "Δ added on top of frozen V_base per infoset per opponent (sorted by BR-net spread).",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(out_path, fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main(skip_tournament: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load agents ───────────────────────────────────────────────────────────
    print("Loading agents ...")
    v2a = UngatedModAgent()
    v2a.load_model(V2A_CKPT)
    v2a.set_train_mode(False)

    v2b = StateGatedModAgent()
    v2b.load_model(V2B_CKPT)
    v2b.set_train_mode(False)
    print(f"  v2a from: {V2A_CKPT}")
    print(f"  v2b from: {V2B_CKPT}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading infoset CSV ...")
    rows = load_csv(CSV_PATH)
    print(f"  {len(rows)} infosets")

    print("Loading prototype stats ...")
    with open(PROTO_PATH) as f:
        proto_raw = json.load(f)
    proto_stats = {k: proto_raw[k] for k in MODEL_ORDER}  # 7-dim lists

    # ── Compute predictions for all infosets ──────────────────────────────────
    print("\nComputing per-infoset predictions ...")
    preds = compute_predictions(rows, v2a, v2b, proto_stats)
    print(f"  Done ({len(preds)} infosets).")

    # ── Section 1: Tournament ────────────────────────────────────────────────
    if not skip_tournament:
        print(f"\n{'='*60}")
        print(f"SECTION 1 — Tournament Evaluation ({TOURNAMENT_ROUNDS:,} rounds/opp)")
        print(f"{'='*60}")
        with open(PRIORS_PATH) as f:
            pool_means = json.load(f)
        tournament = run_tournament(v2a, v2b, pool_means)
        _write_json(os.path.join(OUT_DIR, "tournament.json"), tournament)
        print("\nGenerating tournament figure ...")
        plot_tournament(tournament, os.path.join(OUT_DIR, "fig0_tournament.png"))
    else:
        print("\n[Skipping tournament evaluation]")
        tournament = None

    # ── Section 2: EV Spread Overlay ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SECTION 2 — EV Spread Overlay")
    print(f"{'='*60}")
    plot_ev_spread_overlay(rows, preds,
                           os.path.join(OUT_DIR, "fig1_ev_spread_overlay.png"))

    # ── Section 3: Gate Analysis ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SECTION 3 — Gate Analysis (v2b)")
    print(f"{'='*60}")
    plot_gate_analysis(rows, preds,
                       os.path.join(OUT_DIR, "fig2_gate_analysis.png"))

    # ── Section 4: Modulation Magnitude ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("SECTION 4 — Modulation Magnitude")
    print(f"{'='*60}")
    plot_modulation_magnitude(rows, preds,
                              os.path.join(OUT_DIR, "fig3_modulation_magnitude.png"))

    print(f"\n{'='*60}")
    print(f"All outputs saved to {OUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tournament", action="store_true",
                        help="Skip the 10K-round tournament evaluation")
    args = parser.parse_args()
    main(skip_tournament=args.no_tournament)
