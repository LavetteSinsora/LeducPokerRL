"""
policy_sensitivity.py — Policy & Value Sensitivity to Opponent Embedding

For each of the 486 game states in EV_variation_analysis, inject the state
into the game engine, then query StatAugValueAgent (pool_random, 300K ckpt)
with each of the 6 rule-based opponent archetype embeddings from
shared_data/opponent_prototype_stats.json.

Compares:
  - Real EV spread  : max - min EV across archetypes  (EV_variation_analysis ground truth)
  - Model value spread: max - min predicted V(state) across archetype embeddings

Produces:
  sensitivity_data.json   — per-state raw metrics
  overlay_plot.png        — dual-bar: real vs model value spread, sorted by real spread
  calibration_scatter.png — scatter: real spread vs model spread (calibration)
  acr_breakdown.png       — action change rate by game situation
  sensitivity_curves.png  — per-stat sensitivity curves for canonical states
  case_studies.png        — top-5 highest-divergence states, action value bars

Run from project root:
    python -m preliminary_experiments.opp_stats_input_aug.analysis.policy_sensitivity
"""

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from engine.leduc_game import LeducGame, Action
from preliminary_experiments.ev_variation_extras.code.sim_engine import FixedStateSimulator
from preliminary_experiments.opp_stats_input_aug.agent import StatAugValueAgent

# ── Paths ──────────────────────────────────────────────────────────────────────
CKPT_PATH    = os.path.join(HERE, "..", "outputs", "pool_random", "checkpoint_best.pt")
EV_DATA_PATH = os.path.join(ROOT, "paper", "ev_analysis", "data.json")
PROTO_PATH   = os.path.join(ROOT, "paper", "evaluation", "shared", "data", "opponent_prototype_stats.json")
OUT_DIR      = os.path.join(HERE, "outputs")

# 6 rule-based archetypes (excluding value_based and cfr from prototype stats)
ARCHETYPES = ["tight_passive", "tight_aggressive", "loose_passive",
              "loose_aggressive", "maniac", "random"]

ARCHETYPE_COLORS = {
    "tight_passive":    "#1f77b4",
    "tight_aggressive": "#ff7f0e",
    "loose_passive":    "#9467bd",
    "loose_aggressive": "#e377c2",
    "maniac":           "#8c564b",
    "random":           "#7f7f7f",
}

ACTION_COLORS = {"FOLD": "#d62728", "CALL": "#2ca02c", "RAISE": "#1f77b4"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _get_obs(record):
    """Inject recorded state into LeducGame; return observation for current_player."""
    game = LeducGame()
    sim  = FixedStateSimulator(
        hand0=record["hand0"],
        hand1=record["hand1"],
        pot=record["pot"],
        current_player=record["current_player"],
        rnd=record["round"],
        raises=record["raises"],
        board=record["board"],
    )
    sim._inject(game)
    return game.get_observation(viewer_id=record["current_player"])


def _model_best_value(agent, obs, opp_stats_vec):
    """Max value across legal actions for given opponent stats."""
    evals = agent.get_action_evaluations(obs, opp_stats_vec)
    if not evals:
        return None, None
    best = max(evals, key=lambda e: e["value"])
    action_vals = {e["action"].name: round(float(e["value"]), 5) for e in evals}
    return best["action"].name, action_vals


# ── Core: per-state analysis ───────────────────────────────────────────────────

def run_analysis(agent, ev_data, proto_stats):
    """
    For each unique state in ev_data, compute:
      - real_ev_spread   : max - min EV across 6 archetypes (ground truth)
      - model_value_spread: max - min of max_a V(s',opp) across 6 archetypes
      - best_actions     : argmax action per archetype
      - action_changes   : bool — does the best action differ across any archetype pair?
      - n_unique_actions : int count of distinct best actions

    Returns list of per-state dicts.
    """
    # Group ev_data records by state_id
    state_records = {}
    for r in ev_data["records"]:
        sid = r["state_id"]
        if sid not in state_records:
            state_records[sid] = {"meta": r, "evs": {}}
        if r["opponent"] in ARCHETYPES:
            state_records[sid]["evs"][r["opponent"]] = r["ev"]

    results = []
    n = len(state_records)
    for i, (sid, info) in enumerate(state_records.items()):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n} states processed...")

        meta = info["meta"]
        real_evs = info["evs"]

        # Real EV spread (6 archetypes)
        if len(real_evs) == 6:
            vals = list(real_evs.values())
            real_spread = round(max(vals) - min(vals), 5)
        else:
            real_spread = None

        # Model values per archetype
        obs = _get_obs(meta)
        model_best_vals = {}   # archetype → max_action_value
        model_best_acts = {}   # archetype → best_action_name
        model_all_vals  = {}   # archetype → {action: value}

        for arch in ARCHETYPES:
            stats_vec = np.array(proto_stats[arch], dtype=np.float32)
            best_act, act_vals = _model_best_value(agent, obs, stats_vec)
            if best_act is not None:
                model_best_acts[arch] = best_act
                model_best_vals[arch] = act_vals[best_act]
                model_all_vals[arch]  = act_vals

        # Model value spread
        if len(model_best_vals) == 6:
            mv = list(model_best_vals.values())
            model_spread = round(max(mv) - min(mv), 5)
        else:
            model_spread = None

        # Action change metrics
        unique_acts = set(model_best_acts.values())
        n_unique = len(unique_acts)
        action_changes = n_unique > 1

        results.append({
            "state_id":        sid,
            "round":           meta["round"],
            "hand0":           meta["hand0"],
            "hand1":           meta["hand1"],
            "board":           meta["board"],
            "pot":             meta["pot"],
            "current_player":  meta["current_player"],
            "raises":          meta["raises"],
            "real_ev":         real_evs,
            "real_ev_spread":  real_spread,
            "model_all_values":     model_all_vals,
            "model_best_values":    model_best_vals,
            "model_best_actions":   model_best_acts,
            "model_value_spread":   model_spread,
            "action_changes":       action_changes,
            "n_unique_actions":     n_unique,
        })

    return results


# ── Per-stat sensitivity curves ────────────────────────────────────────────────

def compute_sensitivity_curves(agent, canonical_states, pool_means_vec, stat_names, proto_stats):
    """
    For each canonical state × stat: vary stat from 0→1, hold others at pool_mean.
    Returns dict: {state_id: {stat_name: {action: [values]}}}
    """
    POOL = np.array(pool_means_vec, dtype=np.float32)  # 7-dim pool mean
    STEPS = np.linspace(0.0, 1.0, 40)
    curves = {}

    for state_id, meta in canonical_states.items():
        obs = _get_obs(meta)
        curves[state_id] = {}

        for stat_idx, stat_name in enumerate(stat_names):
            stat_curves = {}
            for t in STEPS:
                stats_vec = POOL.copy()
                stats_vec[stat_idx] = t
                evals = agent.get_action_evaluations(obs, stats_vec)
                for e in evals:
                    aname = e["action"].name
                    if aname not in stat_curves:
                        stat_curves[aname] = {"x": [], "y": []}
                    stat_curves[aname]["x"].append(float(t))
                    stat_curves[aname]["y"].append(float(e["value"]))
            curves[state_id][stat_name] = stat_curves

    return curves


# ── Figures ────────────────────────────────────────────────────────────────────

def fig_overlay(results, out_path):
    """
    Dual-bar overlay: real EV spread (blue) and model value spread (orange),
    sorted by real EV spread. Red triangles mark states where action changes.
    """
    valid = [r for r in results if r["real_ev_spread"] is not None
             and r["model_value_spread"] is not None]
    valid.sort(key=lambda r: r["real_ev_spread"])

    x      = np.arange(len(valid))
    real_s = np.array([r["real_ev_spread"]   for r in valid])
    mod_s  = np.array([r["model_value_spread"] for r in valid])
    changed = np.array([r["action_changes"] for r in valid], dtype=bool)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8),
                                   gridspec_kw={"height_ratios": [4, 1]},
                                   sharex=True)

    # Top: dual bars
    w = 0.45
    ax1.bar(x - w/2, real_s, w, color="#2196F3", alpha=0.75, label="Real EV spread (ground truth)")
    ax1.bar(x + w/2, mod_s,  w, color="#FF9800", alpha=0.75, label="Model value spread (22-dim)")

    # Red dots where action changes
    change_x = x[changed]
    change_y = np.maximum(real_s[changed], mod_s[changed]) + 0.3
    ax1.scatter(change_x, change_y, color="#d62728", marker="v", s=30, zorder=5,
                label=f"Action changes ({changed.sum()} states)")

    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_ylabel("EV / Value spread (chips)")
    ax1.set_title("Per-state: Real EV Spread vs Model Value Spread\n"
                  "(sorted by real EV spread; ▼ = best action changed across opponent profiles)")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # Bottom: round indicator
    rounds = np.array([r["round"] for r in valid])
    ax2.bar(x, np.ones(len(x)), 1.0,
            color=["#bbdefb" if rd == 0 else "#e8f5e9" for rd in rounds],
            alpha=0.9)
    ax2.set_yticks([])
    ax2.set_ylabel("Round")
    pf_patch = mpatches.Patch(color="#bbdefb", label="Pre-flop")
    fl_patch = mpatches.Patch(color="#e8f5e9", label="Flop")
    ax2.legend(handles=[pf_patch, fl_patch], fontsize=8, loc="upper left")
    ax2.set_xlabel("States (sorted by real EV spread →)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_calibration_scatter(results, out_path):
    """Scatter: real EV spread vs model value spread. Diagonal = perfect calibration."""
    valid = [r for r in results if r["real_ev_spread"] is not None
             and r["model_value_spread"] is not None]

    real_s = np.array([r["real_ev_spread"]    for r in valid])
    mod_s  = np.array([r["model_value_spread"] for r in valid])
    n_unique = np.array([r["n_unique_actions"]  for r in valid])
    rounds    = np.array([r["round"]             for r in valid])

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot pre-flop and flop separately
    for rnd, label, marker in [(0, "Pre-flop", "^"), (1, "Flop", "o")]:
        mask = rounds == rnd
        sc = ax.scatter(real_s[mask], mod_s[mask],
                        c=n_unique[mask], cmap="RdYlGn_r",
                        vmin=1, vmax=4, s=40, marker=marker, alpha=0.7, label=label)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("# unique optimal actions", fontsize=9)
    cbar.set_ticks([1, 2, 3, 4])

    lim = max(real_s.max(), mod_s.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.0, label="y = x (perfect calibration)")
    ax.set_xlabel("Real EV spread (ground truth, chips)")
    ax.set_ylabel("Model value spread (22-dim agent, chips)")
    ax.set_title("Calibration: Does the Model's Sensitivity Match Reality?\n"
                 "(color = # archetypes that disagree on optimal action)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_acr_breakdown(results, out_path):
    """Action change rate broken down by game situation."""
    def acr(subset):
        if not subset: return 0.0
        return sum(r["action_changes"] for r in subset) / len(subset)

    def bar_group(ax, labels, values, title, colors=None):
        x = np.arange(len(labels))
        bars = ax.bar(x, values,
                      color=colors if colors else ["#2196F3"] * len(labels),
                      alpha=0.8)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.005,
                    f"{h:.1%}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, min(1.0, max(values) * 1.3 + 0.05) if values else 0.2)
        ax.set_ylabel("Action change rate")
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Action Change Rate: When Does Opponent Embedding Change the Decision?",
                 fontsize=12, fontweight="bold")

    # By round
    by_round = {"Pre-flop": [r for r in results if r["round"] == 0],
                "Flop":     [r for r in results if r["round"] == 1]}
    bar_group(axes[0,0], list(by_round.keys()),
              [acr(v) for v in by_round.values()],
              "By round", ["#bbdefb", "#e8f5e9"])

    # By raises
    by_raises = {f"raises={k}": [r for r in results if r["raises"] == k]
                 for k in [0, 1, 2]}
    bar_group(axes[0,1], list(by_raises.keys()),
              [acr(v) for v in by_raises.values()],
              "By raises this round")

    # By my hand (current player's hand)
    def cp_hand(r):
        return r["hand0"] if r["current_player"] == 0 else r["hand1"]

    by_hand = {h: [r for r in results if cp_hand(r) == h] for h in ["J", "Q", "K"]}
    bar_group(axes[1,0], list(by_hand.keys()),
              [acr(v) for v in by_hand.values()],
              "By my hole card",
              ["#ffcdd2", "#fff9c4", "#c8e6c9"])

    # By board card (flop only)
    flop = [r for r in results if r["round"] == 1]
    by_board = {b: [r for r in flop if r["board"] == b] for b in ["J", "Q", "K"]}
    bar_group(axes[1,1], list(by_board.keys()),
              [acr(v) for v in by_board.values()],
              "By board card (flop states only)",
              ["#ffcdd2", "#fff9c4", "#c8e6c9"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_sensitivity_curves(curves, canonical_labels, stat_names, out_path):
    """Per-stat sensitivity curves for canonical states."""
    n_states = len(curves)
    n_stats  = len(stat_names)
    fig, axes = plt.subplots(n_states, n_stats,
                             figsize=(4 * n_stats, 3 * n_states),
                             squeeze=False)
    fig.suptitle("Per-Stat Sensitivity Curves\n"
                 "(each stat varied 0→1 while others held at pool mean)",
                 fontsize=11, fontweight="bold")

    for row, (state_id, state_curves) in enumerate(curves.items()):
        for col, stat_name in enumerate(stat_names):
            ax = axes[row][col]
            stat_data = state_curves.get(stat_name, {})
            for action_name, xy in stat_data.items():
                ax.plot(xy["x"], xy["y"],
                        color=ACTION_COLORS.get(action_name, "gray"),
                        linewidth=1.8, label=action_name)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_xlabel(stat_name if row == n_states - 1 else "", fontsize=7)
            ax.set_ylabel("Value" if col == 0 else "", fontsize=8)
            if row == 0:
                ax.set_title(stat_name.replace("_", "\n"), fontsize=8)
            if col == 0:
                ax.set_ylabel(canonical_labels[state_id][:20], fontsize=7)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_case_studies(results, out_path, top_n=5):
    """Top-N states by n_unique_actions: action value bars per archetype."""
    # Rank by n_unique_actions desc, then by max(real_ev_spread) desc
    valid = [r for r in results
             if r["model_all_values"] and r["real_ev_spread"] is not None]
    valid.sort(key=lambda r: (-r["n_unique_actions"], -r["real_ev_spread"]))
    top = valid[:top_n]

    fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 5), squeeze=False)
    fig.suptitle("Case Studies: Top States by Action Disagreement Across Opponent Profiles",
                 fontsize=11, fontweight="bold")

    for col, record in enumerate(top):
        ax = axes[0][col]
        all_vals = record["model_all_values"]
        best_acts = record["model_best_actions"]

        # Collect all action names
        all_actions = set()
        for av in all_vals.values():
            all_actions |= set(av.keys())
        all_actions = sorted(all_actions)

        x = np.arange(len(ARCHETYPES))
        n_actions = len(all_actions)
        width = 0.8 / n_actions

        for i, act in enumerate(all_actions):
            vals = [all_vals.get(arch, {}).get(act, np.nan) for arch in ARCHETYPES]
            offset = (i - n_actions / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, color=ACTION_COLORS.get(act, "gray"),
                   alpha=0.8, label=act)

        # Mark best action per archetype with a star
        for j, arch in enumerate(ARCHETYPES):
            best = best_acts.get(arch)
            if best:
                bval = all_vals.get(arch, {}).get(best, np.nan)
                if not np.isnan(bval):
                    ax.plot(j, bval + 0.05, "*", color="black", markersize=8, zorder=5)

        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        state_id = record["state_id"]
        real_s = record["real_ev_spread"]
        mod_s  = record["model_value_spread"]
        ax.set_title(f"{state_id}\nreal_spread={real_s:.2f}  model_spread={mod_s:.2f}",
                     fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("_", "\n")[:8] for a in ARCHETYPES],
                           fontsize=6, rotation=0)
        ax.set_ylabel("Value (chips)" if col == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading model and data...")
    agent = StatAugValueAgent(model_path=CKPT_PATH)
    agent.set_train_mode(False)

    ev_data    = _load_json(EV_DATA_PATH)
    proto_raw  = _load_json(PROTO_PATH)
    proto_stats = {k: v for k, v in proto_raw.items() if k in ARCHETYPES}

    # Pool means: average of all 6 archetype vectors
    pool_means_vec = np.mean(
        [np.array(proto_stats[a], dtype=np.float32) for a in ARCHETYPES], axis=0
    ).tolist()

    stat_names = proto_raw["_meta"]["feature_order"]  # 7 stat names

    print(f"\nRunning per-state analysis over {ev_data['metadata']['n_states']} states...")
    results = run_analysis(agent, ev_data, proto_stats)

    # Save raw data
    out_data = os.path.join(OUT_DIR, "sensitivity_data.json")
    with open(out_data, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw data → {out_data}")

    # ── Summary stats ──────────────────────────────────────────────────────────
    valid = [r for r in results if r["real_ev_spread"] is not None
             and r["model_value_spread"] is not None]
    n_changed = sum(r["action_changes"] for r in valid)
    print(f"\n{'─'*55}")
    print(f"  States with action changes  : {n_changed}/{len(valid)} ({n_changed/len(valid):.1%})")
    print(f"  Mean real EV spread         : {np.mean([r['real_ev_spread'] for r in valid]):.3f}")
    print(f"  Mean model value spread     : {np.mean([r['model_value_spread'] for r in valid]):.3f}")

    # Correlation between real and model spreads
    real_arr = np.array([r["real_ev_spread"]    for r in valid])
    mod_arr  = np.array([r["model_value_spread"] for r in valid])
    corr     = np.corrcoef(real_arr, mod_arr)[0, 1]
    print(f"  Correlation (real vs model) : {corr:.3f}")
    print(f"{'─'*55}")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    fig_overlay(results, os.path.join(OUT_DIR, "overlay_plot.png"))
    fig_calibration_scatter(results, os.path.join(OUT_DIR, "calibration_scatter.png"))
    fig_acr_breakdown(results, os.path.join(OUT_DIR, "acr_breakdown.png"))
    fig_case_studies(results, os.path.join(OUT_DIR, "case_studies.png"))

    # Canonical states for sensitivity curves:
    # preflop Q (ante only, cp=0), flop Q vs board K (cp=0, raises=0)
    all_metas = {r["state_id"]: r for r in results}
    canonical = {}
    canonical_labels = {}

    for sid, r in all_metas.items():
        cp_hand = r["hand0"] if r["current_player"] == 0 else r["hand1"]
        # Pre-flop, Q, cp=0, raises=0, pot=[1,1]
        if (r["round"] == 0 and cp_hand == "Q"
                and r["raises"] == 0 and r["pot"] == [1, 1]):
            canonical[sid] = r
            canonical_labels[sid] = "Pre-flop Q\npot=[1,1] r=0"
            break

    for sid, r in all_metas.items():
        cp_hand = r["hand0"] if r["current_player"] == 0 else r["hand1"]
        # Flop, Q, board=K, cp=0, raises=0
        if (r["round"] == 1 and cp_hand == "Q" and r["board"] == "K"
                and r["raises"] == 0 and r["pot"] == [3, 3]):
            canonical[sid] = r
            canonical_labels[sid] = "Flop Q (board K)\npot=[3,3] r=0"
            break

    # Vary the 6 rate stats (index 0-5); confidence (index 6) is fixed at 0.96
    sens_stat_names = stat_names[:6]

    if canonical:
        print(f"\nComputing sensitivity curves for {len(canonical)} canonical states...")
        curves = compute_sensitivity_curves(
            agent, canonical, pool_means_vec, sens_stat_names, proto_stats)
        fig_sensitivity_curves(
            curves, canonical_labels, sens_stat_names,
            os.path.join(OUT_DIR, "sensitivity_curves.png"))

    print("\nAll figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
