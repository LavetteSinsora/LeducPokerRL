"""
plot_overlay.py — Overlay ModulatedValue predictions on EV variation figures.

Produces 3 publication-ready figures comparing:
  - Ground truth EV (MC rollout, from data.json)
  - ValueBased network prediction (no opponent info)
  - ModulatedValue network prediction (conditioned on each opponent's stats)

Figures:
  figA_ev_spread_comparison.png  — State-level EV std: ground truth vs modulated predictions
  figB_high_variance_overlay.png — Top high-variance states: bars + prediction markers
  figC_prediction_range.png      — All states sorted by CFR EV: envelope comparison

Run from project root:
    python -m preliminary_experiments.ev_variation_extras.modulated_value_agent_analysis.plot_overlay
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats as scipy_stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

DATA_PATH   = os.path.join(ROOT, "paper", "ev_analysis", "data.json")
PRED_PATH   = os.path.join(HERE, "predictions.json")
OUT_DIR     = HERE
DPI         = 150

RULE_BASED_KEYS = [
    "tight_passive",
    "tight_aggressive",
    "loose_passive",
    "loose_aggressive",
    "maniac",
    "random",
]

OPPONENT_LABELS = {
    "tight_passive":    "Tight Passive",
    "tight_aggressive": "Tight Aggressive",
    "loose_passive":    "Loose Passive",
    "loose_aggressive": "Loose Aggressive",
    "maniac":           "Maniac",
    "random":           "Random",
}

# Consistent color palette (ColorBrewer-inspired, same order as analyze.py)
_PALETTE = [
    '#1f77b4',   # tight_passive    — blue
    '#d62728',   # tight_aggressive — red
    '#2ca02c',   # loose_passive    — green
    '#ff7f0e',   # loose_aggressive — orange
    '#9467bd',   # maniac           — purple
    '#8c564b',   # random           — brown
]
OPP_COLORS = {k: _PALETTE[i] for i, k in enumerate(RULE_BASED_KEYS)}

COLOR_GT        = '#555555'   # ground truth bars — dark gray
COLOR_VB        = '#e377c2'   # value_based prediction — pink
COLOR_MOD       = '#17becf'   # modulated prediction envelope — teal


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_ground_truth(path: str) -> tuple:
    """
    Returns:
      matrix[state_id][opp_key] = {"ev": float, "ev_std": float, "n": int}
      meta[state_id]             = {round, hand0, hand1, board, pot, current_player, raises}
    """
    with open(path) as f:
        data = json.load(f)
    matrix, meta = {}, {}
    for rec in data["records"]:
        sid = rec["state_id"]
        if sid not in matrix:
            matrix[sid] = {}
            meta[sid] = {k: rec[k] for k in
                         ["round", "hand0", "hand1", "board", "pot", "current_player", "raises"]}
        matrix[sid][rec["opponent"]] = {
            "ev": rec["ev"], "ev_std": rec["ev_std"], "n": rec["n"]
        }
    return matrix, meta


def load_predictions(path: str) -> dict:
    """
    Returns:
      preds[state_id][opp_key] = {"modulated_pred": float, "value_based_pred": float}
    """
    with open(path) as f:
        data = json.load(f)
    preds = {}
    for rec in data["records"]:
        sid = rec["state_id"]
        if sid not in preds:
            preds[sid] = {}
        preds[sid][rec["opponent"]] = {
            "modulated_pred":   rec["modulated_pred"],
            "value_based_pred": rec["value_based_pred"],
        }
    return preds


def _state_label(m: dict) -> str:
    rnd   = "Pre" if m["round"] == 0 else "Flop"
    board = m["board"] if m["board"] else "—"
    return f"{rnd} {m['hand0']}v{m['hand1']}\n{board} [{m['pot'][0]},{m['pot'][1]}]"


def _save(path: str):
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Derived statistics
# ---------------------------------------------------------------------------

def compute_per_state_stats(gt_matrix, pred_matrix, meta):
    """
    For each state compute:
      - gt_ev[opp]          ground truth EV per opponent (rule-based 6)
      - gt_std_cross        std of gt_ev across the 6 rule-based opponents
      - mod_pred[opp]       modulated prediction per opponent
      - mod_std_cross       std of mod_pred across the 6 opponents
      - vb_pred             value_based prediction (constant across opponents)
      - gt_ev_cfr           ground truth EV vs CFR
    """
    results = []
    for sid in gt_matrix:
        gt_row   = gt_matrix[sid]
        pred_row = pred_matrix.get(sid, {})
        m        = meta[sid]

        gt_evs   = [gt_row[k]["ev"] for k in RULE_BASED_KEYS if k in gt_row]
        if len(gt_evs) < 2:
            continue

        mod_preds = [pred_row[k]["modulated_pred"] for k in RULE_BASED_KEYS if k in pred_row]
        vb_pred   = pred_row[RULE_BASED_KEYS[0]]["value_based_pred"] if pred_row else float("nan")

        results.append({
            "state_id":        sid,
            "meta":            m,
            "gt_ev":           {k: gt_row[k]["ev"]   for k in RULE_BASED_KEYS if k in gt_row},
            "gt_ev_std_cross": float(np.std(gt_evs, ddof=0)),
            "mod_pred":        {k: pred_row[k]["modulated_pred"]   for k in RULE_BASED_KEYS if k in pred_row},
            "mod_std_cross":   float(np.std(mod_preds, ddof=0)) if len(mod_preds) >= 2 else float("nan"),
            "vb_pred":         vb_pred,
            "gt_ev_cfr":       gt_row.get("cfr", {}).get("ev", float("nan")),
        })

    results.sort(key=lambda x: x["gt_ev_std_cross"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Fig A: EV spread scatter + sorted line comparison
# ---------------------------------------------------------------------------

def plot_figA(state_stats: list, out_path: str):
    """
    Two-panel figure:
      Left:  Scatter of ground truth EV std vs modulated prediction std per state.
      Right: States sorted by GT std; two lines show GT std and mod-pred std.
    """
    gt_stds  = np.array([s["gt_ev_std_cross"] for s in state_stats])
    mod_stds = np.array([s["mod_std_cross"]   for s in state_stats])

    # Pearson + Spearman correlations
    pearson_r,  pval_p = scipy_stats.pearsonr(gt_stds, mod_stds)
    spearman_r, pval_s = scipy_stats.spearmanr(gt_stds, mod_stds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: scatter ---
    ax = axes[0]
    ax.scatter(gt_stds, mod_stds, s=12, alpha=0.45, color=COLOR_MOD, edgecolors="none")
    lims = [min(gt_stds.min(), mod_stds.min()) - 0.02,
            max(gt_stds.max(), mod_stds.max()) + 0.02]
    ax.plot(lims, lims, color="gray", lw=0.8, linestyle="--", label="y = x (ideal)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Ground Truth EV σ (across 6 rule-based opponents)")
    ax.set_ylabel("Modulated Prediction σ (across 6 opponents' stats)")
    ax.set_title(
        f"EV Spread: Ground Truth vs Modulated Prediction\n"
        f"Pearson r = {pearson_r:.3f}  |  Spearman ρ = {spearman_r:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    # --- Right: sorted lines ---
    ax2 = axes[1]
    x   = np.arange(len(state_stats))
    ax2.plot(x, gt_stds,  color=COLOR_GT,  lw=1.2, label="Ground Truth σ (MC rollout)")
    ax2.plot(x, mod_stds, color=COLOR_MOD, lw=1.2, linestyle="--",
             label="Modulated Prediction σ")
    ax2.fill_between(x, np.minimum(gt_stds, mod_stds), np.maximum(gt_stds, mod_stds),
                     alpha=0.15, color="gray", label="Gap region")
    ax2.set_xlabel("State (sorted by Ground Truth σ, descending)")
    ax2.set_ylabel("EV σ across opponents (chips)")
    ax2.set_title("Variance Spread: GT vs Modulated (sorted)", fontsize=10)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    _save(out_path)

    print(f"    Pearson r = {pearson_r:.4f} (p={pval_p:.2e}), "
          f"Spearman ρ = {spearman_r:.4f} (p={pval_s:.2e})")


# ---------------------------------------------------------------------------
# Fig B: High-variance states — bars + prediction markers
# ---------------------------------------------------------------------------

def plot_figB(state_stats: list, out_path: str, n_show: int = 7):
    """
    For the top n_show states by ground truth EV σ:
      - Grouped bars: ground truth EV per opponent
      - Diamond markers: modulated_value prediction per opponent
      - Horizontal dashed line: value_based baseline prediction
    """
    selected  = state_stats[:n_show]
    n_states  = len(selected)
    n_opp     = len(RULE_BASED_KEYS)
    bar_w     = 0.75 / n_opp
    x         = np.arange(n_states)

    fig, ax = plt.subplots(figsize=(max(13, n_states * 1.8), 6))

    for i, opp_key in enumerate(RULE_BASED_KEYS):
        gt_evs = [s["gt_ev"].get(opp_key, float("nan")) for s in selected]
        offsets = x + (i - n_opp / 2 + 0.5) * bar_w

        # Ground truth bars
        ax.bar(
            offsets, gt_evs, bar_w * 0.88,
            color=OPP_COLORS[opp_key], alpha=0.60,
            label=f"GT: {OPPONENT_LABELS[opp_key]}",
        )

        # Modulated prediction markers (diamonds)
        mod_preds = [s["mod_pred"].get(opp_key, float("nan")) for s in selected]
        ax.scatter(
            offsets, mod_preds,
            marker="D", s=40, color=OPP_COLORS[opp_key],
            edgecolors="black", linewidths=0.6, zorder=5,
        )

    # Value-based baseline (constant per state, no opponent conditioning)
    for j, s in enumerate(selected):
        vb = s["vb_pred"]
        ax.hlines(
            vb, j - 0.4, j + 0.4,
            colors=COLOR_VB, linewidths=1.8, linestyles="--", zorder=6,
        )

    ax.axhline(0, color="black", lw=0.7, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([_state_label(s["meta"]) for s in selected],
                       fontsize=7.5, rotation=20, ha="right")
    ax.set_xlabel("Game State (hand0 vs hand1 | board | pot)")
    ax.set_ylabel("EV / Predicted Value (chips)")
    ax.set_title(
        f"High-Variance States: Ground Truth EV vs Network Predictions\n"
        f"Bars = GT EV (MC rollout) · Diamonds = Modulated Prediction · Dashed = ValueBased Baseline",
        fontsize=10,
    )

    # Legend: one entry per opponent (GT bars only, to avoid clutter)
    gt_handles = [mpatches.Patch(color=OPP_COLORS[k], alpha=0.6,
                                 label=OPPONENT_LABELS[k]) for k in RULE_BASED_KEYS]
    vb_handle  = plt.Line2D([0], [0], color=COLOR_VB, lw=1.8, linestyle="--",
                             label="ValueBased Prediction")
    mod_handle = plt.Line2D([0], [0], marker="D", color="black", lw=0, markersize=6,
                             label="Modulated Prediction")
    ax.legend(handles=gt_handles + [vb_handle, mod_handle],
              loc="upper right", fontsize=7.5, ncol=2,
              framealpha=0.9, edgecolor="lightgray")

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    _save(out_path)


# ---------------------------------------------------------------------------
# Fig C: Prediction range — all states sorted by GT EV vs CFR
# ---------------------------------------------------------------------------

def plot_figC(state_stats: list, out_path: str):
    """
    States sorted by ground truth EV vs CFR (ascending).
    Shows:
      - Ground truth EV vs CFR (black solid, reference)
      - ValueBased prediction (pink dashed)
      - Modulated prediction envelope: min/max band across 6 opponent stats,
        plus mean modulated prediction (teal solid)
    """
    # Filter states that have CFR ground truth
    filtered = [s for s in state_stats if not np.isnan(s["gt_ev_cfr"])]
    filtered.sort(key=lambda s: s["gt_ev_cfr"])

    x       = np.arange(len(filtered))
    gt_cfr  = np.array([s["gt_ev_cfr"] for s in filtered])
    vb_pred = np.array([s["vb_pred"]   for s in filtered])

    mod_vals = np.array([
        [s["mod_pred"].get(k, float("nan")) for k in RULE_BASED_KEYS]
        for s in filtered
    ])  # shape: (n_states, 6)

    mod_mean = np.nanmean(mod_vals, axis=1)
    mod_min  = np.nanmin(mod_vals,  axis=1)
    mod_max  = np.nanmax(mod_vals,  axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Ground truth vs CFR
    ax.plot(x, gt_cfr, color=COLOR_GT, lw=1.5, label="GT EV vs CFR (ground truth)", zorder=4)

    # ValueBased prediction
    ax.plot(x, vb_pred, color=COLOR_VB, lw=1.2, linestyle="--",
            label="ValueBased prediction (no stats)", zorder=3)

    # Modulated: mean line + min/max shaded envelope
    ax.fill_between(x, mod_min, mod_max,
                    color=COLOR_MOD, alpha=0.22,
                    label="Modulated pred range (min–max across 6 opp stats)")
    ax.plot(x, mod_mean, color=COLOR_MOD, lw=1.4,
            label="Modulated pred mean (across 6 opp stats)", zorder=3)

    # Per-opponent modulated lines (thin, colored)
    for i, k in enumerate(RULE_BASED_KEYS):
        mod_k = mod_vals[:, i]
        ax.plot(x, mod_k, color=OPP_COLORS[k], lw=0.5, alpha=0.5, zorder=2)

    ax.axhline(0, color="black", lw=0.6, linestyle="--", alpha=0.4)
    ax.set_xlabel("State (sorted by GT EV vs CFR, ascending)")
    ax.set_ylabel("EV / Predicted Value (chips)")
    ax.set_title(
        "Prediction Range Comparison: Ground Truth vs ValueBased vs Modulated\n"
        "(thin lines = per-opponent modulated predictions; teal band = min/max envelope)",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, edgecolor="lightgray")
    plt.tight_layout()
    _save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading ground truth from {DATA_PATH}")
    gt_matrix, meta = load_ground_truth(DATA_PATH)

    print(f"Loading predictions from {PRED_PATH}")
    pred_matrix = load_predictions(PRED_PATH)

    print("Computing per-state statistics ...")
    state_stats = compute_per_state_stats(gt_matrix, pred_matrix, meta)
    print(f"  {len(state_stats)} states processed.")

    gt_stds  = [s["gt_ev_std_cross"] for s in state_stats]
    mod_stds = [s["mod_std_cross"]   for s in state_stats]
    print(f"\n  GT EV σ (cross-opp):         mean={np.mean(gt_stds):.4f}  "
          f"median={np.median(gt_stds):.4f}")
    print(f"  Modulated Pred σ (cross-opp): mean={np.mean(mod_stds):.4f}  "
          f"median={np.median(mod_stds):.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nGenerating figures → {OUT_DIR}/")

    print("  [Fig A] EV spread comparison ...")
    plot_figA(state_stats,
              out_path=os.path.join(OUT_DIR, "figA_ev_spread_comparison.png"))

    print("  [Fig B] High-variance overlay ...")
    plot_figB(state_stats,
              out_path=os.path.join(OUT_DIR, "figB_high_variance_overlay.png"))

    print("  [Fig C] Prediction range comparison ...")
    plot_figC(state_stats,
              out_path=os.path.join(OUT_DIR, "figC_prediction_range.png"))

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
