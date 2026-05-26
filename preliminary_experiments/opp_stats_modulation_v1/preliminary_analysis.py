"""
opp_stats_modulation_v1 — Preliminary EV Residual Analysis
============================================================
Validates that the supervised residual approach (Variant B) has enough signal
before committing to full training.

Three checks:
  1. Base calibration: how well does V_base(s) correlate with EV(s, cfr)?
     (CFR ≈ near-optimal; high correlation means base is well-calibrated.)

  2. Residual magnitude: for each opponent, compute the mean absolute residual
     |EV_infoset(s, opp) − V_base(s)| across all info-set states.
     Large residuals = more signal for the modulation head to learn.

  3. Residual discriminability: are per-opponent residuals distinct enough
     that the modulation head can learn to differentiate them?
     Measured by opponent separation (between-opponent variance of residuals
     vs within-opponent variance).

Usage:
    python preliminary_analysis.py
    python preliminary_analysis.py --save  # write results to outputs/preliminary/

Outputs printed to stdout; optional JSON + plots saved to outputs/preliminary/.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from preliminary_experiments.opp_stats_modulation_v1.agent import StatModValueAgent
from preliminary_experiments.opp_stats_modulation_v1.train import (
    _encode_game_state, _card_removal_probs, OPPONENT_KEYS,
)

CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}


# ── data loading ───────────────────────────────────────────────────────────────

def load_ev_data():
    path = os.path.join(HERE, "..", "EV_variation_analysis", "data.json")
    if not os.path.exists(path):
        print(f"EV data not found: {path}")
        sys.exit(1)
    with open(path) as f:
        raw = json.load(f)
    data = raw["records"] if isinstance(raw, dict) and "records" in raw else raw
    print(f"Loaded {len(data)} EV records from {path}")
    return data


# ── info-set aggregation (mirrors _build_supervised_dataset) ──────────────────

def aggregate_ev_infoset(ev_data):
    """
    Aggregate per-hand EV data to info-set level using card-removal weighting.

    Returns:
        records: list of {
            'state_key': (my_hand, board, pot, cp, raises),
            'opponent':  str,
            'ev_infoset': float,
        }
    """
    # Group by (my_hand, board, pot, cp, raises, opponent)
    groups = {}
    for rec in ev_data:
        cp = rec["current_player"]
        my_hand  = rec["hand0"] if cp == 0 else rec["hand1"]
        opp_hand = rec["hand1"] if cp == 0 else rec["hand0"]
        board    = rec.get("board")
        key = (my_hand, board, tuple(rec["pot"]), cp, rec["raises"], rec["opponent"])
        groups.setdefault(key, {})[opp_hand] = \
            groups.get(key, {}).get(opp_hand, []) + [rec["ev"]]

    records = []
    for (my_hand, board, pot, cp, raises, opp), hand_evs in groups.items():
        probs = _card_removal_probs(my_hand, board)
        ev_infoset = 0.0
        total_w    = 0.0
        for opp_hand, ev_list in hand_evs.items():
            p = probs.get(opp_hand, 0.0)
            ev_infoset += p * (sum(ev_list) / len(ev_list))
            total_w    += p
        if total_w > 0:
            ev_infoset /= total_w

        # minimal obs-like object for encoding
        class _Obs:
            pass
        obs = _Obs()
        obs.player_hand     = my_hand
        obs.board           = board
        obs.pot             = list(pot)
        obs.current_player  = cp
        obs.current_round   = 0 if board is None else 1
        obs.is_finished     = False
        obs.raises_this_round = raises

        records.append({
            "state_key":   (my_hand, board, pot, cp, raises),
            "opponent":    opp,
            "ev_infoset":  ev_infoset,
            "_obs":        obs,
        })
    return records


# ── analysis ──────────────────────────────────────────────────────────────────

def run_analysis(save: bool = False):
    ev_data = load_ev_data()
    records = aggregate_ev_infoset(ev_data)

    agent = StatModValueAgent()   # loads frozen base
    agent.set_train_mode(False)

    # compute V_base for each unique info-set state
    state_to_vbase = {}
    for rec in records:
        sk = rec["state_key"]
        if sk not in state_to_vbase:
            obs = rec["_obs"]
            game_enc = _encode_game_state(obs, viewer_id=obs.current_player).unsqueeze(0)
            with torch.no_grad():
                v_base = agent.base(game_enc).item()
            state_to_vbase[sk] = v_base

    # compute residuals
    for rec in records:
        rec["v_base"]   = state_to_vbase[rec["state_key"]]
        rec["residual"] = rec["ev_infoset"] - rec["v_base"]

    # ── Check 1: Base calibration vs CFR ──────────────────────────────────────
    cfr_records = [r for r in records if r["opponent"] == "cfr"]
    if cfr_records:
        ev_cfr    = np.array([r["ev_infoset"] for r in cfr_records])
        v_base_cfr = np.array([r["v_base"]     for r in cfr_records])
        corr = np.corrcoef(ev_cfr, v_base_cfr)[0, 1]
        mae  = np.mean(np.abs(ev_cfr - v_base_cfr))
        bias = np.mean(ev_cfr - v_base_cfr)
        print(f"\n{'='*60}")
        print(f"  Check 1: Base calibration vs CFR (near-optimal)")
        print(f"{'='*60}")
        print(f"  Pearson correlation  : {corr:+.4f}")
        print(f"  Mean abs error (MAE) : {mae:.4f} chips")
        print(f"  Mean bias            : {bias:+.4f} chips  (+ = base underestimates)")
        print(f"  Interpretation: corr > 0.7 → base is well-aligned with optimal EV")
    else:
        print("  No CFR records found in EV data.")
        corr, mae, bias = None, None, None

    # ── Check 2: Residual magnitude per opponent ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Check 2: Mean |residual| per opponent")
    print(f"  (Large → more signal for modulation head to learn)")
    print(f"{'='*60}")
    opp_residuals = {}
    for rec in records:
        opp_residuals.setdefault(rec["opponent"], []).append(rec["residual"])

    opp_stats_table = {}
    for opp in sorted(opp_residuals):
        res = np.array(opp_residuals[opp])
        opp_stats_table[opp] = {
            "mean_residual":     float(np.mean(res)),
            "mean_abs_residual": float(np.mean(np.abs(res))),
            "std_residual":      float(np.std(res)),
            "n":                 len(res),
        }
        print(f"  {opp:<22}  mean={np.mean(res):+.3f}  "
              f"mae={np.mean(np.abs(res)):.3f}  std={np.std(res):.3f}  "
              f"n={len(res)}")

    # ── Check 3: Residual discriminability ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Check 3: Between-opponent vs within-opponent residual variance")
    print(f"  (F-ratio > 1 → opponents are distinguishable by residual)")
    print(f"{'='*60}")
    all_opp_means = np.array([opp_stats_table[o]["mean_residual"]
                               for o in opp_stats_table])
    all_opp_stds  = np.array([opp_stats_table[o]["std_residual"]
                               for o in opp_stats_table])
    between_var   = np.var(all_opp_means)
    within_var    = np.mean(all_opp_stds ** 2)
    f_ratio       = between_var / within_var if within_var > 0 else float("inf")
    print(f"  Between-opponent variance (of means): {between_var:.5f}")
    print(f"  Within-opponent variance (mean std²): {within_var:.5f}")
    print(f"  F-ratio                             : {f_ratio:.3f}")
    print(f"  Interpretation: F > 0.1 suggests meaningful discriminability")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    viable = (corr is not None and corr > 0.5) and (f_ratio > 0.1)
    print(f"  Base-CFR correlation : {'GOOD' if (corr or 0) > 0.5 else 'WEAK'} ({corr:.3f})")
    print(f"  F-ratio              : {'GOOD' if f_ratio > 0.1 else 'WEAK'} ({f_ratio:.3f})")
    print(f"  Supervised Variant B : {'VIABLE ✓' if viable else 'MAY STRUGGLE — check distributions'}")

    if save:
        out_dir = os.path.join(HERE, "outputs", "preliminary")
        os.makedirs(out_dir, exist_ok=True)
        result = {
            "base_cfr_correlation": corr,
            "base_cfr_mae": mae,
            "base_cfr_bias": bias,
            "f_ratio": f_ratio,
            "between_opponent_variance": between_var,
            "within_opponent_variance": within_var,
            "per_opponent": opp_stats_table,
            "viable_for_supervised": viable,
        }
        with open(os.path.join(out_dir, "preliminary_results.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results saved → {out_dir}/preliminary_results.json")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            opps    = list(opp_stats_table.keys())
            means   = [opp_stats_table[o]["mean_residual"] for o in opps]
            stds    = [opp_stats_table[o]["std_residual"]  for o in opps]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # residual distribution per opponent
            ax = axes[0]
            for i, opp in enumerate(opps):
                res = np.array(opp_residuals[opp])
                ax.scatter([i] * len(res), res, alpha=0.3, s=8)
            ax.errorbar(range(len(opps)), means, yerr=stds, fmt="o",
                        color="red", linewidth=2, capsize=5, label="mean ± std")
            ax.axhline(0, color="black", linestyle="--", alpha=0.5)
            ax.set_xticks(range(len(opps)))
            ax.set_xticklabels(opps, rotation=30, ha="right")
            ax.set_ylabel("EV_infoset − V_base  (residual)")
            ax.set_title("Residual per Opponent")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # base vs oracle scatter (CFR)
            if cfr_records:
                ax2 = axes[1]
                ax2.scatter(v_base_cfr, ev_cfr, alpha=0.5, s=12)
                lims = [min(v_base_cfr.min(), ev_cfr.min()),
                        max(v_base_cfr.max(), ev_cfr.max())]
                ax2.plot(lims, lims, "r--", linewidth=1.5, label="y=x (ideal)")
                ax2.set_xlabel("V_base(s)")
                ax2.set_ylabel("EV_infoset(s, cfr)")
                ax2.set_title(f"Base Calibration vs CFR  (r={corr:.3f})")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "preliminary_analysis.png"), dpi=150)
            plt.close(fig)
            print(f"  Plot saved → {out_dir}/preliminary_analysis.png")
        except ImportError:
            print("  matplotlib not available — skipping plots")

    return viable


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Save results JSON and plots to outputs/preliminary/")
    args = parser.parse_args()
    run_analysis(save=args.save)
