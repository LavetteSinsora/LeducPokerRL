"""
analyze.py — EV variance analysis and figure generation.

Reads data.json, pivots to a state × opponent EV matrix, computes
cross-opponent variance statistics, and produces 5 publication-ready figures.

Run from project root:
    python -m preliminary_experiments.ev_variation_extras.analysis.analyze
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe for headless runs)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATA_PATH = "OpponentModeling/EV_variation_analysis/data.json"
FIG_DIR   = "OpponentModeling/EV_variation_analysis/analysis"
DPI = 150

OPPONENT_KEYS = [
    'tight_passive',
    'tight_aggressive',
    'loose_passive',
    'loose_aggressive',
    'maniac',
    'random',
    'value_based',
    'cfr',
]

OPPONENT_LABELS = {
    'tight_passive':    'Tight Passive',
    'tight_aggressive': 'Tight Aggressive',
    'loose_passive':    'Loose Passive',
    'loose_aggressive': 'Loose Aggressive',
    'maniac':           'Maniac',
    'random':           'Random',
    'value_based':      'ValueBased (self)',
    'cfr':              'CFR (Nash)',
}

# 8 visually distinct colors (ColorBrewer-inspired)
_PALETTE = [
    '#1f77b4',   # tight_passive   — blue
    '#d62728',   # tight_aggressive — red
    '#2ca02c',   # loose_passive   — green
    '#ff7f0e',   # loose_aggressive — orange
    '#9467bd',   # maniac           — purple
    '#8c564b',   # random           — brown
    '#e377c2',   # value_based      — pink
    '#17becf',   # cfr              — teal
]
OPP_COLORS = {k: _PALETTE[i] for i, k in enumerate(OPPONENT_KEYS)}


# ---------------------------------------------------------------------------
# Data loading and pivot
# ---------------------------------------------------------------------------

def load_data(path: str = DATA_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def build_matrix(records: list) -> tuple:
    """
    Pivot flat records into:
      matrix[state_id][opponent] = {"ev": float, "ev_std": float, "n": int}
      meta[state_id]             = {round, hand0, hand1, board, pot, current_player, raises}
    """
    matrix = {}
    meta   = {}
    for rec in records:
        sid = rec['state_id']
        if sid not in matrix:
            matrix[sid] = {}
            meta[sid] = {k: rec[k] for k in
                         ['round', 'hand0', 'hand1', 'board',
                          'pot', 'current_player', 'raises']}
        matrix[sid][rec['opponent']] = {
            'ev':     rec['ev'],
            'ev_std': rec['ev_std'],
            'n':      rec['n'],
        }
    return matrix, meta


# ---------------------------------------------------------------------------
# Per-state statistics
# ---------------------------------------------------------------------------

def compute_state_stats(matrix: dict, meta: dict) -> list:
    """
    For each state compute cross-opponent EV statistics.
    Returns a list of dicts sorted by ev_std_cross (descending).
    """
    stats = []
    for sid, opp_data in matrix.items():
        present = [k for k in OPPONENT_KEYS if k in opp_data]
        if not present:
            continue

        evs = np.array([opp_data[k]['ev'] for k in present])

        ev_vs_self = opp_data.get('value_based', {}).get('ev', float('nan'))
        ev_vs_cfr  = opp_data.get('cfr',         {}).get('ev', float('nan'))

        if not (np.isnan(ev_vs_self) or np.isnan(ev_vs_cfr)):
            opt_dev = float(abs(ev_vs_self - ev_vs_cfr))
        else:
            opt_dev = float('nan')

        stats.append({
            "state_id":     sid,
            **meta[sid],
            # per-opponent EV and within-rollout std
            "ev_by_opponent":   {k: opp_data.get(k, {}).get('ev',     float('nan')) for k in OPPONENT_KEYS},
            "ev_std_rollout":   {k: opp_data.get(k, {}).get('ev_std', float('nan')) for k in OPPONENT_KEYS},
            "n_rollouts":       {k: opp_data.get(k, {}).get('n',      0)            for k in OPPONENT_KEYS},
            # cross-opponent summary
            "ev_mean_cross":    float(evs.mean()),
            "ev_std_cross":     float(evs.std(ddof=0)),   # population std across opponents
            "ev_min_cross":     float(evs.min()),
            "ev_max_cross":     float(evs.max()),
            "optimality_deviation": opt_dev,
        })

    stats.sort(key=lambda x: x['ev_std_cross'], reverse=True)
    return stats


def compute_aggregate(state_stats: list) -> dict:
    stds   = [s['ev_std_cross'] for s in state_stats]
    devs   = [s['optimality_deviation'] for s in state_stats
              if not np.isnan(s['optimality_deviation'])]

    mean_ev_by_opp = {}
    for k in OPPONENT_KEYS:
        vals = [s['ev_by_opponent'][k] for s in state_stats
                if not np.isnan(s['ev_by_opponent'][k])]
        mean_ev_by_opp[k] = float(np.mean(vals)) if vals else float('nan')

    return {
        "n_states":                  len(state_stats),
        "n_preflop":                 sum(1 for s in state_stats if s['round'] == 0),
        "n_flop":                    sum(1 for s in state_stats if s['round'] == 1),
        "mean_ev_std_cross":         float(np.mean(stds)),
        "median_ev_std_cross":       float(np.median(stds)),
        "pct75_ev_std_cross":        float(np.percentile(stds, 75)),
        "pct25_ev_std_cross":        float(np.percentile(stds, 25)),
        "mean_optimality_deviation": float(np.mean(devs)) if devs else float('nan'),
        "mean_ev_by_opponent":       mean_ev_by_opp,
    }


# ---------------------------------------------------------------------------
# Helper: state x-tick label
# ---------------------------------------------------------------------------

def _state_label(s: dict) -> str:
    rnd   = "Pre" if s['round'] == 0 else "Flop"
    board = s['board'] if s['board'] else "—"
    return f"{rnd} {s['hand0']}v{s['hand1']}\n{board} [{s['pot'][0]},{s['pot'][1]}]"


# ---------------------------------------------------------------------------
# Figure 1 & 2: Grouped bar charts (high / low variance states)
# ---------------------------------------------------------------------------

def plot_grouped_bars(state_stats, which, n_show=7, out_path=None):
    """
    Grouped bar chart showing ValueBasedAgent EV vs each opponent
    for the n_show states with highest (which='high') or
    lowest (which='low') cross-opponent EV std.

    Error bars: ±1 standard error = within-rollout ev_std / sqrt(n).
    """
    if which == 'high':
        selected = state_stats[:n_show]
        title    = f"High-Variance States: Value Agent EV by Opponent (top {n_show} by σ)"
    else:
        selected = list(reversed(state_stats[-n_show:]))
        title    = f"Low-Variance States: Value Agent EV by Opponent (bottom {n_show} by σ)"

    n_states = len(selected)
    n_opp    = len(OPPONENT_KEYS)
    bar_w    = 0.8 / n_opp
    x        = np.arange(n_states)

    fig, ax = plt.subplots(figsize=(max(12, n_states * 1.6), 5))

    for i, opp_key in enumerate(OPPONENT_KEYS):
        evs = [s['ev_by_opponent'][opp_key] for s in selected]
        ses = []
        for s in selected:
            std = s['ev_std_rollout'][opp_key]
            n   = s['n_rollouts'][opp_key]
            ses.append(std / np.sqrt(n) if n > 1 else 0.0)

        offsets = x + (i - n_opp / 2 + 0.5) * bar_w
        ax.bar(
            offsets, evs, bar_w * 0.9,
            label=OPPONENT_LABELS[opp_key],
            color=OPP_COLORS[opp_key],
            alpha=0.85,
            yerr=ses, capsize=2,
            error_kw=dict(lw=0.8, ecolor='#333333'),
        )

    ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([_state_label(s) for s in selected],
                       fontsize=7.5, rotation=20, ha='right')
    ax.set_xlabel("Game State (hand0 vs hand1 | board | pot)")
    ax.set_ylabel("EV (chips)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper right', fontsize=7.5, ncol=2,
              framealpha=0.9, edgecolor='lightgray')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    _save_or_show(out_path)


# ---------------------------------------------------------------------------
# Figure 3: Distribution of cross-opponent EV std
# ---------------------------------------------------------------------------

def plot_ev_std_distribution(state_stats, aggregate, out_path=None):
    """Histogram of cross-opponent EV std across all states."""
    stds       = [s['ev_std_cross'] for s in state_stats]
    mean_val   = aggregate['mean_ev_std_cross']
    median_val = aggregate['median_ev_std_cross']
    n_states   = len(stds)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(stds, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(mean_val,   color='#d62728', linestyle='--', linewidth=1.5,
               label=f"Mean = {mean_val:.3f}")
    ax.axvline(median_val, color='#ff7f0e', linestyle='--', linewidth=1.5,
               label=f"Median = {median_val:.3f}")
    ax.set_xlabel("Cross-Opponent EV Std (chips)")
    ax.set_ylabel("Number of States")
    ax.set_title(
        f"Distribution of EV Variance Across Opponents\n"
        f"({n_states} states, {len(OPPONENT_KEYS)} opponents each)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_or_show(out_path)


# ---------------------------------------------------------------------------
# Figure 4: Optimality proxy — value_based vs CFR EV
# ---------------------------------------------------------------------------

def plot_optimality_proxy(state_stats, aggregate, out_path=None):
    """
    Line plot sorted by EV(vs CFR) ascending.
    Shows EV(vs self) vs EV(vs CFR) per state with shaded deviation area.
    """
    sorted_states = sorted(state_stats, key=lambda s: s['ev_by_opponent']['cfr'])
    cfr_evs  = [s['ev_by_opponent']['cfr']         for s in sorted_states]
    self_evs = [s['ev_by_opponent']['value_based']  for s in sorted_states]
    x = np.arange(len(sorted_states))

    mean_dev = aggregate['mean_optimality_deviation']

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, cfr_evs,  color=OPP_COLORS['cfr'],         lw=1.3,
            label=OPPONENT_LABELS['cfr'], zorder=3)
    ax.plot(x, self_evs, color=OPP_COLORS['value_based'],  lw=1.3,
            label=OPPONENT_LABELS['value_based'], zorder=3)
    ax.fill_between(
        x,
        np.minimum(cfr_evs, self_evs),
        np.maximum(cfr_evs, self_evs),
        alpha=0.18, color='gray', label='Deviation region', zorder=2,
    )
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.4, zorder=1)
    ax.set_xlabel("State (sorted by EV vs CFR, ascending)")
    ax.set_ylabel("EV of ValueBasedAgent (chips)")
    ax.set_title(
        f"Optimality Proxy: ValueBased (self-play) vs CFR Opponent\n"
        f"Mean |deviation| = {mean_dev:.3f} chips across {len(sorted_states)} states",
        fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    _save_or_show(out_path)


# ---------------------------------------------------------------------------
# Figure 5: Mean EV by opponent
# ---------------------------------------------------------------------------

def plot_mean_ev_by_opponent(aggregate, out_path=None):
    """Horizontal bar chart of mean EV vs each opponent, sorted descending."""
    mean_ev     = aggregate['mean_ev_by_opponent']
    sorted_keys = sorted(OPPONENT_KEYS, key=lambda k: mean_ev[k], reverse=True)
    labels      = [OPPONENT_LABELS[k] for k in sorted_keys]
    values      = [mean_ev[k] for k in sorted_keys]
    colors      = [OPP_COLORS[k] for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, values, color=colors, alpha=0.85, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    for bar, val in zip(bars, values):
        pad = 0.015
        ax.text(
            val + (pad if val >= 0 else -pad),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va='center',
            ha='left' if val >= 0 else 'right',
            fontsize=8.5,
        )

    ax.set_xlabel("Mean EV of ValueBasedAgent (chips)")
    ax.set_title(
        f"Mean EV of ValueBasedAgent vs Each Opponent\n"
        f"(averaged over {aggregate['n_states']} states)", fontsize=11)
    plt.tight_layout()
    _save_or_show(out_path)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save_or_show(out_path):
    if out_path:
        plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"Loading {DATA_PATH} ...")
    data    = load_data()
    records = data['records']
    print(f"  {len(records)} records, {data['metadata']['n_states']} states, "
          f"{data['metadata']['n_opponents']} opponents.")

    matrix, meta = build_matrix(records)
    state_stats  = compute_state_stats(matrix, meta)
    aggregate    = compute_aggregate(state_stats)

    # Print summary
    print(f"\n--- Aggregate Statistics ---")
    print(f"  n_states:               {aggregate['n_states']}")
    print(f"  n_preflop / n_flop:     {aggregate['n_preflop']} / {aggregate['n_flop']}")
    print(f"  mean EV std (cross-opp): {aggregate['mean_ev_std_cross']:.4f}  "
          f"(median: {aggregate['median_ev_std_cross']:.4f})")
    print(f"  IQR [25th, 75th]:       [{aggregate['pct25_ev_std_cross']:.4f}, "
          f"{aggregate['pct75_ev_std_cross']:.4f}]")
    print(f"  mean optimality dev:    {aggregate['mean_optimality_deviation']:.4f}")
    print(f"\n  Mean EV vs each opponent:")
    for k in OPPONENT_KEYS:
        print(f"    {OPPONENT_LABELS[k]:22s}: {aggregate['mean_ev_by_opponent'][k]:+.4f}")

    print(f"\n  Top 5 high-variance states:")
    for s in state_stats[:5]:
        print(f"    {s['state_id']:40s}  σ={s['ev_std_cross']:.4f}")

    print(f"\n  Top 5 low-variance states:")
    for s in reversed(state_stats[-5:]):
        print(f"    {s['state_id']:40s}  σ={s['ev_std_cross']:.4f}")

    # Generate figures
    print(f"\nGenerating figures → {FIG_DIR}/")
    plot_grouped_bars(state_stats, 'high',
        out_path=os.path.join(FIG_DIR, 'fig1_high_variance.png'))
    plot_grouped_bars(state_stats, 'low',
        out_path=os.path.join(FIG_DIR, 'fig2_low_variance.png'))
    plot_ev_std_distribution(state_stats, aggregate,
        out_path=os.path.join(FIG_DIR, 'fig3_distribution.png'))
    plot_optimality_proxy(state_stats, aggregate,
        out_path=os.path.join(FIG_DIR, 'fig4_optimality.png'))
    plot_mean_ev_by_opponent(aggregate,
        out_path=os.path.join(FIG_DIR, 'fig5_mean_ev.png'))

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
