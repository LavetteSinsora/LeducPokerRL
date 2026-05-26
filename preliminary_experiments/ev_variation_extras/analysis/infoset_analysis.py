"""
infoset_analysis.py — EV variance at the information-set level.

A game *state* fully specifies both players' hands.  The *infoset* is what
the value agent actually observes: (my_hand, board, pot, round, raises,
current_player).  Multiple states share the same infoset — they differ only
in the opponent's private card, which the value agent cannot see.

EV of an infoset against a given opponent is the hand-removal-weighted
average of the per-state EVs:

    EV(I, opp) = Σ_{opp_hand} P(opp_hand | my_hand, board) · EV(s, opp)

where P(opp_hand | my_hand, board) is the card-removal probability derived
from the 6-card Leduc deck after removing my_hand (and board, on the flop).

Motivation: measuring whether the value modulation head (opponent-conditioned
value function) is actually needed.  If cross-opponent EV std at the infoset
level is large, the value agent genuinely needs to modulate its value
estimates by opponent.  If it is near zero, a fixed (opponent-agnostic) value
function suffices.

Run from project root:
    python -m preliminary_experiments.ev_variation_extras.analysis.infoset_analysis
"""

import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
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

_PALETTE = [
    '#1f77b4',   # tight_passive
    '#d62728',   # tight_aggressive
    '#2ca02c',   # loose_passive
    '#ff7f0e',   # loose_aggressive
    '#9467bd',   # maniac
    '#8c564b',   # random
    '#e377c2',   # value_based
    '#17becf',   # cfr
]
OPP_COLORS = {k: _PALETTE[i] for i, k in enumerate(OPPONENT_KEYS)}

FULL_DECK = ['J', 'J', 'Q', 'Q', 'K', 'K']


# ---------------------------------------------------------------------------
# Hand-removal weights
# ---------------------------------------------------------------------------

def hand_removal_weights(my_hand: str, board: str | None) -> dict:
    """
    Returns {opp_hand_rank: probability} for the value agent holding my_hand
    on a board (or None for pre-flop).

    Derivation: start from full 6-card Leduc deck, remove my_hand (and board
    on the flop).  The opponent is equally likely to hold any remaining card.

    Pre-flop:  remaining = deck − my_hand  (5 cards)
    Flop:      remaining = deck − my_hand − board  (4 cards)

    Returns only ranks with positive probability.
    """
    remaining = list(FULL_DECK)
    remaining.remove(my_hand)
    if board is not None:
        remaining.remove(board)
    counts = Counter(remaining)
    total = len(remaining)   # 5 pre-flop, 4 on flop
    return {rank: count / total for rank, count in counts.items()}


# ---------------------------------------------------------------------------
# Infoset key
# ---------------------------------------------------------------------------

def infoset_key(rec: dict) -> tuple:
    """
    Canonical infoset key from a data.json record.

    An infoset is the value agent's observable context:
        (my_hand, board, pot_tuple, round, raises, current_player)

    my_hand = hand0 if current_player == 0 else hand1
    """
    cp = rec['current_player']
    my_hand = rec['hand0'] if cp == 0 else rec['hand1']
    return (
        my_hand,
        rec['board'],              # None for pre-flop
        tuple(rec['pot']),         # make hashable
        rec['round'],
        rec['raises'],
        cp,
    )


def opp_hand_from_rec(rec: dict) -> str:
    """Opponent's private card from a data.json record."""
    cp = rec['current_player']
    return rec['hand1'] if cp == 0 else rec['hand0']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_group(path: str = DATA_PATH) -> tuple:
    """
    Load data.json and group records into infosets.

    Returns:
        infoset_states:   {ikey: {opp_hand: {opponent: {ev, ev_std, n}}}}
        infoset_meta:     {ikey: {my_hand, board, pot, round, raises, cp}}
    """
    with open(path) as f:
        data = json.load(f)

    # First pass: collect per-(state, opponent) ev
    # state key: (hand0, hand1, board, tuple(pot), round, raises, cp)
    state_ev = defaultdict(dict)   # state_key -> opponent -> {ev, ev_std, n}
    state_to_ikey = {}             # state_key -> infoset_key

    for rec in data['records']:
        sk = (
            rec['hand0'], rec['hand1'], rec['board'],
            tuple(rec['pot']), rec['round'], rec['raises'], rec['current_player'],
        )
        ikey = infoset_key(rec)
        state_ev[sk][rec['opponent']] = {
            'ev': rec['ev'], 'ev_std': rec['ev_std'], 'n': rec['n'],
        }
        state_to_ikey[sk] = ikey

    # Build infoset_states: ikey -> opp_hand -> opponent -> ev data
    infoset_states = defaultdict(dict)
    infoset_meta   = {}

    for sk, opp_data in state_ev.items():
        hand0, hand1, board, pot_t, rnd, raises, cp = sk
        ikey     = state_to_ikey[sk]
        my_hand  = hand0 if cp == 0 else hand1
        opp_hand = hand1 if cp == 0 else hand0

        infoset_states[ikey][opp_hand] = opp_data   # {opponent: {ev,...}}

        if ikey not in infoset_meta:
            infoset_meta[ikey] = {
                'my_hand': my_hand,
                'board':   board,
                'pot':     list(pot_t),
                'round':   rnd,
                'raises':  raises,
                'cp':      cp,
            }

    return dict(infoset_states), infoset_meta


# ---------------------------------------------------------------------------
# Infoset EV computation
# ---------------------------------------------------------------------------

def compute_infoset_ev(
    ikey: tuple,
    opp_hand_data: dict,
    infoset_meta: dict,
) -> dict:
    """
    Compute weighted EV for each opponent across all states in the infoset.

    opp_hand_data: {opp_hand_rank: {opponent_key: {ev, ev_std, n}}}
    Returns: {opponent_key: weighted_ev}
    """
    meta    = infoset_meta[ikey]
    weights = hand_removal_weights(meta['my_hand'], meta['board'])

    # Normalize weights to the opp_hand ranks actually present in data
    # (should already sum to ~1.0 over present ranks)
    present_ranks  = set(opp_hand_data.keys())
    weight_sum = sum(weights.get(r, 0.0) for r in present_ranks)

    infoset_ev = {}
    for opp_key in OPPONENT_KEYS:
        weighted_ev = 0.0
        for opp_hand_rank, per_opp in opp_hand_data.items():
            w = weights.get(opp_hand_rank, 0.0) / weight_sum  # renormalized
            ev = per_opp.get(opp_key, {}).get('ev', float('nan'))
            if np.isnan(ev):
                continue
            weighted_ev += w * ev
        infoset_ev[opp_key] = weighted_ev

    return infoset_ev


def compute_all_infoset_stats(infoset_states: dict, infoset_meta: dict) -> list:
    """
    Compute per-infoset statistics and return a list sorted by
    cross-opponent EV std (descending).
    """
    stats = []
    for ikey, opp_hand_data in infoset_states.items():
        meta   = infoset_meta[ikey]
        ev_by_opp = compute_infoset_ev(ikey, opp_hand_data, infoset_meta)

        evs = np.array([ev_by_opp[k] for k in OPPONENT_KEYS
                        if not np.isnan(ev_by_opp.get(k, float('nan')))])

        ev_vs_self = ev_by_opp.get('value_based', float('nan'))
        ev_vs_cfr  = ev_by_opp.get('cfr',         float('nan'))
        if not (np.isnan(ev_vs_self) or np.isnan(ev_vs_cfr)):
            opt_dev = float(abs(ev_vs_self - ev_vs_cfr))
        else:
            opt_dev = float('nan')

        stats.append({
            'ikey':         ikey,
            **meta,
            'n_states':     len(opp_hand_data),
            'ev_by_opp':    ev_by_opp,
            'ev_mean_cross': float(evs.mean()),
            'ev_std_cross':  float(evs.std(ddof=0)),
            'ev_min_cross':  float(evs.min()),
            'ev_max_cross':  float(evs.max()),
            'optimality_deviation': opt_dev,
        })

    stats.sort(key=lambda x: x['ev_std_cross'], reverse=True)
    return stats


def compute_infoset_aggregate(stats: list, state_stds: list | None = None) -> dict:
    stds = [s['ev_std_cross'] for s in stats]
    devs = [s['optimality_deviation'] for s in stats
            if not np.isnan(s['optimality_deviation'])]

    mean_ev_by_opp = {}
    for k in OPPONENT_KEYS:
        vals = [s['ev_by_opp'][k] for s in stats
                if not np.isnan(s['ev_by_opp'].get(k, float('nan')))]
        mean_ev_by_opp[k] = float(np.mean(vals)) if vals else float('nan')

    agg = {
        'n_infosets':                len(stats),
        'n_preflop':                 sum(1 for s in stats if s['round'] == 0),
        'n_flop':                    sum(1 for s in stats if s['round'] == 1),
        'mean_ev_std_cross':         float(np.mean(stds)),
        'median_ev_std_cross':       float(np.median(stds)),
        'pct75_ev_std_cross':        float(np.percentile(stds, 75)),
        'pct25_ev_std_cross':        float(np.percentile(stds, 25)),
        'mean_optimality_deviation': float(np.mean(devs)) if devs else float('nan'),
        'mean_ev_by_opponent':       mean_ev_by_opp,
    }
    return agg


# ---------------------------------------------------------------------------
# Helper: infoset label for axes
# ---------------------------------------------------------------------------

def _infoset_label(s: dict) -> str:
    rnd   = "Pre" if s['round'] == 0 else "Flop"
    board = s['board'] if s['board'] else "—"
    pot   = s['pot']
    return f"{rnd} {s['my_hand']}|{board}\n[{pot[0]},{pot[1]}] r{s['raises']}"


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _save_or_show(out_path):
    if out_path:
        plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Figure I1: Distribution of infoset cross-opponent EV std
# ---------------------------------------------------------------------------

def plot_infoset_distribution(stats, aggregate, out_path=None):
    """Histogram of cross-opponent EV std across all infosets."""
    stds       = [s['ev_std_cross'] for s in stats]
    mean_val   = aggregate['mean_ev_std_cross']
    median_val = aggregate['median_ev_std_cross']
    n_infosets = len(stds)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(stds, bins=25, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(mean_val,   color='#d62728', linestyle='--', linewidth=1.5,
               label=f"Mean = {mean_val:.3f}")
    ax.axvline(median_val, color='#ff7f0e', linestyle='--', linewidth=1.5,
               label=f"Median = {median_val:.3f}")
    ax.set_xlabel("Cross-Opponent EV Std at Infoset Level (chips)")
    ax.set_ylabel("Number of Infosets")
    ax.set_title(
        f"Distribution of EV Variance Across Opponents — Infoset Level\n"
        f"({n_infosets} infosets, {len(OPPONENT_KEYS)} opponents each)",
        fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_or_show(out_path)


# ---------------------------------------------------------------------------
# Figure I2 & I3: Grouped bars (high / low variance infosets)
# ---------------------------------------------------------------------------

def plot_infoset_grouped_bars(stats, which, n_show=7, out_path=None):
    """
    Grouped bar chart showing ValueBasedAgent EV vs each opponent
    for the n_show infosets with highest ('high') or lowest ('low')
    cross-opponent EV std.
    """
    if which == 'high':
        selected = stats[:n_show]
        title    = f"High-Variance Infosets: Value Agent EV by Opponent (top {n_show} by σ)"
    else:
        selected = list(reversed(stats[-n_show:]))
        title    = f"Low-Variance Infosets: Value Agent EV by Opponent (bottom {n_show} by σ)"

    n_infosets = len(selected)
    n_opp      = len(OPPONENT_KEYS)
    bar_w      = 0.8 / n_opp
    x          = np.arange(n_infosets)

    fig, ax = plt.subplots(figsize=(max(12, n_infosets * 1.6), 5))

    for i, opp_key in enumerate(OPPONENT_KEYS):
        evs     = [s['ev_by_opp'][opp_key] for s in selected]
        offsets = x + (i - n_opp / 2 + 0.5) * bar_w
        ax.bar(
            offsets, evs, bar_w * 0.9,
            label=OPPONENT_LABELS[opp_key],
            color=OPP_COLORS[opp_key],
            alpha=0.85,
        )

    ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([_infoset_label(s) for s in selected],
                       fontsize=7.5, rotation=20, ha='right')
    ax.set_xlabel("Infoset (my hand | board | pot | raises)")
    ax.set_ylabel("Weighted EV (chips)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper right', fontsize=7.5, ncol=2,
              framealpha=0.9, edgecolor='lightgray')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    _save_or_show(out_path)


# ---------------------------------------------------------------------------
# Figure I4: Infoset vs State level variance comparison
# ---------------------------------------------------------------------------

def plot_variance_comparison(state_stds, infoset_stds, out_path=None):
    """
    Overlaid histograms comparing cross-opponent EV std at the state level
    vs the infoset level.

    The infoset distribution is expected to be narrower because hand-removal
    weighted averaging across opponent hands smooths out the state-level
    variance.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(0, max(max(state_stds), max(infoset_stds)) + 0.05, 30)

    ax.hist(state_stds, bins=bins, color='steelblue', edgecolor='white',
            alpha=0.6, label=f"State level  (n={len(state_stds)})")
    ax.hist(infoset_stds, bins=bins, color='darkorange', edgecolor='white',
            alpha=0.6, label=f"Infoset level (n={len(infoset_stds)})")

    ax.axvline(np.mean(state_stds),   color='steelblue', linestyle='--', lw=1.5,
               label=f"State mean = {np.mean(state_stds):.3f}")
    ax.axvline(np.mean(infoset_stds), color='darkorange', linestyle='--', lw=1.5,
               label=f"Infoset mean = {np.mean(infoset_stds):.3f}")

    ax.set_xlabel("Cross-Opponent EV Std (chips)")
    ax.set_ylabel("Count")
    ax.set_title(
        "EV Variance Distribution: State Level vs Infoset Level\n"
        "Narrowing shows how opponent-hand averaging reduces apparent variance",
        fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_or_show(out_path)


# ---------------------------------------------------------------------------
# Figure I5: Mean EV by opponent at infoset level
# ---------------------------------------------------------------------------

def plot_infoset_mean_ev_by_opponent(aggregate, out_path=None):
    """Horizontal bar chart of mean infoset-level EV vs each opponent."""
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

    ax.set_xlabel("Mean Weighted EV of ValueBasedAgent (chips)")
    ax.set_title(
        f"Mean Infoset-Level EV of ValueBasedAgent vs Each Opponent\n"
        f"(averaged over {aggregate['n_infosets']} infosets)",
        fontsize=11)
    plt.tight_layout()
    _save_or_show(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"Loading {DATA_PATH} ...")
    infoset_states, infoset_meta = load_and_group(DATA_PATH)
    print(f"  Grouped into {len(infoset_states)} infosets.")

    # Count pre-flop vs flop
    n_pf   = sum(1 for m in infoset_meta.values() if m['round'] == 0)
    n_flop = sum(1 for m in infoset_meta.values() if m['round'] == 1)
    print(f"  Pre-flop infosets: {n_pf}, Flop infosets: {n_flop}")

    # Show weight example
    ex_key = list(infoset_meta.keys())[0]
    ex_meta = infoset_meta[ex_key]
    ex_w = hand_removal_weights(ex_meta['my_hand'], ex_meta['board'])
    print(f"\n  Example infoset key: {ex_key}")
    print(f"  Hand-removal weights: {ex_w}  (sum={sum(ex_w.values()):.4f})")

    print("\nComputing infoset EV stats...")
    stats     = compute_all_infoset_stats(infoset_states, infoset_meta)
    aggregate = compute_infoset_aggregate(stats)

    print(f"\n--- Infoset-Level Aggregate Statistics ---")
    print(f"  n_infosets:              {aggregate['n_infosets']}")
    print(f"  n_preflop / n_flop:      {aggregate['n_preflop']} / {aggregate['n_flop']}")
    print(f"  mean EV std (cross-opp): {aggregate['mean_ev_std_cross']:.4f}  "
          f"(median: {aggregate['median_ev_std_cross']:.4f})")
    print(f"  IQR [25th, 75th]:        [{aggregate['pct25_ev_std_cross']:.4f}, "
          f"{aggregate['pct75_ev_std_cross']:.4f}]")
    print(f"  mean optimality dev:     {aggregate['mean_optimality_deviation']:.4f}")
    print(f"\n  Mean infoset EV vs each opponent:")
    for k in OPPONENT_KEYS:
        print(f"    {OPPONENT_LABELS[k]:22s}: {aggregate['mean_ev_by_opponent'][k]:+.4f}")

    print(f"\n  Top 5 high-variance infosets:")
    for s in stats[:5]:
        print(f"    {str(s['ikey']):65s}  σ={s['ev_std_cross']:.4f}")

    print(f"\n  Top 5 low-variance infosets:")
    for s in reversed(stats[-5:]):
        print(f"    {str(s['ikey']):65s}  σ={s['ev_std_cross']:.4f}")

    # Load state-level stds for comparison figure
    with open(DATA_PATH) as f:
        raw_data = json.load(f)
    from preliminary_experiments.ev_variation_extras.analysis.analyze import (
        build_matrix, compute_state_stats,
    )
    matrix, state_meta = build_matrix(raw_data['records'])
    state_stats  = compute_state_stats(matrix, state_meta)
    state_stds   = [s['ev_std_cross'] for s in state_stats]
    infoset_stds = [s['ev_std_cross'] for s in stats]

    print(f"\nState-level mean EV std:   {np.mean(state_stds):.4f}")
    print(f"Infoset-level mean EV std: {np.mean(infoset_stds):.4f}")
    reduction_pct = 100 * (1 - np.mean(infoset_stds) / np.mean(state_stds))
    print(f"Variance reduction:        {reduction_pct:.1f}%")

    # Generate figures
    print(f"\nGenerating figures → {FIG_DIR}/")
    plot_infoset_distribution(stats, aggregate,
        out_path=os.path.join(FIG_DIR, 'fig_i1_distribution.png'))
    plot_infoset_grouped_bars(stats, 'high',
        out_path=os.path.join(FIG_DIR, 'fig_i2_high_variance.png'))
    plot_infoset_grouped_bars(stats, 'low',
        out_path=os.path.join(FIG_DIR, 'fig_i3_low_variance.png'))
    plot_variance_comparison(state_stds, infoset_stds,
        out_path=os.path.join(FIG_DIR, 'fig_i4_comparison.png'))
    plot_infoset_mean_ev_by_opponent(aggregate,
        out_path=os.path.join(FIG_DIR, 'fig_i5_mean_ev.png'))

    print("\nAll infoset figures saved.")


if __name__ == "__main__":
    main()
