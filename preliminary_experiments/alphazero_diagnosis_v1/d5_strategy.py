"""
D5: Strategy Profile — Hand-Stratified Action Distributions

Runs 2000 greedy games per checkpoint and computes action distributions
broken down by (hand, round, player position).

Key signals:
  - Hand-rank ordering: raise_rate(K) > raise_rate(Q) > raise_rate(J)
    A poker agent with no hand discrimination is highly exploitable.
  - Strategy entropy: how uniform is the action distribution per (hand, round)?
    High entropy → agent plays the same regardless of cards.
  - Raise rate spread (K_raise - J_raise) should be positive and large.

Output:
  outputs/d5_raise_rates.png      — raise rate per hand across checkpoints
  outputs/d5_action_bars.png      — stacked bar charts (fold/call/raise) per checkpoint
  outputs/d5_entropy.png          — per-hand action distribution entropy over time
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import entropy as scipy_entropy

from diagnose import (
    CHECKPOINT_EPISODES, load_checkpoint, run_greedy_games,
    ensure_output_dir, OUTPUT_DIR, HAND_LABELS, COLORS,
)

N_GAMES = 2000


def compute_strategy_stats(records):
    """
    Compute action distribution per (hand, round).
    Returns: dict[(hand, round)] → {fold: float, call: float, raise: float}
    """
    counts = defaultdict(lambda: [0, 0, 0])  # fold, call, raise

    for rec in records:
        for dec in rec.decisions:
            key = (dec.hand, dec.round)
            counts[key][dec.action_taken] += 1

    stats = {}
    for key, cnt in counts.items():
        total = sum(cnt)
        if total == 0:
            continue
        stats[key] = {
            'fold':  cnt[0] / total,
            'call':  cnt[1] / total,
            'raise': cnt[2] / total,
            'total': total,
        }
    return stats


def compute_per_hand_raise_rates(stats, round_idx=None):
    """
    Extract raise rate per hand (J/Q/K).
    If round_idx is None, aggregate across both rounds.
    """
    rates = {}
    for hand in HAND_LABELS:
        if round_idx is None:
            # Aggregate across rounds (weighted by count)
            total_raise = 0
            total = 0
            for rnd in [0, 1]:
                key = (hand, rnd)
                if key in stats:
                    s = stats[key]
                    total_raise += s['raise'] * s['total']
                    total += s['total']
            rates[hand] = total_raise / total if total > 0 else 0.0
        else:
            key = (hand, round_idx)
            rates[hand] = stats.get(key, {}).get('raise', 0.0)
    return rates


def distribution_entropy(stats_entry):
    """Shannon entropy (nats) of [fold, call, raise] distribution."""
    probs = [stats_entry['fold'], stats_entry['call'], stats_entry['raise']]
    probs = [max(p, 1e-8) for p in probs]
    return float(scipy_entropy(probs))


def main():
    ensure_output_dir()

    all_stats = {}
    all_raise_by_hand = {}  # ep → {hand → raise_rate}

    for ep in CHECKPOINT_EPISODES:
        print(f"D5: ep {ep:,} — playing {N_GAMES} greedy games...")
        config, state_enc, belief_net, q_net = load_checkpoint(ep)
        records = run_greedy_games(state_enc, belief_net, q_net, config, n_games=N_GAMES)
        stats = compute_strategy_stats(records)
        all_stats[ep] = stats

        raise_rates = compute_per_hand_raise_rates(stats)
        all_raise_by_hand[ep] = raise_rates
        spread = raise_rates.get('K', 0) - raise_rates.get('J', 0)
        print(f"  J={raise_rates.get('J',0):.3f}  Q={raise_rates.get('Q',0):.3f}  K={raise_rates.get('K',0):.3f}  spread(K-J)={spread:.3f}")

    # ── Plot 1: Raise rate per hand over checkpoints ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('D5: Raise Rate by Hand — Preflop vs Postflop', fontsize=13)

    round_names = ['Preflop (round 0)', 'Postflop (round 1)']
    hand_colors = {'J': '#E53935', 'Q': '#FB8C00', 'K': '#43A047'}
    eps_k = [ep / 1000 for ep in CHECKPOINT_EPISODES]

    for rnd_idx, ax in enumerate(axes):
        for hand in HAND_LABELS:
            rates = [compute_per_hand_raise_rates(all_stats[ep], round_idx=rnd_idx).get(hand, 0)
                     for ep in CHECKPOINT_EPISODES]
            ax.plot(eps_k, rates, 'o-', color=hand_colors[hand], lw=2, ms=8, label=f'Hand={hand}')

        ax.set_xlabel('Episode (thousands)', fontsize=10)
        ax.set_ylabel('Raise Rate', fontsize=10)
        ax.set_title(round_names[rnd_idx], fontsize=11)
        ax.set_ylim(0, 0.6)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(1/3, color='gray', ls=':', lw=1, label='random baseline')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d5_raise_rates.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # ── Plot 2: Stacked bar charts per checkpoint ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('D5: Action Distribution by Hand and Round\n(J/Q/K × Preflop/Postflop)', fontsize=13)

    bar_colors = {'fold': '#EF5350', 'call': '#FFA726', 'raise': '#42A5F5'}

    for idx, ep in enumerate(CHECKPOINT_EPISODES):
        ax = axes[idx // 3][idx % 3]
        stats = all_stats[ep]

        hands = HAND_LABELS
        rounds = [0, 1]
        x_labels = [f'{h}\nR{r}' for h in hands for r in rounds]
        x = np.arange(len(x_labels))

        fold_vals  = []
        call_vals  = []
        raise_vals = []

        for hand in hands:
            for rnd in rounds:
                key = (hand, rnd)
                s = stats.get(key, {'fold': 0.33, 'call': 0.33, 'raise': 0.33})
                fold_vals.append(s['fold'])
                call_vals.append(s['call'])
                raise_vals.append(s['raise'])

        ax.bar(x, fold_vals,  label='fold',  color=bar_colors['fold'],  alpha=0.85)
        ax.bar(x, call_vals,  label='call',  color=bar_colors['call'],  alpha=0.85,
               bottom=fold_vals)
        ax.bar(x, raise_vals, label='raise', color=bar_colors['raise'], alpha=0.85,
               bottom=[f+c for f, c in zip(fold_vals, call_vals)])

        # Annotate raise rate on top of each bar
        for xi, (fv, cv, rv) in enumerate(zip(fold_vals, call_vals, raise_vals)):
            ax.text(xi, fv + cv + rv + 0.02, f'{rv:.2f}', ha='center', va='bottom',
                    fontsize=7, color='#1565C0', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Fraction', fontsize=8)
        ax.set_title(f'ep {ep:,}', fontsize=10)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d5_action_bars.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 3: Strategy entropy over training ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('D5: Action Distribution Entropy per Hand\n(higher = more uniform / less differentiated)', fontsize=12)

    for hand in HAND_LABELS:
        entropies = []
        for ep in CHECKPOINT_EPISODES:
            stats = all_stats[ep]
            # Average entropy across both rounds (preflop + postflop)
            ents = []
            for rnd in [0, 1]:
                key = (hand, rnd)
                if key in stats:
                    ents.append(distribution_entropy(stats[key]))
            entropies.append(np.mean(ents) if ents else 0.0)
        ax.plot(eps_k, entropies, 'o-', color=hand_colors[hand], lw=2, ms=8, label=f'Hand={hand}')

    # Uniform distribution entropy = log(3) ≈ 1.099
    ax.axhline(np.log(3), color='gray', ls='--', lw=1.5, label='uniform (log 3 ≈ 1.10)')
    ax.set_xlabel('Episode (thousands)', fontsize=10)
    ax.set_ylabel('Shannon Entropy (nats)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d5_entropy.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 4: Raise rate spread (K - J) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title('D5: Raise Rate Spread (K − J)\n(positive = hand-differentiated; near 0 = collapsed)', fontsize=12)

    spreads = [all_raise_by_hand[ep].get('K', 0) - all_raise_by_hand[ep].get('J', 0)
               for ep in CHECKPOINT_EPISODES]
    bars = ax.bar(eps_k, spreads, color=['#4CAF50' if s > 0.05 else '#F44336' for s in spreads],
                  width=6, alpha=0.85)
    ax.axhline(0, color='black', lw=1)
    ax.set_xlabel('Episode (thousands)', fontsize=10)
    ax.set_ylabel('Raise Rate Spread (K − J)', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    for bar, s in zip(bars, spreads):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{s:+.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d5_raise_spread.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── D5 Summary ──────────────────────────────────────────────────────")
    print(f"{'Episode':>10}  {'J raise':>9}  {'Q raise':>9}  {'K raise':>9}  {'K-J spread':>12}")
    for ep in CHECKPOINT_EPISODES:
        r = all_raise_by_hand[ep]
        spread = r.get('K', 0) - r.get('J', 0)
        print(f"{ep:>10,}  {r.get('J',0):>9.3f}  {r.get('Q',0):>9.3f}  {r.get('K',0):>9.3f}  {spread:>12.3f}")


if __name__ == '__main__':
    main()
