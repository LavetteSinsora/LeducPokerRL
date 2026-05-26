"""
D6: Belief Portfolio Coherence — Are b_opp[J/Q/K] Meaningfully Different?

Each player maintains a "belief portfolio":
  b_opp[k] = opponent's belief about ME, assuming opponent holds card k

During PIMC search, for each imagined opponent hand h_j, we use b_opp[h_j]
as the opponent's action model. If all three b_opp vectors are identical,
imagining different opponent hands gives the same opponent behavior — the
"portfolio" provides no value and PIMC degenerates.

Metrics:
  - Portfolio diversity: mean pairwise L2 distance between b_opp[J], b_opp[Q], b_opp[K]
    at each decision step.
    Low diversity → all three "imagined opponents" behave identically.
  - Per-card entropy: entropy of each b_opp[k] (how confident is "opponent-if-K" about me?)
  - b_mine vs b_opp[true_opp_hand] alignment:
    Does my belief about opponent's hand (b_mine) agree with their self-knowledge?
    They "know" they hold some card k, so they use b_opp[k] which reflects their
    belief about me. b_mine should converge to match what "they" actually see.

Output:
  outputs/d6_diversity.png   — portfolio diversity per decision step
  outputs/d6_entropy.png     — entropy of b_opp[J/Q/K] per step
  outputs/d6_collapse.png    — pairwise L2 distances over training episodes
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import entropy as scipy_entropy

from diagnose import (
    CHECKPOINT_EPISODES, load_checkpoint, run_greedy_games,
    ensure_output_dir, OUTPUT_DIR, HAND_LABELS, COLORS, CARD_TO_IDX,
)

N_GAMES = 800


def pairwise_l2(vecs):
    """Mean pairwise L2 distance among a list of vectors (list of 3, each len-3)."""
    dists = []
    n = len(vecs)
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(np.array(vecs[i]) - np.array(vecs[j]))))
    return float(np.mean(dists)) if dists else 0.0


def analyze_portfolio(records):
    """
    At each decision, record:
      - pairwise L2 among b_opp[J], b_opp[Q], b_opp[K]
      - entropy of each b_opp[k]
      - entropy of b_mine
    Bucketed by decision_idx.
    """
    diversity_by_step  = defaultdict(list)   # step → [pairwise_L2, ...]
    entropy_by_step_k  = defaultdict(lambda: defaultdict(list))  # step → card_k → [entropy]
    bmine_entropy_by_step = defaultdict(list)

    for rec in records:
        for dec in rec.decisions:
            step = dec.decision_idx
            b_opp = dec.b_opp  # [[b_J], [b_Q], [b_K]] each len-3

            # Portfolio diversity
            diversity_by_step[step].append(pairwise_l2(b_opp))

            # Per-card entropy
            for k_idx, k_name in enumerate(HAND_LABELS):
                probs = [max(p, 1e-8) for p in b_opp[k_idx]]
                entropy_by_step_k[step][k_name].append(float(scipy_entropy(probs)))

            # b_mine entropy
            bmine_probs = [max(p, 1e-8) for p in dec.b_mine]
            bmine_entropy_by_step[step].append(float(scipy_entropy(bmine_probs)))

    steps = sorted(diversity_by_step.keys())
    diversity     = {s: float(np.mean(diversity_by_step[s]))         for s in steps}
    entropy_k     = {s: {k: float(np.mean(entropy_by_step_k[s][k])) for k in HAND_LABELS} for s in steps}
    bmine_entropy = {s: float(np.mean(bmine_entropy_by_step[s]))     for s in steps}
    return steps, diversity, entropy_k, bmine_entropy


def main():
    ensure_output_dir()

    ep_results = {}

    for ep in CHECKPOINT_EPISODES:
        print(f"D6: ep {ep:,} — playing {N_GAMES} games...")
        config, state_enc, belief_net, q_net = load_checkpoint(ep)
        records = run_greedy_games(state_enc, belief_net, q_net, config, n_games=N_GAMES)
        steps, diversity, entropy_k, bmine_entropy = analyze_portfolio(records)
        ep_results[ep] = {
            'steps': steps, 'diversity': diversity,
            'entropy_k': entropy_k, 'bmine_entropy': bmine_entropy,
        }
        avg_div = np.mean(list(diversity.values())) if diversity else 0
        print(f"  avg portfolio diversity: {avg_div:.4f}")

    ep_colors = dict(zip(CHECKPOINT_EPISODES, COLORS))

    # ── Plot 1: Portfolio diversity per decision step ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title('D6: Belief Portfolio Diversity — Mean Pairwise L2(b_opp[J], b_opp[Q], b_opp[K])\n'
                 '(higher = three imagined opponents behave differently; near 0 = portfolio collapsed)', fontsize=10)

    for ep in CHECKPOINT_EPISODES:
        r = ep_results[ep]
        steps = r['steps']
        divs  = [r['diversity'].get(s, 0) for s in steps]
        ax.plot(steps, divs, 'o-', color=ep_colors[ep], lw=2, ms=7, label=f'ep {ep:,}')

    ax.set_xlabel('Decision Index', fontsize=10)
    ax.set_ylabel('Mean Pairwise L2 Distance', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d6_diversity.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # ── Plot 2: Entropy of b_opp[k] per step ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('D6: b_opp Entropy per Decision Step', fontsize=12)

    hand_colors = {'J': '#E53935', 'Q': '#FB8C00', 'K': '#43A047'}
    ep_ls       = ['-', '--', '-.', ':', '-', '--']

    # Left: b_mine entropy
    ax = axes[0]
    ax.set_title('b_mine entropy over decision steps\n(should decrease as game reveals info)', fontsize=10)
    for ep, ls in zip(CHECKPOINT_EPISODES, ep_ls):
        r = ep_results[ep]
        steps = r['steps']
        ents  = [r['bmine_entropy'].get(s, 0) for s in steps]
        ax.plot(steps, ents, marker='o', ls=ls, color=ep_colors[ep], lw=2, ms=6, label=f'ep {ep:,}')
    ax.axhline(np.log(3), color='gray', ls=':', lw=1.5, label='uniform (log 3)')
    ax.set_xlabel('Decision Index', fontsize=9)
    ax.set_ylabel('Entropy (nats)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: b_opp[K] entropy at each step (K is most informative)
    ax = axes[1]
    ax.set_title('b_opp[K] entropy over decision steps\n(opponent\'s belief about me, if they hold K)', fontsize=10)
    for ep, ls in zip(CHECKPOINT_EPISODES, ep_ls):
        r = ep_results[ep]
        steps = r['steps']
        ents  = [r['entropy_k'].get(s, {}).get('K', 0) for s in steps]
        ax.plot(steps, ents, marker='o', ls=ls, color=ep_colors[ep], lw=2, ms=6, label=f'ep {ep:,}')
    ax.axhline(np.log(3), color='gray', ls=':', lw=1.5, label='uniform (log 3)')
    ax.set_xlabel('Decision Index', fontsize=9)
    ax.set_ylabel('Entropy (nats)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d6_entropy.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 3: Portfolio diversity over training episodes ────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title('D6: Belief Portfolio Diversity over Training\n'
                 '(aggregate mean across all decision steps; collapse → PIMC search is degenerate)', fontsize=11)

    eps_k = [ep / 1000 for ep in CHECKPOINT_EPISODES]
    avg_divs = []
    for ep in CHECKPOINT_EPISODES:
        r = ep_results[ep]
        vals = list(r['diversity'].values())
        avg_divs.append(float(np.mean(vals)) if vals else 0.0)

    bars = ax.bar(eps_k, avg_divs,
                  color=['#1565C0' if d > 0.05 else '#B71C1C' for d in avg_divs],
                  width=6, alpha=0.85)
    ax.set_xlabel('Episode (thousands)', fontsize=10)
    ax.set_ylabel('Avg Pairwise L2 Distance', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    for bar, d in zip(bars, avg_divs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{d:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d6_collapse.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── D6 Summary: avg portfolio diversity per episode ──────────────────")
    for ep, div in zip(CHECKPOINT_EPISODES, avg_divs):
        print(f"  ep {ep:>7,}: avg_diversity={div:.4f}")


if __name__ == '__main__':
    main()
