"""
D7: Belief Trace — Visualizing Belief Evolution Across a Game

Replays a FIXED canonical event sequence through each checkpoint's BeliefNet
and tracks how b_mine (Player 0's belief about Player 1's hand) changes.

Canonical game:
  Player 0 holds J,  Player 1 holds K  (true opponent hand = K, index 2)
  Event sequence:
    1. P0 calls   (check)
    2. P1 raises             ← b_mine UPDATES (opponent acted)
    3. P0 calls   (calls raise)
    4. Deal Q                ← b_mine UPDATES (deal event)
    5. P0 checks  (postflop)
    6. P1 raises  (postflop) ← b_mine UPDATES (opponent acted)
    7. P0 calls   (calls)

The same 7 events are replayed through every checkpoint. Because the belief
update is driven by BeliefNet weights, different checkpoints yield different
belief trajectories — this directly shows how the belief network's interpretation
of identical actions changes over training.

Key questions:
  - Does b_K (prob on true hand) rise after P1 raises?
  - Does belief become more confident (lower entropy) as the game deepens?
  - Or does BeliefNet update in the wrong direction?

Output:
  outputs/d7_belief_trace.png    — 6-row panel (one per checkpoint), bar charts per step
  outputs/d7_belief_lines.png    — line plots of b_J / b_Q / b_K vs step per checkpoint
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from diagnose import (
    CHECKPOINT_EPISODES, load_checkpoint, ensure_output_dir, OUTPUT_DIR,
    HAND_LABELS, COLORS,
)

import sys, os
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preliminary_experiments.alphazero.state_encoder import action_event_id, deal_event_id, CARD_TO_IDX
from preliminary_experiments.alphazero.belief import make_belief_state, update_belief_state
from scipy.stats import entropy as scipy_entropy

# ── Canonical event sequence ──────────────────────────────────────────────────
# Player 0 holds J,  Player 1 holds K
P0_HAND = 'J'
P1_HAND = 'K'   # true opponent hand
TRUE_OPP_IDX = CARD_TO_IDX[P1_HAND]   # = 2 (K)

CANONICAL_EVENTS = [
    # (event_id, actor, label)
    (action_event_id(0, 1), 0,    "P0 calls\n(check)"),
    (action_event_id(1, 2), 1,    "P1 raises\n★ b_mine↑"),
    (action_event_id(0, 1), 0,    "P0 calls\nraise"),
    (deal_event_id('Q'),    None,  "Deal Q\n★ b_mine↑"),
    (action_event_id(0, 1), 0,    "P0 checks\n(postflop)"),
    (action_event_id(1, 2), 1,    "P1 raises\n★ b_mine↑"),
    (action_event_id(0, 1), 0,    "P0 calls"),
]
EVENT_LABELS = ["Initial"] + [e[2] for e in CANONICAL_EVENTS]
N_STEPS = len(CANONICAL_EVENTS) + 1   # initial + 7 events = 8 snapshots

# Steps where b_mine actually changes (opponent acts or deal)
BELIEF_UPDATE_STEPS = {2, 4, 6}   # 1-indexed after initial (events 2, 4, 6)


def trace_belief(state_enc, belief_net, config):
    """
    Replay the canonical event sequence for player 0 (holding J).
    Returns: list of 8 belief vectors [b_J, b_Q, b_K], one per step (initial + 7 events).
    """
    with torch.no_grad():
        bs = make_belief_state(P0_HAND, config.d_model)
        traces = [bs.b_mine.tolist()]   # step 0: informed prior

        for eid, actor, _ in CANONICAL_EVENTS:
            e_prime, P_new = state_enc.encode_event(eid, bs.P_current, bs.P_history)
            update_belief_state(bs, actor, player_i=0, e_prime=e_prime, P_new=P_new,
                                belief_net=belief_net)
            traces.append(bs.b_mine.tolist())

    return traces  # list of 8 × [b_J, b_Q, b_K]


def main():
    ensure_output_dir()

    all_traces = {}
    print("D7: Tracing belief over canonical game sequence...")
    for ep in CHECKPOINT_EPISODES:
        config, state_enc, belief_net, q_net = load_checkpoint(ep)
        traces = trace_belief(state_enc, belief_net, config)
        all_traces[ep] = traces
        # Quick summary: b_K at start, after P1-raise, after deal, after postflop-raise
        bK_vals = [traces[0][2], traces[2][2], traces[4][2], traces[6][2]]
        print(f"  ep {ep:>7,}: b_K @ [init, post-raise1, post-deal, post-raise2] = "
              f"[{bK_vals[0]:.3f}, {bK_vals[1]:.3f}, {bK_vals[2]:.3f}, {bK_vals[3]:.3f}]")

    ep_colors = dict(zip(CHECKPOINT_EPISODES, COLORS))
    hand_colors = {'J': '#E53935', 'Q': '#FB8C00', 'K': '#43A047'}
    bar_colors  = [hand_colors['J'], hand_colors['Q'], hand_colors['K']]
    step_xs = list(range(N_STEPS))

    # ── Plot 1: Bar chart panel (6 rows × 8 columns) ─────────────────────────
    fig, axes = plt.subplots(len(CHECKPOINT_EPISODES), N_STEPS,
                             figsize=(18, 10), sharey=True)
    fig.suptitle(
        f'D7: Belief Trace — Player 0 holds J, Opponent holds K (★ = steps where b_mine updates)\n'
        f'Each cell: belief distribution [b_J, b_Q, b_K]. Green bar = true hand (K).',
        fontsize=11
    )

    for row_i, ep in enumerate(CHECKPOINT_EPISODES):
        traces = all_traces[ep]
        for col_j, belief in enumerate(traces):
            ax = axes[row_i][col_j]

            colors = [hand_colors['J'], hand_colors['Q'], hand_colors['K']]
            # Highlight true hand with green
            colors[TRUE_OPP_IDX] = '#1B5E20'

            bars = ax.bar([0, 1, 2], belief, color=colors, width=0.7, alpha=0.85)
            ax.set_ylim(0, 1.05)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['J', 'Q', 'K'], fontsize=7)
            ax.tick_params(axis='y', labelsize=6)

            # Mark belief-update steps with border
            if col_j > 0 and col_j in BELIEF_UPDATE_STEPS:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#1565C0')
                    spine.set_linewidth(2)

            # Column headers (event labels) on top row
            if row_i == 0:
                ax.set_title(EVENT_LABELS[col_j], fontsize=7, pad=2)

            # Row labels (checkpoint) on leftmost column
            if col_j == 0:
                ax.set_ylabel(f'ep {ep//1000}K', fontsize=8, rotation=90, labelpad=2)

            # Annotate true hand probability
            ax.text(TRUE_OPP_IDX, belief[TRUE_OPP_IDX] + 0.04,
                    f'{belief[TRUE_OPP_IDX]:.2f}', ha='center', fontsize=6,
                    color='#1B5E20', fontweight='bold')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d7_belief_trace.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # ── Plot 2: Line plots of b_J, b_Q, b_K over steps ───────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        'D7: Belief Lines — b_J / b_Q / b_K vs Event Step\n'
        '(true hand = K; shaded band = belief-update events 2/4/6)',
        fontsize=12
    )

    for idx, ep in enumerate(CHECKPOINT_EPISODES):
        ax = axes[idx // 3][idx % 3]
        traces = np.array(all_traces[ep])   # (8, 3)

        for hand_i, (hname, hcolor) in enumerate(hand_colors.items()):
            lw = 2.5 if hname == 'K' else 1.5
            ls = '-' if hname == 'K' else '--'
            ax.plot(step_xs, traces[:, hand_i], lw=lw, ls=ls,
                    color=hcolor, marker='o', ms=5, label=f'b_{hname}')

        # Shade belief-update steps
        for update_step in BELIEF_UPDATE_STEPS:
            ax.axvspan(update_step - 0.3, update_step + 0.3, alpha=0.12, color='#1565C0')

        ax.axhline(1/3, color='gray', ls=':', lw=1, alpha=0.6, label='1/3 prior')
        ax.set_xticks(step_xs)
        ax.set_xticklabels([f'{i}' for i in step_xs], fontsize=8)
        ax.set_ylim(0, 0.85)
        ax.set_xlabel('Event step', fontsize=9)
        ax.set_ylabel('Probability', fontsize=9)
        ax.set_title(f'ep {ep:,}', fontsize=10)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.25)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d7_belief_lines.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 3: b_K trajectory comparison — all checkpoints on one plot ───────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('D7: b_K (Probability on True Hand K) Over Game Steps\n'
                 '(Blue shaded = steps where b_mine updates; dashed = 1/3 random)', fontsize=12)

    for ep in CHECKPOINT_EPISODES:
        traces = np.array(all_traces[ep])
        bK = traces[:, TRUE_OPP_IDX]
        ax1.plot(step_xs, bK, 'o-', color=ep_colors[ep], lw=2, ms=7, label=f'ep {ep:,}')

    for update_step in BELIEF_UPDATE_STEPS:
        ax1.axvspan(update_step - 0.3, update_step + 0.3, alpha=0.12, color='#1565C0')
    ax1.axhline(1/3, color='gray', ls='--', lw=1.5, label='random (1/3)')
    ax1.axhline(0.4, color='purple', ls=':', lw=1, label='informed prior (0.4)')
    ax1.set_xlabel('Event step', fontsize=10)
    ax1.set_ylabel('P(opponent = K)', fontsize=10)
    ax1.set_title('b_K trajectory (K is the true opponent hand)', fontsize=10)
    ax1.set_ylim(0, 0.9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Entropy over steps
    for ep in CHECKPOINT_EPISODES:
        traces = np.array(all_traces[ep])
        ents = [float(scipy_entropy([max(p, 1e-8) for p in row])) for row in traces]
        ax2.plot(step_xs, ents, 'o-', color=ep_colors[ep], lw=2, ms=7, label=f'ep {ep:,}')

    for update_step in BELIEF_UPDATE_STEPS:
        ax2.axvspan(update_step - 0.3, update_step + 0.3, alpha=0.12, color='#1565C0')
    ax2.axhline(np.log(3), color='gray', ls='--', lw=1.5, label='uniform (log 3)')
    ax2.axhline(1.054, color='purple', ls=':', lw=1, label='informed prior (1.054)')
    ax2.set_xlabel('Event step', fontsize=10)
    ax2.set_ylabel('Entropy (nats)', fontsize=10)
    ax2.set_title('Belief entropy (lower = more confident)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d7_bK_and_entropy.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── D7 Summary: b_K at key steps (true hand = K) ────────────────────")
    print(f"{'Episode':>10}  {'init':>6}  {'→raise1':>9}  {'→deal':>7}  {'→raise2':>9}  {'final':>7}  {'direction'}")
    for ep in CHECKPOINT_EPISODES:
        traces = all_traces[ep]
        bK = [t[TRUE_OPP_IDX] for t in traces]
        # Direction: does b_K rise after P1 raises?
        rise1 = "↑" if bK[2] > bK[1] else "↓"
        rise2 = "↑" if bK[6] > bK[5] else "↓"
        direction = f"raise1:{rise1}  raise2:{rise2}"
        print(f"{ep:>10,}  {bK[0]:>6.3f}  {bK[2]:>9.3f}  {bK[4]:>7.3f}  {bK[6]:>9.3f}  {bK[7]:>7.3f}  {direction}")


if __name__ == '__main__':
    main()
