"""
D8: Q-Value Tracking at Canonical States + Value-Based Comparison

Defines 4 canonical game states and computes:
  (A) AZ Q-network: Q(P_t, hand, b_mine) → [Q_fold, Q_call, Q_raise] for each checkpoint
  (B) Value-Based agent: V(s') after each action (1-step lookahead) — fixed reference

By plotting both on the same axis we can see:
  - Whether AZ Q-values oscillate across checkpoints (instability signal)
  - Whether AZ agrees or disagrees with value_based agent recommendations
  - Whether AZ greedy policy (argmax Q) matches value_based policy

Canonical states (all from Player 0's perspective):

  A: Opening preflop,  hand=J  → P_t=zeros, b=prior(J)  [weak hand, first to act]
  B: Opening preflop,  hand=K  → P_t=zeros, b=prior(K)  [strong hand, first to act]
  C: P0 responds to P1's preflop raise,  hand=J
     event history: [P0_call, P1_raise]  → P_t after 2 events, b_mine updated by P1's raise
  D: P0 opens postflop after both checked + deal=Q,  hand=K
     event history: [P0_call, P1_call, deal_Q]

Output:
  outputs/d8_qvals.png           — Q-values per canonical state × checkpoint
  outputs/d8_policy.png          — greedy action per canonical state × checkpoint
  outputs/d8_vs_valuebased.png   — AZ argmax vs value_based argmax comparison
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preliminary_experiments.alphazero.state_encoder import action_event_id, deal_event_id, CARD_TO_IDX
from preliminary_experiments.alphazero.belief import make_belief_state, update_belief_state
from preliminary_experiments.alphazero.agent import hand_onehot
from engine.leduc_game import Action, LeducGame
from engine.observation import Observation
from agents.value_based.agent import ValueBasedAgent

from diagnose import (
    CHECKPOINT_EPISODES, load_checkpoint, ensure_output_dir, OUTPUT_DIR,
    HAND_LABELS, COLORS,
)

ACTION_NAMES = ['fold', 'call', 'raise']
ACTION_COLORS = {'fold': '#E53935', 'call': '#FB8C00', 'raise': '#43A047'}


# ── Canonical state definitions ───────────────────────────────────────────────

def _obs(hand, board, pot, current_round, raises=0, player=0):
    """Construct a minimal Observation for value_based evaluation."""
    legal = [Action.FOLD, Action.CALL]
    if raises < 2:
        legal.append(Action.RAISE)
    return Observation(
        player_hand=hand, board=board, pot=pot,
        current_player=player, current_round=current_round,
        legal_actions=legal, is_finished=False,
        raises_this_round=raises,
    )


CANONICAL_STATES = [
    {
        'id':     'A',
        'name':   'Opening (hand=J)\nPreflop, no history',
        'events': [],       # no events; P_t = zeros, b = prior(J)
        'hand':   'J',
        'obs':    _obs('J', None, [1, 1], current_round=0, raises=0),
    },
    {
        'id':     'B',
        'name':   'Opening (hand=K)\nPreflop, no history',
        'events': [],
        'hand':   'K',
        'obs':    _obs('K', None, [1, 1], current_round=0, raises=0),
    },
    {
        'id':     'C',
        'name':   'Respond to P1 raise (hand=J)\nPreflop: P0 checked, P1 raised',
        # P0 called (check), P1 raised → pot=[1,3], P0 must act
        'events': [
            (action_event_id(0, 1), 0),  # P0 calls (check)
            (action_event_id(1, 2), 1),  # P1 raises → b_mine updates
        ],
        'hand':   'J',
        'obs':    _obs('J', None, [1, 3], current_round=0, raises=1),
    },
    {
        'id':     'D',
        'name':   'Open postflop (hand=K)\nBoard=Q; preflop: P0 checked, P1 checked',
        # P0 checks, P1 checks → round ends → deal Q
        'events': [
            (action_event_id(0, 1), 0),  # P0 calls (check)
            (action_event_id(1, 1), 1),  # P1 calls (check) → round ends + deal
            (deal_event_id('Q'),    None), # deal Q → b_mine updates
        ],
        'hand':   'K',
        'obs':    _obs('K', 'Q', [1, 1], current_round=1, raises=0),
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def replay_to_belief(events, hand, state_enc, belief_net, config):
    """Replay a sequence of (event_id, actor) events and return player 0's BeliefState."""
    with torch.no_grad():
        bs = make_belief_state(hand, config.d_model)
        for eid, actor in events:
            e_prime, P_new = state_enc.encode_event(eid, bs.P_current, bs.P_history)
            update_belief_state(bs, actor, player_i=0, e_prime=e_prime,
                                P_new=P_new, belief_net=belief_net)
    return bs


def compute_az_qvals(state_def, state_enc, belief_net, q_net, config):
    """Compute Q-values for a canonical state. Returns {0: Q_fold, 1: Q_call, 2: Q_raise}."""
    bs = replay_to_belief(state_def['events'], state_def['hand'],
                          state_enc, belief_net, config)
    with torch.no_grad():
        q_vals = q_net(bs.P_current, hand_onehot(state_def['hand']), bs.b_mine)
    return q_vals.tolist()   # [Q_fold, Q_call, Q_raise]


def compute_vb_values(state_def, vb_agent):
    """
    Compute value_based 1-step lookahead values for each action.
    Returns: dict action_id → value (from V(s') or fold-immediate).
    """
    obs = state_def['obs']
    evals = vb_agent.get_action_evaluations(obs)
    return {int(e['action']): e['value'] for e in evals}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dir()

    # Load value_based agent (fixed reference across all checkpoints)
    vb_path = os.path.join(_ROOT, 'agents', 'value_based', 'checkpoint.pt')
    vb_agent = ValueBasedAgent(model_path=vb_path)
    vb_agent.set_train_mode(False)
    print("D8: Loaded value_based agent")

    # Compute value_based values for each canonical state (fixed across checkpoints)
    vb_values = {}
    for s in CANONICAL_STATES:
        vb_values[s['id']] = compute_vb_values(s, vb_agent)

    # Compute AZ Q-values for each checkpoint × canonical state
    az_qvals = {s['id']: [] for s in CANONICAL_STATES}   # sid → list of [Q_fold, Q_call, Q_raise]

    for ep in CHECKPOINT_EPISODES:
        print(f"D8: ep {ep:,}...")
        config, state_enc, belief_net, q_net = load_checkpoint(ep)
        for s in CANONICAL_STATES:
            q = compute_az_qvals(s, state_enc, belief_net, q_net, config)
            az_qvals[s['id']].append(q)
            legal_ids = [int(a) for a in s['obs'].legal_actions]
            greedy = max(legal_ids, key=lambda aid: q[aid])
            print(f"  State {s['id']} ({s['hand']}): "
                  f"fold={q[0]:+.3f}  call={q[1]:+.3f}  raise={q[2]:+.3f}  "
                  f"→ greedy={ACTION_NAMES[greedy]}")

    ep_colors = dict(zip(CHECKPOINT_EPISODES, COLORS))
    eps_k = [ep / 1000 for ep in CHECKPOINT_EPISODES]

    # ── Plot 1: Q-values over checkpoints for each canonical state ─────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('D8: AZ Q-Values at Canonical States Over Training\n'
                 '(oscillation = unstable Q-network; colored dashes = value_based V(s\'))',
                 fontsize=12)

    for idx, s in enumerate(CANONICAL_STATES):
        ax = axes[idx // 2][idx % 2]
        qvals = np.array(az_qvals[s['id']])   # (6, 3)
        legal_ids = [int(a) for a in s['obs'].legal_actions]

        for action_id in range(3):
            if action_id not in legal_ids:
                continue
            aname = ACTION_NAMES[action_id]
            ax.plot(eps_k, qvals[:, action_id], 'o-',
                    color=ACTION_COLORS[aname], lw=2, ms=8, label=f'Q_{aname} (AZ)')

        # Value_based V(s') as horizontal dashed lines
        vb_v = vb_values[s['id']]
        for action_id, val in vb_v.items():
            aname = ACTION_NAMES[action_id]
            ax.axhline(val, color=ACTION_COLORS[aname], ls='--', lw=1.5, alpha=0.7)
            ax.text(eps_k[-1] + 1, val, f'VB_{aname}={val:+.2f}',
                    va='center', fontsize=7, color=ACTION_COLORS[aname])

        # Mark greedy action at each checkpoint
        for ep_i, (ep, q_row) in enumerate(zip(CHECKPOINT_EPISODES, qvals)):
            best = max(legal_ids, key=lambda aid: q_row[aid])
            ax.scatter([ep / 1000], [q_row[best]], s=120, zorder=5,
                       color=ACTION_COLORS[ACTION_NAMES[best]],
                       edgecolors='black', linewidths=1.5)

        ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)
        ax.set_xlabel('Episode (thousands)', fontsize=9)
        ax.set_ylabel('Q-value (chips)', fontsize=9)
        ax.set_title(s['name'], fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d8_qvals.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # ── Plot 2: Greedy action (policy) per state over checkpoints ─────────────
    fig, axes = plt.subplots(1, len(CANONICAL_STATES), figsize=(16, 4))
    fig.suptitle('D8: AZ Greedy Policy at Canonical States (argmax Q)\n'
                 '— and value_based recommendation (dashed line) —', fontsize=12)

    action_to_int = {'fold': 0, 'call': 1, 'raise': 2}
    action_to_y   = {0: 0, 1: 1, 2: 2}   # y-axis position

    for idx, s in enumerate(CANONICAL_STATES):
        ax = axes[idx]
        qvals = np.array(az_qvals[s['id']])
        legal_ids = [int(a) for a in s['obs'].legal_actions]

        # AZ greedy action at each checkpoint
        greedy_actions = [max(legal_ids, key=lambda aid: q[aid]) for q in qvals]
        colors = [ACTION_COLORS[ACTION_NAMES[a]] for a in greedy_actions]
        ax.scatter(eps_k, greedy_actions, c=colors, s=200, zorder=5, marker='D')
        ax.plot(eps_k, greedy_actions, 'k-', lw=1, alpha=0.3)

        # Value_based recommended action
        vb_v = vb_values[s['id']]
        if vb_v:
            vb_best = max(vb_v, key=lambda aid: vb_v[aid])
            ax.axhline(vb_best, color='black', ls='--', lw=2, alpha=0.6,
                       label=f'VB: {ACTION_NAMES[vb_best]}')

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['fold', 'call', 'raise'], fontsize=9)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel('Episode (thousands)', fontsize=9)
        ax.set_title(s['name'], fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.3)

        # Shade disagreement regions (AZ ≠ value_based)
        if vb_v:
            for ep_i, act in enumerate(greedy_actions):
                if act != vb_best:
                    ax.axvspan(eps_k[ep_i] - 5, eps_k[ep_i] + 5,
                               alpha=0.15, color='red')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d8_policy.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 3: Q-value range (max-min) per state — stability proxy ───────────
    fig, axes = plt.subplots(1, len(CANONICAL_STATES), figsize=(14, 4))
    fig.suptitle('D8: Q-Value Range (max − min across actions) Over Training\n'
                 '(narrow range = degenerate signal; high = action differentiation)',
                 fontsize=12)

    for idx, s in enumerate(CANONICAL_STATES):
        ax = axes[idx]
        qvals = np.array(az_qvals[s['id']])
        legal_ids = [int(a) for a in s['obs'].legal_actions]
        ranges = [qvals[i, legal_ids].max() - qvals[i, legal_ids].min()
                  for i in range(len(CHECKPOINT_EPISODES))]

        bars = ax.bar(eps_k, ranges, width=6,
                      color=[ep_colors[ep] for ep in CHECKPOINT_EPISODES], alpha=0.85)
        for bar, r in zip(bars, ranges):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{r:.2f}', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('Episode (thousands)', fontsize=9)
        ax.set_ylabel('Q range (chips)', fontsize=9)
        ax.set_title(s['name'], fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d8_qval_range.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n── D8 Summary: greedy action per canonical state per checkpoint ──────")
    header = f"{'Episode':>10}" + "".join(f"  {s['id']:>6}" for s in CANONICAL_STATES)
    print(header)
    print(f"{'VB baseline':>10}" + "".join(
        f"  {ACTION_NAMES[max(vb_values[s['id']], key=lambda a: vb_values[s['id']][a])]:>6}"
        if vb_values[s['id']] else "     ?" for s in CANONICAL_STATES))
    print("—" * (10 + 8 * len(CANONICAL_STATES)))

    for ep_i, ep in enumerate(CHECKPOINT_EPISODES):
        row = f"{ep:>10,}"
        for s in CANONICAL_STATES:
            q = az_qvals[s['id']][ep_i]
            legal_ids = [int(a) for a in s['obs'].legal_actions]
            best = max(legal_ids, key=lambda aid: q[aid])
            row += f"  {ACTION_NAMES[best]:>6}"
        print(row)


if __name__ == '__main__':
    main()
