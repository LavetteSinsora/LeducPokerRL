"""
D2: Q* vs Q Agreement — Is the Q-Network Tracking Its Training Targets?

Compares two Q estimates at each decision point:
  Q_net(s):  direct network output (what the network has learned)
  Q_star(s): PIMC search result   (the training target actually used)

Key measurements:
  1. MSE(Q_net, Q_star) per checkpoint — divergence = network failed to fit targets
  2. Q_star value range (max - min across legal actions) — narrow = degenerate signal
  3. Q_net hand-stratified values — does the network output higher Q for K > Q > J?
  4. Rank agreement: does argmax(Q_net) == argmax(Q_star)?

NOTE: This script requires PIMC search (k=5 rollouts per action per imagined hand).
      It is significantly slower than D1-D6.
      Default: 150 games per checkpoint. Reduce N_GAMES to speed up.

Usage:
  python d2_q_agreement.py [--games N]

Output:
  outputs/d2_mse.png          — MSE(Q_net, Q_star) per checkpoint
  outputs/d2_scatter.png      — scatter Q_net vs Q_star per checkpoint
  outputs/d2_qstar_range.png  — Q_star value range (signal quality) per checkpoint
  outputs/d2_rank.png         — argmax agreement rate per checkpoint
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from preliminary_experiments.alphazero.belief import make_belief_state, update_belief_state
from preliminary_experiments.alphazero.state_encoder import action_event_id, deal_event_id, IDX_TO_CARD, CARD_TO_IDX
from preliminary_experiments.alphazero.agent import hand_onehot
from preliminary_experiments.alphazero.rollout import pimc_search
from engine.leduc_game import LeducGame, Action

from diagnose import (
    CHECKPOINT_EPISODES, load_checkpoint, ensure_output_dir, OUTPUT_DIR,
    HAND_LABELS, COLORS,
)

DEFAULT_N_GAMES = 150
K_ROLLOUTS = 5  # fewer than training (10) to keep it tractable


@torch.no_grad()
def run_pimc_comparison_games(state_enc, belief_net, q_net, config, n_games):
    """
    Play n_games. At each decision, compute both Q_star (PIMC) and Q_net (direct).
    Returns list of comparison records.
    """
    results = []

    for game_i in range(n_games):
        if (game_i + 1) % 25 == 0:
            print(f"    game {game_i + 1}/{n_games}...")

        game = LeducGame()
        game.reset()
        hands = list(game.player_hands)
        bs = [make_belief_state(hands[p], config.d_model) for p in range(2)]

        while not game.is_finished:
            p = game.current_player
            legal = game.get_legal_actions()
            legal_ids = [int(a) for a in legal]
            obs = game.get_observation(viewer_id=p)

            # PIMC Q_star
            q_star = pimc_search(
                obs=obs,
                player_i=p,
                h_i=hands[p],
                bs_i=bs[p],
                legal_actions=legal,
                state_enc=state_enc,
                belief_net=belief_net,
                q_net=q_net,
                k=K_ROLLOUTS,
                T=config.temperature,
            )

            # Direct Q_net
            q_net_vals = q_net(bs[p].P_current, hand_onehot(hands[p]), bs[p].b_mine)

            # Record only legal action values
            q_star_legal = q_star[legal_ids].tolist()
            q_net_legal  = q_net_vals[legal_ids].tolist()

            results.append({
                'hand': hands[p],
                'opp_hand': hands[1 - p],
                'round': game.current_round,
                'q_star': q_star_legal,
                'q_net':  q_net_legal,
                'legal_ids': legal_ids,
                'q_star_argmax': int(torch.argmax(q_star[torch.tensor(legal_ids)]).item()),
                'q_net_argmax':  int(torch.argmax(q_net_vals[torch.tensor(legal_ids)]).item()),
            })

            # Greedy action (use Q_star for actual play to stay consistent with training)
            best_id = legal_ids[int(torch.argmax(q_star[torch.tensor(legal_ids)]).item())]
            actor = game.current_player
            pre_board = game.board
            _, _, done, _ = game.step(Action(best_id))

            act_eid = action_event_id(actor, best_id)
            deal_eid = None
            if game.board is not None and game.board != pre_board:
                deal_eid = deal_event_id(game.board)

            for pi in range(2):
                e_prime, P_new = state_enc.encode_event(act_eid, bs[pi].P_current, bs[pi].P_history)
                update_belief_state(bs[pi], actor, pi, e_prime, P_new, belief_net)

            if deal_eid is not None:
                for pi in range(2):
                    e_prime_d, P_new_d = state_enc.encode_event(deal_eid, bs[pi].P_current, bs[pi].P_history)
                    update_belief_state(bs[pi], None, pi, e_prime_d, P_new_d, belief_net)

            if done:
                break

    return results


def compute_metrics(records):
    """
    From a list of comparison records, compute:
      mse:          MSE between Q_net and Q_star (on legal actions)
      rank_agree:   fraction where argmax(Q_net) == argmax(Q_star)
      qstar_range:  mean (max - min) of Q_star across legal actions
      qnet_range:   mean (max - min) of Q_net across legal actions
    Also per-hand Q_net mean by hand.
    """
    mse_vals = []
    rank_agree = []
    qstar_ranges = []
    qnet_ranges  = []
    qnet_by_hand = defaultdict(list)

    for r in records:
        qs  = np.array(r['q_star'])
        qn  = np.array(r['q_net'])
        mse_vals.append(float(np.mean((qs - qn) ** 2)))
        rank_agree.append(r['q_star_argmax'] == r['q_net_argmax'])
        qstar_ranges.append(float(qs.max() - qs.min()))
        qnet_ranges.append(float(qn.max() - qn.min()))
        for q_val in qn:
            qnet_by_hand[r['hand']].append(q_val)

    return {
        'mse':         float(np.mean(mse_vals)),
        'rank_agree':  float(np.mean(rank_agree)),
        'qstar_range': float(np.mean(qstar_ranges)),
        'qnet_range':  float(np.mean(qnet_ranges)),
        'qnet_mean_by_hand': {h: float(np.mean(qnet_by_hand[h])) for h in HAND_LABELS
                               if h in qnet_by_hand},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=DEFAULT_N_GAMES,
                        help='Games per checkpoint (default 150; more = slower but more accurate)')
    args = parser.parse_args()

    ensure_output_dir()

    ep_records  = {}
    ep_metrics  = {}

    for ep in CHECKPOINT_EPISODES:
        print(f"D2: ep {ep:,} — playing {args.games} games with PIMC search (k={K_ROLLOUTS})...")
        config, state_enc, belief_net, q_net = load_checkpoint(ep)
        records = run_pimc_comparison_games(state_enc, belief_net, q_net, config, args.games)
        metrics = compute_metrics(records)
        ep_records[ep] = records
        ep_metrics[ep] = metrics
        print(f"  MSE={metrics['mse']:.4f}  rank_agree={metrics['rank_agree']:.3f}  "
              f"Q*_range={metrics['qstar_range']:.4f}  Q_net_range={metrics['qnet_range']:.4f}")

    ep_colors = dict(zip(CHECKPOINT_EPISODES, COLORS))
    eps_k = [ep / 1000 for ep in CHECKPOINT_EPISODES]

    # ── Plot 1: MSE and rank agreement over training ───────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('D2: Q-Network vs Q* (PIMC) Agreement', fontsize=13)

    mses  = [ep_metrics[ep]['mse']        for ep in CHECKPOINT_EPISODES]
    ranks = [ep_metrics[ep]['rank_agree'] for ep in CHECKPOINT_EPISODES]

    ax1.bar(eps_k, mses, width=6, color=[ep_colors[ep] for ep in CHECKPOINT_EPISODES], alpha=0.85)
    ax1.set_xlabel('Episode (thousands)', fontsize=10)
    ax1.set_ylabel('MSE(Q_net, Q*)', fontsize=10)
    ax1.set_title('MSE between network Q and PIMC Q*\n(lower = network fits training targets better)', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    for x, y in zip(eps_k, mses):
        ax1.text(x, y + max(mses) * 0.02, f'{y:.4f}', ha='center', va='bottom', fontsize=9)

    ax2.bar(eps_k, ranks, width=6, color=[ep_colors[ep] for ep in CHECKPOINT_EPISODES], alpha=0.85)
    ax2.axhline(1/3, color='gray', ls='--', lw=1.5, label='random (1/3)')
    ax2.set_xlabel('Episode (thousands)', fontsize=10)
    ax2.set_ylabel('Rank Agreement Rate', fontsize=10)
    ax2.set_title('argmax(Q_net) == argmax(Q*)\n(higher = network picks same best action as search)', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(fontsize=9)
    for x, y in zip(eps_k, ranks):
        ax2.text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d2_mse.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # ── Plot 2: Scatter Q_net vs Q_star per checkpoint ────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('D2: Q_net vs Q* Scatter (colored by hand)\n'
                 '(diagonal = perfect agreement; spread = divergence)', fontsize=12)

    hand_colors = {'J': '#E53935', 'Q': '#FB8C00', 'K': '#43A047'}

    for idx, ep in enumerate(CHECKPOINT_EPISODES):
        ax = axes[idx // 3][idx % 3]
        recs = ep_records[ep]

        for hand in HAND_LABELS:
            xs, ys = [], []
            for r in recs:
                if r['hand'] == hand:
                    xs.extend(r['q_star'])
                    ys.extend(r['q_net'])
            if xs:
                ax.scatter(xs, ys, c=hand_colors[hand], alpha=0.3, s=15, label=f'{hand}')

        # Diagonal reference
        all_vals = [v for r in recs for v in r['q_star'] + r['q_net']]
        if all_vals:
            mn, mx = min(all_vals), max(all_vals)
            ax.plot([mn, mx], [mn, mx], 'k--', lw=1, alpha=0.5, label='y=x')

        ax.set_xlabel('Q* (PIMC)', fontsize=8)
        ax.set_ylabel('Q_net (network)', fontsize=8)
        ax.set_title(f'ep {ep:,}  MSE={ep_metrics[ep]["mse"]:.4f}', fontsize=10)
        if idx == 0:
            ax.legend(fontsize=7, markerscale=2)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d2_scatter.png'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 3: Q_star value range over training ──────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title('D2: Q* Value Range (max − min across legal actions)\n'
                 '(narrow = degenerate training signal; wide = search provides clear guidance)', fontsize=11)

    qstar_ranges = [ep_metrics[ep]['qstar_range'] for ep in CHECKPOINT_EPISODES]
    qnet_ranges  = [ep_metrics[ep]['qnet_range']  for ep in CHECKPOINT_EPISODES]
    x = np.array(eps_k)
    width = 2.5

    ax.bar(x - width/2, qstar_ranges, width=width, label='Q* range (PIMC)', color='#1565C0', alpha=0.85)
    ax.bar(x + width/2, qnet_ranges,  width=width, label='Q_net range', color='#B71C1C', alpha=0.85)
    ax.set_xlabel('Episode (thousands)', fontsize=10)
    ax.set_ylabel('Value Range (max − min)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/d2_qstar_range.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── D2 Summary ──────────────────────────────────────────────────────")
    print(f"{'Episode':>10}  {'MSE':>8}  {'Rank Agree':>12}  {'Q* Range':>10}  {'Qnet Range':>12}")
    for ep in CHECKPOINT_EPISODES:
        m = ep_metrics[ep]
        print(f"{ep:>10,}  {m['mse']:>8.4f}  {m['rank_agree']:>12.3f}  "
              f"{m['qstar_range']:>10.4f}  {m['qnet_range']:>12.4f}")


if __name__ == '__main__':
    main()
