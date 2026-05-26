"""
Shared utilities for AlphaZero agent diagnosis.

Provides:
  - load_checkpoint(episode) → (config, state_enc, belief_net, q_net)
  - run_greedy_games(...)    → list[GameRecord]
  - CHECKPOINT_EPISODES, OUTPUT_DIR
"""

import os
import sys
import torch
from dataclasses import dataclass
from typing import List, Tuple

# ── Project root on path ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preliminary_experiments.alphazero.config import AZConfig
from preliminary_experiments.alphazero.state_encoder import (
    StateEncoder, action_event_id, deal_event_id,
    CARD_TO_IDX, IDX_TO_CARD,
)
from preliminary_experiments.alphazero.belief import BeliefNet, make_belief_state, update_belief_state
from preliminary_experiments.alphazero.agent import QNet, hand_onehot
from engine.leduc_game import LeducGame, Action


# ── Constants ─────────────────────────────────────────────────────────────────

CHECKPOINT_EPISODES = [70000, 80000, 90000, 100000, 110000, 120000]
_CHECKPOINT_DIR = os.path.join(_ROOT, 'outputs', 'alphazero_v1')
OUTPUT_DIR = os.path.join(_HERE, 'outputs')

EVENT_LABELS = [
    'P0_fold', 'P0_call', 'P0_raise',
    'P1_fold', 'P1_call', 'P1_raise',
    'deal_J',  'deal_Q',  'deal_K',
]

HAND_LABELS = ['J', 'Q', 'K']
ACTION_LABELS = ['fold', 'call', 'raise']
COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0', '#00BCD4']


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def checkpoint_path(episode: int) -> str:
    return os.path.join(_CHECKPOINT_DIR, f'checkpoint_ep{episode:07d}.pt')


def load_checkpoint(episode: int) -> Tuple[AZConfig, StateEncoder, BeliefNet, QNet]:
    """Load networks from a checkpoint. Returns eval-mode networks."""
    config = AZConfig()
    state_enc = StateEncoder(config)
    belief_net = BeliefNet(config)
    q_net = QNet(config)

    path = checkpoint_path(episode)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    state_enc.load_state_dict(ckpt['state_enc'])
    belief_net.load_state_dict(ckpt['belief_net'])
    q_net.load_state_dict(ckpt['q_net'])

    state_enc.eval()
    belief_net.eval()
    q_net.eval()
    return config, state_enc, belief_net, q_net


# ── Game record types ─────────────────────────────────────────────────────────

@dataclass
class DecisionInfo:
    """All information captured at one decision point."""
    player: int
    hand: str               # J, Q, K (this player's hand)
    opp_hand: str           # opponent's actual hand (ground truth)
    decision_idx: int       # 0-indexed count across all decisions in game
    round: int              # 0=preflop, 1=postflop
    action_taken: int       # 0=fold, 1=call, 2=raise
    legal_actions: list
    q_vals: List[float]     # [Q_fold, Q_call, Q_raise] from network
    P_t: List[float]        # public state d-vector at this decision
    b_mine: List[float]     # [b_J, b_Q, b_K] — belief about opponent's hand
    b_opp: List[List[float]]  # [[b_opp_J], [b_opp_Q], [b_opp_K]] each (3,)


@dataclass
class GameRecord:
    """Complete record of one greedy self-play game."""
    hands: List[str]                    # [hand_p0, hand_p1]
    decisions: List[DecisionInfo]
    final_rewards: List[float]          # [reward_p0, reward_p1]
    P_at_events: List[List[float]]      # P_t after each event (includes initial zeros)
    event_ids: List[int]                # event ID for each event


# ── Greedy game runner ────────────────────────────────────────────────────────

def run_greedy_games(
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    q_net: QNet,
    config: AZConfig,
    n_games: int = 1000,
) -> List[GameRecord]:
    """
    Play n_games using greedy Q-network (argmax, no PIMC search).

    Returns detailed GameRecord for each game including per-decision
    belief states, Q-values, and public state vectors.
    """
    records = []

    with torch.no_grad():
        for _ in range(n_games):
            game = LeducGame()
            game.reset()
            hands = list(game.player_hands)

            # Per-player belief states
            bs = [make_belief_state(hands[p], config.d_model) for p in range(2)]

            decisions: List[DecisionInfo] = []
            P_at_events: List[List[float]] = [bs[0].P_current.tolist()]
            event_ids: List[int] = []
            decision_idx = 0
            final_rewards = [0.0, 0.0]

            while not game.is_finished:
                p = game.current_player
                legal = game.get_legal_actions()
                legal_ids = [int(a) for a in legal]

                # Capture state BEFORE action
                P_t = bs[p].P_current.clone()
                b_mine = bs[p].b_mine.clone()
                b_opp_snap = [b.clone() for b in bs[p].b_opp]

                q_vals = q_net(P_t, hand_onehot(hands[p]), b_mine)

                # Greedy: argmax among legal actions
                best_action_id = max(legal_ids, key=lambda aid: q_vals[aid].item())

                decisions.append(DecisionInfo(
                    player=p,
                    hand=hands[p],
                    opp_hand=hands[1 - p],
                    decision_idx=decision_idx,
                    round=game.current_round,
                    action_taken=best_action_id,
                    legal_actions=legal_ids,
                    q_vals=q_vals.tolist(),
                    P_t=P_t.tolist(),
                    b_mine=b_mine.tolist(),
                    b_opp=[b.tolist() for b in b_opp_snap],
                ))
                decision_idx += 1

                # Step game
                actor = game.current_player
                pre_board = game.board
                _, rewards, done, _ = game.step(Action(best_action_id))

                act_eid = action_event_id(actor, best_action_id)
                deal_eid = None
                if game.board is not None and game.board != pre_board:
                    deal_eid = deal_event_id(game.board)

                event_ids.append(act_eid)

                # Update both players' beliefs for the action event
                for pi in range(2):
                    e_prime, P_new = state_enc.encode_event(
                        act_eid, bs[pi].P_current, bs[pi].P_history
                    )
                    update_belief_state(bs[pi], actor, pi, e_prime, P_new, belief_net)
                P_at_events.append(bs[0].P_current.tolist())

                # Update beliefs for deal event if it occurred
                if deal_eid is not None:
                    event_ids.append(deal_eid)
                    for pi in range(2):
                        e_prime_d, P_new_d = state_enc.encode_event(
                            deal_eid, bs[pi].P_current, bs[pi].P_history
                        )
                        update_belief_state(bs[pi], None, pi, e_prime_d, P_new_d, belief_net)
                    P_at_events.append(bs[0].P_current.tolist())

                if done:
                    final_rewards = list(rewards)
                    break

            records.append(GameRecord(
                hands=hands,
                decisions=decisions,
                final_rewards=final_rewards,
                P_at_events=P_at_events,
                event_ids=event_ids,
            ))

    return records
