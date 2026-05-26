"""
AlphaZero v5 — Position-Symmetric Training.

Three targeted fixes over v4, all motivated by the belief-accuracy diagnosis:

  Fix 1 — Belief loss for both players
           _belief_loss_only previously computed CE loss only at P0 decision steps.
           P1's BeliefNet got zero gradient → P1 belief accuracy ≈ random (33%).
           Now CE loss is computed at ALL decision steps regardless of player.

  Fix 2 — Alternating player roles
           Every episode randomly assigns az_player ∈ {0, 1}.  When az_player=1,
           the AZ agent sits in P1 and the opponent takes P0.  This gives the
           replay buffer P0/P1-balanced Q* targets and trains Q-net on both.

  Fix 3 — Position bit in Q-net
           Q-net input extended by 1 bit: [P_t | h_oh | b_mine | pos_bit].
           Allows the Q-net to learn explicitly position-dependent strategies.
           pos_bit is stored in the replay record and propagated into fast_pimc_search.

  Speed — k=20 rollouts (was k=30, saves ~33% episode time).
           Replay buffer compensates for increased Q* variance.

All v4 stability mechanisms retained: target Q-net, replay buffer (50K / batch 256),
asymmetric T (T_self=0.5, T_opp=2.0), entropy bonus (λ=0.01).

Usage:
    python experiments/alphazero_v5/train_v5.py
    python experiments/alphazero_v5/train_v5.py --smoke
    python experiments/alphazero_v5/train_v5.py --resume experiments/alphazero_v5/outputs/checkpoint.pt
"""

import argparse
import copy
import json
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preliminary_experiments.alphazero.config import AZConfig
from preliminary_experiments.alphazero.state_encoder import (
    StateEncoder, CARD_TO_IDX, IDX_TO_CARD,
    action_event_id, deal_event_id,
)
from preliminary_experiments.alphazero.belief import (
    BeliefNet, BeliefState, make_belief_state,
    update_belief_state, informed_prior,
)
from preliminary_experiments.alphazero.agent import AZAgent, hand_onehot, masked_softmax
from preliminary_experiments.alphazero.fast_rollout import _H_ONEHOT, _MASK_ALL, _MASK_NO_R, _bs_copy, RolloutGame
from preliminary_experiments.alphazero.rollout import _step_game
from preliminary_experiments.alphazero.tournament import az_tournament_checkpointer
from preliminary_experiments.alphazero.eval import evaluate, analyze_raise_rates

from agents.base import BaseAgent
from engine.leduc_game import LeducGame, Action


# ── v5 config ─────────────────────────────────────────────────────────────────

V5_CONFIG = AZConfig(
    d_model=4,
    state_hidden=(8,),
    belief_hidden=(8, 8),
    q_hidden=(32, 32),
    k_rollouts=20,              # reduced from 30 for speed (~33% faster episodes)
    temperature=1.0,
    T_self=0.5,
    T_opp=2.0,
    lambda_entropy=0.01,
    target_sync_freq=500,
    replay_buffer_size=50_000,
    replay_batch_size=256,
    n_episodes=200_000,
    lr=1e-3,
    lambda_belief=0.1,
)


# ── Fix 3: Position-aware Q-net ───────────────────────────────────────────────

# Pre-built position-bit tensors (one per player, shared across calls).
_POS_BIT = [torch.tensor([0.0]), torch.tensor([1.0])]


class QNetV5(nn.Module):
    """
    Q-net with 1-bit position feature appended to input.
    Input: [P_t (d) | h_onehot (3) | b_mine (3) | pos_bit (1)]  = d+7 dim.

    Backward-compatible: when called without pos_bit (e.g. from fast_rollout
    internals) the position is treated as 0 (P0 default).  This only affects
    rollout action sampling, not the training signal.
    """

    def __init__(self, config: AZConfig):
        super().__init__()
        in_dim = config.d_model + 3 + 3 + 1   # +1 vs original QNet

        layers: list = []
        curr = in_dim
        for h in config.q_hidden:
            layers += [nn.Linear(curr, h), nn.ReLU()]
            curr = h
        layers.append(nn.Linear(curr, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        P_t:      torch.Tensor,               # (d,) or (B, d)
        h_onehot: torch.Tensor,               # (3,) or (B, 3)
        b:        torch.Tensor,               # (3,) or (B, 3)
        pos_bit:  Optional[torch.Tensor] = None,  # (1,) or (B, 1) – None → zeros
    ) -> torch.Tensor:
        if pos_bit is None:
            # Backward-compatible default — used by fast_rollout internals
            pos_bit = _POS_BIT[0].expand_as(P_t[..., :1]) if P_t.dim() > 1 \
                      else _POS_BIT[0]
        x = torch.cat([P_t, h_onehot, b, pos_bit], dim=-1)
        return self.mlp(x)


# ── Position-aware PIMC search ────────────────────────────────────────────────
#
# fast_rollout._sample calls q_net(P, h_onehot, b) without pos_bit.
# We override the search loop here to inject pos_bit for the az_player's
# own decisions; rollout opponent sampling uses the default (pos_bit=None → 0).

def _sample_v5(
    raises: int,
    h: str,
    b: torch.Tensor,
    P: torch.Tensor,
    q_net: QNetV5,
    T: float,
    pos_bit: torch.Tensor,
) -> Action:
    """Q-net forward with position bit + masked softmax sampling."""
    q = q_net(P, _H_ONEHOT[h], b, pos_bit)
    mask = _MASK_ALL if raises < 2 else _MASK_NO_R
    return Action(torch.multinomial(F.softmax((q + mask) / T, dim=-1), 1).item())


def _rollout_v5(
    game: RolloutGame,
    player_i: int,
    h_i: str,
    h_j: str,
    bs: BeliefState,
    b_opp_j: torch.Tensor,
    first_action: Action,
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    q_net: QNetV5,
    T: float,
    T_self: Optional[float] = None,
    T_opp:  Optional[float] = None,
) -> float:
    """Single rollout — position-aware version of fast_rollout._rollout."""
    t_i = T_self if T_self is not None else T
    t_j = T_opp  if T_opp  is not None else T
    pb_i = _POS_BIT[player_i]        # pos_bit for player_i
    pb_j = _POS_BIT[1 - player_i]   # pos_bit for opponent j

    _, done, act_eid, deal_eid, _ = game.step(first_action)
    if done:
        return game.get_reward()[player_i]

    e_prime, P_new = state_enc.encode_event(act_eid, bs.P_current, bs.P_history)
    b_opp_j = belief_net(b_opp_j, bs.P_current, e_prime)
    bs.P_history.append(P_new)
    bs.P_current = P_new

    if deal_eid is not None:
        e_d, P_new_d = state_enc.encode_event(deal_eid, bs.P_current, bs.P_history)
        b_stack = torch.stack([bs.b_mine, b_opp_j])
        P_exp   = bs.P_current.unsqueeze(0).expand(2, -1)
        e_exp   = e_d.unsqueeze(0).expand(2, -1)
        out = F.softmax(
            belief_net.mlp(torch.cat([b_stack, P_exp, e_exp], dim=-1)), dim=-1
        )
        bs.b_mine, b_opp_j = out[0], out[1]
        bs.P_history.append(P_new_d)
        bs.P_current = P_new_d

    while not game.is_finished:
        actor = game.current_player
        if actor == player_i:
            action = _sample_v5(game.raises_this_round, h_i, bs.b_mine, bs.P_current, q_net, t_i, pb_i)
        else:
            action = _sample_v5(game.raises_this_round, h_j, b_opp_j,   bs.P_current, q_net, t_j, pb_j)

        _, done, act_eid, deal_eid, _ = game.step(action)
        if done:
            return game.get_reward()[player_i]

        e_prime, P_new = state_enc.encode_event(act_eid, bs.P_current, bs.P_history)
        if actor != player_i:
            bs.b_mine = belief_net(bs.b_mine, bs.P_current, e_prime)
        else:
            b_opp_j = belief_net(b_opp_j, bs.P_current, e_prime)
        bs.P_history.append(P_new)
        bs.P_current = P_new

        if deal_eid is not None:
            e_d, P_new_d = state_enc.encode_event(deal_eid, bs.P_current, bs.P_history)
            b_stack = torch.stack([bs.b_mine, b_opp_j])
            P_exp   = bs.P_current.unsqueeze(0).expand(2, -1)
            e_exp   = e_d.unsqueeze(0).expand(2, -1)
            out = F.softmax(
                belief_net.mlp(torch.cat([b_stack, P_exp, e_exp], dim=-1)), dim=-1
            )
            bs.b_mine, b_opp_j = out[0], out[1]
            bs.P_history.append(P_new_d)
            bs.P_current = P_new_d

    return game.get_reward()[player_i]


def pimc_search_v5(
    obs,
    player_i: int,
    h_i: str,
    bs_i: BeliefState,
    legal_actions: list,
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    q_net: QNetV5,
    k: int = 10,
    T: float = 1.0,
    T_self: Optional[float] = None,
    T_opp:  Optional[float] = None,
) -> torch.Tensor:
    """Position-aware PIMC search (same semantics as fast_pimc_search)."""
    q_star = torch.full((3,), float('-inf'))
    rg = RolloutGame()

    with torch.no_grad():
        for hj_idx, h_j in enumerate(IDX_TO_CARD):
            weight = bs_i.b_mine[hj_idx].item()
            if weight < 1e-8:
                continue

            b_opp_j_base = bs_i.b_opp[hj_idx]
            rg.init_from_obs(obs, player_i, h_i, h_j)
            init_packed = rg.pack()

            for action in legal_actions:
                aid   = int(action)
                total = 0.0
                for _ in range(k):
                    rg.restore(init_packed)
                    total += _rollout_v5(
                        game=rg, player_i=player_i,
                        h_i=h_i, h_j=h_j,
                        bs=_bs_copy(bs_i), b_opp_j=b_opp_j_base,
                        first_action=action,
                        state_enc=state_enc, belief_net=belief_net, q_net=q_net,
                        T=T, T_self=T_self, T_opp=T_opp,
                    )

                avg = total / k
                if q_star[aid] == float('-inf'):
                    q_star[aid] = 0.0
                q_star[aid] += weight * avg

    return q_star


# ── Decision record with position bit ─────────────────────────────────────────

@dataclass
class DecisionRecordV5:
    """Replay buffer entry — includes pos_bit for position-aware Q-net."""
    step:         int
    player:       int
    legal_actions: list
    q_star:       torch.Tensor    # shape (3,)   [no-grad]
    P_t:          torch.Tensor    # shape (d,)   [no-grad]
    h_onehot:     torch.Tensor    # shape (3,)   [no-grad]
    b_mine:       torch.Tensor    # shape (3,)   [no-grad]
    pos_bit:      torch.Tensor    # shape (1,)   [no-grad]  ← new
    legal_ids:    List[int]


@dataclass
class EpisodeV5:
    hands:     List[str]
    events:    List[tuple]
    decisions: List[DecisionRecordV5]


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.buf: deque = deque(maxlen=maxlen)

    def add(self, records: List[DecisionRecordV5]) -> None:
        for r in records:
            self.buf.append(r)

    def sample(self, n: int) -> List[DecisionRecordV5]:
        n = min(n, len(self.buf))
        return [self.buf[i] for i in random.sample(range(len(self.buf)), n)]

    def __len__(self) -> int:
        return len(self.buf)


# ── Fix 2: Episode play with configurable az_player ───────────────────────────

def _play_episode_v5(
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    q_net_target: QNetV5,
    config: AZConfig,
    opponent: BaseAgent,
    az_player: int,          # 0 or 1 — which seat the AZ agent occupies
) -> EpisodeV5:
    """
    Play one game with the AZ agent in seat `az_player` and the opponent
    in the other seat.  PIMC uses q_net_target (frozen).

    Decisions are recorded for az_player only and include pos_bit.
    """
    game = LeducGame()
    game.reset()
    hands = list(game.player_hands)

    with torch.no_grad():
        bs = [make_belief_state(hands[p], config.d_model) for p in range(2)]

    pb_az = _POS_BIT[az_player]   # pre-built pos_bit tensor
    events:    List[tuple]            = []
    decisions: List[DecisionRecordV5] = []

    with torch.no_grad():
        while not game.is_finished:
            p   = game.current_player
            obs = game.get_observation(viewer_id=p)
            legal = game.get_legal_actions()

            if p == az_player:
                q_star = pimc_search_v5(
                    obs=obs, player_i=p, h_i=hands[p], bs_i=bs[p],
                    legal_actions=legal, state_enc=state_enc,
                    belief_net=belief_net, q_net=q_net_target,
                    k=config.k_rollouts, T=config.temperature,
                    T_self=config.T_self, T_opp=config.T_opp,
                )
                decisions.append(DecisionRecordV5(
                    step=len(events),
                    player=p,
                    legal_actions=legal,
                    q_star=q_star.clone(),
                    P_t=bs[p].P_current.detach().clone(),
                    h_onehot=_H_ONEHOT[hands[p]].clone(),
                    b_mine=bs[p].b_mine.detach().clone(),
                    pos_bit=pb_az.clone(),
                    legal_ids=[int(a) for a in legal],
                ))
                probs  = masked_softmax(q_star, legal, config.temperature)
                action = Action(torch.multinomial(probs, 1).item())
            else:
                opponent.set_train_mode(False)
                action = opponent.select_action(obs)

            _, done, act_eid, deal_eid, actor = _step_game(game, action)
            events.append((act_eid, actor))
            if deal_eid is not None:
                events.append((deal_eid, None))

            for pi in range(2):
                e_prime, P_new = state_enc.encode_event(
                    act_eid, bs[pi].P_current, bs[pi].P_history
                )
                update_belief_state(bs[pi], actor, pi, e_prime, P_new, belief_net)
                if deal_eid is not None:
                    e_d, P_new_d = state_enc.encode_event(
                        deal_eid, bs[pi].P_current, bs[pi].P_history
                    )
                    update_belief_state(bs[pi], None, pi, e_d, P_new_d, belief_net)

    return EpisodeV5(hands=hands, events=events, decisions=decisions)


# ── Fix 1: Belief loss for ALL decision steps ─────────────────────────────────

def _belief_loss_both_players(
    episode: EpisodeV5,
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    config: AZConfig,
) -> torch.Tensor:
    """
    Replay episode with gradients; compute belief CE loss at EVERY decision
    step (both P0 and P1), not just az_player steps.

    Key fix vs v4: the loss gate `if dp.player == 0` is removed.
    Both belief states are already maintained during replay, so this is trivial.
    """
    d     = config.d_model
    hands = episode.hands

    bs = [
        BeliefState(
            b_mine=informed_prior(hands[p]),
            b_opp=[informed_prior(hj) for hj in IDX_TO_CARD],
            P_current=torch.zeros(d),
            P_history=[torch.zeros(d)],
        )
        for p in range(2)
    ]

    dec_at_step: dict = {}
    for dp in episode.decisions:
        dec_at_step.setdefault(dp.step, []).append(dp)

    belief_losses = []

    for event_idx, (event_id, actor) in enumerate(episode.events):
        # Record CE loss at every decision point (regardless of player)
        for dp in dec_at_step.get(event_idx, []):
            p           = dp.player
            h_opp_true  = hands[1 - p]
            h_opp_idx   = CARD_TO_IDX[h_opp_true]
            b_log       = torch.log(bs[p].b_mine[h_opp_idx].clamp(min=1e-8))
            belief_losses.append(-b_log)

        # Advance both belief states (maintains grad path through belief_net)
        for pi in range(2):
            e_prime, P_new = state_enc.encode_event(
                event_id, bs[pi].P_current, bs[pi].P_history
            )
            update_belief_state(bs[pi], actor, pi, e_prime, P_new, belief_net)

    if not belief_losses:
        return torch.tensor(0.0)
    return config.lambda_belief * torch.stack(belief_losses).mean()


# ── Q-net update from replay buffer ──────────────────────────────────────────

def _q_loss_from_buffer(
    batch: List[DecisionRecordV5],
    q_net: QNetV5,
    config: AZConfig,
) -> torch.Tensor:
    """Batched Q-net MSE + entropy loss.  Uses pos_bit from each record."""
    B = len(batch)

    P_t    = torch.stack([e.P_t      for e in batch])    # (B, d)
    h_oh   = torch.stack([e.h_onehot for e in batch])    # (B, 3)
    b      = torch.stack([e.b_mine   for e in batch])    # (B, 3)
    pos    = torch.stack([e.pos_bit  for e in batch])    # (B, 1)
    q_all  = q_net(P_t, h_oh, b, pos)                    # (B, 3)

    mse_total = torch.tensor(0.0)
    ent_total = torch.tensor(0.0)

    for i, entry in enumerate(batch):
        lids = entry.legal_ids
        pred = q_all[i][lids]
        tgt  = entry.q_star[lids].detach()
        mse_total = mse_total + F.mse_loss(pred, tgt)
        pi  = F.softmax(pred, dim=-1)
        ent_total = ent_total + -(pi * torch.log(pi.clamp(min=1e-8))).sum()

    return mse_total / B - config.lambda_entropy * (ent_total / B)


# ── Trainer ───────────────────────────────────────────────────────────────────

class AZTrainerV5:
    """
    v5 trainer: all v4 mechanisms + position-symmetric training.

    Two optimisers (same as v4):
      enc_optimizer : StateEncoder + BeliefNet (online, belief CE loss)
      q_optimizer   : QNetV5         (replay buffer, MSE + entropy + pos_bit)
    """

    def __init__(
        self,
        config:     AZConfig,
        state_enc:  StateEncoder,
        belief_net: BeliefNet,
        q_net:      QNetV5,
        opponents:  List[BaseAgent],
    ):
        self.config     = config
        self.state_enc  = state_enc
        self.belief_net = belief_net
        self.q_net      = q_net
        self.opponents  = opponents

        self.q_net_target = copy.deepcopy(q_net)
        for p in self.q_net_target.parameters():
            p.requires_grad_(False)

        self.enc_optimizer = torch.optim.Adam(
            list(state_enc.parameters()) + list(belief_net.parameters()),
            lr=config.lr,
        )
        self.q_optimizer = torch.optim.Adam(
            q_net.parameters(), lr=config.lr,
        )

        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self.episode_count = 0
        self.loss_history:  List[float] = []

    # ── Core training step ────────────────────────────────────────────────

    def train_one_episode(self) -> float:
        self.state_enc.eval()
        self.belief_net.eval()
        self.q_net_target.eval()
        self.q_net.eval()

        # Fix 2: randomly alternate which seat the AZ agent occupies
        az_player = random.randint(0, 1)
        opponent  = random.choice(self.opponents)
        episode   = _play_episode_v5(
            self.state_enc, self.belief_net, self.q_net_target,
            self.config, opponent, az_player,
        )

        # 1. Update StateEncoder + BeliefNet (Fix 1: CE for both players)
        self.state_enc.train()
        self.belief_net.train()
        self.enc_optimizer.zero_grad()
        b_loss = _belief_loss_both_players(
            episode, self.state_enc, self.belief_net, self.config
        )
        if b_loss.requires_grad:
            b_loss.backward()
            self.enc_optimizer.step()

        # 2. Add decisions to replay buffer; update Q-net from batch
        self.replay_buffer.add(episode.decisions)
        q_loss_val = 0.0
        if len(self.replay_buffer) >= self.config.replay_batch_size:
            batch = self.replay_buffer.sample(self.config.replay_batch_size)
            self.q_net.train()
            self.q_optimizer.zero_grad()
            q_loss = _q_loss_from_buffer(batch, self.q_net, self.config)
            q_loss.backward()
            self.q_optimizer.step()
            q_loss_val = q_loss.item()

        # 3. Sync target Q-net
        freq = self.config.target_sync_freq
        if freq > 0 and self.episode_count % freq == 0:
            self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.episode_count += 1
        total_loss = b_loss.item() + q_loss_val
        self.loss_history.append(total_loss)
        return total_loss

    # ── Training loop ─────────────────────────────────────────────────────

    def train(
        self,
        log_every:        int  = 1000,
        checkpoint_path:  Optional[str] = None,
        checkpoint_every: int  = 5000,
        callback          = None,
    ) -> None:
        import signal

        interrupted = [False]

        def _handle_sigint(sig, frame):
            print("\n[train] Interrupt — saving checkpoint after this episode.")
            interrupted[0] = True

        prev_handler = signal.signal(signal.SIGINT, _handle_sigint)

        try:
            start  = self.episode_count
            target = max(start, self.config.n_episodes)
            t0     = time.time()

            for ep in range(start, target):
                loss_val = self.train_one_episode()

                if callback is not None:
                    callback({"episode": self.episode_count, "loss": loss_val})

                if (ep - start + 1) % log_every == 0:
                    recent  = self.loss_history[-log_every:]
                    avg     = sum(recent) / len(recent)
                    elapsed = time.time() - t0
                    eps_done = ep - start + 1
                    eta_min  = (elapsed / eps_done) * (target - ep - 1) / 60
                    print(
                        f"ep {self.episode_count:>7d}/{target} | "
                        f"avg_loss {avg:.4f} | "
                        f"buf {len(self.replay_buffer):>6d} | "
                        f"elapsed {elapsed/60:.1f}m | ETA {eta_min:.1f}m"
                    )

                if checkpoint_path and self.episode_count % checkpoint_every == 0:
                    self.save(checkpoint_path)
                    print(f"  [ckpt] saved → {checkpoint_path}")

                if interrupted[0]:
                    break

        finally:
            signal.signal(signal.SIGINT, prev_handler)
            if checkpoint_path:
                self.save(checkpoint_path)
                print(f"[train] final checkpoint → {checkpoint_path}")

    # ── Checkpoint I/O ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            'state_enc':     self.state_enc.state_dict(),
            'belief_net':    self.belief_net.state_dict(),
            'q_net':         self.q_net.state_dict(),
            'q_net_target':  self.q_net_target.state_dict(),
            'enc_optimizer': self.enc_optimizer.state_dict(),
            'q_optimizer':   self.q_optimizer.state_dict(),
            'episode':       self.episode_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.state_enc.load_state_dict(ckpt['state_enc'])
        self.belief_net.load_state_dict(ckpt['belief_net'])
        self.q_net.load_state_dict(ckpt['q_net'])
        tgt = ckpt.get('q_net_target', ckpt['q_net'])
        self.q_net_target.load_state_dict(tgt)
        self.enc_optimizer.load_state_dict(ckpt['enc_optimizer'])
        self.q_optimizer.load_state_dict(
            ckpt.get('q_optimizer', ckpt.get('optimizer', {}))
        )
        self.episode_count = ckpt.get('episode', 0)


# ── Opponent pool ─────────────────────────────────────────────────────────────

def load_opponents() -> List[BaseAgent]:
    from agents.heuristic.agent import HeuristicAgent
    from agents.value_based.agent import ValueBasedAgent
    from agents.cfr.agent import CFRAgent

    opponents = [
        HeuristicAgent(),
        ValueBasedAgent(model_path=str(ROOT / "agents" / "value_based" / "checkpoint.pt")),
        CFRAgent(model_path=str(ROOT / "agents" / "cfr" / "checkpoint.pt")),
    ]
    for opp in opponents:
        opp.set_train_mode(False)
    print(f"Loaded {len(opponents)} opponents: [heuristic, value_based, cfr]")
    return opponents


# ── AZAgent wrapper for tournament eval ──────────────────────────────────────

class AZAgentV5(AZAgent):
    """AZAgent subclass using QNetV5 and pimc_search_v5."""

    def select_action(self, obs):
        assert self._bs is not None, "Call new_game() before select_action()"
        legal  = obs.legal_actions
        q_star = pimc_search_v5(
            obs=obs, player_i=self.player_id, h_i=self._hand, bs_i=self._bs,
            legal_actions=legal, state_enc=self.state_enc,
            belief_net=self.belief_net, q_net=self.q_net,
            k=self.config.k_rollouts, T=self.config.temperature,
            T_self=self.config.T_self, T_opp=self.config.T_opp,
        )
        probs = masked_softmax(q_star, legal, self.config.temperature)
        return Action(torch.multinomial(probs, 1).item())


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero v5.")
    parser.add_argument("--episodes",         type=int,   default=200_000)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--lambda-belief",    type=float, default=0.1)
    parser.add_argument("--lambda-entropy",   type=float, default=0.01)
    parser.add_argument("--T-self",           type=float, default=0.5)
    parser.add_argument("--T-opp",            type=float, default=2.0)
    parser.add_argument("--target-sync-freq", type=int,   default=500)
    parser.add_argument("--replay-buffer",    type=int,   default=50_000)
    parser.add_argument("--replay-batch",     type=int,   default=256)
    parser.add_argument("--log-every",        type=int,   default=1000)
    parser.add_argument("--checkpoint-every", type=int,   default=5000)
    parser.add_argument("--eval-every",       type=int,   default=10_000)
    parser.add_argument("--eval-games",       type=int,   default=200)
    parser.add_argument("--resume",           type=str,   default=None)
    parser.add_argument("--smoke",            action="store_true",
                        help="10-episode smoke test.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = parser.parse_args()

    if args.smoke:
        args.episodes       = 10
        args.log_every      = 5
        args.checkpoint_every = 10
        args.eval_every     = 5
        args.eval_games     = 20
        args.replay_buffer  = 500

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(args.output_dir / "checkpoint.pt")
    history_path    = args.output_dir / "train_history.json"
    config_path     = args.output_dir / "train_config.json"

    config = AZConfig(
        d_model=V5_CONFIG.d_model,
        state_hidden=V5_CONFIG.state_hidden,
        belief_hidden=V5_CONFIG.belief_hidden,
        q_hidden=V5_CONFIG.q_hidden,
        k_rollouts=V5_CONFIG.k_rollouts,
        temperature=V5_CONFIG.temperature,
        T_self=args.T_self, T_opp=args.T_opp,
        lambda_entropy=args.lambda_entropy,
        target_sync_freq=args.target_sync_freq,
        replay_buffer_size=args.replay_buffer,
        replay_batch_size=args.replay_batch,
        n_episodes=args.episodes,
        lr=args.lr,
        lambda_belief=args.lambda_belief,
    )

    config_path.write_text(json.dumps({
        "experiment":         "alphazero_v5",
        "d_model":            config.d_model,
        "state_hidden":       list(config.state_hidden),
        "belief_hidden":      list(config.belief_hidden),
        "q_hidden":           list(config.q_hidden),
        "q_net_input_dim":    config.d_model + 7,
        "k_rollouts":         config.k_rollouts,
        "temperature":        config.temperature,
        "T_self":             config.T_self,
        "T_opp":              config.T_opp,
        "lambda_entropy":     config.lambda_entropy,
        "target_sync_freq":   config.target_sync_freq,
        "replay_buffer_size": config.replay_buffer_size,
        "replay_batch_size":  config.replay_batch_size,
        "n_episodes":         config.n_episodes,
        "lr":                 config.lr,
        "lambda_belief":      config.lambda_belief,
        "opponent_pool":      ["heuristic", "value_based", "cfr"],
        "changes_vs_v4":      ["belief_loss_both_players", "alternating_player_roles",
                               "position_bit_qnet", "k_rollouts_20"],
        "resumed_from":       args.resume,
    }, indent=2))
    print(f"Config saved → {config_path}")

    # ── Build networks ────────────────────────────────────────────────────
    state_enc  = StateEncoder(config)
    belief_net = BeliefNet(config)
    q_net      = QNetV5(config)

    opponents = load_opponents()
    trainer   = AZTrainerV5(config, state_enc, belief_net, q_net, opponents)

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume}  (episode {trainer.episode_count})")

    agent0 = AZAgentV5(config, state_enc, belief_net, q_net, player_id=0)
    agent1 = AZAgentV5(config, state_enc, belief_net, q_net, player_id=1)

    # ── Eval setup ────────────────────────────────────────────────────────
    eval_history:  list = []
    eval_log_path = args.output_dir / "eval_history.json"

    if args.eval_every > 0:
        from agents.heuristic.agent import HeuristicAgent
        _heuristic_eval = HeuristicAgent()
        print(f"Periodic eval: every {args.eval_every} ep, "
              f"{args.eval_games} games/opponent.")

    def _run_eval(ep_count):
        agent0.set_train_mode(False)
        r_heu  = evaluate(agent0, _heuristic_eval, "heuristic",
                          args.eval_games, use_search=False)
        rates  = analyze_raise_rates(agent0, n_games=args.eval_games,
                                     use_search=False)
        agent0.set_train_mode(True)
        spread = max(rates.values()) - min(rates.values())
        entry  = {
            "episode":      ep_count,
            "vs_heuristic": r_heu["avg_chips"],
            "raise_J":      rates["J"],
            "raise_Q":      rates["Q"],
            "raise_K":      rates["K"],
            "raise_spread": round(spread, 3),
        }
        eval_history.append(entry)
        eval_log_path.write_text(json.dumps(eval_history, indent=2))
        print(f"  [eval] ep {ep_count:>7d} | vs heuristic {r_heu['avg_chips']:+.3f} | "
              f"raise J/Q/K {rates['J']:.0%}/{rates['Q']:.0%}/{rates['K']:.0%} "
              f"(spread {spread:.0%})")

    history: list = []

    def _history_callback(event):
        history.append(event)
        if len(history) % 100 == 0:
            history_path.write_text(json.dumps(history, indent=2))
        if args.eval_every > 0 and event["episode"] % args.eval_every == 0:
            _run_eval(event["episode"])

    checkpointer = az_tournament_checkpointer(
        agent0=agent0,
        agent1=agent1,
        output_dir=args.output_dir,
        pass_through_callback=_history_callback,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\nStarting alphazero_v5 training: {args.episodes} episodes")
    print(f"Architecture: d={config.d_model}, state{config.state_hidden}, "
          f"belief{config.belief_hidden}, Q{config.q_hidden}+pos_bit")
    print(f"k={config.k_rollouts} rollouts | T_self={config.T_self} T_opp={config.T_opp}")
    print(f"Fixes: belief_loss=both_players | training=alternating_P0P1 | pos_bit=on")
    print(f"Output dir: {args.output_dir}\n")

    t_start = time.time()
    trainer.train(
        log_every=args.log_every,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        callback=checkpointer.callback,
    )
    elapsed = time.time() - t_start

    history_path.write_text(json.dumps(history, indent=2))
    print(f"\nDone. {trainer.episode_count} episodes in {elapsed/60:.1f} min.")
    print(f"Checkpoint : {checkpoint_path}")


if __name__ == "__main__":
    main()
