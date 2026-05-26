"""
AlphaZero v4 — Training Stability + Exploration Fixes.

Four changes vs v3 (architecture unchanged: d=4, state(8,), belief(8,8), Q(32,32), k=30):

  1. Target Q-net  — PIMC rollouts use a frozen q_net_target; live q_net is only
                     updated during the gradient step.  Synced every
                     config.target_sync_freq episodes.  Eliminates the bootstrap
                     non-stationarity that caused oscillation in v1-v3.

  2. Replay buffer — Decision records (P_t, h_oh, b_mine, Q*) are stored in a
                     50K-entry deque.  Q-net is updated from a 256-sample minibatch
                     rather than the single current episode.  Decorrelates updates.

  3. Asymmetric T  — Rollout action sampling uses T_self=0.5 (player_i near-optimal)
                     and T_opp=2.0 (opponent makes mistakes).  This gives Q*(raise|J)
                     a positive contribution when the opponent folds, breaking the
                     pure-strategy bluffing attractor.

  4. Entropy bonus — Q-net loss = MSE − λ_ent·H(π_legal).  Prevents strategy collapse
                     to pure play.  λ_ent=0.01 is small enough to not distort values.

StateEncoder + BeliefNet are still updated online from the current episode
(belief CE loss only).  Q-net is updated from the replay buffer.

Usage:
    python experiments/alphazero_v4/train_v4.py
    python experiments/alphazero_v4/train_v4.py --smoke
    python experiments/alphazero_v4/train_v4.py --resume experiments/alphazero_v4/outputs/checkpoint.pt
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
from preliminary_experiments.alphazero.agent import QNet, AZAgent, hand_onehot, masked_softmax
from preliminary_experiments.alphazero.rollout import _step_game
from preliminary_experiments.alphazero.fast_rollout import fast_pimc_search, _H_ONEHOT
from preliminary_experiments.alphazero.trainer import Episode, _replay_episode
from preliminary_experiments.alphazero.tournament import az_tournament_checkpointer
from preliminary_experiments.alphazero.eval import evaluate, analyze_raise_rates

from agents.base import BaseAgent
from engine.leduc_game import LeducGame, Action


# ── v4 config ─────────────────────────────────────────────────────────────────

V4_CONFIG = AZConfig(
    d_model=4,
    state_hidden=(8,),
    belief_hidden=(8, 8),
    q_hidden=(32, 32),
    k_rollouts=30,
    temperature=1.0,        # action-selection temperature (outside rollouts)
    T_self=0.5,             # rollout temp for player_i
    T_opp=2.0,              # rollout temp for imagined opponent
    lambda_entropy=0.01,
    target_sync_freq=500,
    replay_buffer_size=50_000,
    replay_batch_size=256,
    n_episodes=200_000,
    lr=1e-3,
    lambda_belief=0.1,
)


# ── Extended decision record (adds pre-computed Q-net inputs) ─────────────────

@dataclass
class DecisionRecordV4:
    """Decision record extended with detached tensors for replay buffer."""
    step: int
    player: int
    legal_actions: list
    q_star: torch.Tensor        # shape (3,), from q_net_target  [no-grad]
    P_t: torch.Tensor           # shape (d_model,) detached       [no-grad]
    h_onehot: torch.Tensor      # shape (3,)                      [no-grad]
    b_mine: torch.Tensor        # shape (3,) detached             [no-grad]
    legal_ids: List[int]        # precomputed for fast indexing


@dataclass
class EpisodeV4:
    """Episode with v4 decision records (superset of Episode)."""
    hands: List[str]
    events: List[tuple]
    decisions: List[DecisionRecordV4]

    def as_base_episode(self) -> Episode:
        """Convert to base Episode for _replay_episode (belief training)."""
        from preliminary_experiments.alphazero.trainer import DecisionRecord
        base_decisions = [
            DecisionRecord(
                step=d.step,
                player=d.player,
                legal_actions=d.legal_actions,
                q_star=d.q_star,
            )
            for d in self.decisions
        ]
        return Episode(hands=self.hands, events=self.events, decisions=base_decisions)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size FIFO buffer of DecisionRecordV4 entries."""

    def __init__(self, maxlen: int):
        self.buf: deque = deque(maxlen=maxlen)

    def add(self, records: List[DecisionRecordV4]) -> None:
        for r in records:
            self.buf.append(r)

    def sample(self, n: int) -> List[DecisionRecordV4]:
        n = min(n, len(self.buf))
        indices = random.sample(range(len(self.buf)), n)
        return [self.buf[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buf)


# ── Episode play ──────────────────────────────────────────────────────────────

def _play_episode_v4(
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    q_net_target: QNet,      # frozen — used for Q* generation
    config: AZConfig,
    opponent: BaseAgent,
) -> EpisodeV4:
    """
    Play one game:
      - Player 0 uses fast_pimc_search with q_net_target (generates Q* targets).
      - Player 1 uses the frozen opponent.

    Records P_t and b_mine at each P0 decision (detached, for replay buffer).
    """
    game = LeducGame()
    game.reset()
    hands = list(game.player_hands)

    with torch.no_grad():
        bs = [make_belief_state(hands[p], config.d_model) for p in range(2)]

    events: List[tuple] = []
    decisions: List[DecisionRecordV4] = []

    with torch.no_grad():
        while not game.is_finished:
            p = game.current_player
            obs = game.get_observation(viewer_id=p)
            legal = game.get_legal_actions()

            if p == 0:
                q_star = fast_pimc_search(
                    obs=obs,
                    player_i=p,
                    h_i=hands[p],
                    bs_i=bs[p],
                    legal_actions=legal,
                    state_enc=state_enc,
                    belief_net=belief_net,
                    q_net=q_net_target,          # frozen target net
                    k=config.k_rollouts,
                    T=config.temperature,
                    T_self=config.T_self,
                    T_opp=config.T_opp,
                )
                decisions.append(DecisionRecordV4(
                    step=len(events),
                    player=p,
                    legal_actions=legal,
                    q_star=q_star.clone(),
                    P_t=bs[p].P_current.detach().clone(),
                    h_onehot=_H_ONEHOT[hands[p]].clone(),
                    b_mine=bs[p].b_mine.detach().clone(),
                    legal_ids=[int(a) for a in legal],
                ))
                probs = masked_softmax(q_star, legal, config.temperature)
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

    return EpisodeV4(hands=hands, events=events, decisions=decisions)


# ── Belief-only replay (StateEncoder + BeliefNet training) ────────────────────

def _belief_loss_only(
    episode: EpisodeV4,
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    config: AZConfig,
) -> torch.Tensor:
    """
    Replay episode WITH gradients for StateEncoder + BeliefNet only.
    Computes belief CE loss at every P0 decision point.
    Q-net is NOT called here (Q-net updates come from the replay buffer).
    """
    d = config.d_model
    hands = episode.hands

    P_current = torch.zeros(d)
    P_history = [torch.zeros(d)]

    bs = [
        BeliefState(
            b_mine=informed_prior(hands[p]),
            b_opp=[informed_prior(hj) for hj in IDX_TO_CARD],
            P_current=P_current.clone(),
            P_history=[P_current.clone()],
        )
        for p in range(2)
    ]

    dec_at_step = {}
    for dp in episode.decisions:
        dec_at_step.setdefault(dp.step, []).append(dp)

    belief_losses = []

    for event_idx, (event_id, actor) in enumerate(episode.events):
        for dp in dec_at_step.get(event_idx, []):
            p = dp.player
            h_opp_true = hands[1 - p]
            h_opp_idx  = CARD_TO_IDX[h_opp_true]
            b_log = torch.log(bs[p].b_mine[h_opp_idx].clamp(min=1e-8))
            belief_losses.append(-b_log)

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
    batch: List[DecisionRecordV4],
    q_net: QNet,
    config: AZConfig,
) -> torch.Tensor:
    """
    Batched Q-net MSE loss + entropy bonus from a replay buffer sample.

    Uses batched forward pass (QNet.forward supports (B,*) inputs via
    torch.cat + nn.Linear broadcasting).
    """
    B = len(batch)

    # Batch inputs — all detached (no gradient path through StateEncoder/BeliefNet)
    P_t   = torch.stack([e.P_t      for e in batch])   # (B, d)
    h_oh  = torch.stack([e.h_onehot for e in batch])   # (B, 3)
    b     = torch.stack([e.b_mine   for e in batch])   # (B, 3)
    q_all = q_net(P_t, h_oh, b)                        # (B, 3)

    mse_total = torch.tensor(0.0)
    ent_total = torch.tensor(0.0)

    for i, entry in enumerate(batch):
        lids = entry.legal_ids
        pred = q_all[i][lids]
        tgt  = entry.q_star[lids].detach()

        mse_total = mse_total + F.mse_loss(pred, tgt)

        pi  = F.softmax(pred, dim=-1)
        ent = -(pi * torch.log(pi.clamp(min=1e-8))).sum()
        ent_total = ent_total + ent

    return mse_total / B - config.lambda_entropy * (ent_total / B)


# ── Trainer ───────────────────────────────────────────────────────────────────

class AZTrainerV4:
    """
    v4 trainer: target Q-net + replay buffer + asymmetric T + entropy bonus.

    Two separate optimisers:
      enc_optimizer : StateEncoder + BeliefNet (updated online, belief CE loss)
      q_optimizer   : Q-net (updated from replay buffer, MSE + entropy)
    """

    def __init__(
        self,
        config: AZConfig,
        state_enc: StateEncoder,
        belief_net: BeliefNet,
        q_net: QNet,
        opponents: List[BaseAgent],
    ):
        self.config     = config
        self.state_enc  = state_enc
        self.belief_net = belief_net
        self.q_net      = q_net
        self.opponents  = opponents

        # Frozen target Q-net — initialised as a copy of q_net
        self.q_net_target = copy.deepcopy(q_net)
        for p in self.q_net_target.parameters():
            p.requires_grad_(False)

        # Separate optimisers
        self.enc_optimizer = torch.optim.Adam(
            list(state_enc.parameters()) + list(belief_net.parameters()),
            lr=config.lr,
        )
        self.q_optimizer = torch.optim.Adam(
            q_net.parameters(),
            lr=config.lr,
        )

        # Replay buffer (disabled if replay_buffer_size == 0)
        self.replay_buffer: Optional[ReplayBuffer] = (
            ReplayBuffer(config.replay_buffer_size)
            if config.replay_buffer_size > 0 else None
        )

        self.episode_count = 0
        self.loss_history: List[float] = []

    # ── Core training step ────────────────────────────────────────────────

    def train_one_episode(self) -> float:
        self.state_enc.eval()
        self.belief_net.eval()
        self.q_net_target.eval()   # always eval (no grad, no BN stats)
        self.q_net.eval()          # eval during play phase

        opponent = random.choice(self.opponents)
        episode = _play_episode_v4(
            self.state_enc, self.belief_net, self.q_net_target,
            self.config, opponent,
        )

        # ── 1. Update StateEncoder + BeliefNet (online, belief CE) ────────
        self.state_enc.train()
        self.belief_net.train()
        self.enc_optimizer.zero_grad()
        b_loss = _belief_loss_only(episode, self.state_enc, self.belief_net, self.config)
        # Short games where P0 decides only once have no grad path through belief_net
        if b_loss.requires_grad:
            b_loss.backward()
            self.enc_optimizer.step()

        # ── 2. Add new decision records to replay buffer ───────────────────
        q_loss_val = 0.0
        if self.replay_buffer is not None:
            self.replay_buffer.add(episode.decisions)

            if len(self.replay_buffer) >= self.config.replay_batch_size:
                batch = self.replay_buffer.sample(self.config.replay_batch_size)
                self.q_net.train()
                self.q_optimizer.zero_grad()
                q_loss = _q_loss_from_buffer(batch, self.q_net, self.config)
                q_loss.backward()
                self.q_optimizer.step()
                q_loss_val = q_loss.item()
        else:
            # Fallback: online Q-net update (same as v3, no replay buffer)
            self.q_net.train()
            self.q_optimizer.zero_grad()
            # Reuse _replay_episode for the Q portion (with gradient on q_net)
            from preliminary_experiments.alphazero.trainer import _replay_episode as _re
            full_loss = _re(episode.as_base_episode(), self.state_enc,
                            self.belief_net, self.q_net, self.config)
            full_loss.backward()
            self.q_optimizer.step()
            q_loss_val = full_loss.item()

        # ── 3. Sync target Q-net ───────────────────────────────────────────
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
        log_every: int = 1000,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 5000,
        callback=None,
    ) -> None:
        import signal, time

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
                    buf_size = len(self.replay_buffer) if self.replay_buffer else 0
                    print(
                        f"ep {self.episode_count:>7d}/{target} | "
                        f"avg_loss {avg:.4f} | "
                        f"buf {buf_size:>6d} | "
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
            'state_enc':      self.state_enc.state_dict(),
            'belief_net':     self.belief_net.state_dict(),
            'q_net':          self.q_net.state_dict(),
            'q_net_target':   self.q_net_target.state_dict(),
            'enc_optimizer':  self.enc_optimizer.state_dict(),
            'q_optimizer':    self.q_optimizer.state_dict(),
            'episode':        self.episode_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location='cpu')
        self.state_enc.load_state_dict(ckpt['state_enc'])
        self.belief_net.load_state_dict(ckpt['belief_net'])
        self.q_net.load_state_dict(ckpt['q_net'])
        if 'q_net_target' in ckpt:
            self.q_net_target.load_state_dict(ckpt['q_net_target'])
        else:
            self.q_net_target.load_state_dict(ckpt['q_net'])
        self.enc_optimizer.load_state_dict(ckpt['enc_optimizer'])
        self.q_optimizer.load_state_dict(ckpt.get('q_optimizer',
                                                   ckpt.get('optimizer', {})))
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


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero v4.")
    parser.add_argument("--episodes",          type=int,   default=200_000)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--lambda-belief",     type=float, default=0.1)
    parser.add_argument("--lambda-entropy",    type=float, default=0.01)
    parser.add_argument("--T-self",            type=float, default=0.5)
    parser.add_argument("--T-opp",             type=float, default=2.0)
    parser.add_argument("--target-sync-freq",  type=int,   default=500)
    parser.add_argument("--replay-buffer",     type=int,   default=50_000)
    parser.add_argument("--replay-batch",      type=int,   default=256)
    parser.add_argument("--log-every",         type=int,   default=1000)
    parser.add_argument("--checkpoint-every",  type=int,   default=5000)
    parser.add_argument("--eval-every",        type=int,   default=10_000)
    parser.add_argument("--eval-games",        type=int,   default=200)
    parser.add_argument("--resume",            type=str,   default=None)
    parser.add_argument("--smoke",             action="store_true",
                        help="Tiny budget (10 episodes) to verify pipeline.")
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
        args.replay_buffer  = 500    # small enough to fill quickly in smoke test

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(args.output_dir / "checkpoint.pt")
    history_path    = args.output_dir / "train_history.json"
    config_path     = args.output_dir / "train_config.json"

    config = AZConfig(
        d_model=V4_CONFIG.d_model,
        state_hidden=V4_CONFIG.state_hidden,
        belief_hidden=V4_CONFIG.belief_hidden,
        q_hidden=V4_CONFIG.q_hidden,
        k_rollouts=V4_CONFIG.k_rollouts,
        temperature=V4_CONFIG.temperature,
        T_self=args.T_self,
        T_opp=args.T_opp,
        lambda_entropy=args.lambda_entropy,
        target_sync_freq=args.target_sync_freq,
        replay_buffer_size=args.replay_buffer,
        replay_batch_size=args.replay_batch,
        n_episodes=args.episodes,
        lr=args.lr,
        lambda_belief=args.lambda_belief,
    )

    config_path.write_text(json.dumps({
        "experiment":        "alphazero_v4",
        "d_model":           config.d_model,
        "state_hidden":      list(config.state_hidden),
        "belief_hidden":     list(config.belief_hidden),
        "q_hidden":          list(config.q_hidden),
        "k_rollouts":        config.k_rollouts,
        "temperature":       config.temperature,
        "T_self":            config.T_self,
        "T_opp":             config.T_opp,
        "lambda_entropy":    config.lambda_entropy,
        "target_sync_freq":  config.target_sync_freq,
        "replay_buffer_size": config.replay_buffer_size,
        "replay_batch_size": config.replay_batch_size,
        "n_episodes":        config.n_episodes,
        "lr":                config.lr,
        "lambda_belief":     config.lambda_belief,
        "opponent_pool":     ["heuristic", "value_based", "cfr"],
        "resumed_from":      args.resume,
    }, indent=2))
    print(f"Config saved → {config_path}")

    # ── Networks ──────────────────────────────────────────────────────────
    state_enc  = StateEncoder(config)
    belief_net = BeliefNet(config)
    q_net      = QNet(config)

    opponents = load_opponents()
    trainer   = AZTrainerV4(config, state_enc, belief_net, q_net, opponents)

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume}  (episode {trainer.episode_count})")

    agent0 = AZAgent(config, state_enc, belief_net, q_net, player_id=0)
    agent1 = AZAgent(config, state_enc, belief_net, q_net, player_id=1)

    # ── Eval setup ────────────────────────────────────────────────────────
    eval_history: list = []
    eval_log_path = args.output_dir / "eval_history.json"

    if args.eval_every > 0:
        from agents.heuristic.agent import HeuristicAgent
        _heuristic_eval = HeuristicAgent()
        print(f"Periodic eval: every {args.eval_every} episodes, "
              f"{args.eval_games} games/opponent.")

    def _run_eval(ep_count):
        agent0.set_train_mode(False)
        r_heu  = evaluate(agent0, _heuristic_eval, "heuristic",
                          args.eval_games, use_search=False)
        rates  = analyze_raise_rates(agent0, n_games=args.eval_games,
                                     use_search=False)
        agent0.set_train_mode(True)
        spread = max(rates.values()) - min(rates.values())
        entry = {
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

    # ── Callbacks ─────────────────────────────────────────────────────────
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
    print(f"\nStarting alphazero_v4 training: {args.episodes} episodes")
    print(f"Architecture: d={config.d_model}, state{config.state_hidden}, "
          f"belief{config.belief_hidden}, Q{config.q_hidden}, k={config.k_rollouts}")
    print(f"T_self={config.T_self}, T_opp={config.T_opp}, "
          f"λ_ent={config.lambda_entropy}")
    print(f"target_sync={config.target_sync_freq} ep, "
          f"replay_buf={config.replay_buffer_size}, "
          f"batch={config.replay_batch_size}")
    print(f"Opponent pool: [heuristic, value_based, cfr]")
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
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
