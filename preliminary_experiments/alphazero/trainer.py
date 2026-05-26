"""
Online training loop for the AlphaZero-style agent.

Algorithm per episode:
  1. PLAY  (no-grad): run a self-play game; both players search at every
     decision point to generate Q* targets; record the full event sequence
     and all decision data.
  2. TRAIN (with-grad): replay the event sequence through f_state and
     f_belief to reconstruct P_t and beliefs with gradient tracking;
     compute L_Q + λ·L_belief; update all three modules jointly.

Both players share the same Q_θ, f_state, f_belief (self-play).
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from engine.leduc_game import LeducGame, Action
from engine.observation import Observation

from .config import AZConfig
from .state_encoder import (
    StateEncoder, CARD_TO_IDX, IDX_TO_CARD,
    action_event_id, deal_event_id,
)
from .belief import (
    BeliefNet, BeliefState, make_belief_state,
    update_belief_state, informed_prior,
)
from .agent import QNet, hand_onehot, masked_softmax
from .rollout import pimc_search, _step_game


# ── Episode data structures ──────────────────────────────────────────────────

@dataclass
class DecisionRecord:
    """Data recorded at one player decision point during the play phase."""
    step: int               # number of events in episode.events BEFORE this decision
    player: int             # which player decides
    legal_actions: list     # legal Action list at this point
    q_star: torch.Tensor    # Q*(a), shape (3,), illegal actions = -inf  [no-grad]


@dataclass
class Episode:
    """Complete record of one self-play game."""
    hands: List[str]                 # [hand_p0, hand_p1]
    events: List[tuple]              # [(event_id, actor), ...], actor=None for deal
    decisions: List[DecisionRecord]  # in chronological order


# ── Training logic ───────────────────────────────────────────────────────────

def _replay_episode(
    episode: Episode,
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    q_net: QNet,
    config: AZConfig,
) -> torch.Tensor:
    """
    Replay an episode WITH gradient tracking to compute the training loss.

    Maintains two BeliefStates (one per player) and shared P_t.
    At each decision point: computes Q_θ vs Q* (MSE) and b vs h_true (CE).
    Returns the scalar loss.
    """
    d = config.d_model
    hands = episode.hands

    # Shared public state
    P_current = torch.zeros(d)
    P_history: List[torch.Tensor] = [torch.zeros(d)]

    # Per-player beliefs — LIVE tensors with grad
    # bs[p].b_mine = p's belief about opponent
    # bs[p].b_opp[k] = opponent's belief about p IF opponent holds card k
    bs = [
        BeliefState(
            b_mine=informed_prior(hands[p]),
            b_opp=[informed_prior(hj) for hj in IDX_TO_CARD],
            P_current=P_current.clone(),
            P_history=[P_current.clone()],
        )
        for p in range(2)
    ]

    # Index decisions by step
    dec_at_step = {}
    for dp in episode.decisions:
        dec_at_step.setdefault(dp.step, []).append(dp)

    q_losses = []
    belief_losses = []

    for event_idx, (event_id, actor) in enumerate(episode.events):
        # ── Decision(s) BEFORE this event ────────────────────────────────
        for dp in dec_at_step.get(event_idx, []):
            p = dp.player
            h_opp_true = hands[1 - p]
            h_opp_idx = CARD_TO_IDX[h_opp_true]

            # Q loss: MSE between Q_θ and Q*
            q_vals = q_net(bs[p].P_current, hand_onehot(hands[p]), bs[p].b_mine)
            legal_ids = [int(a) for a in dp.legal_actions]
            q_pred = q_vals[legal_ids]
            q_tgt  = dp.q_star[legal_ids]
            q_losses.append(F.mse_loss(q_pred, q_tgt))

            # Belief CE loss: -log b_{i→j}(h_j_true)
            b_log = torch.log(bs[p].b_mine[h_opp_idx].clamp(min=1e-8))
            belief_losses.append(-b_log)

        # ── Advance public state (shared) ─────────────────────────────────
        e_prime, P_new = state_enc.encode_event(event_id, bs[0].P_current, bs[0].P_history)
        # Note: both players share the same P_t; we use bs[0] as the shared reference
        # and keep bs[1] in sync below.

        # ── Update per-player beliefs ─────────────────────────────────────
        for p in range(2):
            P_before = bs[p].P_current
            # Recompute e_prime for player p's graph (same event, same params → same value)
            e_prime_p, P_new_p = state_enc.encode_event(
                event_id, P_before, bs[p].P_history
            )
            update_belief_state(bs[p], actor, p, e_prime_p, P_new_p, belief_net)

    if not q_losses:
        return torch.tensor(0.0)

    L_q = torch.stack(q_losses).mean()
    L_b = torch.stack(belief_losses).mean()
    return L_q + config.lambda_belief * L_b


# ── Episode play ─────────────────────────────────────────────────────────────

def _play_episode(
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    q_net: QNet,
    config: AZConfig,
) -> Episode:
    """
    Play one complete self-play game with PIMC search at every decision point.
    All network calls are under torch.no_grad().
    Returns an Episode for subsequent training.
    """
    game = LeducGame()
    game.reset()

    hands = list(game.player_hands)  # [hand_p0, hand_p1]

    # Per-player belief states (no-grad)
    with torch.no_grad():
        bs = [make_belief_state(hands[p], config.d_model) for p in range(2)]

    events: List[tuple] = []
    decisions: List[DecisionRecord] = []

    with torch.no_grad():
        while not game.is_finished:
            p = game.current_player
            obs = game.get_observation(viewer_id=p)
            legal = game.get_legal_actions()

            # ── Search ───────────────────────────────────────────────────
            q_star = pimc_search(
                obs=obs,
                player_i=p,
                h_i=hands[p],
                bs_i=bs[p],
                legal_actions=legal,
                state_enc=state_enc,
                belief_net=belief_net,
                q_net=q_net,
                k=config.k_rollouts,
                T=config.temperature,
            )

            decisions.append(DecisionRecord(
                step=len(events),
                player=p,
                legal_actions=legal,
                q_star=q_star.clone(),
            ))

            # ── Sample action ─────────────────────────────────────────────
            probs = masked_softmax(q_star, legal, config.temperature)
            action_idx = torch.multinomial(probs, 1).item()
            action = Action(action_idx)

            # ── Step game ─────────────────────────────────────────────────
            _, done, act_eid, deal_eid, actor = _step_game(game, action)

            events.append((act_eid, actor))
            if deal_eid is not None:
                events.append((deal_eid, None))

            # ── Update both players' beliefs ──────────────────────────────
            for player_idx in range(2):
                e_prime, P_new = state_enc.encode_event(
                    act_eid, bs[player_idx].P_current, bs[player_idx].P_history
                )
                update_belief_state(bs[player_idx], actor, player_idx,
                                    e_prime, P_new, belief_net)

                if deal_eid is not None:
                    e_prime_d, P_new_d = state_enc.encode_event(
                        deal_eid, bs[player_idx].P_current, bs[player_idx].P_history
                    )
                    update_belief_state(bs[player_idx], None, player_idx,
                                        e_prime_d, P_new_d, belief_net)

    return Episode(hands=hands, events=events, decisions=decisions)


# ── Trainer ──────────────────────────────────────────────────────────────────

class AZTrainer:
    """
    Online AlphaZero-style trainer.

    Per episode:
      1. Play one self-play game with PIMC search (no-grad).
      2. Replay episode through networks with grad; compute and minimise loss.
    """

    def __init__(
        self,
        config: AZConfig,
        state_enc: StateEncoder,
        belief_net: BeliefNet,
        q_net: QNet,
    ):
        self.config = config
        self.state_enc = state_enc
        self.belief_net = belief_net
        self.q_net = q_net

        self.optimizer = torch.optim.Adam(
            list(state_enc.parameters())
            + list(belief_net.parameters())
            + list(q_net.parameters()),
            lr=config.lr,
        )

        self.episode_count = 0
        self.loss_history: List[float] = []

    def train(
        self,
        log_every: int = 1000,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 5000,
        callback=None,
    ) -> None:
        """
        Run the training loop for config.n_episodes episodes.

        Args:
            log_every:         Print a progress line every N episodes.
            checkpoint_path:   If set, save a checkpoint here every
                               `checkpoint_every` episodes (and on interrupt).
            checkpoint_every:  How often (in episodes) to write a checkpoint.
            callback:          Optional callable(dict) invoked after every episode
                               with keys {episode, loss}.  Useful for JSON logging.
        """
        import signal, time

        # ── Interrupt handler ─────────────────────────────────────────────
        interrupted = [False]

        def _handle_sigint(sig, frame):
            print("\n[train] Interrupt received — will save checkpoint and exit after this episode.")
            interrupted[0] = True

        prev_handler = signal.signal(signal.SIGINT, _handle_sigint)

        try:
            start  = self.episode_count
            # n_episodes is the *total* target; if we've already passed it, run 0 more.
            target = max(start, self.config.n_episodes)
            t0 = time.time()

            for ep in range(start, target):
                loss_val = self.train_one_episode()
                self.loss_history.append(loss_val)

                if callback is not None:
                    callback({"episode": self.episode_count, "loss": loss_val})

                if (ep - start + 1) % log_every == 0:
                    recent = self.loss_history[-log_every:]
                    avg = sum(recent) / len(recent)
                    elapsed = time.time() - t0
                    eps_done = ep - start + 1
                    eps_remaining = target - ep - 1
                    eta_s = (elapsed / eps_done) * eps_remaining if eps_done else 0
                    eta_min = eta_s / 60
                    print(
                        f"ep {self.episode_count:>7d}/{target} | "
                        f"avg_loss {avg:.4f} | "
                        f"elapsed {elapsed/60:.1f}m | "
                        f"ETA {eta_min:.1f}m"
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

    def train_one_episode(self) -> float:
        """Play + train on one episode. Returns scalar loss."""
        self.state_enc.eval()
        self.belief_net.eval()
        self.q_net.eval()

        episode = _play_episode(
            self.state_enc, self.belief_net, self.q_net, self.config
        )

        self.state_enc.train()
        self.belief_net.train()
        self.q_net.train()

        self.optimizer.zero_grad()
        loss = _replay_episode(episode, self.state_enc, self.belief_net, self.q_net, self.config)
        loss.backward()
        self.optimizer.step()

        self.episode_count += 1
        return loss.item()

    def save(self, path: str) -> None:
        torch.save({
            'state_enc':  self.state_enc.state_dict(),
            'belief_net': self.belief_net.state_dict(),
            'q_net':      self.q_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'episode':    self.episode_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location='cpu')
        self.state_enc.load_state_dict(ckpt['state_enc'])
        self.belief_net.load_state_dict(ckpt['belief_net'])
        self.q_net.load_state_dict(ckpt['q_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.episode_count = ckpt.get('episode', 0)


# ── Entry point ──────────────────────────────────────────────────────────────

def build_agent_and_trainer(config: AZConfig | None = None):
    """Convenience factory: returns (trainer, agent_p0, agent_p1)."""
    from .agent import AZAgent

    if config is None:
        config = AZConfig()

    state_enc  = StateEncoder(config)
    belief_net = BeliefNet(config)
    q_net      = QNet(config)

    trainer = AZTrainer(config, state_enc, belief_net, q_net)

    agent0 = AZAgent(config, state_enc, belief_net, q_net, player_id=0)
    agent1 = AZAgent(config, state_enc, belief_net, q_net, player_id=1)

    return trainer, agent0, agent1
