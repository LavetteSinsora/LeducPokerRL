"""
Q-network and AZAgent for the AlphaZero-style agent.

Q_θ(P_t, h_i, b_{i→j}) → [Q_fold, Q_call, Q_raise]
  input:  8 (P_t)  +  3 (h_i one-hot)  +  3 (belief)  =  14 dim
  hidden: [64, 64, 64]
  output: 3 Q-values (masked to legal actions at decision time)

AZAgent inherits BaseAgent and runs PIMC search in select_action().
It maintains a BeliefState internally and expects the game to provide
the acting player's identity so the belief can be updated after each action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.leduc_game import Action
from engine.observation import Observation
from agents.base import BaseAgent

from .config import AZConfig
from .state_encoder import (
    StateEncoder, make_initial_state,
    CARD_TO_IDX, IDX_TO_CARD,
    action_event_id, deal_event_id,
)
from .belief import BeliefNet, BeliefState, make_belief_state, update_belief_state


# ── Q-network ────────────────────────────────────────────────────────────────

class QNet(nn.Module):
    """
    Belief-conditioned Q-network.
    Input: [P_t (d) | h_i one-hot (3) | b_{i→j} (3)]  → 3 Q-values.
    """

    def __init__(self, config: AZConfig):
        super().__init__()
        d = config.d_model
        in_dim = d + 3 + 3

        layers = []
        curr = in_dim
        for h in config.q_hidden:
            layers += [nn.Linear(curr, h), nn.ReLU()]
            curr = h
        layers.append(nn.Linear(curr, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        P_t: torch.Tensor,      # (d,)
        h_onehot: torch.Tensor, # (3,)
        b: torch.Tensor,        # (3,)
    ) -> torch.Tensor:
        """Returns Q-values for all three actions, shape (3,)."""
        x = torch.cat([P_t, h_onehot, b], dim=-1)
        return self.mlp(x)


def hand_onehot(h: str) -> torch.Tensor:
    v = torch.zeros(3)
    v[CARD_TO_IDX[h]] = 1.0
    return v


def masked_softmax(q_vals: torch.Tensor, legal_actions: list, T: float) -> torch.Tensor:
    """Softmax over legal actions only; illegal actions get probability 0."""
    mask = torch.full((3,), float('-inf'))
    for a in legal_actions:
        mask[int(a)] = 0.0
    return F.softmax((q_vals + mask) / T, dim=-1)


# ── Agent ────────────────────────────────────────────────────────────────────

class AZAgent(BaseAgent):
    """
    AlphaZero-style agent with PIMC search and belief tracking.

    During a session the agent must be notified of every game event
    (including the opponent's actions) via observe_event(), so that its
    BeliefState stays up to date.

    select_action() runs PIMC search and returns the sampled action.
    """

    def __init__(
        self,
        config: AZConfig,
        state_enc: StateEncoder,
        belief_net: BeliefNet,
        q_net: QNet,
        player_id: int = 0,
    ):
        self.config = config
        self.state_enc = state_enc
        self.belief_net = belief_net
        self.q_net = q_net
        self.player_id = player_id

        # Per-game state — reset at the start of each hand
        self._bs: BeliefState | None = None
        self._hand: str | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def new_game(self, hand: str) -> None:
        """Call at the start of each new hand with the agent's private card."""
        self._hand = hand
        self._bs = make_belief_state(hand, d_model=self.config.d_model)

    def observe_event(self, actor: int | None, event_id: int) -> None:
        """
        Update internal state after any game event (own or opponent action, deal).

        actor: player who acted (0 or 1), or None for a deal event.
        event_id: from action_event_id() or deal_event_id().
        """
        assert self._bs is not None, "Call new_game() before observe_event()"
        with torch.no_grad():
            e_prime, P_new = self.state_enc.encode_event(
                event_id, self._bs.P_current, self._bs.P_history
            )
            update_belief_state(
                self._bs, actor, self.player_id, e_prime, P_new, self.belief_net
            )

    # ── BaseAgent interface ──────────────────────────────────────────────────

    def select_action(self, obs: Observation) -> Action:
        """
        Run PIMC search and sample action proportional to softmax(Q*/T).

        Requires new_game() to have been called for the current hand.
        The caller is responsible for calling observe_event() after this
        action is executed.
        """
        from .rollout import pimc_search  # local import to avoid circular dep

        assert self._bs is not None, "Call new_game() before select_action()"

        legal = obs.legal_actions
        q_star = pimc_search(
            obs=obs,
            player_i=self.player_id,
            h_i=self._hand,
            bs_i=self._bs,
            legal_actions=legal,
            state_enc=self.state_enc,
            belief_net=self.belief_net,
            q_net=self.q_net,
            k=self.config.k_rollouts,
            T=self.config.temperature,
        )

        probs = masked_softmax(q_star, legal, self.config.temperature)
        action_idx = torch.multinomial(probs, 1).item()
        return Action(action_idx)

    def get_action_evaluations(self, obs: Observation) -> list:
        from .rollout import pimc_search
        if self._bs is None:
            return []
        q_star = pimc_search(
            obs=obs,
            player_i=self.player_id,
            h_i=self._hand,
            bs_i=self._bs,
            legal_actions=obs.legal_actions,
            state_enc=self.state_enc,
            belief_net=self.belief_net,
            q_net=self.q_net,
            k=self.config.k_rollouts,
            T=self.config.temperature,
        )
        return q_star.tolist()

    def save_model(self, path: str) -> None:
        torch.save({
            'state_enc': self.state_enc.state_dict(),
            'belief_net': self.belief_net.state_dict(),
            'q_net': self.q_net.state_dict(),
        }, path)

    def load_model(self, path: str) -> None:
        ckpt = torch.load(path, map_location='cpu')
        self.state_enc.load_state_dict(ckpt['state_enc'])
        self.belief_net.load_state_dict(ckpt['belief_net'])
        self.q_net.load_state_dict(ckpt['q_net'])

    def set_train_mode(self, mode: bool) -> None:
        if mode:
            self.state_enc.train(); self.belief_net.train(); self.q_net.train()
        else:
            self.state_enc.eval(); self.belief_net.eval(); self.q_net.eval()
