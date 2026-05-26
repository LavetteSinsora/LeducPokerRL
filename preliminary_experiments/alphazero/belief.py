"""
Belief system for the AlphaZero-style agent.

Each player i maintains four belief vectors at all times:

    b_mine          : i's distribution over j's private hand   (3-dim simplex)
    b_opp[0/1/2]    : IF j holds J/Q/K, j's distribution over i's hand   (3-dim simplex each)

The three b_opp vectors are the "opponent belief portfolio" — used during PIMC
search when we imagine each possible opponent hand.

Update rules:
    - j acts → b_mine is updated     (i observes j's action)
    - i acts → all b_opp[k] updated  (i models how j's belief would shift)
    - DEAL    → all four updated      (both players observe the board card)

Initialization:
    Beliefs start from the informed prior given own hand h_i.
    Knowing h_i = J means 1 of the 5 remaining cards is J, 2 are Q, 2 are K.
"""

import copy
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .state_encoder import CARD_TO_IDX, IDX_TO_CARD


# ── Prior ────────────────────────────────────────────────────────────────────

def informed_prior(h: str) -> torch.Tensor:
    """
    Marginal distribution over opponent's hand given own hand h.

    Leduc deck: 2×J, 2×Q, 2×K.  After dealing h to self, 5 cards remain.
    If h = J: remaining = J×1, Q×2, K×2  →  [1/5, 2/5, 2/5]
    Similarly for Q and K (same formula, different index).
    """
    prior = torch.full((3,), 2.0 / 5.0)
    prior[CARD_TO_IDX[h]] = 1.0 / 5.0
    return prior


# ── Belief network ───────────────────────────────────────────────────────────

class BeliefNet(nn.Module):
    """
    f_belief: (b_prev, P_t, e'_t) → b_next

    Input:  3 (prior belief) + d (P_t) + d (e'_t) = 3 + 2d
    Output: 3-dim simplex (softmax over logits)
    """

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        in_dim = 3 + d + d

        layers = []
        curr = in_dim
        for h in config.belief_hidden:
            layers += [nn.Linear(curr, h), nn.ReLU()]
            curr = h
        layers.append(nn.Linear(curr, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        b: torch.Tensor,        # (3,)
        P_t: torch.Tensor,      # (d,)
        e_prime: torch.Tensor,  # (d,)
    ) -> torch.Tensor:
        """Returns updated belief, shape (3,)."""
        x = torch.cat([b, P_t, e_prime], dim=-1)
        return F.softmax(self.mlp(x), dim=-1)


# ── Belief state ─────────────────────────────────────────────────────────────

@dataclass
class BeliefState:
    """
    All belief and public-state information held by one player during gameplay.

    This object is intentionally *separated* from the neural networks —
    it holds the running tensor values only.  Call BeliefState.copy() to
    get an independent snapshot before starting a rollout.
    """
    b_mine: torch.Tensor          # (3,) — my belief about opponent's hand
    b_opp: List[torch.Tensor]     # [b_opp_J, b_opp_Q, b_opp_K], each (3,)
                                  # b_opp[k] = opponent's belief about me IF opponent holds card k
    P_current: torch.Tensor       # (d,) — current public state encoding
    P_history: List[torch.Tensor] # [P_0, P_1, ..., P_current], each (d,)

    def copy(self) -> "BeliefState":
        """Return a detached deep copy suitable for use in rollouts."""
        return BeliefState(
            b_mine=self.b_mine.detach().clone(),
            b_opp=[b.detach().clone() for b in self.b_opp],
            P_current=self.P_current.detach().clone(),
            P_history=[p.detach().clone() for p in self.P_history],
        )


def make_belief_state(h_i: str, d_model: int = 8) -> BeliefState:
    """Initialise a BeliefState for player holding h_i at game start."""
    P0 = torch.zeros(d_model)
    return BeliefState(
        b_mine=informed_prior(h_i),
        b_opp=[informed_prior(hj) for hj in IDX_TO_CARD],  # j's prior if j holds J/Q/K
        P_current=P0.clone(),
        P_history=[P0],
    )


# ── Belief update helpers ────────────────────────────────────────────────────

def update_belief_state(
    bs: BeliefState,
    actor: int,             # which player just acted (None for deal events)
    player_i: int,          # which player owns this BeliefState
    e_prime: torch.Tensor,  # contextualized event embedding from state encoder
    P_new: torch.Tensor,    # new public state after event
    belief_net: BeliefNet,
) -> None:
    """
    Update bs in-place after observing one event (action or deal).

    actor == None  → deal event: update all four belief vectors
    actor == j     → opponent acted: update b_mine only
    actor == i     → I acted: update all b_opp[k]
    Then advance P_current and P_history.
    """
    P_before = bs.P_current  # state *before* this event (used as input to f_belief)

    if actor is None:
        # Deal event: both players learn from the community card
        bs.b_mine = belief_net(bs.b_mine, P_before, e_prime)
        for k in range(3):
            bs.b_opp[k] = belief_net(bs.b_opp[k], P_before, e_prime)
    elif actor != player_i:
        # Opponent acted: update my belief about them
        bs.b_mine = belief_net(bs.b_mine, P_before, e_prime)
    else:
        # I acted: update the opponent's hypothetical beliefs about me
        for k in range(3):
            bs.b_opp[k] = belief_net(bs.b_opp[k], P_before, e_prime)

    bs.P_history.append(P_new)
    bs.P_current = P_new
