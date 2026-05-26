"""
Public state encoder for AlphaZero-style agent.

f_state: (P_t, e'_t) → P_{t+1}
  where e'_t = CatEmbed(e_t) + CrossAttn(Q=CatEmbed(e_t), KV=P_{1:t})

The encoder is *stateless* — P_current and P_history are maintained by the caller,
making it straightforward to deep-copy for rollouts.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

# ── Card and event constants ─────────────────────────────────────────────────

CARD_TO_IDX = {'J': 0, 'Q': 1, 'K': 2}
IDX_TO_CARD = ['J', 'Q', 'K']
N_CARDS = 3

# Event IDs:
#   player ∈ {0, 1}, action ∈ {FOLD=0, CALL=1, RAISE=2}  → player * 3 + action
#   deal card ∈ {J, Q, K}                                  → 6 + CARD_TO_IDX[card]
N_EVENTS = 9


def action_event_id(player: int, action_value: int) -> int:
    """Map (player, action) → event ID in [0, 5]."""
    return player * 3 + action_value


def deal_event_id(card: str) -> int:
    """Map community card → event ID in [6, 8]."""
    return 6 + CARD_TO_IDX[card]


# ── State encoder ────────────────────────────────────────────────────────────

class StateEncoder(nn.Module):
    """
    Stateless recurrent public-state encoder.

    Usage:
        e_prime, P_new = encoder.encode_event(event_id, P_current, P_history)
        P_history.append(P_new)
        P_current = P_new
    """

    def __init__(self, config):
        super().__init__()
        d = config.d_model

        self.event_embed = nn.Embedding(config.n_events, d)

        # Single-head cross-attention:
        #   Q  = current event embedding  (1 × d)
        #   KV = P_history                (T × d)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=1, batch_first=True
        )

        # State transition MLP: [P_t (d) ; e'_t (d)] → P_{t+1} (d)
        layers = []
        in_dim = 2 * d
        for h in config.state_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, d))
        self.state_mlp = nn.Sequential(*layers)

    def encode_event(
        self,
        event_id: int,
        P_current: torch.Tensor,        # (d,)
        P_history: List[torch.Tensor],  # [P_0, ..., P_t], each (d,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one event.

        Returns:
            e_prime : contextualized event embedding, shape (d,)
            P_new   : updated public state,           shape (d,)
        """
        # Categorical embedding
        eid = torch.tensor([event_id], dtype=torch.long, device=P_current.device)
        e_base = self.event_embed(eid)          # (1, d)

        # Cross-attention: Q = e_base, KV = stacked P_history
        KV = torch.stack(P_history, dim=0).unsqueeze(0)  # (1, T, d)
        Q  = e_base.unsqueeze(0)                           # (1, 1, d)
        context, _ = self.cross_attn(Q, KV, KV)            # (1, 1, d)

        e_prime = e_base.squeeze(0) + context.squeeze(0).squeeze(0)  # (d,)

        # State transition
        combined = torch.cat([P_current, e_prime], dim=-1)  # (2d,)
        P_new = self.state_mlp(combined)                     # (d,)

        return e_prime, P_new


def make_initial_state(d_model: int, device: torch.device = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Return (P_current, P_history) at game start.
    P_0 = zeros(d) — no public information before the first action.
    """
    if device is None:
        device = torch.device('cpu')
    P0 = torch.zeros(d_model, device=device)
    return P0.clone(), [P0]
