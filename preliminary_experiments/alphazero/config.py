from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AZConfig:
    # ── Shared model dimension ──────────────────────────────────────────────
    d_model: int = 8
    n_events: int = 9  # 9 event types (6 player actions + 3 deal events)

    # ── f_state MLP hidden widths (input=2*d, output=d) ────────────────────
    state_hidden: Tuple[int, ...] = (16, 16)

    # ── f_belief MLP hidden widths (input=3+d+d=19, output=3) ──────────────
    belief_hidden: Tuple[int, ...] = (32, 32)

    # ── Q-network hidden widths (input=d+3+3=14, output=3) ─────────────────
    q_hidden: Tuple[int, ...] = (64, 64, 64)

    # ── Search ──────────────────────────────────────────────────────────────
    k_rollouts: int = 10
    temperature: float = 1.0        # softmax temperature for action selection
    T_self: float = 1.0             # rollout temp for searching player (v4+)
    T_opp:  float = 1.0             # rollout temp for imagined opponent  (v4+)

    # ── Training ────────────────────────────────────────────────────────────
    n_episodes: int = 200_000
    lr: float = 1e-3
    lambda_belief: float = 0.1      # weight for belief CE loss
    lambda_entropy: float = 0.0     # entropy bonus weight (0 = disabled)

    # ── Stabilisation (v4+) ─────────────────────────────────────────────────
    target_sync_freq: int = 0       # sync frozen Q-net target every N ep (0=off)
    replay_buffer_size: int = 0     # decision replay buffer capacity   (0=off)
    replay_batch_size: int = 256    # minibatch size for Q-net update
