"""
Opponent hand posterior belief network for Leduc Hold'em.

Phase 1: BeliefNetwork — GRU that outputs P(opp_rank | I_t).
Phase 2: BeliefPosteriorValueAgent — freeze BeliefNetwork, train value net
          with belief vector as extra input (15 + 3 = 18-dim).
"""

from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from agents.base import BaseAgent
from agents.value_based.agent import ValueBasedAgent, ValueNetwork
from engine.leduc_game import Action, LeducGame
from engine.observation import Observation

CARD_MAP = {"J": 0, "Q": 1, "K": 2}
ACTION_MAP = {"FOLD": 0, "CALL": 1, "RAISE": 2}
MAX_SEQ_LEN = 8   # max actions per Leduc hand (4 pre-flop + 4 flop)
CTX_DIM = 7       # hand(3) + board(4)
TOKEN_DIM = 4     # is_opponent(1) + action_onehot(3)
GRU_HIDDEN = 32
NUM_RANKS = 3     # J, Q, K


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_static_context(my_hand: str, board: Optional[str]) -> torch.Tensor:
    """7-dim static context: [hand_onehot(3), board_onehot(4)]."""
    hand_vec = torch.zeros(3)
    hand_vec[CARD_MAP[my_hand]] = 1.0
    board_vec = torch.zeros(4)
    board_vec[CARD_MAP.get(board, 3)] = 1.0   # index 3 = None/pre-flop
    return torch.cat([hand_vec, board_vec])


def encode_action_tokens(
    action_history: tuple,
    viewer_id: int,
    max_len: int = MAX_SEQ_LEN,
) -> Tuple[torch.Tensor, int]:
    """
    Returns (tokens, seq_len).
    tokens: (max_len, TOKEN_DIM) float tensor, zero-padded.
    seq_len: number of real events.
    """
    tokens = torch.zeros(max_len, TOKEN_DIM)
    seq_len = min(len(action_history), max_len)
    for i, (player, action_name) in enumerate(action_history[:max_len]):
        is_opp = float(int(player) != viewer_id)
        action_oh = [0.0, 0.0, 0.0]
        action_oh[ACTION_MAP[action_name]] = 1.0
        tokens[i] = torch.tensor([is_opp] + action_oh)
    return tokens, seq_len


def compute_legal_mask(my_hand: str, board: Optional[str]) -> torch.Tensor:
    """
    Boolean mask [J, Q, K]: True if that rank can legally be opponent's hand.
    Deck has 2 of each rank; remove my_hand and board (if present).
    """
    remaining = [2, 2, 2]
    remaining[CARD_MAP[my_hand]] -= 1
    if board is not None:
        remaining[CARD_MAP[board]] -= 1
    return torch.tensor([r > 0 for r in remaining], dtype=torch.bool)


# ---------------------------------------------------------------------------
# BeliefNetwork
# ---------------------------------------------------------------------------

class ContextInitNet(nn.Module):
    """Projects static context (7-dim) to GRU initial hidden state (hidden_size)."""
    def __init__(self, ctx_dim: int = CTX_DIM, hidden_size: int = GRU_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden_size),
            nn.ReLU(),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        # ctx: (batch, ctx_dim) → (1, batch, hidden_size) for GRU
        return self.net(ctx).unsqueeze(0)


class BeliefNetwork(nn.Module):
    """
    GRU-based posterior estimator: P(opp_rank | I_t).

    Forward pass:
      ctx   : (batch, 7)   static context (my_hand + board one-hots)
      tokens: (batch, T, 4) event tokens, zero-padded to T=MAX_SEQ_LEN
      seq_lens: (batch,) int  actual sequence lengths (for correct final state)

    Returns:
      probs : (batch, 3) posterior distribution over J/Q/K (masked + softmax)
      logits: (batch, 3) raw logits before masking (for loss computation)
    """

    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        hidden_size: int = GRU_HIDDEN,
        ctx_dim: int = CTX_DIM,
        num_ranks: int = NUM_RANKS,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.init_net = ContextInitNet(ctx_dim, hidden_size)
        self.gru = nn.GRU(token_dim, hidden_size, batch_first=True)
        self.output_head = nn.Linear(hidden_size, num_ranks)

    def forward(
        self,
        ctx: torch.Tensor,
        tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ctx      : (B, 7)
        tokens   : (B, T, 4)
        seq_lens : (B,) int64 — actual number of real events
        mask     : (B, 3) bool — True where rank is legal (optional; if None, no masking)
        """
        h0 = self.init_net(ctx)   # (1, B, hidden_size)

        batch_size = tokens.size(0)
        max_len = tokens.size(1)

        if max_len == 0 or seq_lens.max().item() == 0:
            # No events — use h0 directly
            h_final = h0.squeeze(0)   # (B, hidden_size)
        else:
            _, h_n = self.gru(tokens, h0)   # h_n: (1, B, hidden_size)
            # Collect the hidden state at each sequence's true end
            # For efficiency, gather from all timestep outputs instead
            output, _ = self.gru(tokens, h0)   # output: (B, T, hidden_size)
            # Index the output at (seq_len - 1) for each sample; if seq_len==0 use h0
            idx = (seq_lens - 1).clamp(min=0).long()  # (B,)
            idx_expanded = idx.view(batch_size, 1, 1).expand(batch_size, 1, self.hidden_size)
            h_final = output.gather(1, idx_expanded).squeeze(1)  # (B, hidden_size)
            # For samples with seq_len==0, override with h0
            zero_mask = (seq_lens == 0)
            if zero_mask.any():
                h_final[zero_mask] = h0.squeeze(0)[zero_mask]

        logits = self.output_head(h_final)  # (B, 3)

        if mask is not None:
            # Where mask is False, set logit to -inf to zero probability
            logits_masked = logits.clone()
            logits_masked[~mask] = float("-inf")
        else:
            logits_masked = logits

        probs = torch.softmax(logits_masked, dim=-1)
        return probs, logits_masked

    def predict_single(
        self,
        my_hand: str,
        board: Optional[str],
        action_history: tuple,
        viewer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method for single-sample inference (no batching)."""
        ctx = encode_static_context(my_hand, board).unsqueeze(0)
        tokens, seq_len = encode_action_tokens(action_history, viewer_id)
        tokens = tokens.unsqueeze(0)
        seq_lens = torch.tensor([seq_len], dtype=torch.long)
        mask = compute_legal_mask(my_hand, board).unsqueeze(0)

        with torch.no_grad():
            probs, logits = self.forward(ctx, tokens, seq_lens, mask)
        return probs.squeeze(0), logits.squeeze(0)


# ---------------------------------------------------------------------------
# BeliefPosteriorValueAgent (Phase 2)
# ---------------------------------------------------------------------------

class BeliefPosteriorValueAgent(ValueBasedAgent):
    """
    Value agent augmented with a frozen belief network.

    At each decision point:
      1. Run frozen BeliefNetwork → 3-dim belief vector b = P(J, Q, K | I_t)
      2. Concatenate to standard 15-dim state encoding → 18-dim
      3. Feed through a new value network (trained from scratch)

    The BeliefNetwork is loaded from a saved Phase-1 checkpoint and frozen.
    """

    BASE_INPUT_SIZE = 15   # standard ValueBasedAgent encoding
    BELIEF_DIM = 3
    FULL_INPUT_SIZE = BASE_INPUT_SIZE + BELIEF_DIM  # 18

    def __init__(
        self,
        belief_model_path: str = None,
        model_path: str = None,
        temperature: float = 1.0,
    ):
        # Build belief network (not calling super().__init__ yet to control input_size)
        self.temperature = temperature
        self.train_mode = False

        self.belief_net = BeliefNetwork()
        if belief_model_path and Path(belief_model_path).exists():
            state = torch.load(belief_model_path, weights_only=False)
            if isinstance(state, dict) and "belief_net" in state:
                self.belief_net.load_state_dict(state["belief_net"])
            else:
                self.belief_net.load_state_dict(state)
        # Freeze belief network
        for p in self.belief_net.parameters():
            p.requires_grad = False
        self.belief_net.eval()

        # Value network over 18-dim input
        self.input_size = self.FULL_INPUT_SIZE
        self.model = ValueNetwork(self.input_size)

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.belief_net.eval()   # always frozen
        self.model.train(mode)

    def _get_belief(self, obs: Observation, viewer_id: int) -> torch.Tensor:
        """Returns 3-dim belief vector P(J, Q, K | I_t)."""
        action_history = obs.action_history if obs.action_history is not None else ()
        probs, _ = self.belief_net.predict_single(
            my_hand=obs.player_hand,
            board=obs.board,
            action_history=action_history,
            viewer_id=viewer_id,
        )
        return probs  # (3,)

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """18-dim: [base_15, belief_3]."""
        if viewer_id is None:
            viewer_id = obs.current_player
        base = super().encode_observation(obs, viewer_id=viewer_id)  # (1, 15)
        belief = self._get_belief(obs, viewer_id).unsqueeze(0)       # (1, 3)
        return torch.cat([base, belief], dim=-1)                      # (1, 18)

    def save_model(self, path: str) -> None:
        torch.save({
            "value_net": self.model.state_dict(),
            "belief_net": self.belief_net.state_dict(),
        }, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        if isinstance(checkpoint, dict) and "value_net" in checkpoint:
            self.model.load_state_dict(checkpoint["value_net"])
            if "belief_net" in checkpoint:
                self.belief_net.load_state_dict(checkpoint["belief_net"])
        else:
            self.model.load_state_dict(checkpoint)
