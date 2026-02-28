"""
Distributional Value Agent for Leduc Hold'em.

Uses two separate networks:
  - Value network: identical to ValueBasedAgent's ValueNetwork (predicts E[V(s)])
  - Variance network: separate MLP that predicts Var[V(s)]

Risk-sensitive decision making via:
    score(a) = E[V(post(s,a))] - beta * Std[V(post(s,a))]

By keeping the value network identical and independent of the variance
network, we ensure the mean predictions are as good as the scalar agent,
while the variance head adds risk-sensitivity on top.
"""

import torch
import torch.nn as nn
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from .base import BaseAgent


class ValueNetwork(nn.Module):
    """Same architecture as ValueBasedAgent's ValueNetwork."""
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VarianceNetwork(nn.Module):
    """Separate MLP for predicting variance of returns."""
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.net(x))


class DualHeadModel(nn.Module):
    """Container for the two separate networks."""
    def __init__(self, input_size: int = 15, hidden_size: int = 64):
        super().__init__()
        self.value_net = ValueNetwork(input_size, hidden_size)
        self.var_net = VarianceNetwork(input_size, hidden_size)

    def forward(self, x: torch.Tensor):
        mean = self.value_net(x)
        var = self.var_net(x)
        return mean, var


class DistributionalValueAgent(BaseAgent):
    """
    Risk-sensitive agent with separate value and variance networks.
    Selects actions via: argmax_a [mean(a) - beta * std(a)]
    """

    CARD_MAP = {'J': 0, 'Q': 1, 'K': 2}
    MAX_CHIPS = 13

    def __init__(self, model_path: str = None, temperature: float = 1.0,
                 risk_beta: float = 0.5, n_quantiles: int = 10):
        self.input_size = 15
        self.temperature = temperature
        self.risk_beta = risk_beta
        self.n_quantiles = n_quantiles
        self.train_mode = False

        self.taus = torch.tensor([(2 * i + 1) / (2 * n_quantiles)
                                  for i in range(n_quantiles)])

        self.model = DualHeadModel(self.input_size)
        if model_path:
            self.load_model(model_path)
        self.model.eval()

    def set_train_mode(self, mode: bool):
        self.train_mode = mode
        self.model.train(mode)

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, weights_only=True))

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        if viewer_id is None:
            viewer_id = obs.current_player

        hand_idx = self.CARD_MAP.get(obs.player_hand)
        hand_vec = torch.zeros(3)
        if hand_idx is not None:
            hand_vec[hand_idx] = 1.0

        board_idx = self.CARD_MAP.get(obs.board, 3)
        board_vec = torch.zeros(4)
        board_vec[board_idx] = 1.0

        p0_pot, p1_pot = obs.pot
        pot_rel = [p0_pot, p1_pot] if viewer_id == 0 else [p1_pot, p0_pot]
        pot_vec = torch.tensor(pot_rel, dtype=torch.float32) / self.MAX_CHIPS

        features = torch.tensor([
            1.0 if viewer_id == obs.current_player else 0.0,
            float(viewer_id),
            float(obs.current_round),
            1.0 if obs.is_finished else 0.0,
            1.0 if (obs.board is not None and obs.player_hand == obs.board) else 0.0,
            obs.raises_this_round / 2.0,
        ], dtype=torch.float32)

        return torch.cat([hand_vec, board_vec, pot_vec, features]).unsqueeze(0)

    def _get_mean_std(self, obs: Observation, viewer_id: int):
        encoded = self.encode_observation(obs, viewer_id=viewer_id)
        with torch.no_grad():
            mean, var = self.model(encoded)
            return mean.item(), var.item() ** 0.5

    def get_action_evaluations(self, obs: Observation) -> list:
        evaluations = []
        current_p = obs.current_player

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)

            if done and action == Action.FOLD:
                fold_val = -float(obs.pot[current_p])
                mean, std = fold_val, 0.0
            else:
                mean, std = self._get_mean_std(post_obs, viewer_id=current_p)

            risk_score = mean - self.risk_beta * std
            encoded = self.encode_observation(post_obs, viewer_id=current_p)

            evaluations.append({
                "action": action,
                "value": risk_score,
                "mean": mean,
                "std": std,
                "is_terminal": done,
                "encoded": encoded,
            })
        return evaluations

    def select_action(self, obs: Observation) -> Action:
        results = self.get_action_evaluations(obs)
        if not results:
            return Action.FOLD

        try:
            if self.train_mode:
                # During training, use mean only for exploration (risk-neutral)
                values = torch.tensor([r["mean"] for r in results])
                probs = torch.softmax(values / self.temperature, dim=0)
                idx = torch.multinomial(probs, 1).item()
                return results[idx]["action"]
            else:
                return max(results, key=lambda x: x["value"])["action"]
        except Exception as e:
            print(f"Error in DistributionalValueAgent selection: {e}")
            return results[0]["action"]
