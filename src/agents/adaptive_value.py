import torch
from dataclasses import replace
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from .value_based import ValueBasedAgent, ValueNetwork


class AdaptiveValueAgent(ValueBasedAgent):
    """
    Value-based agent that exploits opponent tendencies.

    Extends ValueBasedAgent with 4 additional input features derived from
    cross-hand opponent behavior statistics:
      - fold_rate, raise_rate, fold_to_raise_rate, confidence (hands_observed)

    When opponent_stats is not available (e.g. single-hand evaluation),
    defaults to [0.5, 0.5, 0.5, 0.0] — uninformative prior, zero confidence.

    Total input: 15 (base) + 4 (stats) = 19 features.
    """

    STATS_SIZE = 4

    def __init__(self, model_path: str = None, temperature: float = 1.0):
        # Bypass ValueBasedAgent.__init__ to set input_size=19
        self.input_size = 15 + self.STATS_SIZE
        self.temperature = temperature
        self.train_mode = False

        self.model = ValueNetwork(self.input_size)
        if model_path:
            self.load_model(model_path)
        self.model.eval()

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Encodes base features (15) + opponent stats (4) = 19 features."""
        base = super().encode_observation(obs, viewer_id)  # [1, 15]

        if obs.opponent_stats is not None and hasattr(obs.opponent_stats, 'to_feature_vector'):
            stats_vec = torch.tensor(obs.opponent_stats.to_feature_vector(),
                                     dtype=torch.float32)
        else:
            # No stats available — uninformative default
            stats_vec = torch.tensor([0.5, 0.5, 0.5, 0.0], dtype=torch.float32)

        return torch.cat([base.squeeze(0), stats_vec]).unsqueeze(0)  # [1, 19]

    def get_action_evaluations(self, obs: Observation) -> list:
        """1-step lookahead, carrying opponent_stats into simulated states.

        The parent's default discards opponent_stats when calling
        LeducGame.simulate_action(), which always returns opponent_stats=None.
        We inject the current obs.opponent_stats back into every simulated
        post-state so the value network actually receives meaningful stats.
        """
        evaluations = []
        current_p = obs.current_player

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)

            # Carry opponent_stats forward so the 19-feature encoding is real.
            if obs.opponent_stats is not None:
                post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)

            if done and action == Action.FOLD:
                val = -float(obs.pot[current_p])
            else:
                val = self._get_value(post_obs, viewer_id=current_p)

            encoded = self.encode_observation(post_obs, viewer_id=current_p)
            evaluations.append({
                "action": action,
                "value": val,
                "is_terminal": done,
                "encoded": encoded,
            })
        return evaluations
