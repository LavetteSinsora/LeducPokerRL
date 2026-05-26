import torch
import torch.nn as nn
from dataclasses import replace
from src.engine.leduc_game import Action, LeducGame
from src.engine.observation import Observation
from .value_based import ValueBasedAgent, ValueNetwork


class ModulationNetwork(nn.Module):
    """Network that produces opponent-specific value adjustments."""

    def __init__(self, input_size=19, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


class GateNetwork(nn.Module):
    """Confidence-gated network that controls modulation strength.

    Takes opponent stats (4-dim) as input and outputs a gate value in [0, 1].
    Should output low values when confidence is low (opponent stats unreliable).
    """

    def __init__(self, input_size=4, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class ModulatedValueAgent(ValueBasedAgent):
    """
    Value agent with gated opponent-specific modulation.

    Uses a frozen pretrained base network for universal value estimation,
    a modulation network for opponent-specific adjustments, and a
    confidence gate to control modulation strength.

    Architecture:
        V(s, opp) = V_base(s) + g(opp_stats) * Delta(s, opp_stats)

    where:
        V_base  = frozen 15-dim value network (pretrained)
        Delta   = trainable modulation network (19-dim input: game state + stats)
        g       = trainable gate network (4-dim input: opponent stats only)
    """

    STATS_SIZE = 4

    def __init__(self, model_path=None, temperature=1.0, base_model_path=None):
        # Set up base (15-dim) - will be frozen
        self.input_size = 15  # base encoding only
        self.temperature = temperature
        self.train_mode = False

        self.model = ValueNetwork(self.input_size)  # base network
        self.mod_net = ModulationNetwork(15 + self.STATS_SIZE)  # modulation
        self.gate_net = GateNetwork(self.STATS_SIZE)  # gate

        if base_model_path:
            self.model.load_state_dict(torch.load(base_model_path))

        # Freeze base network
        for p in self.model.parameters():
            p.requires_grad = False

        if model_path:
            self.load_model(model_path)

        self.model.eval()

    def encode_observation(self, obs: Observation, viewer_id: int = None) -> torch.Tensor:
        """Returns base encoding (15-dim) like parent ValueBasedAgent."""
        return super().encode_observation(obs, viewer_id)

    def _encode_stats(self, obs: Observation) -> torch.Tensor:
        """Returns opponent stats as a 4-dim tensor."""
        if obs.opponent_stats is not None and hasattr(obs.opponent_stats, 'to_feature_vector'):
            stats_vec = torch.tensor(
                obs.opponent_stats.to_feature_vector(), dtype=torch.float32
            )
        else:
            # No stats available -- uninformative default, zero confidence
            stats_vec = torch.tensor([0.5, 0.5, 0.5, 0.0], dtype=torch.float32)
        return stats_vec

    def _get_value(self, obs: Observation, viewer_id: int) -> float:
        """Computes V_base + gate * delta for the modulated value estimate."""
        base_enc = self.encode_observation(obs, viewer_id=viewer_id)  # [1, 15]
        stats_vec = self._encode_stats(obs)  # [4]

        with torch.no_grad():
            v_base = self.model(base_enc)  # [1, 1]

            # Modulation input: concat base encoding and stats
            mod_input = torch.cat([base_enc.squeeze(0), stats_vec]).unsqueeze(0)  # [1, 19]
            delta = self.mod_net(mod_input)  # [1, 1]

            # Gate input: stats only
            gate = self.gate_net(stats_vec.unsqueeze(0))  # [1, 1]

            value = v_base + gate * delta  # [1, 1]
            return value.item()

    def get_action_evaluations(self, obs: Observation) -> list:
        """1-step lookahead carrying stats forward (like AdaptiveValueAgent)."""
        evaluations = []
        current_p = obs.current_player

        for action in obs.legal_actions:
            post_obs, done = LeducGame.simulate_action(obs, action)

            # Carry opponent_stats forward into simulated post-states
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

    def set_train_mode(self, mode: bool):
        """Toggle train/eval mode. Base stays frozen; mod/gate toggle."""
        self.train_mode = mode
        # Base network always stays in eval mode (frozen)
        self.model.eval()
        self.mod_net.train(mode)
        self.gate_net.train(mode)

    def save_model(self, path: str) -> None:
        """Save all three networks (base + mod + gate)."""
        torch.save(
            {
                "base": self.model.state_dict(),
                "mod": self.mod_net.state_dict(),
                "gate": self.gate_net.state_dict(),
            },
            path,
        )

    def load_model(self, path: str) -> None:
        """Load model weights. Handles backward-compatible loading.

        If the file contains a dict with 'base', 'mod', 'gate' keys,
        loads all three networks. If it is just base weights (backward
        compatibility), loads only the base network.
        """
        data = torch.load(path)
        if isinstance(data, dict) and "base" in data:
            self.model.load_state_dict(data["base"])
            self.mod_net.load_state_dict(data["mod"])
            self.gate_net.load_state_dict(data["gate"])
        else:
            # Backward-compatible: file contains only base network weights
            self.model.load_state_dict(data)
