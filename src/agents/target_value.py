import copy
import torch
from .value_based import ValueBasedAgent, ValueNetwork


class TargetValueAgent(ValueBasedAgent):
    """
    Value agent with a frozen target network for stable TD learning.

    Same architecture as ValueBasedAgent (15-dim input, 64-hidden).
    The target_model is a frozen copy of the main model, updated every
    K gradient steps by copying the main model's weights.

    At inference time, behavior is identical to ValueBasedAgent
    (uses the main model for action selection).
    """

    def __init__(self, model_path=None, temperature=1.0):
        # Pre-create target network so load_model() can populate it
        # (super().__init__ may call load_model before we get control back)
        self._target_initialized = False
        super().__init__(model_path=model_path, temperature=temperature)
        if not self._target_initialized:
            # No model_path was provided, so load_model was never called.
            # Create target as a copy of the (randomly initialized) main model.
            self.target_model = copy.deepcopy(self.model)
            self._freeze_target()
        del self._target_initialized

    def _freeze_target(self):
        """Put target model in eval mode and disable gradients."""
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

    def sync_target(self):
        """Copy main model weights to target model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def get_target_value(self, obs, viewer_id):
        """Get value estimate from the frozen target network."""
        encoded = self.encode_observation(obs, viewer_id=viewer_id)
        with torch.no_grad():
            return self.target_model(encoded).item()

    def save_model(self, path):
        """Save both main and target model weights."""
        torch.save({
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
        }, path)

    def load_model(self, path):
        """Load both main and target model weights."""
        # Ensure target_model exists (may be called during __init__ before it's set)
        if not hasattr(self, 'target_model'):
            self.target_model = copy.deepcopy(self.model)

        checkpoint = torch.load(path)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            self.target_model.load_state_dict(checkpoint['target_model'])
        else:
            # Backwards compatible: load as regular state dict
            self.model.load_state_dict(checkpoint)
            self.sync_target()

        self._freeze_target()
        self._target_initialized = True
