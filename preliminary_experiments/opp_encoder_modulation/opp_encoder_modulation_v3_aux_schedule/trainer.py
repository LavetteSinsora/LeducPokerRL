"""Trainer that delays and ramps in the auxiliary action-prediction loss."""

from experiments.opp_encoder_modulation_v2.trainer import OpponentEncoderModulationTrainer as V2Trainer


class OpponentEncoderModulationTrainer(V2Trainer):
    """Keep v2 intact, but schedule the auxiliary loss instead of applying it immediately."""

    def __init__(
        self,
        agent,
        action_loss_weight: float = 0.5,
        aux_warmup_sessions: int = 10000,
        aux_ramp_sessions: int = 10000,
        **kwargs,
    ):
        self.target_action_loss_weight = action_loss_weight
        self.aux_warmup_sessions = aux_warmup_sessions
        self.aux_ramp_sessions = aux_ramp_sessions
        super().__init__(agent, action_loss_weight=0.0, **kwargs)

    def current_action_loss_weight(self) -> float:
        if self.sessions_played <= self.aux_warmup_sessions:
            return 0.0
        if self.aux_ramp_sessions <= 0:
            return self.target_action_loss_weight
        progress = (self.sessions_played - self.aux_warmup_sessions) / self.aux_ramp_sessions
        progress = max(0.0, min(1.0, progress))
        return self.target_action_loss_weight * progress

    def update_model(self, batch_data: list) -> float:
        self.action_loss_weight = self.current_action_loss_weight()
        loss = super().update_model(batch_data)
        self.last_metrics["scheduled_action_loss_weight"] = self.action_loss_weight
        return loss
