from .actor_critic import ActorCriticAgent


class EntropyACAgent(ActorCriticAgent):
    """Actor-critic agent with entropy regularization.

    Same network architecture as ActorCriticAgent.
    The difference is in EntropyACTrainer which adds -beta*H(pi) to the loss,
    encouraging mixed strategies that are harder to exploit.
    """
    pass
