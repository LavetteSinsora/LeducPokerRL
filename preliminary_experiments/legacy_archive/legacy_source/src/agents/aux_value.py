from .value_based import ValueBasedAgent


class AuxValueAgent(ValueBasedAgent):
    """
    Value agent trained with a pre-action Bellman consistency auxiliary loss.

    At inference time this is identical to ValueBasedAgent: same 15-dim
    encoding, same ValueNetwork, same Boltzmann/greedy action selection.
    No opponent-stats features are included (+4 dims are absent).

    The difference lives entirely in AuxValueTrainer, which adds an auxiliary
    supervised loss at each decision step:

        V(pre-action state) ← max_a V(post-action state_a)   [detached]

    This forces the network's representation to satisfy Bellman consistency
    at the current player's own decision nodes, providing additional gradient
    signal without altering the TD(0) bootstrap chain on post-action states.
    """
    pass
