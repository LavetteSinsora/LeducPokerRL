from .value_based import ValueBasedAgent


class NStepValueAgent(ValueBasedAgent):
    """Value agent trained with n-step returns instead of TD(0).

    Identical architecture and action selection to ValueBasedAgent.
    The difference is entirely in NStepValueTrainer, which computes
    n-step TD targets: instead of bootstrapping from V(s_{t+1}), it
    bootstraps from V(s_{t+n}) or uses the terminal reward when fewer
    than n steps remain. With typical Leduc chain lengths of 2-4 steps
    and n=3, most targets use the actual terminal reward -- giving
    cleaner gradient signal than TD(0) bootstrapping.
    """
    pass
