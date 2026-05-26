from .agent import AdaptiveValueAgent
from .trainer import AdaptiveTrainer

AGENT_META = {
    "id": "adaptive_value",
    "display_name": "Adaptive Value Agent",
    "description": "Value network augmented with opponent session statistics.",
    "agent_class": AdaptiveValueAgent,
    "is_trainable": True,
    "category": "rl",
    "trainer_class": AdaptiveTrainer,
}
