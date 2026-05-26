from .agent import ValueBasedAgent
from .trainer import SelfPlayTrainer

AGENT_META = {
    "id": "value_based",
    "display_name": "Value Based Agent",
    "description": "TD(0) value-network baseline trained in self-play.",
    "agent_class": ValueBasedAgent,
    "is_trainable": True,
    "category": "rl",
    "trainer_class": SelfPlayTrainer,
}
