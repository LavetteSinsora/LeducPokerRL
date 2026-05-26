from .agent import CFRAgent
from .trainer import CFRTrainer

AGENT_META = {
    "id": "cfr",
    "display_name": "CFR Agent",
    "description": "Tabular CFR+ Nash-equilibrium reference agent.",
    "agent_class": CFRAgent,
    "is_trainable": True,
    "category": "game_theory",
    "trainer_class": CFRTrainer,
}
