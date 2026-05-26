from .agent import ModulatedValueAgent
from .trainer import ModulatedValueTrainer

AGENT_META = {
    "id": "modulated_value",
    "display_name": "Modulated Value Agent",
    "description": "Frozen value baseline plus gated opponent-specific modulation.",
    "agent_class": ModulatedValueAgent,
    "is_trainable": True,
    "category": "rl",
    "trainer_class": ModulatedValueTrainer,
}
