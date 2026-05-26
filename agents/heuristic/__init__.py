from .agent import HeuristicAgent

AGENT_META = {
    "id": "heuristic",
    "display_name": "Heuristic Agent",
    "description": "Rule-based baseline with hand-crafted betting logic.",
    "agent_class": HeuristicAgent,
    "is_trainable": False,
    "category": "rule_based",
    "checkpoint_name": None,
}
