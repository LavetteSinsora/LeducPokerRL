from agents import registry
from engine.leduc_game import LeducGame


def test_registry_discovers_promoted_agents():
    ids = {meta.id for meta in registry.list_agents()}
    assert {"heuristic", "value_based", "adaptive_value", "modulated_value", "cfr"} <= ids


def test_promoted_agents_can_select_actions():
    game = LeducGame()
    obs = game.get_observation(viewer_id=0)

    for agent_id in ["heuristic", "value_based", "adaptive_value", "modulated_value"]:
        agent = registry.create(agent_id)
        assert agent.select_action(obs) in obs.legal_actions
