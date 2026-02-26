"""
Extended Adaptive Agent — Unit & Integration Tests

Tests the ExtendedAdaptiveAgent, which is a trivial pass-through subclass
of AdaptiveValueAgent used as a null hypothesis control for Round 3.
Even though the agent has no new logic, we comprehensively verify that
it inherits and exercises every capability of the parent class.

Run with: python -m pytest tests/unit/test_extended_adaptive.py -v
"""

import math
import os
import tempfile
import pytest
import torch

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.engine.poker_session import PokerSession
from src.agents.base import BaseAgent
from src.agents.value_based import ValueBasedAgent, ValueNetwork
from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.extended_adaptive import ExtendedAdaptiveAgent
from src.training.adaptive_trainer import AdaptiveTrainer


# =============================================================================
# 1. Class hierarchy
# =============================================================================

class TestClassHierarchy:
    """ExtendedAdaptiveAgent must be a subclass of AdaptiveValueAgent,
    ValueBasedAgent, and BaseAgent."""

    def test_is_subclass_of_adaptive_value_agent(self):
        assert issubclass(ExtendedAdaptiveAgent, AdaptiveValueAgent)

    def test_is_subclass_of_value_based_agent(self):
        assert issubclass(ExtendedAdaptiveAgent, ValueBasedAgent)

    def test_is_subclass_of_base_agent(self):
        assert issubclass(ExtendedAdaptiveAgent, BaseAgent)

    def test_isinstance_checks(self):
        agent = ExtendedAdaptiveAgent()
        assert isinstance(agent, AdaptiveValueAgent)
        assert isinstance(agent, ValueBasedAgent)
        assert isinstance(agent, BaseAgent)

    def test_is_not_same_class_as_parent(self):
        """Ensure it is a distinct class, not an alias."""
        assert ExtendedAdaptiveAgent is not AdaptiveValueAgent


# =============================================================================
# 2. Network architecture (input_size=19, hidden=64)
# =============================================================================

class TestNetworkArchitecture:

    def test_input_size_is_19(self):
        agent = ExtendedAdaptiveAgent()
        assert agent.input_size == 19

    def test_uses_value_network(self):
        agent = ExtendedAdaptiveAgent()
        assert isinstance(agent.model, ValueNetwork)

    def test_hidden_size_default_64(self):
        """The ValueNetwork default hidden_size is 64."""
        agent = ExtendedAdaptiveAgent()
        first_layer = agent.model.net[0]
        assert first_layer.in_features == 19
        assert first_layer.out_features == 64

    def test_output_size_is_1(self):
        """Value network outputs a single scalar."""
        agent = ExtendedAdaptiveAgent()
        last_layer = agent.model.net[-1]
        assert last_layer.out_features == 1

    def test_forward_pass_shape(self):
        agent = ExtendedAdaptiveAgent()
        x = torch.randn(1, 19)
        with torch.no_grad():
            out = agent.model(x)
        assert out.shape == (1, 1)

    def test_batch_forward_pass(self):
        agent = ExtendedAdaptiveAgent()
        x = torch.randn(8, 19)
        with torch.no_grad():
            out = agent.model(x)
        assert out.shape == (8, 1)


# =============================================================================
# 3. Encoding produces 19 features
# =============================================================================

class TestEncoding:

    def test_encoding_shape_from_game(self):
        """Encoding a live game observation should produce [1, 19]."""
        agent = ExtendedAdaptiveAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 19)

    def test_encoding_shape_with_stats(self):
        """Encoding with opponent stats should still produce [1, 19]."""
        from src.engine.poker_session import OpponentStats
        agent = ExtendedAdaptiveAgent()
        stats = OpponentStats()
        for _ in range(5):
            stats.record_action("FOLD", was_facing_raise=False)
            stats.record_hand_complete()

        obs = Observation(
            player_hand="K", board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            opponent_stats=stats,
        )
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 19)

    def test_encoding_without_stats_uses_defaults(self):
        """Without opponent_stats, last 4 features should be [0.5, 0.5, 0.5, 0.0]."""
        agent = ExtendedAdaptiveAgent()
        obs = Observation(
            player_hand="Q", board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
        )
        encoded = agent.encode_observation(obs)
        assert encoded[0, 15].item() == pytest.approx(0.5)
        assert encoded[0, 16].item() == pytest.approx(0.5)
        assert encoded[0, 17].item() == pytest.approx(0.5)
        assert encoded[0, 18].item() == pytest.approx(0.0)

    def test_encoding_with_stats_values(self):
        """Stats features should appear in positions 15-18."""
        from src.engine.poker_session import OpponentStats
        agent = ExtendedAdaptiveAgent()
        stats = OpponentStats()
        for _ in range(10):
            stats.record_action("FOLD", was_facing_raise=False)
            stats.record_hand_complete()

        obs = Observation(
            player_hand="K", board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            opponent_stats=stats,
        )
        encoded = agent.encode_observation(obs)
        fv = stats.to_feature_vector()
        for i, val in enumerate(fv):
            assert encoded[0, 15 + i].item() == pytest.approx(val, abs=1e-5)


# =============================================================================
# 4. Uses ValueNetwork (same as parent)
# =============================================================================

class TestUsesValueNetwork:

    def test_model_is_value_network(self):
        agent = ExtendedAdaptiveAgent()
        assert isinstance(agent.model, ValueNetwork)

    def test_model_is_same_class_as_parent(self):
        parent_agent = AdaptiveValueAgent()
        child_agent = ExtendedAdaptiveAgent()
        assert type(parent_agent.model) is type(child_agent.model)


# =============================================================================
# 5. Action selection works (Boltzmann train, greedy eval)
# =============================================================================

class TestActionSelection:

    def test_select_action_returns_legal_action(self):
        agent = ExtendedAdaptiveAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        action = agent.select_action(obs)
        assert isinstance(action, Action)
        assert action in obs.legal_actions

    def test_greedy_eval_mode(self):
        """In eval mode, agent should always pick the highest-value action."""
        agent = ExtendedAdaptiveAgent()
        agent.set_train_mode(False)
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        actions = [agent.select_action(obs) for _ in range(20)]
        assert len(set(a.value for a in actions)) == 1, "Greedy should always pick same action"

    def test_boltzmann_train_mode(self):
        """In train mode, agent should use Boltzmann exploration (may vary)."""
        agent = ExtendedAdaptiveAgent()
        agent.set_train_mode(True)
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        actions = [agent.select_action(obs) for _ in range(100)]
        for a in actions:
            assert a in obs.legal_actions

    def test_plays_full_game(self):
        """Agent should play a complete game without errors."""
        agent = ExtendedAdaptiveAgent()
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            obs = game.get_observation()
            action = agent.select_action(obs)
            game.step(action)
        assert game.is_finished

    def test_get_action_evaluations(self):
        agent = ExtendedAdaptiveAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        evals = agent.get_action_evaluations(obs)
        assert len(evals) == len(obs.legal_actions)
        for e in evals:
            assert "action" in e
            assert "value" in e
            assert "encoded" in e
            assert e["encoded"].shape == (1, 19)


# =============================================================================
# 6. Save/load round-trip
# =============================================================================

class TestSaveLoadRoundTrip:

    def test_save_and_load(self):
        """Model should produce identical outputs after save/load."""
        agent = ExtendedAdaptiveAgent()
        x = torch.randn(1, 19)

        with torch.no_grad():
            original_output = agent.model(x).clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            agent.save_model(path)

            loaded_agent = ExtendedAdaptiveAgent(model_path=path)
            with torch.no_grad():
                loaded_output = loaded_agent.model(x)

        torch.testing.assert_close(original_output, loaded_output)

    def test_save_creates_file(self):
        agent = ExtendedAdaptiveAgent()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            agent.save_model(path)
            assert os.path.exists(path)

    def test_load_changes_weights(self):
        """Loading a saved model should change the agent weights."""
        agent1 = ExtendedAdaptiveAgent()
        agent2 = ExtendedAdaptiveAgent()

        with torch.no_grad():
            for p in agent1.model.parameters():
                p.fill_(42.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            agent1.save_model(path)
            agent2.load_model(path)

        for p1, p2 in zip(agent1.model.parameters(), agent2.model.parameters()):
            assert torch.equal(p1, p2)


# =============================================================================
# 7. Training loop with AdaptiveTrainer runs without errors
# =============================================================================

class TestTrainingLoop:

    def test_training_completes(self):
        """A short training run should complete with finite loss."""
        agent = ExtendedAdaptiveAgent()
        trainer = AdaptiveTrainer(agent, hands_per_session=10)

        losses = []
        trainer.train(
            num_episodes=3,
            batch_size=10,
            callback=lambda d: losses.append(d["loss"]) if d["type"] == "batch_update" else None,
        )

        assert len(losses) > 0
        for loss in losses:
            assert math.isfinite(loss), f"Loss should be finite, got {loss}"

    def test_collect_episode_returns_session_data(self):
        """collect_episode should return a list of (chains, rewards) tuples."""
        agent = ExtendedAdaptiveAgent()
        trainer = AdaptiveTrainer(agent, hands_per_session=5)
        agent.set_train_mode(True)

        session_data = trainer.collect_episode()
        assert isinstance(session_data, list)
        assert len(session_data) == 5

        for chains, rewards in session_data:
            assert len(chains) == 2
            assert len(rewards) == 2
            assert rewards[0] + rewards[1] == pytest.approx(0.0)

    def test_trainer_uses_agent(self):
        """The trainer should reference our ExtendedAdaptiveAgent instance."""
        agent = ExtendedAdaptiveAgent()
        trainer = AdaptiveTrainer(agent)
        assert trainer.agent is agent


# =============================================================================
# 8. debug_episode() returns expected format
# =============================================================================

class TestDebugEpisode:

    def test_debug_episode_structure(self):
        """debug_episode should return a dict with trace and session analytics."""
        agent = ExtendedAdaptiveAgent()
        trainer = AdaptiveTrainer(agent, hands_per_session=10)
        result = trainer.debug_episode()

        assert "trace" in result
        assert "final_rewards" in result
        assert "session_analytics" in result
        assert "eval_type" in result
        assert result["eval_type"] == "value"

    def test_debug_episode_has_steps(self):
        agent = ExtendedAdaptiveAgent()
        trainer = AdaptiveTrainer(agent, hands_per_session=10)
        result = trainer.debug_episode()
        assert len(result["trace"]) > 0

    def test_debug_episode_step_fields(self):
        """Each step in the trace should contain expected keys."""
        agent = ExtendedAdaptiveAgent()
        trainer = AdaptiveTrainer(agent, hands_per_session=10)
        result = trainer.debug_episode()

        for step in result["trace"]:
            assert "player_id" in step
            assert "observation" in step
            assert "opponent_stats" in step
            assert "evaluations" in step
            assert "selected_action" in step
            assert "true_value" in step
            assert "prediction_error" in step

    def test_debug_episode_rewards_zero_sum(self):
        agent = ExtendedAdaptiveAgent()
        trainer = AdaptiveTrainer(agent, hands_per_session=10)
        result = trainer.debug_episode()
        rewards = result["final_rewards"]
        assert rewards[0] + rewards[1] == pytest.approx(0.0)


# =============================================================================
# 9. update_params() works
# =============================================================================

class TestUpdateParams:

    def test_update_lr(self):
        agent = ExtendedAdaptiveAgent()
        trainer = AdaptiveTrainer(agent, learning_rate=1e-3)
        trainer.update_params({"lr": 5e-4})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == 5e-4

    def test_update_hands_per_session(self):
        agent = ExtendedAdaptiveAgent()
        trainer = AdaptiveTrainer(agent, hands_per_session=30)
        trainer.update_params({"hands_per_session": 50})
        assert trainer.hands_per_session == 50


# =============================================================================
# 10. Registry integration (check id="extended_adaptive")
# =============================================================================

class TestRegistryIntegration:

    def test_registered_in_registry(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("extended_adaptive")
        assert metadata is not None

    def test_registry_id(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("extended_adaptive")
        assert metadata.id == "extended_adaptive"

    def test_registry_display_name(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("extended_adaptive")
        assert metadata.display_name == "Extended Adaptive AI"

    def test_registry_is_trainable(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("extended_adaptive")
        assert metadata.is_trainable is True

    def test_registry_requires_model_path(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("extended_adaptive")
        assert metadata.requires_model_path is True

    def test_registry_category(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("extended_adaptive")
        assert metadata.category == "rl"

    def test_registry_trainer_class(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("extended_adaptive")
        assert metadata.trainer_class is AdaptiveTrainer

    def test_registry_create(self):
        from src.agents.registry import registry
        agent = registry.create("extended_adaptive")
        assert isinstance(agent, ExtendedAdaptiveAgent)
        assert isinstance(agent, AdaptiveValueAgent)

    def test_appears_in_trainable_agents(self):
        from src.agents.registry import registry
        trainable = registry.get_trainable_agents()
        ids = [a.id for a in trainable]
        assert "extended_adaptive" in ids


# =============================================================================
# 11. Shares no state with parent class instances (independent objects)
# =============================================================================

class TestInstanceIndependence:

    def test_separate_model_instances(self):
        """Extended and parent agents should have independent model instances."""
        parent = AdaptiveValueAgent()
        child = ExtendedAdaptiveAgent()
        assert parent.model is not child.model

    def test_weight_modification_is_independent(self):
        """Modifying one agent model should not affect the other."""
        parent = AdaptiveValueAgent()
        child = ExtendedAdaptiveAgent()

        x = torch.randn(1, 19)
        with torch.no_grad():
            original_parent_output = parent.model(x).clone()

        with torch.no_grad():
            for p in child.model.parameters():
                p.fill_(99.0)

        with torch.no_grad():
            after_output = parent.model(x)
        torch.testing.assert_close(original_parent_output, after_output)

    def test_train_mode_independent(self):
        """Setting train mode on one should not affect the other."""
        parent = AdaptiveValueAgent()
        child = ExtendedAdaptiveAgent()

        child.set_train_mode(True)
        assert child.train_mode is True
        assert parent.train_mode is False

    def test_temperature_independent(self):
        """Temperature should be independent between instances."""
        parent = AdaptiveValueAgent(temperature=0.5)
        child = ExtendedAdaptiveAgent(temperature=2.0)
        assert parent.temperature == 0.5
        assert child.temperature == 2.0

    def test_multiple_extended_agents_independent(self):
        """Two ExtendedAdaptiveAgent instances should be independent."""
        agent1 = ExtendedAdaptiveAgent()
        agent2 = ExtendedAdaptiveAgent()
        assert agent1.model is not agent2.model

        with torch.no_grad():
            for p in agent1.model.parameters():
                p.fill_(0.0)

        x = torch.randn(1, 19)
        with torch.no_grad():
            out2 = agent2.model(x)
