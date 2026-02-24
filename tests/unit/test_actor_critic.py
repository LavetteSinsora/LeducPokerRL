"""
Actor-Critic Agent Unit Tests

Tests cover:
  1. Agent inherits BaseAgent
  2. Network output shapes: probs = (batch, 3), value = (batch, 1)
  3. select_action returns legal Action
  4. Encoding shape matches input_size (15)
  5. save/load model roundtrip
  6. Trainer.collect_episode() returns expected structure
  7. Trainer.update_model() produces finite loss
  8. Training loop (10 episodes) completes
  9. Agent registered in registry
 10. Agent plays full games without errors

Run with: python -m pytest tests/unit/test_actor_critic.py -v
"""

import pytest
import torch
import os
import tempfile
import math

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.base import BaseAgent


# =============================================================================
# 1. Agent inherits BaseAgent
# =============================================================================

class TestActorCriticAgentContract:
    def test_inherits_base_agent(self):
        from src.agents.actor_critic import ActorCriticAgent
        assert issubclass(ActorCriticAgent, BaseAgent)

    def test_instantiation(self):
        from src.agents.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent()
        assert agent.input_size == 15
        assert agent.train_mode is False


# =============================================================================
# 2. Network output shapes
# =============================================================================

class TestActorCriticNetwork:
    def test_output_shapes_single(self):
        from src.agents.actor_critic import ActorCriticNetwork
        net = ActorCriticNetwork(input_size=15, hidden_size=64)
        x = torch.randn(1, 15)
        probs, value = net(x)
        assert probs.shape == (1, 3), f"Expected probs shape (1, 3), got {probs.shape}"
        assert value.shape == (1, 1), f"Expected value shape (1, 1), got {value.shape}"

    def test_output_shapes_batch(self):
        from src.agents.actor_critic import ActorCriticNetwork
        net = ActorCriticNetwork(input_size=15, hidden_size=64)
        x = torch.randn(8, 15)
        probs, value = net(x)
        assert probs.shape == (8, 3), f"Expected probs shape (8, 3), got {probs.shape}"
        assert value.shape == (8, 1), f"Expected value shape (8, 1), got {value.shape}"

    def test_probs_sum_to_one(self):
        from src.agents.actor_critic import ActorCriticNetwork
        net = ActorCriticNetwork(input_size=15, hidden_size=64)
        x = torch.randn(4, 15)
        probs, _ = net(x)
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_probs_non_negative(self):
        from src.agents.actor_critic import ActorCriticNetwork
        net = ActorCriticNetwork(input_size=15, hidden_size=64)
        x = torch.randn(4, 15)
        probs, _ = net(x)
        assert (probs >= 0).all(), "All probabilities should be non-negative"

    def test_hidden_size_64(self):
        """Verify the network uses hidden size 64 as specified."""
        from src.agents.actor_critic import ActorCriticNetwork
        net = ActorCriticNetwork(input_size=15, hidden_size=64)
        # Check backbone layer sizes
        assert net.backbone[0].in_features == 15
        assert net.backbone[0].out_features == 64
        assert net.backbone[2].in_features == 64
        assert net.backbone[2].out_features == 64
        # Check head input sizes
        assert net.policy_head.in_features == 64
        assert net.value_head.in_features == 64


# =============================================================================
# 3. select_action returns legal Action
# =============================================================================

class TestSelectAction:
    def test_returns_legal_action(self):
        from src.agents.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        action = agent.select_action(obs)
        assert isinstance(action, Action)
        assert action in obs.legal_actions

    def test_returns_legal_action_train_mode(self):
        from src.agents.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent()
        agent.set_train_mode(True)
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        # Run multiple times since train mode samples stochastically
        for _ in range(20):
            action = agent.select_action(obs)
            assert isinstance(action, Action)
            assert action in obs.legal_actions

    def test_returns_legal_action_when_raise_not_available(self):
        """When max raises reached, agent should still return a legal action."""
        from src.agents.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent()
        game = LeducGame()
        game.reset()
        # Force max raises
        game.step(Action.RAISE)  # P0 raises
        game.step(Action.RAISE)  # P1 raises
        obs = game.get_observation()
        assert Action.RAISE not in obs.legal_actions
        action = agent.select_action(obs)
        assert action in obs.legal_actions


# =============================================================================
# 4. Encoding shape matches input_size (15)
# =============================================================================

class TestEncoding:
    def test_encoding_shape(self):
        from src.agents.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 15), f"Expected shape (1, 15), got {encoded.shape}"

    def test_encoding_matches_input_size(self):
        from src.agents.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.shape[1] == agent.input_size


# =============================================================================
# 5. save/load model roundtrip
# =============================================================================

class TestSaveLoad:
    def test_save_load_roundtrip(self):
        from src.agents.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent()

        # Get initial predictions
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        with torch.no_grad():
            probs_before, value_before = agent.model(encoded)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_model(path)

            # Create a new agent and load
            agent2 = ActorCriticAgent()
            agent2.load_model(path)

            with torch.no_grad():
                probs_after, value_after = agent2.model(encoded)

            torch.testing.assert_close(probs_before, probs_after)
            torch.testing.assert_close(value_before, value_after)
        finally:
            os.unlink(path)

    def test_load_from_constructor(self):
        from src.agents.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_model(path)
            agent2 = ActorCriticAgent(model_path=path)
            # Should not raise
            assert agent2 is not None
        finally:
            os.unlink(path)


# =============================================================================
# 6. Trainer.collect_episode() returns expected structure
# =============================================================================

class TestTrainerCollectEpisode:
    def test_collect_episode_structure(self):
        from src.agents.actor_critic import ActorCriticAgent
        from src.training.actor_critic_trainer import ActorCriticTrainer
        agent = ActorCriticAgent()
        trainer = ActorCriticTrainer(agent)

        episode = trainer.collect_episode()

        assert "log_probs" in episode
        assert "values" in episode
        assert "rewards" in episode

        # log_probs and values are per-player lists
        assert len(episode["log_probs"]) == 2
        assert len(episode["values"]) == 2

        # rewards is a 2-element list
        assert len(episode["rewards"]) == 2

        # At least one player should have taken actions
        total_actions = len(episode["log_probs"][0]) + len(episode["log_probs"][1])
        assert total_actions > 0

        # log_probs and values should have the same length per player
        for p in [0, 1]:
            assert len(episode["log_probs"][p]) == len(episode["values"][p])

    def test_collect_episode_values_are_tensors(self):
        from src.agents.actor_critic import ActorCriticAgent
        from src.training.actor_critic_trainer import ActorCriticTrainer
        agent = ActorCriticAgent()
        trainer = ActorCriticTrainer(agent)

        episode = trainer.collect_episode()

        for p in [0, 1]:
            for lp in episode["log_probs"][p]:
                assert isinstance(lp, torch.Tensor)
            for v in episode["values"][p]:
                assert isinstance(v, torch.Tensor)


# =============================================================================
# 7. Trainer.update_model() produces finite loss
# =============================================================================

class TestTrainerUpdateModel:
    def test_update_model_finite_loss(self):
        from src.agents.actor_critic import ActorCriticAgent
        from src.training.actor_critic_trainer import ActorCriticTrainer
        agent = ActorCriticAgent()
        trainer = ActorCriticTrainer(agent)

        batch = [trainer.collect_episode() for _ in range(4)]
        loss = trainer.update_model(batch)

        assert isinstance(loss, float)
        assert math.isfinite(loss), f"Loss should be finite, got {loss}"

    def test_update_model_changes_parameters(self):
        from src.agents.actor_critic import ActorCriticAgent
        from src.training.actor_critic_trainer import ActorCriticTrainer
        agent = ActorCriticAgent()
        trainer = ActorCriticTrainer(agent)

        # Snapshot parameters before
        params_before = {
            name: p.clone() for name, p in agent.model.named_parameters()
        }

        batch = [trainer.collect_episode() for _ in range(8)]
        trainer.update_model(batch)

        # At least some parameters should have changed
        any_changed = False
        for name, p in agent.model.named_parameters():
            if not torch.equal(p, params_before[name]):
                any_changed = True
                break
        assert any_changed, "Parameters should change after update"


# =============================================================================
# 8. Training loop (10 episodes) completes
# =============================================================================

class TestTrainingLoop:
    def test_short_training_completes(self):
        from src.agents.actor_critic import ActorCriticAgent
        from src.training.actor_critic_trainer import ActorCriticTrainer
        agent = ActorCriticAgent()
        trainer = ActorCriticTrainer(agent)

        events = []
        trainer.train(
            num_episodes=10,
            batch_size=5,
            callback=lambda d: events.append(d),
        )

        # Should have produced some batch_update events
        batch_events = [e for e in events if e["type"] == "batch_update"]
        assert len(batch_events) > 0

    def test_training_with_save(self):
        from src.agents.actor_critic import ActorCriticAgent
        from src.training.actor_critic_trainer import ActorCriticTrainer
        agent = ActorCriticAgent()
        trainer = ActorCriticTrainer(agent)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.train(num_episodes=10, batch_size=5, save_path=path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# =============================================================================
# 9. Agent registered in registry
# =============================================================================

class TestRegistry:
    def test_registered_in_registry(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("actor_critic")
        assert metadata is not None
        assert metadata.is_trainable is True
        assert metadata.category == "rl"
        assert metadata.display_name == "Actor-Critic AI"

    def test_create_from_registry(self):
        from src.agents.registry import registry
        agent = registry.create("actor_critic")
        assert agent is not None
        assert isinstance(agent, BaseAgent)

    def test_trainer_class_set(self):
        from src.agents.registry import registry
        from src.training.actor_critic_trainer import ActorCriticTrainer
        metadata = registry.get_metadata("actor_critic")
        assert metadata.trainer_class is ActorCriticTrainer


# =============================================================================
# 10. Agent plays full games without errors
# =============================================================================

class TestFullGameplay:
    def test_plays_full_game_eval_mode(self):
        from src.agents.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent()
        agent.set_train_mode(False)
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            obs = game.get_observation()
            action = agent.select_action(obs)
            game.step(action)
        assert game.is_finished

    def test_plays_full_game_train_mode(self):
        from src.agents.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent()
        agent.set_train_mode(True)
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            obs = game.get_observation()
            action = agent.select_action(obs)
            game.step(action)
        assert game.is_finished

    def test_plays_many_games_without_error(self):
        from src.agents.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent()
        for _ in range(50):
            game = LeducGame()
            game.reset()
            while not game.is_finished:
                obs = game.get_observation()
                action = agent.select_action(obs)
                game.step(action)
            assert game.is_finished

    def test_get_action_evaluations(self):
        from src.agents.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        evals = agent.get_action_evaluations(obs)
        assert len(evals) == len(obs.legal_actions)
        for e in evals:
            assert "action" in e
            assert "probability" in e
            assert "value_estimate" in e
            assert "raw_probability" in e

    def test_debug_episode(self):
        from src.agents.actor_critic import ActorCriticAgent
        from src.training.actor_critic_trainer import ActorCriticTrainer
        agent = ActorCriticAgent()
        trainer = ActorCriticTrainer(agent)
        result = trainer.debug_episode()
        assert "trace" in result
        assert "final_rewards" in result
        assert "eval_type" in result
        assert result["eval_type"] == "actor_critic"
        assert len(result["trace"]) > 0
        # Each step should have value_estimate
        for step in result["trace"]:
            assert "value_estimate" in step


# =============================================================================
# 11. update_params
# =============================================================================

class TestUpdateParams:
    def test_update_learning_rate(self):
        from src.agents.actor_critic import ActorCriticAgent
        from src.training.actor_critic_trainer import ActorCriticTrainer
        agent = ActorCriticAgent()
        trainer = ActorCriticTrainer(agent, learning_rate=1e-3)

        trainer.update_params({"lr": 5e-4})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == 5e-4

    def test_update_value_coeff(self):
        from src.agents.actor_critic import ActorCriticAgent
        from src.training.actor_critic_trainer import ActorCriticTrainer
        agent = ActorCriticAgent()
        trainer = ActorCriticTrainer(agent, value_coeff=0.5)

        trainer.update_params({"value_coeff": 1.0})
        assert trainer.value_coeff == 1.0
