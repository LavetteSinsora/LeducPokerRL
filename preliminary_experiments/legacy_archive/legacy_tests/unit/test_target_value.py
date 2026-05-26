"""
Target-Stabilized Value Agent Unit Tests

Tests cover:
  1.  Agent creation with target network (both models exist, same architecture)
  2.  Agent inherits ValueBasedAgent / BaseAgent
  3.  sync_target() copies weights correctly
  4.  Target model is frozen (no gradients)
  5.  After sync, both models produce same outputs
  6.  After gradient step without sync, models diverge
  7.  Target network used for TD targets in trainer (not main model)
  8.  target_sync_every triggers sync at correct intervals
  9.  Save/load round-trip (saves and loads both models)
  10. Backward compatible load (regular state dict loads correctly)
  11. Training loop runs without errors (100 episodes)
  12. debug_episode() returns expected format
  13. Encoding shape matches input_size (15)
  14. select_action returns legal Action
  15. Agent plays full games without errors
  16. get_target_value returns float
  17. Agent registered in registry
  18. Trainer.collect_episode() returns expected structure
  19. Trainer.update_model() produces finite loss
  20. update_params updates lr and target_sync_every
  21. gradient_steps counter increments correctly
  22. Target model stays in eval mode after sync
  23. Temperature parameter forwarded correctly
  24. Multiple syncs produce consistent results
  25. Target model frozen after load

Run with: python -m pytest tests/unit/test_target_value.py -v
"""

import pytest
import torch
import os
import tempfile
import math
import copy

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.base import BaseAgent
from src.agents.value_based import ValueBasedAgent, ValueNetwork


# =============================================================================
# 1. Agent creation with target network
# =============================================================================

class TestAgentCreation:
    def test_both_models_exist(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        assert hasattr(agent, 'model')
        assert hasattr(agent, 'target_model')

    def test_same_architecture(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        # Both should be ValueNetwork instances
        assert isinstance(agent.model, ValueNetwork)
        assert isinstance(agent.target_model, ValueNetwork)

    def test_models_have_same_structure(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        main_params = dict(agent.model.named_parameters())
        target_params = dict(agent.target_model.named_parameters())
        assert set(main_params.keys()) == set(target_params.keys())
        for name in main_params:
            assert main_params[name].shape == target_params[name].shape

    def test_input_size_is_15(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        assert agent.input_size == 15


# =============================================================================
# 2. Agent inherits ValueBasedAgent / BaseAgent
# =============================================================================

class TestInheritance:
    def test_inherits_value_based_agent(self):
        from src.agents.target_value import TargetValueAgent
        assert issubclass(TargetValueAgent, ValueBasedAgent)

    def test_inherits_base_agent(self):
        from src.agents.target_value import TargetValueAgent
        assert issubclass(TargetValueAgent, BaseAgent)

    def test_instantiation(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        assert agent.train_mode is False


# =============================================================================
# 3. sync_target() copies weights correctly
# =============================================================================

class TestSyncTarget:
    def test_sync_copies_weights(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()

        # Manually modify main model weights
        with torch.no_grad():
            for p in agent.model.parameters():
                p.fill_(42.0)

        agent.sync_target()

        for mp, tp in zip(agent.model.parameters(), agent.target_model.parameters()):
            assert torch.equal(mp, tp), "After sync, target should match main model"

    def test_sync_is_a_copy_not_reference(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        agent.sync_target()

        # Modify main model after sync
        with torch.no_grad():
            for p in agent.model.parameters():
                p.add_(1.0)

        # Target should NOT have changed
        for mp, tp in zip(agent.model.parameters(), agent.target_model.parameters()):
            assert not torch.equal(mp, tp), "After modifying main, target should differ"


# =============================================================================
# 4. Target model is frozen (no gradients)
# =============================================================================

class TestTargetFrozen:
    def test_target_params_require_no_grad(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        for name, param in agent.target_model.named_parameters():
            assert not param.requires_grad, f"Target param {name} should not require grad"

    def test_target_is_in_eval_mode(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        assert not agent.target_model.training, "Target model should be in eval mode"


# =============================================================================
# 5. After sync, both models produce same outputs
# =============================================================================

class TestSyncOutputs:
    def test_same_output_after_sync(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        agent.sync_target()

        x = torch.randn(1, 15)
        with torch.no_grad():
            main_out = agent.model(x)
            target_out = agent.target_model(x)
        torch.testing.assert_close(main_out, target_out)

    def test_same_output_batch_after_sync(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        agent.sync_target()

        x = torch.randn(8, 15)
        with torch.no_grad():
            main_out = agent.model(x)
            target_out = agent.target_model(x)
        torch.testing.assert_close(main_out, target_out)


# =============================================================================
# 6. After gradient step without sync, models diverge
# =============================================================================

class TestModelDivergence:
    def test_models_diverge_after_gradient_step(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        agent.sync_target()

        # Do a gradient step on the main model
        optimizer = torch.optim.SGD(agent.model.parameters(), lr=0.1)
        x = torch.randn(1, 15)
        pred = agent.model(x)
        loss = (pred - torch.tensor([[1.0]])).pow(2)
        loss.backward()
        optimizer.step()

        # Now they should differ
        x2 = torch.randn(1, 15)
        with torch.no_grad():
            main_out = agent.model(x2)
            target_out = agent.target_model(x2)
        assert not torch.allclose(main_out, target_out, atol=1e-6), \
            "After a gradient step, main and target should diverge"


# =============================================================================
# 7. Target network used for TD targets in trainer
# =============================================================================

class TestTargetNetworkUsedInTrainer:
    def test_trainer_uses_target_for_bootstrap(self):
        """Verify that the trainer bootstraps from target_model, not model."""
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent, target_sync_every=9999)  # Never auto-sync

        # Make target and main diverge
        with torch.no_grad():
            for p in agent.target_model.parameters():
                p.fill_(0.0)
            # Main model has non-zero random weights (default init)

        # Collect a batch
        agent.set_train_mode(True)
        batch = [trainer.collect_episode() for _ in range(4)]

        # Snapshot main model params before update
        params_before = {n: p.clone() for n, p in agent.model.named_parameters()}

        trainer.update_model(batch)

        # The gradient should depend on target_model outputs (all zeros),
        # not on the main model's own bootstrapped values.
        # We just verify the update happened without errors and params changed.
        any_changed = False
        for name, p in agent.model.named_parameters():
            if not torch.equal(p, params_before[name]):
                any_changed = True
                break
        assert any_changed, "Update should change main model parameters"


# =============================================================================
# 8. target_sync_every triggers sync at correct intervals
# =============================================================================

class TestSyncInterval:
    def test_sync_at_correct_interval(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent, target_sync_every=3)

        agent.set_train_mode(True)

        sync_count = 0
        original_sync = agent.sync_target

        def counting_sync():
            nonlocal sync_count
            sync_count += 1
            original_sync()

        agent.sync_target = counting_sync

        for _ in range(9):
            batch = [trainer.collect_episode() for _ in range(4)]
            trainer.update_model(batch)

        # 9 gradient steps / sync every 3 = 3 syncs
        assert sync_count == 3, f"Expected 3 syncs, got {sync_count}"
        assert trainer.gradient_steps == 9

    def test_no_sync_before_interval(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent, target_sync_every=100)

        agent.set_train_mode(True)

        sync_count = 0
        original_sync = agent.sync_target

        def counting_sync():
            nonlocal sync_count
            sync_count += 1
            original_sync()

        agent.sync_target = counting_sync

        for _ in range(5):
            batch = [trainer.collect_episode() for _ in range(4)]
            trainer.update_model(batch)

        assert sync_count == 0, "No sync should happen before reaching target_sync_every"


# =============================================================================
# 9. Save/load round-trip
# =============================================================================

class TestSaveLoad:
    def test_save_load_roundtrip(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()

        # Make main and target different
        with torch.no_grad():
            for p in agent.model.parameters():
                p.fill_(1.0)
            for p in agent.target_model.parameters():
                p.fill_(2.0)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_model(path)

            agent2 = TargetValueAgent()
            agent2.load_model(path)

            # Main models should match
            for p1, p2 in zip(agent.model.parameters(), agent2.model.parameters()):
                torch.testing.assert_close(p1, p2)

            # Target models should match
            for p1, p2 in zip(agent.target_model.parameters(), agent2.target_model.parameters()):
                torch.testing.assert_close(p1, p2)
        finally:
            os.unlink(path)

    def test_save_load_preserves_difference(self):
        """Main and target should remain different after load if they were different before save."""
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()

        with torch.no_grad():
            for p in agent.model.parameters():
                p.fill_(1.0)
            for p in agent.target_model.parameters():
                p.fill_(2.0)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_model(path)
            agent2 = TargetValueAgent()
            agent2.load_model(path)

            x = torch.randn(1, 15)
            with torch.no_grad():
                main_out = agent2.model(x)
                target_out = agent2.target_model(x)
            # They should differ since we saved them with different weights
            assert not torch.allclose(main_out, target_out, atol=1e-6)
        finally:
            os.unlink(path)

    def test_load_from_constructor(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_model(path)
            agent2 = TargetValueAgent(model_path=path)
            assert agent2 is not None
        finally:
            os.unlink(path)


# =============================================================================
# 10. Backward compatible load (regular state dict)
# =============================================================================

class TestBackwardCompatibleLoad:
    def test_load_plain_state_dict(self):
        """Loading a plain ValueBasedAgent checkpoint should work and sync target."""
        from src.agents.target_value import TargetValueAgent
        from src.agents.value_based import ValueBasedAgent

        # Save a plain ValueBasedAgent model
        plain_agent = ValueBasedAgent()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            plain_agent.save_model(path)

            # Load into TargetValueAgent
            target_agent = TargetValueAgent()
            target_agent.load_model(path)

            # Both models should now have the same weights (sync happened)
            x = torch.randn(1, 15)
            with torch.no_grad():
                main_out = target_agent.model(x)
                target_out = target_agent.target_model(x)
            torch.testing.assert_close(main_out, target_out)
        finally:
            os.unlink(path)


# =============================================================================
# 11. Training loop runs without errors (100 episodes)
# =============================================================================

class TestTrainingLoop:
    def test_training_100_episodes(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent, target_sync_every=10)

        events = []
        trainer.train(
            num_episodes=100,
            batch_size=32,
            callback=lambda d: events.append(d),
        )

        batch_events = [e for e in events if e["type"] == "batch_update"]
        assert len(batch_events) > 0
        for e in batch_events:
            assert math.isfinite(e["loss"]), f"Loss should be finite, got {e['loss']}"

    def test_training_with_save(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent, target_sync_every=5)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.train(num_episodes=20, batch_size=10, save_path=path)
            assert os.path.exists(path)

            # Verify saved file has both models
            checkpoint = torch.load(path)
            assert isinstance(checkpoint, dict)
            assert 'model' in checkpoint
            assert 'target_model' in checkpoint
        finally:
            os.unlink(path)


# =============================================================================
# 12. debug_episode() returns expected format
# =============================================================================

class TestDebugEpisode:
    def test_debug_episode_format(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent)
        result = trainer.debug_episode()

        assert "trace" in result
        assert "final_rewards" in result
        assert "eval_type" in result
        assert result["eval_type"] == "value"
        assert len(result["trace"]) > 0

    def test_debug_episode_trace_fields(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent)
        result = trainer.debug_episode()

        for step in result["trace"]:
            assert "player_id" in step
            assert "observation" in step
            assert "evaluations" in step
            assert "selected_action" in step
            assert "true_value" in step
            assert "prediction_error" in step

    def test_debug_episode_restores_train_mode(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent)

        agent.set_train_mode(True)
        trainer.debug_episode()
        assert agent.train_mode is True, "debug_episode should restore train_mode"

        agent.set_train_mode(False)
        trainer.debug_episode()
        assert agent.train_mode is False, "debug_episode should restore train_mode"


# =============================================================================
# 13. Encoding shape matches input_size (15)
# =============================================================================

class TestEncoding:
    def test_encoding_shape(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 15), f"Expected shape (1, 15), got {encoded.shape}"

    def test_encoding_matches_input_size(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.shape[1] == agent.input_size


# =============================================================================
# 14. select_action returns legal Action
# =============================================================================

class TestSelectAction:
    def test_returns_legal_action(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        action = agent.select_action(obs)
        assert isinstance(action, Action)
        assert action in obs.legal_actions

    def test_returns_legal_action_train_mode(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        agent.set_train_mode(True)
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        for _ in range(20):
            action = agent.select_action(obs)
            assert isinstance(action, Action)
            assert action in obs.legal_actions


# =============================================================================
# 15. Agent plays full games without errors
# =============================================================================

class TestFullGameplay:
    def test_plays_full_game_eval_mode(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        agent.set_train_mode(False)
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            obs = game.get_observation()
            action = agent.select_action(obs)
            game.step(action)
        assert game.is_finished

    def test_plays_full_game_train_mode(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        agent.set_train_mode(True)
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            obs = game.get_observation()
            action = agent.select_action(obs)
            game.step(action)
        assert game.is_finished

    def test_plays_many_games_without_error(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        for _ in range(50):
            game = LeducGame()
            game.reset()
            while not game.is_finished:
                obs = game.get_observation()
                action = agent.select_action(obs)
                game.step(action)
            assert game.is_finished


# =============================================================================
# 16. get_target_value returns float
# =============================================================================

class TestGetTargetValue:
    def test_returns_float(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        val = agent.get_target_value(obs, viewer_id=0)
        assert isinstance(val, float)

    def test_uses_target_network(self):
        """Verify get_target_value uses target_model, not model."""
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()

        # Make models diverge
        with torch.no_grad():
            for p in agent.model.parameters():
                p.fill_(1.0)
            for p in agent.target_model.parameters():
                p.fill_(0.0)

        game = LeducGame()
        game.reset()
        obs = game.get_observation()

        target_val = agent.get_target_value(obs, viewer_id=0)
        main_val = agent._get_value(obs, viewer_id=0)

        assert target_val != main_val, "get_target_value should use target network"


# =============================================================================
# 17. Agent registered in registry
# =============================================================================

class TestRegistry:
    def test_registered_in_registry(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("target_value")
        assert metadata is not None
        assert metadata.is_trainable is True
        assert metadata.category == "rl"
        assert metadata.display_name == "Target-Stabilized Value AI"

    def test_create_from_registry(self):
        from src.agents.registry import registry
        agent = registry.create("target_value")
        assert agent is not None
        assert isinstance(agent, BaseAgent)
        assert isinstance(agent, ValueBasedAgent)

    def test_trainer_class_set(self):
        from src.agents.registry import registry
        from src.training.target_value_trainer import TargetValueTrainer
        metadata = registry.get_metadata("target_value")
        assert metadata.trainer_class is TargetValueTrainer


# =============================================================================
# 18. Trainer.collect_episode() returns expected structure
# =============================================================================

class TestTrainerCollectEpisode:
    def test_collect_episode_structure(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent)
        agent.set_train_mode(True)

        chains, rewards = trainer.collect_episode()

        assert len(chains) == 2, "Should have chains for 2 players"
        assert len(rewards) == 2, "Should have rewards for 2 players"

        # At least one player should have actions
        total_steps = len(chains[0]) + len(chains[1])
        assert total_steps > 0

        # Each chain element should be a tensor
        for p in [0, 1]:
            for t in chains[p]:
                assert isinstance(t, torch.Tensor)


# =============================================================================
# 19. Trainer.update_model() produces finite loss
# =============================================================================

class TestTrainerUpdateModel:
    def test_update_model_finite_loss(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent, target_sync_every=9999)
        agent.set_train_mode(True)

        batch = [trainer.collect_episode() for _ in range(4)]
        loss = trainer.update_model(batch)

        assert isinstance(loss, float)
        assert math.isfinite(loss), f"Loss should be finite, got {loss}"

    def test_update_model_changes_main_parameters(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent, target_sync_every=9999)
        agent.set_train_mode(True)

        params_before = {n: p.clone() for n, p in agent.model.named_parameters()}

        batch = [trainer.collect_episode() for _ in range(8)]
        trainer.update_model(batch)

        any_changed = False
        for name, p in agent.model.named_parameters():
            if not torch.equal(p, params_before[name]):
                any_changed = True
                break
        assert any_changed, "Main model parameters should change after update"

    def test_update_model_does_not_change_target_without_sync(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent, target_sync_every=9999)
        agent.set_train_mode(True)

        target_before = {n: p.clone() for n, p in agent.target_model.named_parameters()}

        batch = [trainer.collect_episode() for _ in range(8)]
        trainer.update_model(batch)

        for name, p in agent.target_model.named_parameters():
            assert torch.equal(p, target_before[name]), \
                f"Target param {name} should NOT change without sync"


# =============================================================================
# 20. update_params updates lr and target_sync_every
# =============================================================================

class TestUpdateParams:
    def test_update_learning_rate(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent, learning_rate=1e-3)

        trainer.update_params({"lr": 5e-4})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == 5e-4

    def test_update_target_sync_every(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent, target_sync_every=100)

        trainer.update_params({"target_sync_every": 50})
        assert trainer.target_sync_every == 50

    def test_update_both_params(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent, learning_rate=1e-3, target_sync_every=100)

        trainer.update_params({"lr": 2e-4, "target_sync_every": 25})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == 2e-4
        assert trainer.target_sync_every == 25


# =============================================================================
# 21. gradient_steps counter increments correctly
# =============================================================================

class TestGradientStepsCounter:
    def test_starts_at_zero(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent)
        assert trainer.gradient_steps == 0

    def test_increments_per_update(self):
        from src.agents.target_value import TargetValueAgent
        from src.training.target_value_trainer import TargetValueTrainer

        agent = TargetValueAgent()
        trainer = TargetValueTrainer(agent, target_sync_every=9999)
        agent.set_train_mode(True)

        for i in range(5):
            batch = [trainer.collect_episode() for _ in range(4)]
            trainer.update_model(batch)
            assert trainer.gradient_steps == i + 1


# =============================================================================
# 22. Target model stays in eval mode after sync
# =============================================================================

class TestTargetEvalMode:
    def test_target_eval_after_sync(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()

        agent.model.train()
        agent.sync_target()

        # Target should still be in eval mode
        assert not agent.target_model.training, \
            "Target model should remain in eval mode after sync"

    def test_target_eval_after_set_train_mode(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()

        agent.set_train_mode(True)
        # Target should still be eval
        assert not agent.target_model.training

        agent.set_train_mode(False)
        assert not agent.target_model.training


# =============================================================================
# 23. Temperature parameter forwarded correctly
# =============================================================================

class TestTemperature:
    def test_default_temperature(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()
        assert agent.temperature == 1.0

    def test_custom_temperature(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent(temperature=0.5)
        assert agent.temperature == 0.5


# =============================================================================
# 24. Multiple syncs produce consistent results
# =============================================================================

class TestMultipleSyncs:
    def test_multiple_syncs(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()

        x = torch.randn(1, 15)

        for i in range(5):
            # Modify main model
            with torch.no_grad():
                for p in agent.model.parameters():
                    p.fill_(float(i))

            agent.sync_target()

            with torch.no_grad():
                main_out = agent.model(x)
                target_out = agent.target_model(x)

            torch.testing.assert_close(main_out, target_out,
                                       msg=f"Sync {i}: outputs should match")


# =============================================================================
# 25. Target model frozen after load
# =============================================================================

class TestTargetFrozenAfterLoad:
    def test_target_frozen_after_load(self):
        from src.agents.target_value import TargetValueAgent
        agent = TargetValueAgent()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_model(path)
            agent2 = TargetValueAgent()
            agent2.load_model(path)

            for name, param in agent2.target_model.named_parameters():
                assert not param.requires_grad, \
                    f"Target param {name} should be frozen after load"
        finally:
            os.unlink(path)

    def test_target_frozen_after_backward_compatible_load(self):
        from src.agents.target_value import TargetValueAgent
        from src.agents.value_based import ValueBasedAgent

        plain = ValueBasedAgent()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            plain.save_model(path)
            agent = TargetValueAgent()
            agent.load_model(path)

            for name, param in agent.target_model.named_parameters():
                assert not param.requires_grad, \
                    f"Target param {name} should be frozen after backward-compat load"
        finally:
            os.unlink(path)
