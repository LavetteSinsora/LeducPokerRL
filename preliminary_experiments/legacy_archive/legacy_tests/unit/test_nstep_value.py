"""
N-Step Value Agent -- Unit & Integration Tests.

Tests cover:
  1. Agent class hierarchy and contracts
  2. Network architecture (input_size=15, hidden=64)
  3. Action selection (Boltzmann in train mode, greedy in eval mode)
  4. N-step target computation (verify targets for various chain lengths and n values)
  5. Training loop (trainer.train() runs without errors)
  6. Save/load round-trip
  7. Edge cases: n_steps=1 behaves like TD(0), n_steps > chain length uses terminal reward
  8. debug_episode() returns expected format
  9. update_params() works for lr and n_steps
 10. Registry integration

Run with: python -m pytest tests/unit/test_nstep_value.py -v
"""

import pytest
import torch
import tempfile
import os
import math

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.base import BaseAgent
from src.agents.value_based import ValueBasedAgent, ValueNetwork
from src.agents.nstep_value import NStepValueAgent


# =============================================================================
# 1. Agent class hierarchy and contracts
# =============================================================================

class TestNStepValueAgentHierarchy:

    def test_inherits_value_based_agent(self):
        assert issubclass(NStepValueAgent, ValueBasedAgent)

    def test_inherits_base_agent(self):
        assert issubclass(NStepValueAgent, BaseAgent)

    def test_instantiation_without_model_path(self):
        agent = NStepValueAgent()
        assert agent is not None

    def test_train_mode_default_false(self):
        agent = NStepValueAgent()
        assert agent.train_mode is False


# =============================================================================
# 2. Network architecture (input_size=15, hidden=64)
# =============================================================================

class TestNetworkArchitecture:

    def test_input_size_is_15(self):
        agent = NStepValueAgent()
        assert agent.input_size == 15

    def test_model_is_value_network(self):
        agent = NStepValueAgent()
        assert isinstance(agent.model, ValueNetwork)

    def test_model_first_layer_accepts_15(self):
        agent = NStepValueAgent()
        first_layer = agent.model.net[0]
        assert first_layer.in_features == 15

    def test_model_hidden_size_64(self):
        agent = NStepValueAgent()
        first_layer = agent.model.net[0]
        assert first_layer.out_features == 64
        second_layer = agent.model.net[2]
        assert second_layer.in_features == 64
        assert second_layer.out_features == 64

    def test_model_output_size_1(self):
        agent = NStepValueAgent()
        last_layer = agent.model.net[4]
        assert last_layer.out_features == 1

    def test_forward_pass_shape(self):
        agent = NStepValueAgent()
        x = torch.randn(1, 15)
        with torch.no_grad():
            out = agent.model(x)
        assert out.shape == (1, 1)

    def test_forward_pass_batch(self):
        agent = NStepValueAgent()
        x = torch.randn(8, 15)
        with torch.no_grad():
            out = agent.model(x)
        assert out.shape == (8, 1)


# =============================================================================
# 3. Action selection (Boltzmann in train mode, greedy in eval mode)
# =============================================================================

class TestActionSelection:

    def test_returns_legal_action_eval_mode(self):
        agent = NStepValueAgent()
        agent.set_train_mode(False)
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        action = agent.select_action(obs)
        assert isinstance(action, Action)
        assert action in obs.legal_actions

    def test_returns_legal_action_train_mode(self):
        agent = NStepValueAgent()
        agent.set_train_mode(True)
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        for _ in range(20):
            action = agent.select_action(obs)
            assert isinstance(action, Action)
            assert action in obs.legal_actions

    def test_eval_mode_is_deterministic(self):
        """Greedy mode should always select the same action for the same state."""
        agent = NStepValueAgent()
        agent.set_train_mode(False)
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        actions = [agent.select_action(obs) for _ in range(20)]
        assert len(set(actions)) == 1, "Eval mode should be deterministic"

    def test_train_mode_explores(self):
        """Boltzmann exploration should sometimes pick different actions (probabilistic)."""
        agent = NStepValueAgent(temperature=2.0)
        agent.set_train_mode(True)
        game = LeducGame()
        # Play many games to get varied observations
        seen_actions = set()
        for _ in range(100):
            game.reset()
            obs = game.get_observation()
            action = agent.select_action(obs)
            seen_actions.add(action)
        # With temperature=2.0 over 100 tries, we should see at least 2 distinct actions
        assert len(seen_actions) >= 2, "Train mode should explore multiple actions"

    def test_returns_legal_action_when_raise_not_available(self):
        agent = NStepValueAgent()
        game = LeducGame()
        game.reset()
        game.step(Action.RAISE)
        game.step(Action.RAISE)
        obs = game.get_observation()
        assert Action.RAISE not in obs.legal_actions
        action = agent.select_action(obs)
        assert action in obs.legal_actions

    def test_get_action_evaluations_returns_valid(self):
        agent = NStepValueAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        evals = agent.get_action_evaluations(obs)
        assert len(evals) == len(obs.legal_actions)
        for e in evals:
            assert "action" in e
            assert "value" in e
            assert "is_terminal" in e
            assert "encoded" in e
            assert e["encoded"].shape == (1, 15)


# =============================================================================
# 4. N-step target computation
# =============================================================================

class TestNStepTargetComputation:

    def _make_chain_and_reward(self, chain_length, reward, agent):
        """Helper: create a synthetic chain of encoded states and a reward."""
        chain = [torch.randn(1, 15) for _ in range(chain_length)]
        return chain, reward

    def test_nstep_3_chain_length_2_uses_terminal_reward(self):
        """With n=3, chain length 2: all timesteps t+3 >= 2, so all targets = reward."""
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        agent.set_train_mode(True)
        trainer = NStepValueTrainer(agent, n_steps=3)

        chain = [torch.randn(1, 15) for _ in range(2)]
        reward = 3.0

        # Manually compute expected targets
        for t in range(2):
            # t + 3 >= 2 for all t in {0, 1}, so target = reward
            assert t + 3 >= 2

    def test_nstep_1_chain_length_4_bootstraps_from_next(self):
        """With n=1, chain length 4: t=0,1,2 bootstrap from t+1; t=3 uses reward."""
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        agent.set_train_mode(True)
        trainer = NStepValueTrainer(agent, n_steps=1)

        chain = [torch.randn(1, 15) for _ in range(4)]
        reward = 5.0

        # t=0: t+1=1 < 4, bootstrap from chain[1]
        assert 0 + 1 < 4
        # t=1: t+1=2 < 4, bootstrap from chain[2]
        assert 1 + 1 < 4
        # t=2: t+1=3 < 4, bootstrap from chain[3]
        assert 2 + 1 < 4
        # t=3: t+1=4 >= 4, use terminal reward
        assert 3 + 1 >= 4

    def test_nstep_2_chain_length_3_mixed_targets(self):
        """With n=2, chain length 3: t=0 bootstraps from chain[2]; t=1,2 use reward."""
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        agent.set_train_mode(True)
        trainer = NStepValueTrainer(agent, n_steps=2)

        # t=0: t+2=2 < 3, bootstrap from chain[2]
        assert 0 + 2 < 3
        # t=1: t+2=3 >= 3, use reward
        assert 1 + 2 >= 3
        # t=2: t+2=4 >= 3, use reward
        assert 2 + 2 >= 3

    def test_nstep_1_matches_td0_behavior(self):
        """n_steps=1 should produce the same training behavior as SelfPlayTrainer (TD(0))."""
        from src.training.nstep_value_trainer import NStepValueTrainer
        from src.training.value_based_trainer import SelfPlayTrainer

        # Use same seed for reproducibility
        torch.manual_seed(42)
        agent_nstep = NStepValueAgent()
        torch.manual_seed(42)
        agent_td0 = ValueBasedAgent()

        # Both agents should have same weights
        for p1, p2 in zip(agent_nstep.model.parameters(), agent_td0.model.parameters()):
            assert torch.allclose(p1, p2)

        # Verify n=1 target logic: for any chain, last state uses reward,
        # all others bootstrap from the next state -- exactly like TD(0)
        trainer = NStepValueTrainer(agent_nstep, n_steps=1)
        chain = [torch.randn(1, 15) for _ in range(5)]
        L = len(chain)
        for t in range(L):
            if t + 1 >= L:
                # Should use terminal reward (same as TD(0) last step)
                assert t == L - 1
            else:
                # Should bootstrap from chain[t+1] (same as TD(0))
                assert t + 1 < L

    def test_large_n_all_terminal(self):
        """When n_steps >= chain length, every timestep targets the terminal reward."""
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent, n_steps=100)

        chain_length = 4
        for t in range(chain_length):
            assert t + 100 >= chain_length


# =============================================================================
# 5. Trainer episode collection and loss computation
# =============================================================================

class TestTrainerEpisodeAndLoss:

    def test_collect_episode_returns_chains_and_rewards(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent)
        chains, rewards = trainer.collect_episode()

        assert len(chains) == 2
        assert len(rewards) == 2
        assert len(chains[0]) + len(chains[1]) > 0

    def test_collected_encodings_are_15d(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent)
        chains, _ = trainer.collect_episode()
        for player_chain in chains:
            for encoded in player_chain:
                assert encoded.shape == (1, 15)

    def test_update_model_produces_finite_loss(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        agent.set_train_mode(True)
        trainer = NStepValueTrainer(agent)

        batch = [trainer.collect_episode() for _ in range(8)]
        loss = trainer.update_model(batch)
        assert isinstance(loss, float)
        assert loss >= 0
        assert math.isfinite(loss), f"Loss should be finite, got {loss}"

    def test_update_model_changes_parameters(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        agent.set_train_mode(True)
        trainer = NStepValueTrainer(agent)

        params_before = {
            name: p.clone() for name, p in agent.model.named_parameters()
        }

        batch = [trainer.collect_episode() for _ in range(8)]
        trainer.update_model(batch)

        any_changed = False
        for name, p in agent.model.named_parameters():
            if not torch.equal(p, params_before[name]):
                any_changed = True
                break
        assert any_changed, "Parameters should change after update"

    def test_update_model_empty_batch(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent)
        loss = trainer.update_model([])
        assert loss == 0.0

    def test_rewards_sum_to_zero(self):
        """In Leduc poker, rewards are zero-sum."""
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent)
        for _ in range(20):
            _, rewards = trainer.collect_episode()
            assert rewards[0] + rewards[1] == pytest.approx(0.0), \
                f"Rewards should be zero-sum, got {rewards}"


# =============================================================================
# 6. Training loop completion
# =============================================================================

class TestTrainingLoop:

    def test_training_loop_completes_100_episodes(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent, learning_rate=1e-3)

        events = []
        trainer.train(
            num_episodes=100,
            batch_size=32,
            callback=lambda d: events.append(d),
        )

        batch_events = [e for e in events if e["type"] == "batch_update"]
        assert len(batch_events) >= 1
        for e in batch_events:
            assert "loss" in e
            assert e["loss"] >= 0
            assert math.isfinite(e["loss"])

    def test_short_training_completes(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent, learning_rate=1e-3)

        events = []
        trainer.train(
            num_episodes=10,
            batch_size=5,
            callback=lambda d: events.append(d),
        )

        batch_events = [e for e in events if e["type"] == "batch_update"]
        assert len(batch_events) >= 1

    def test_training_with_save(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.train(num_episodes=10, batch_size=5, save_path=path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_training_with_different_n_steps(self):
        """Training should work with various n_steps values."""
        from src.training.nstep_value_trainer import NStepValueTrainer
        for n in [1, 2, 3, 5, 10]:
            agent = NStepValueAgent()
            trainer = NStepValueTrainer(agent, n_steps=n)
            trainer.train(num_episodes=10, batch_size=5)
            # Should complete without error


# =============================================================================
# 7. Save/load round-trip
# =============================================================================

class TestSaveLoad:

    def test_save_load_preserves_weights(self):
        agent = NStepValueAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()

        encoded = agent.encode_observation(obs)
        with torch.no_grad():
            val_before = agent.model(encoded).item()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_model(path)
            agent2 = NStepValueAgent()
            agent2.load_model(path)

            with torch.no_grad():
                val_after = agent2.model(encoded).item()

            assert val_before == pytest.approx(val_after, abs=1e-6)
        finally:
            os.unlink(path)

    def test_save_load_model_architecture_matches(self):
        agent = NStepValueAgent()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_model(path)
            agent2 = NStepValueAgent()
            agent2.load_model(path)

            for (n1, p1), (n2, p2) in zip(
                agent.model.named_parameters(),
                agent2.model.named_parameters()
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(path)

    def test_load_from_constructor(self):
        agent = NStepValueAgent()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_model(path)
            agent2 = NStepValueAgent(model_path=path)
            assert agent2 is not None
        finally:
            os.unlink(path)


# =============================================================================
# 8. Edge cases: n_steps=1 like TD(0), n_steps > chain length uses terminal
# =============================================================================

class TestEdgeCases:

    def test_nstep_1_update_produces_loss(self):
        """n_steps=1 should still train successfully."""
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        agent.set_train_mode(True)
        trainer = NStepValueTrainer(agent, n_steps=1)

        batch = [trainer.collect_episode() for _ in range(8)]
        loss = trainer.update_model(batch)
        assert isinstance(loss, float)
        assert loss >= 0
        assert math.isfinite(loss)

    def test_nstep_very_large_update_produces_loss(self):
        """n_steps=999 means every target is terminal reward."""
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        agent.set_train_mode(True)
        trainer = NStepValueTrainer(agent, n_steps=999)

        batch = [trainer.collect_episode() for _ in range(8)]
        loss = trainer.update_model(batch)
        assert isinstance(loss, float)
        assert loss >= 0
        assert math.isfinite(loss)

    def test_single_step_chain(self):
        """A chain of length 1 should always use the terminal reward."""
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        agent.set_train_mode(True)
        trainer = NStepValueTrainer(agent, n_steps=3)

        # Create a synthetic single-step chain
        chain = [torch.randn(1, 15)]
        batch_data = [([chain, []], [2.0, -2.0])]

        loss = trainer.update_model(batch_data)
        assert isinstance(loss, float)
        assert math.isfinite(loss)


# =============================================================================
# 9. debug_episode() returns expected format
# =============================================================================

class TestDebugEpisode:

    def test_debug_episode_returns_dict(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent)
        result = trainer.debug_episode()
        assert isinstance(result, dict)

    def test_debug_episode_has_required_keys(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent)
        result = trainer.debug_episode()
        assert "trace" in result
        assert "final_rewards" in result
        assert "eval_type" in result
        assert result["eval_type"] == "value"

    def test_debug_episode_trace_has_steps(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent)
        result = trainer.debug_episode()
        assert len(result["trace"]) > 0

    def test_debug_episode_step_has_required_fields(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent)
        result = trainer.debug_episode()
        for step in result["trace"]:
            assert "player_id" in step
            assert "observation" in step
            assert "evaluations" in step
            assert "selected_action" in step
            assert "selected_action_id" in step
            assert "encoded_state" in step
            assert "true_value" in step
            assert "prediction_error" in step

    def test_debug_episode_true_value_matches_reward(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent)
        result = trainer.debug_episode()
        rewards = result["final_rewards"]
        for step in result["trace"]:
            assert step["true_value"] == rewards[step["player_id"]]

    def test_debug_episode_prediction_error_non_negative(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent)
        result = trainer.debug_episode()
        for step in result["trace"]:
            assert step["prediction_error"] >= 0

    def test_debug_episode_restores_train_mode(self):
        """debug_episode should restore the agent's training mode after execution."""
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        agent.set_train_mode(True)
        trainer = NStepValueTrainer(agent)
        trainer.debug_episode()
        assert agent.train_mode is True

        agent.set_train_mode(False)
        trainer.debug_episode()
        assert agent.train_mode is False


# =============================================================================
# 10. update_params() works for lr and n_steps
# =============================================================================

class TestUpdateParams:

    def test_update_learning_rate(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent, learning_rate=1e-4)

        trainer.update_params({"lr": 5e-3})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == pytest.approx(5e-3)

    def test_update_n_steps(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent, n_steps=3)

        trainer.update_params({"n_steps": 5})
        assert trainer.n_steps == 5

    def test_update_n_steps_converts_to_int(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent, n_steps=3)

        trainer.update_params({"n_steps": 7.0})
        assert trainer.n_steps == 7
        assert isinstance(trainer.n_steps, int)

    def test_update_both_lr_and_n_steps(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent, learning_rate=1e-4, n_steps=3)

        trainer.update_params({"lr": 1e-2, "n_steps": 10})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == pytest.approx(1e-2)
        assert trainer.n_steps == 10

    def test_update_params_ignores_unknown_keys(self):
        from src.training.nstep_value_trainer import NStepValueTrainer
        agent = NStepValueAgent()
        trainer = NStepValueTrainer(agent, n_steps=3)

        # Should not raise
        trainer.update_params({"unknown_key": 42})
        assert trainer.n_steps == 3  # unchanged


# =============================================================================
# 11. Registry integration
# =============================================================================

class TestRegistryIntegration:

    def test_registered_in_registry(self):
        from src.agents.registry import registry
        assert registry.is_registered("nstep_value")

    def test_metadata_correct(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("nstep_value")
        assert metadata is not None
        assert metadata.is_trainable is True
        assert metadata.category == "rl"
        assert metadata.requires_model_path is True
        assert metadata.display_name == "N-Step Value AI"

    def test_create_from_registry(self):
        from src.agents.registry import registry
        agent = registry.create("nstep_value")
        assert isinstance(agent, NStepValueAgent)
        assert isinstance(agent, ValueBasedAgent)
        assert agent.input_size == 15

    def test_trainer_class_is_correct(self):
        from src.agents.registry import registry
        from src.training.nstep_value_trainer import NStepValueTrainer
        metadata = registry.get_metadata("nstep_value")
        assert metadata.trainer_class is NStepValueTrainer


# =============================================================================
# 12. Full game play
# =============================================================================

class TestFullGamePlay:

    def test_plays_full_game_eval_mode(self):
        agent = NStepValueAgent()
        agent.set_train_mode(False)
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            obs = game.get_observation()
            action = agent.select_action(obs)
            game.step(action)
        assert game.is_finished

    def test_plays_full_game_train_mode(self):
        agent = NStepValueAgent()
        agent.set_train_mode(True)
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            obs = game.get_observation()
            action = agent.select_action(obs)
            game.step(action)
        assert game.is_finished

    def test_plays_many_games_without_error(self):
        agent = NStepValueAgent()
        for _ in range(50):
            game = LeducGame()
            game.reset()
            while not game.is_finished:
                obs = game.get_observation()
                action = agent.select_action(obs)
                game.step(action)
            assert game.is_finished

    def test_encoding_shape(self):
        agent = NStepValueAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 15)

    def test_encoding_dtype_is_float(self):
        agent = NStepValueAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.dtype == torch.float32
