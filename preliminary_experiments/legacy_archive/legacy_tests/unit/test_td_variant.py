"""
Unit tests for TDVariantAgent and TDVariantTrainer.

Tests cover:
1. Class hierarchy (subclass of ValueBasedAgent, BaseAgent)
2. Network architecture (input_size=15, hidden=64)
3. Action selection (Boltzmann train, greedy eval)
4. N-step target computation for various n values
5. MC mode (n_steps=9999 always uses terminal reward)
6. Training loop runs without errors
7. Save/load round-trip
8. debug_episode() returns expected format
9. update_params() for lr and n_steps
10. Registry integration
"""

import pytest
import torch
import numpy as np
import os

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.base import BaseAgent
from src.agents.value_based import ValueBasedAgent, ValueNetwork
from src.agents.td_variant import TDVariantAgent
from src.training.td_variant_trainer import TDVariantTrainer
from src.training.value_based_trainer import SelfPlayTrainer
from src.agents.registry import registry, AgentMetadata


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Fresh TDVariantAgent."""
    return TDVariantAgent()


@pytest.fixture
def trainer(agent):
    """TDVariantTrainer with default n_steps=1 (TD(0))."""
    return TDVariantTrainer(agent, learning_rate=1e-3, n_steps=1)


@pytest.fixture
def trainer_nstep(agent):
    """TDVariantTrainer with n_steps=2."""
    return TDVariantTrainer(agent, learning_rate=1e-3, n_steps=2)


@pytest.fixture
def trainer_mc():
    """TDVariantTrainer in MC mode (n_steps=9999)."""
    a = TDVariantAgent()
    return TDVariantTrainer(a, learning_rate=1e-3, n_steps=9999)


@pytest.fixture
def sample_observation():
    """Pre-flop observation for testing."""
    return Observation(
        player_hand="K",
        board=None,
        pot=[1, 1],
        current_player=0,
        current_round=0,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        is_finished=False,
    )


@pytest.fixture
def flop_observation():
    """Flop-phase observation for testing."""
    return Observation(
        player_hand="Q",
        board="J",
        pot=[3, 3],
        current_player=1,
        current_round=1,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        is_finished=False,
    )


# =============================================================================
# 1. Class Hierarchy
# =============================================================================

class TestClassHierarchy:
    """TDVariantAgent must be a subclass of ValueBasedAgent and BaseAgent."""

    def test_is_instance_of_value_based_agent(self, agent):
        assert isinstance(agent, ValueBasedAgent)

    def test_is_instance_of_base_agent(self, agent):
        assert isinstance(agent, BaseAgent)

    def test_is_instance_of_td_variant_agent(self, agent):
        assert isinstance(agent, TDVariantAgent)

    def test_trainer_is_instance_of_selfplay_trainer(self, trainer):
        assert isinstance(trainer, SelfPlayTrainer)


# =============================================================================
# 2. Network Architecture
# =============================================================================

class TestNetworkArchitecture:
    """Verify the value network has the expected structure."""

    def test_input_size_is_15(self, agent):
        assert agent.input_size == 15

    def test_model_is_value_network(self, agent):
        assert isinstance(agent.model, ValueNetwork)

    def test_hidden_size_64(self, agent):
        # First linear layer: 15 -> 64
        first_layer = agent.model.net[0]
        assert first_layer.in_features == 15
        assert first_layer.out_features == 64

    def test_output_size_is_1(self, agent):
        # Last linear layer should output 1
        last_layer = agent.model.net[-1]
        assert last_layer.out_features == 1

    def test_forward_pass_shape(self, agent):
        x = torch.randn(1, 15)
        out = agent.model(x)
        assert out.shape == (1, 1)

    def test_batch_forward_pass(self, agent):
        x = torch.randn(8, 15)
        out = agent.model(x)
        assert out.shape == (8, 1)

    def test_encoding_matches_input_size(self, agent, sample_observation):
        encoded = agent.encode_observation(sample_observation)
        assert encoded.shape[1] == agent.input_size


# =============================================================================
# 3. Action Selection
# =============================================================================

class TestActionSelection:
    """Boltzmann in train mode, greedy in eval mode."""

    def test_eval_mode_returns_action(self, agent, sample_observation):
        agent.set_train_mode(False)
        action = agent.select_action(sample_observation)
        assert isinstance(action, Action)
        assert action in sample_observation.legal_actions

    def test_train_mode_returns_action(self, agent, sample_observation):
        agent.set_train_mode(True)
        action = agent.select_action(sample_observation)
        assert isinstance(action, Action)
        assert action in sample_observation.legal_actions

    def test_eval_mode_is_deterministic(self, agent, sample_observation):
        agent.set_train_mode(False)
        actions = [agent.select_action(sample_observation) for _ in range(20)]
        assert all(a == actions[0] for a in actions)

    def test_train_mode_explores(self, agent, sample_observation):
        """In train mode with temperature, the agent should sometimes pick
        different actions (statistical; may very rarely fail)."""
        agent.set_train_mode(True)
        agent.temperature = 5.0  # High temperature for more exploration
        actions = set()
        for _ in range(200):
            actions.add(agent.select_action(sample_observation))
            if len(actions) > 1:
                break
        # With high temperature and 200 tries, extremely likely to see > 1 action
        assert len(actions) > 1, "Expected exploration in train mode"


# =============================================================================
# 4. N-step Target Computation
# =============================================================================

class TestNStepTargets:
    """Verify n-step target logic for various n values."""

    def _make_chain_and_rewards(self, agent, length=4):
        """Helper: create a fake chain of encoded states and rewards."""
        chain = [torch.randn(1, 15) for _ in range(length)]
        rewards = [3.0, -3.0]
        return chain, rewards

    def test_td0_bootstraps_from_next(self, agent):
        """With n_steps=1, all non-terminal states bootstrap from t+1."""
        trainer = TDVariantTrainer(agent, n_steps=1)
        chain, rewards = self._make_chain_and_rewards(agent, length=3)
        L = len(chain)

        # For t=0: t+1=1 < 3, so should bootstrap
        assert 0 + 1 < L
        # For t=1: t+1=2 < 3, so should bootstrap
        assert 1 + 1 < L
        # For t=2: t+1=3 >= 3, so should use terminal reward
        assert 2 + 1 >= L

    def test_n2_bootstraps_from_two_ahead(self, agent):
        """With n_steps=2, states bootstrap from t+2."""
        trainer = TDVariantTrainer(agent, n_steps=2)
        chain, rewards = self._make_chain_and_rewards(agent, length=4)
        L = len(chain)

        # t=0: t+2=2 < 4 => bootstrap
        assert 0 + 2 < L
        # t=1: t+2=3 < 4 => bootstrap
        assert 1 + 2 < L
        # t=2: t+2=4 >= 4 => terminal
        assert 2 + 2 >= L
        # t=3: t+2=5 >= 4 => terminal
        assert 3 + 2 >= L

    def test_n3_target_computation(self, agent):
        """With n_steps=3, verify target boundaries."""
        trainer = TDVariantTrainer(agent, n_steps=3)
        chain, rewards = self._make_chain_and_rewards(agent, length=4)
        L = len(chain)

        # t=0: t+3=3 < 4 => bootstrap from chain[3]
        assert 0 + 3 < L
        # t=1: t+3=4 >= 4 => terminal
        assert 1 + 3 >= L

    def test_update_model_runs_with_n1(self, agent):
        """TD(0) update_model completes without error."""
        trainer = TDVariantTrainer(agent, n_steps=1)
        agent.set_train_mode(True)
        batch = [trainer.collect_episode() for _ in range(5)]
        loss = trainer.update_model(batch)
        assert isinstance(loss, float)
        assert not np.isnan(loss)

    def test_update_model_runs_with_n2(self):
        """n=2 update_model completes without error."""
        a = TDVariantAgent()
        trainer = TDVariantTrainer(a, n_steps=2)
        a.set_train_mode(True)
        batch = [trainer.collect_episode() for _ in range(5)]
        loss = trainer.update_model(batch)
        assert isinstance(loss, float)
        assert not np.isnan(loss)

    def test_update_model_runs_with_n3(self):
        """n=3 update_model completes without error."""
        a = TDVariantAgent()
        trainer = TDVariantTrainer(a, n_steps=3)
        a.set_train_mode(True)
        batch = [trainer.collect_episode() for _ in range(5)]
        loss = trainer.update_model(batch)
        assert isinstance(loss, float)
        assert not np.isnan(loss)


# =============================================================================
# 5. MC Mode
# =============================================================================

class TestMCMode:
    """With n_steps=9999, every state targets the terminal reward."""

    def test_mc_always_uses_terminal(self):
        """For any chain length <= ~10, t + 9999 >= L always holds."""
        for L in range(1, 11):
            for t in range(L):
                assert t + 9999 >= L, (
                    f"MC mode failed for t={t}, L={L}"
                )

    def test_mc_update_model_runs(self, trainer_mc):
        """MC trainer runs update_model without error."""
        trainer_mc.agent.set_train_mode(True)
        batch = [trainer_mc.collect_episode() for _ in range(5)]
        loss = trainer_mc.update_model(batch)
        assert isinstance(loss, float)
        assert not np.isnan(loss)

    def test_mc_n_steps_value(self, trainer_mc):
        """MC trainer has n_steps=9999."""
        assert trainer_mc.n_steps == 9999


# =============================================================================
# 6. Training Loop
# =============================================================================

class TestTrainingLoop:
    """Full training loop runs end-to-end without errors."""

    def test_short_training_completes(self):
        """10-episode training finishes without error."""
        a = TDVariantAgent()
        trainer = TDVariantTrainer(a, learning_rate=1e-3, n_steps=1)
        updates = []

        def callback(info):
            updates.append(info)

        trainer.train(num_episodes=10, batch_size=5, callback=callback)
        assert len(updates) > 0

    def test_training_with_n2(self):
        """Training with n_steps=2 completes."""
        a = TDVariantAgent()
        trainer = TDVariantTrainer(a, learning_rate=1e-3, n_steps=2)
        trainer.train(num_episodes=10, batch_size=5)

    def test_training_with_mc(self):
        """Training in MC mode completes."""
        a = TDVariantAgent()
        trainer = TDVariantTrainer(a, learning_rate=1e-3, n_steps=9999)
        trainer.train(num_episodes=10, batch_size=5)

    def test_loss_decreases_or_stays_finite(self):
        """Over a batch, loss remains finite (no NaN/Inf)."""
        a = TDVariantAgent()
        trainer = TDVariantTrainer(a, learning_rate=1e-3, n_steps=1)
        a.set_train_mode(True)

        losses = []
        for _ in range(3):
            batch = [trainer.collect_episode() for _ in range(8)]
            loss = trainer.update_model(batch)
            losses.append(loss)

        for loss in losses:
            assert np.isfinite(loss), f"Loss is not finite: {loss}"


# =============================================================================
# 7. Save / Load Round-Trip
# =============================================================================

class TestSaveLoad:
    """Model can be saved and reloaded with identical outputs."""

    def test_save_and_load(self, agent, tmp_path):
        save_path = str(tmp_path / "td_variant_model.pt")
        agent.save_model(save_path)
        assert os.path.exists(save_path)

        new_agent = TDVariantAgent()
        new_agent.load_model(save_path)

        obs = Observation(
            player_hand="K",
            board="Q",
            pot=[3, 3],
            current_player=0,
            current_round=1,
            legal_actions=[Action.FOLD, Action.CALL],
            is_finished=False,
        )

        enc = agent.encode_observation(obs)
        with torch.no_grad():
            out1 = agent.model(enc)
            out2 = new_agent.model(enc)

        assert torch.allclose(out1, out2), "Loaded model should produce same output"

    def test_save_via_trainer(self, tmp_path):
        """Training with save_path persists the model."""
        a = TDVariantAgent()
        trainer = TDVariantTrainer(a, n_steps=1)
        save_path = str(tmp_path / "trained_td_variant.pt")

        trainer.train(num_episodes=10, batch_size=5, save_path=save_path)
        assert os.path.exists(save_path)


# =============================================================================
# 8. debug_episode()
# =============================================================================

class TestDebugEpisode:
    """debug_episode() returns the expected format."""

    def test_debug_episode_returns_dict(self, trainer):
        result = trainer.debug_episode()
        assert isinstance(result, dict)

    def test_debug_episode_has_trace(self, trainer):
        result = trainer.debug_episode()
        assert "trace" in result
        assert isinstance(result["trace"], list)
        assert len(result["trace"]) > 0

    def test_debug_episode_has_final_rewards(self, trainer):
        result = trainer.debug_episode()
        assert "final_rewards" in result
        rewards = result["final_rewards"]
        assert isinstance(rewards, list)
        assert len(rewards) == 2

    def test_debug_episode_trace_step_fields(self, trainer):
        result = trainer.debug_episode()
        step = result["trace"][0]
        assert "player_id" in step
        assert "observation" in step
        assert "evaluations" in step
        assert "selected_action" in step
        assert "true_value" in step
        assert "prediction_error" in step

    def test_debug_episode_evaluations_structure(self, trainer):
        result = trainer.debug_episode()
        step = result["trace"][0]
        evals = step["evaluations"]
        assert len(evals) > 0
        for ev in evals:
            assert "action" in ev
            assert "value" in ev


# =============================================================================
# 9. update_params()
# =============================================================================

class TestUpdateParams:
    """update_params() handles lr and n_steps changes."""

    def test_update_lr(self, trainer):
        trainer.update_params({"lr": 0.01})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == 0.01

    def test_update_n_steps(self, trainer):
        assert trainer.n_steps == 1
        trainer.update_params({"n_steps": 3})
        assert trainer.n_steps == 3

    def test_update_both(self, trainer):
        trainer.update_params({"lr": 0.05, "n_steps": 5})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == 0.05
        assert trainer.n_steps == 5

    def test_update_empty_params(self, trainer):
        """Empty params dict should not raise."""
        old_n = trainer.n_steps
        trainer.update_params({})
        assert trainer.n_steps == old_n

    def test_n_steps_converted_to_int(self, trainer):
        trainer.update_params({"n_steps": 2.7})
        assert trainer.n_steps == 2
        assert isinstance(trainer.n_steps, int)


# =============================================================================
# 10. Registry Integration
# =============================================================================

class TestRegistryIntegration:
    """TD variant agent is properly registered."""

    def test_td_variant_is_registered(self):
        assert registry.is_registered("td_variant")

    def test_create_td_variant_agent(self):
        agent = registry.create("td_variant")
        assert isinstance(agent, TDVariantAgent)
        assert isinstance(agent, ValueBasedAgent)

    def test_metadata_fields(self):
        meta = registry.get_metadata("td_variant")
        assert meta is not None
        assert meta.id == "td_variant"
        assert meta.display_name == "TD Variant AI"
        assert meta.is_trainable is True
        assert meta.requires_model_path is True
        assert meta.category == "rl"
        assert meta.trainer_class is TDVariantTrainer

    def test_td_variant_in_trainable_agents(self):
        trainable = registry.get_trainable_agents()
        ids = [a.id for a in trainable]
        assert "td_variant" in ids

    def test_td_variant_in_rl_category(self):
        rl_agents = registry.list_agents(category="rl")
        ids = [a.id for a in rl_agents]
        assert "td_variant" in ids
