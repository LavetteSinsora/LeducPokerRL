"""
Unit tests for ModulatedValueAgent — gated base/modulation architecture.

Tests cover:
1.  Class hierarchy
2.  Base network is frozen (requires_grad=False for all base params)
3.  Mod and gate networks are trainable
4.  Gate output is in [0, 1] (sigmoid)
5.  With gate=0, output equals base value
6.  Modulation changes output when gate > 0
7.  Encoding produces correct dimensionality
8.  Stats carry-forward in get_action_evaluations
9.  Training loop only updates mod/gate, not base
10. Save/load round-trip (all three networks)
11. debug_episode()
12. Registry integration
13. Backward-compatible loading (base-only weights)
"""

import pytest
import torch
import torch.nn as nn
import copy
import os
import tempfile

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.engine.poker_session import PokerSession, OpponentStats
from src.agents.base import BaseAgent
from src.agents.value_based import ValueBasedAgent, ValueNetwork
from src.preliminary_experiments.promoted_registry.modulated_value import (
    ModulatedValueAgent,
    ModulationNetwork,
    GateNetwork,
)
from src.training.modulated_value_trainer import ModulatedValueTrainer


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    return ModulatedValueAgent()


@pytest.fixture
def obs_with_stats():
    stats = OpponentStats()
    stats.total_actions = 10
    stats.fold_count = 3
    stats.raise_count = 4
    stats.call_count = 3
    stats.hands_observed = 5
    return Observation(
        player_hand='K',
        board=None,
        pot=[1, 1],
        current_player=0,
        current_round=0,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        is_finished=False,
        opponent_stats=stats,
    )


@pytest.fixture
def obs_no_stats():
    return Observation(
        player_hand='Q',
        board='J',
        pot=[3, 3],
        current_player=1,
        current_round=1,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        is_finished=False,
    )


# ── 1. Class hierarchy ───────────────────────────────────────────────────

class TestClassHierarchy:
    def test_inherits_value_based(self, agent):
        assert isinstance(agent, ValueBasedAgent)

    def test_inherits_base_agent(self, agent):
        assert isinstance(agent, BaseAgent)

    def test_has_three_networks(self, agent):
        assert hasattr(agent, 'model')
        assert hasattr(agent, 'mod_net')
        assert hasattr(agent, 'gate_net')


# ── 2. Base network is frozen ────────────────────────────────────────────

class TestBaseFrozen:
    def test_all_base_params_frozen(self, agent):
        for name, p in agent.model.named_parameters():
            assert not p.requires_grad, f"Base param {name} should be frozen"

    def test_base_is_value_network(self, agent):
        assert isinstance(agent.model, ValueNetwork)


# ── 3. Mod and gate are trainable ────────────────────────────────────────

class TestTrainableNetworks:
    def test_mod_net_params_trainable(self, agent):
        for name, p in agent.mod_net.named_parameters():
            assert p.requires_grad, f"Mod param {name} should be trainable"

    def test_gate_net_params_trainable(self, agent):
        for name, p in agent.gate_net.named_parameters():
            assert p.requires_grad, f"Gate param {name} should be trainable"


# ── 4. Gate output in [0, 1] ─────────────────────────────────────────────

class TestGateOutput:
    def test_gate_output_range(self, agent):
        for _ in range(20):
            x = torch.randn(1, 4)
            g = agent.gate_net(x)
            assert g.item() >= 0.0
            assert g.item() <= 1.0

    def test_gate_sigmoid(self):
        gate = GateNetwork()
        x = torch.randn(1, 4)
        out = gate(x)
        assert 0.0 <= out.item() <= 1.0


# ── 5. With gate=0, output equals base value ─────────────────────────────

class TestGateZero:
    def test_gate_zero_equals_base(self, agent, obs_with_stats):
        # Force gate to output 0
        with torch.no_grad():
            # Set gate bias very negative so sigmoid -> 0
            for p in agent.gate_net.parameters():
                p.zero_()
            # Set the final bias very negative
            agent.gate_net.net[-2].bias.fill_(-100.0)

        base_enc = agent.encode_observation(obs_with_stats, viewer_id=0)
        with torch.no_grad():
            v_base = agent.model(base_enc).item()

        v_mod = agent._get_value(obs_with_stats, viewer_id=0)

        assert abs(v_mod - v_base) < 1e-4, (
            f"With gate=0, modulated value {v_mod} should equal base {v_base}"
        )


# ── 6. Modulation changes output when gate > 0 ──────────────────────────

class TestModulation:
    def test_modulation_changes_value(self, agent, obs_with_stats):
        base_enc = agent.encode_observation(obs_with_stats, viewer_id=0)
        with torch.no_grad():
            v_base = agent.model(base_enc).item()

        # Force gate to ~1
        with torch.no_grad():
            for p in agent.gate_net.parameters():
                p.zero_()
            agent.gate_net.net[-2].bias.fill_(100.0)
            # Force modulation to produce a large nonzero delta
            for p in agent.mod_net.parameters():
                p.zero_()
            agent.mod_net.net[-1].bias.fill_(5.0)

        v_mod = agent._get_value(obs_with_stats, viewer_id=0)
        # Delta should be ~5.0, gate ~1.0, so v_mod ~ v_base + 5.0
        assert abs(v_mod - v_base) > 1.0, (
            f"With gate=1 and large delta, value should differ. "
            f"base={v_base}, modulated={v_mod}"
        )


# ── 7. Encoding dimensionality ──────────────────────────────────────────

class TestEncoding:
    def test_base_encoding_15_dim(self, agent, obs_with_stats):
        enc = agent.encode_observation(obs_with_stats, viewer_id=0)
        assert enc.shape == (1, 15)

    def test_stats_encoding_4_dim(self, agent, obs_with_stats):
        stats = agent._encode_stats(obs_with_stats)
        assert stats.shape == (4,)

    def test_stats_default_no_stats(self, agent, obs_no_stats):
        stats = agent._encode_stats(obs_no_stats)
        assert stats.shape == (4,)
        # Default: [0.5, 0.5, 0.5, 0.0]
        expected = torch.tensor([0.5, 0.5, 0.5, 0.0])
        assert torch.allclose(stats, expected)


# ── 8. Stats carry-forward in get_action_evaluations ────────────────────

class TestStatsCarryForward:
    def test_evaluations_use_stats(self, agent, obs_with_stats):
        evals = agent.get_action_evaluations(obs_with_stats)
        assert len(evals) > 0
        for e in evals:
            assert 'action' in e
            assert 'value' in e
            assert 'encoded' in e

    def test_evaluations_have_correct_encoding_shape(self, agent, obs_with_stats):
        evals = agent.get_action_evaluations(obs_with_stats)
        for e in evals:
            assert e['encoded'].shape == (1, 15)


# ── 9. Training only updates mod/gate, not base ─────────────────────────

class TestTrainingFreezes:
    def test_training_preserves_base(self, agent):
        trainer = ModulatedValueTrainer(agent, learning_rate=1e-3)

        # Snapshot base weights
        base_before = {n: p.clone() for n, p in agent.model.named_parameters()}

        # Snapshot mod/gate weights
        mod_before = {n: p.clone() for n, p in agent.mod_net.named_parameters()}
        gate_before = {n: p.clone() for n, p in agent.gate_net.named_parameters()}

        # Run a few training steps
        agent.set_train_mode(True)
        batch_data = []
        for _ in range(5):
            session_data = trainer.collect_episode()
            batch_data.extend(session_data)

        if batch_data:
            trainer.update_model(batch_data)

        # Base weights should be unchanged
        for name, p in agent.model.named_parameters():
            assert torch.equal(p, base_before[name]), (
                f"Base param {name} changed during training!"
            )

        # At least some mod or gate weights should have changed
        mod_changed = any(
            not torch.equal(p, mod_before[n])
            for n, p in agent.mod_net.named_parameters()
        )
        gate_changed = any(
            not torch.equal(p, gate_before[n])
            for n, p in agent.gate_net.named_parameters()
        )
        assert mod_changed or gate_changed, (
            "Neither mod_net nor gate_net updated during training"
        )


# ── 10. Save/load round-trip ─────────────────────────────────────────────

class TestSaveLoad:
    def test_save_load_roundtrip(self, agent, obs_with_stats):
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name

        try:
            # Get value before save
            v_before = agent._get_value(obs_with_stats, viewer_id=0)

            agent.save_model(save_path)

            # Create a new agent and load
            new_agent = ModulatedValueAgent()
            new_agent.load_model(save_path)

            v_after = new_agent._get_value(obs_with_stats, viewer_id=0)
            assert abs(v_before - v_after) < 1e-5, (
                f"Value mismatch after load: {v_before} vs {v_after}"
            )
        finally:
            os.unlink(save_path)

    def test_save_contains_all_networks(self, agent):
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name

        try:
            agent.save_model(save_path)
            data = torch.load(save_path)
            assert isinstance(data, dict)
            assert 'base' in data
            assert 'mod' in data
            assert 'gate' in data
        finally:
            os.unlink(save_path)


# ── 11. debug_episode ────────────────────────────────────────────────────

class TestDebugEpisode:
    def test_debug_episode_returns_dict(self, agent):
        trainer = ModulatedValueTrainer(agent)
        result = trainer.debug_episode()
        assert isinstance(result, dict)
        assert 'trace' in result
        assert 'final_rewards' in result
        assert 'eval_type' in result
        assert result['eval_type'] == 'value'

    def test_debug_episode_has_stats(self, agent):
        trainer = ModulatedValueTrainer(agent)
        result = trainer.debug_episode()
        assert 'session_analytics' in result

    def test_debug_episode_trace_has_steps(self, agent):
        trainer = ModulatedValueTrainer(agent)
        result = trainer.debug_episode()
        assert len(result['trace']) > 0
        step = result['trace'][0]
        assert 'player_id' in step
        assert 'evaluations' in step
        assert 'selected_action' in step


# ── 12. Registry integration ─────────────────────────────────────────────

class TestRegistryIntegration:
    def test_registered_in_registry(self):
        from src.agents.registry import registry
        assert registry.is_registered('modulated_value')

    def test_create_from_registry(self):
        from src.agents.registry import registry
        agent = registry.create('modulated_value')
        assert isinstance(agent, ModulatedValueAgent)

    def test_metadata(self):
        from src.agents.registry import registry
        meta = registry.get_metadata('modulated_value')
        assert meta is not None
        assert meta.display_name == 'Modulated Value AI'
        assert meta.is_trainable is True
        assert meta.requires_model_path is True
        assert meta.category == 'rl'
        assert meta.trainer_class is ModulatedValueTrainer


# ── 13. Backward-compatible loading (base-only weights) ──────────────────

class TestBackwardCompatibleLoading:
    def test_load_base_only_weights(self, agent, obs_with_stats):
        """When given a file with only base weights, load just the base."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name

        try:
            # Save only base network weights (like a ValueBasedAgent would)
            torch.save(agent.model.state_dict(), save_path)

            # Create new agent and load base-only file
            new_agent = ModulatedValueAgent()
            new_agent.load_model(save_path)  # should not crash

            # Base should match
            for (n1, p1), (n2, p2) in zip(
                agent.model.named_parameters(),
                new_agent.model.named_parameters(),
            ):
                assert torch.equal(p1, p2), f"Base param {n1} mismatch"

            # Agent should still function
            v = new_agent._get_value(obs_with_stats, viewer_id=0)
            assert isinstance(v, float)
        finally:
            os.unlink(save_path)


# ── Additional: Full game play ───────────────────────────────────────────

class TestFullGamePlay:
    def test_plays_full_game(self, agent):
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            obs = game.get_observation()
            action = agent.select_action(obs)
            assert action in obs.legal_actions
            game.step(action)
        assert game.is_finished

    def test_plays_multiple_games(self, agent):
        game = LeducGame()
        for _ in range(20):
            game.reset()
            while not game.is_finished:
                obs = game.get_observation()
                action = agent.select_action(obs)
                game.step(action)
            assert game.is_finished

    def test_select_action_in_train_mode(self, agent, obs_with_stats):
        agent.set_train_mode(True)
        action = agent.select_action(obs_with_stats)
        assert isinstance(action, Action)
        assert action in obs_with_stats.legal_actions

    def test_set_train_mode(self, agent):
        agent.set_train_mode(True)
        assert agent.train_mode is True
        # Base should still be in eval mode
        assert not agent.model.training
        # Mod and gate should be in train mode
        assert agent.mod_net.training
        assert agent.gate_net.training

        agent.set_train_mode(False)
        assert agent.train_mode is False
        assert not agent.model.training
        assert not agent.mod_net.training
        assert not agent.gate_net.training
