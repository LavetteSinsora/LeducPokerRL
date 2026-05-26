"""
Action-History Value Agent — Unit & Integration Tests.

Tests cover:
  1. Agent class hierarchy and contracts
  2. Encoding shape and feature correctness
  3. Action history feature differentiation
  4. Round-splitting logic
  5. Full game play
  6. Save/load model roundtrip
  7. Trainer episode collection and loss computation
  8. Training loop completion
  9. Registry integration

Run with: python -m pytest tests/unit/test_history_value.py -v
"""

import pytest
import torch
import tempfile
import os

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.base import BaseAgent
from src.agents.value_based import ValueBasedAgent
from src.agents.history_value import HistoryValueAgent


# =============================================================================
# 1. Agent class hierarchy
# =============================================================================

class TestHistoryValueAgentHierarchy:

    def test_inherits_value_based_agent(self):
        assert issubclass(HistoryValueAgent, ValueBasedAgent)

    def test_inherits_base_agent(self):
        assert issubclass(HistoryValueAgent, BaseAgent)

    def test_input_size_is_31(self):
        agent = HistoryValueAgent()
        assert agent.input_size == 31

    def test_model_first_layer_accepts_31(self):
        agent = HistoryValueAgent()
        # First linear layer should accept 31 inputs
        first_layer = agent.model.net[0]
        assert first_layer.in_features == 31


# =============================================================================
# 2. Encoding shape
# =============================================================================

class TestEncodingShape:

    def test_encoding_shape_is_1x31(self):
        agent = HistoryValueAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 31)

    def test_encoding_with_history(self):
        agent = HistoryValueAgent()
        game = LeducGame()
        game.reset()
        game.step(Action.CALL)
        game.step(Action.RAISE)
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 31)

    def test_encoding_dtype_is_float(self):
        agent = HistoryValueAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.dtype == torch.float32


# =============================================================================
# 3. History features differentiation
# =============================================================================

class TestHistoryFeatures:

    def test_no_history_gives_zero_history_features(self):
        """When action_history is empty, the last 16 features should all be zero."""
        agent = HistoryValueAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        history_features = encoded[0, 15:]  # Last 16 features
        assert history_features.shape == (16,)
        assert torch.all(history_features == 0.0)

    def test_none_history_gives_zero_features(self):
        """When action_history is None (backward compat), history features are zero."""
        agent = HistoryValueAgent()
        obs = Observation(
            player_hand='K', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            action_history=None,
        )
        encoded = agent.encode_observation(obs)
        history_features = encoded[0, 15:]
        assert torch.all(history_features == 0.0)

    def test_different_histories_produce_different_features(self):
        """
        'Opponent raised then I called' vs 'I called then opponent raised'
        should produce different feature encodings, even with the same pot.
        """
        agent = HistoryValueAgent()

        # Scenario A: P0 raises, P1 calls
        obs_a = Observation(
            player_hand='Q', board=None, pot=[3, 3],
            current_player=0, current_round=1,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            raises_this_round=0,
            action_history=((0, "RAISE"), (1, "CALL")),
        )

        # Scenario B: P0 calls, P1 raises, P0 calls
        obs_b = Observation(
            player_hand='Q', board=None, pot=[3, 3],
            current_player=0, current_round=1,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            raises_this_round=0,
            action_history=((0, "CALL"), (1, "RAISE"), (0, "CALL")),
        )

        enc_a = agent.encode_observation(obs_a, viewer_id=0)
        enc_b = agent.encode_observation(obs_b, viewer_id=0)

        # Base features (0:15) should be the same
        assert torch.allclose(enc_a[0, :15], enc_b[0, :15])
        # History features (15:31) should differ
        assert not torch.allclose(enc_a[0, 15:], enc_b[0, 15:])

    def test_viewer_perspective_changes_encoding(self):
        """Same history encoded from P0's vs P1's perspective should differ."""
        agent = HistoryValueAgent()

        history = ((0, "RAISE"), (1, "CALL"))
        obs = Observation(
            player_hand='Q', board=None, pot=[3, 3],
            current_player=0, current_round=1,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            action_history=history,
        )

        enc_p0 = agent._encode_action_history(history, viewer_id=0)
        enc_p1 = agent._encode_action_history(history, viewer_id=1)

        # Player/opponent action counts should be swapped
        assert not torch.allclose(enc_p0, enc_p1)


# =============================================================================
# 4. Round splitting
# =============================================================================

class TestRoundSplitting:

    def test_empty_history(self):
        agent = HistoryValueAgent()
        rounds = agent._split_into_rounds(())
        assert len(rounds) == 2
        assert rounds[0] == []
        assert rounds[1] == []

    def test_single_round_actions(self):
        """P0 calls, P1 calls -> round 0 ends, no round 1 actions."""
        agent = HistoryValueAgent()
        history = ((0, "CALL"), (1, "CALL"))
        rounds = agent._split_into_rounds(history)
        assert len(rounds[0]) == 2
        assert len(rounds[1]) == 0

    def test_raise_and_call_ends_round(self):
        """P0 raises, P1 calls -> round 0 ends."""
        agent = HistoryValueAgent()
        history = ((0, "RAISE"), (1, "CALL"))
        rounds = agent._split_into_rounds(history)
        assert len(rounds[0]) == 2
        assert len(rounds[1]) == 0

    def test_two_rounds_of_actions(self):
        """P0 calls, P1 calls (round 0 ends), then P0 raises, P1 calls (round 1)."""
        agent = HistoryValueAgent()
        history = (
            (0, "CALL"), (1, "CALL"),   # Round 0
            (0, "RAISE"), (1, "CALL"),  # Round 1
        )
        rounds = agent._split_into_rounds(history)
        assert len(rounds[0]) == 2
        assert len(rounds[1]) == 2

    def test_fold_terminates_splitting(self):
        agent = HistoryValueAgent()
        history = ((0, "FOLD"),)
        rounds = agent._split_into_rounds(history)
        assert len(rounds[0]) == 1
        assert rounds[0][0] == (0, "FOLD")

    def test_complex_round_with_re_raises(self):
        """P0 raises, P1 raises, P0 calls -> round 0 ends."""
        agent = HistoryValueAgent()
        history = ((0, "RAISE"), (1, "RAISE"), (0, "CALL"))
        rounds = agent._split_into_rounds(history)
        assert len(rounds[0]) == 3
        assert len(rounds[1]) == 0


# =============================================================================
# 5. Full game play
# =============================================================================

class TestGamePlay:

    def test_agent_plays_complete_game(self):
        agent = HistoryValueAgent()
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            obs = game.get_observation()
            action = agent.select_action(obs)
            assert isinstance(action, Action)
            assert action in obs.legal_actions
            game.step(action)
        assert game.is_finished

    def test_agent_plays_multiple_games(self):
        agent = HistoryValueAgent()
        for _ in range(10):
            game = LeducGame()
            game.reset()
            while not game.is_finished:
                obs = game.get_observation()
                action = agent.select_action(obs)
                game.step(action)
            assert game.is_finished

    def test_get_action_evaluations_returns_valid(self):
        agent = HistoryValueAgent()
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
            assert e["encoded"].shape == (1, 31)

    def test_get_action_evaluations_extends_history(self):
        """Simulated post-states should have extended action_history."""
        agent = HistoryValueAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        evals = agent.get_action_evaluations(obs)

        # Each evaluation's encoded state should have non-trivial history
        # features (since the simulated action is appended to the history)
        for e in evals:
            history_part = e["encoded"][0, 15:]
            # At least one history feature should be non-zero (the action was added)
            assert not torch.all(history_part == 0.0), \
                f"History features should be non-zero after simulating {e['action'].name}"


# =============================================================================
# 6. Save/load roundtrip
# =============================================================================

class TestSaveLoad:

    def test_save_load_preserves_weights(self):
        agent = HistoryValueAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()

        # Get prediction before save
        encoded = agent.encode_observation(obs)
        with torch.no_grad():
            val_before = agent.model(encoded).item()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_model(path)
            agent2 = HistoryValueAgent()
            agent2.load_model(path)

            with torch.no_grad():
                val_after = agent2.model(encoded).item()

            assert val_before == pytest.approx(val_after, abs=1e-6)
        finally:
            os.unlink(path)

    def test_save_load_model_architecture_matches(self):
        agent = HistoryValueAgent()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_model(path)
            agent2 = HistoryValueAgent()
            agent2.load_model(path)

            # Verify all parameters match
            for (n1, p1), (n2, p2) in zip(
                agent.model.named_parameters(),
                agent2.model.named_parameters()
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(path)


# =============================================================================
# 7. Trainer episode collection and loss
# =============================================================================

class TestHistoryValueTrainer:

    def test_collect_episode_returns_chains_and_rewards(self):
        from src.training.history_value_trainer import HistoryValueTrainer
        agent = HistoryValueAgent()
        trainer = HistoryValueTrainer(agent)
        chains, rewards = trainer.collect_episode()

        assert len(chains) == 2  # Two players
        assert len(rewards) == 2
        # At least one player should have taken actions
        assert len(chains[0]) + len(chains[1]) > 0

    def test_collected_encodings_are_31d(self):
        from src.training.history_value_trainer import HistoryValueTrainer
        agent = HistoryValueAgent()
        trainer = HistoryValueTrainer(agent)
        chains, _ = trainer.collect_episode()
        for player_chain in chains:
            for encoded in player_chain:
                assert encoded.shape == (1, 31)

    def test_update_model_produces_finite_loss(self):
        from src.training.history_value_trainer import HistoryValueTrainer
        agent = HistoryValueAgent()
        agent.set_train_mode(True)
        trainer = HistoryValueTrainer(agent)

        batch = []
        for _ in range(8):
            batch.append(trainer.collect_episode())

        loss = trainer.update_model(batch)
        assert isinstance(loss, float)
        assert loss >= 0
        assert not (loss != loss)  # Not NaN

    def test_training_loop_completes(self):
        """10 episodes of training should complete without error."""
        from src.training.history_value_trainer import HistoryValueTrainer
        agent = HistoryValueAgent()
        trainer = HistoryValueTrainer(agent, learning_rate=1e-3)

        events = []
        trainer.train(
            num_episodes=10,
            batch_size=5,
            callback=lambda d: events.append(d),
        )

        batch_events = [e for e in events if e["type"] == "batch_update"]
        assert len(batch_events) >= 1
        for e in batch_events:
            assert "loss" in e
            assert e["loss"] >= 0

    def test_update_params_changes_lr(self):
        from src.training.history_value_trainer import HistoryValueTrainer
        agent = HistoryValueAgent()
        trainer = HistoryValueTrainer(agent, learning_rate=1e-4)

        trainer.update_params({"lr": 5e-3})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == pytest.approx(5e-3)


# =============================================================================
# 8. Registry integration
# =============================================================================

class TestRegistryIntegration:

    def test_registered_in_registry(self):
        from src.agents.registry import registry
        assert registry.is_registered("history_value")

    def test_metadata_correct(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("history_value")
        assert metadata is not None
        assert metadata.is_trainable is True
        assert metadata.category == "rl"
        assert metadata.requires_model_path is True

    def test_create_from_registry(self):
        from src.agents.registry import registry
        agent = registry.create("history_value")
        assert isinstance(agent, HistoryValueAgent)
        assert isinstance(agent, ValueBasedAgent)
        assert agent.input_size == 31

    def test_trainer_class_is_correct(self):
        from src.agents.registry import registry
        from src.training.history_value_trainer import HistoryValueTrainer
        metadata = registry.get_metadata("history_value")
        assert metadata.trainer_class is HistoryValueTrainer


# =============================================================================
# 9. Scalability design validation
# =============================================================================

class TestScalabilityDesign:

    def test_features_per_round_constant(self):
        """Verify the encoding uses a fixed number of features per round."""
        assert HistoryValueAgent.FEATURES_PER_ROUND == 8

    def test_history_size_matches_formula(self):
        """HISTORY_SIZE should equal FEATURES_PER_ROUND * NUM_ROUNDS."""
        assert HistoryValueAgent.HISTORY_SIZE == (
            HistoryValueAgent.FEATURES_PER_ROUND * HistoryValueAgent.NUM_ROUNDS
        )

    def test_total_input_size_correct(self):
        """Total input = 15 base + HISTORY_SIZE."""
        agent = HistoryValueAgent()
        assert agent.input_size == 15 + HistoryValueAgent.HISTORY_SIZE
