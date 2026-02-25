"""
Adaptive-History Combo Agent -- Unit & Integration Tests.

Tests cover:
  1. Agent class hierarchy and architecture
  2. Encoding shape and feature layout
  3. Opponent stats encoding (positions 15-18)
  4. Action history encoding (positions 19-34)
  5. Combined features carry-forward in get_action_evaluations()
  6. Full game play
  7. Save/load model roundtrip
  8. Trainer episode collection and loss computation
  9. Training loop completion
 10. Debug episode format
 11. Registry integration
 12. Edge cases

Run with: python -m pytest tests/unit/test_adaptive_history.py -v
"""

import pytest
import torch
import tempfile
import os

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.engine.poker_session import PokerSession, OpponentStats
from src.agents.base import BaseAgent
from src.agents.value_based import ValueBasedAgent
from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.adaptive_history import AdaptiveHistoryAgent


# =============================================================================
# 1. Agent class hierarchy and architecture
# =============================================================================

class TestAgentHierarchy:

    def test_inherits_adaptive_value_agent(self):
        assert issubclass(AdaptiveHistoryAgent, AdaptiveValueAgent)

    def test_inherits_value_based_agent(self):
        assert issubclass(AdaptiveHistoryAgent, ValueBasedAgent)

    def test_inherits_base_agent(self):
        assert issubclass(AdaptiveHistoryAgent, BaseAgent)

    def test_input_size_is_35(self):
        agent = AdaptiveHistoryAgent()
        assert agent.input_size == 35

    def test_model_first_layer_accepts_35(self):
        agent = AdaptiveHistoryAgent()
        first_layer = agent.model.net[0]
        assert first_layer.in_features == 35

    def test_model_hidden_size_is_128(self):
        """Wider network: hidden_size doubled from 64 to 128."""
        agent = AdaptiveHistoryAgent()
        first_layer = agent.model.net[0]
        assert first_layer.out_features == 128

    def test_model_output_is_1(self):
        agent = AdaptiveHistoryAgent()
        last_layer = agent.model.net[-1]
        assert last_layer.out_features == 1


# =============================================================================
# 2. Encoding shape and feature layout
# =============================================================================

class TestEncodingShape:

    def test_encoding_shape_is_1x35(self):
        agent = AdaptiveHistoryAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 35)

    def test_encoding_with_stats_and_history(self):
        agent = AdaptiveHistoryAgent()
        stats = OpponentStats()
        stats.record_action("FOLD", was_facing_raise=False)
        stats.record_hand_complete()

        obs = Observation(
            player_hand='K', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            opponent_stats=stats,
            action_history=((0, "CALL"), (1, "RAISE")),
        )
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 35)

    def test_encoding_dtype_is_float(self):
        agent = AdaptiveHistoryAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.dtype == torch.float32


# =============================================================================
# 3. Opponent stats encoding (positions 15-18)
# =============================================================================

class TestOpponentStatsEncoding:

    def test_no_stats_gives_default_prior(self):
        """When opponent_stats is None, positions 15-18 should be [0.5, 0.5, 0.5, 0.0]."""
        agent = AdaptiveHistoryAgent()
        obs = Observation(
            player_hand='Q', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            opponent_stats=None,
            action_history=None,
        )
        encoded = agent.encode_observation(obs)
        stats_features = encoded[0, 15:19]
        expected = torch.tensor([0.5, 0.5, 0.5, 0.0])
        assert torch.allclose(stats_features, expected)

    def test_stats_correctly_encoded(self):
        """Opponent stats should appear at positions 15-18."""
        agent = AdaptiveHistoryAgent()
        stats = OpponentStats()
        # Record some actions to get non-default stats
        for _ in range(10):
            stats.record_action("FOLD", was_facing_raise=True)
        for _ in range(5):
            stats.record_action("RAISE", was_facing_raise=False)
        for _ in range(5):
            stats.record_action("CALL", was_facing_raise=False)
        stats.hands_observed = 20

        obs = Observation(
            player_hand='Q', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            opponent_stats=stats,
            action_history=None,
        )
        encoded = agent.encode_observation(obs)
        stats_features = encoded[0, 15:19].tolist()

        expected = stats.to_feature_vector()
        for i in range(4):
            assert stats_features[i] == pytest.approx(expected[i], abs=1e-6)

    def test_different_stats_produce_different_encodings(self):
        agent = AdaptiveHistoryAgent()

        stats_passive = OpponentStats()
        for _ in range(10):
            stats_passive.record_action("FOLD", was_facing_raise=False)

        stats_aggressive = OpponentStats()
        for _ in range(10):
            stats_aggressive.record_action("RAISE", was_facing_raise=False)

        base_obs = Observation(
            player_hand='Q', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            action_history=None,
        )

        from dataclasses import replace
        obs_passive = replace(base_obs, opponent_stats=stats_passive)
        obs_aggressive = replace(base_obs, opponent_stats=stats_aggressive)

        enc_passive = agent.encode_observation(obs_passive)
        enc_aggressive = agent.encode_observation(obs_aggressive)

        # Stats features should differ
        assert not torch.allclose(enc_passive[0, 15:19], enc_aggressive[0, 15:19])
        # Base features should be the same
        assert torch.allclose(enc_passive[0, :15], enc_aggressive[0, :15])


# =============================================================================
# 4. Action history encoding (positions 19-34)
# =============================================================================

class TestActionHistoryEncoding:

    def test_no_history_gives_zero_history_features(self):
        """When action_history is empty, positions 19-34 should all be zero."""
        agent = AdaptiveHistoryAgent()
        obs = Observation(
            player_hand='K', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            action_history=(),
        )
        encoded = agent.encode_observation(obs)
        history_features = encoded[0, 19:]
        assert history_features.shape == (16,)
        assert torch.all(history_features == 0.0)

    def test_none_history_gives_zero_features(self):
        """When action_history is None, history features are zero."""
        agent = AdaptiveHistoryAgent()
        obs = Observation(
            player_hand='K', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            action_history=None,
        )
        encoded = agent.encode_observation(obs)
        history_features = encoded[0, 19:]
        assert torch.all(history_features == 0.0)

    def test_different_histories_produce_different_features(self):
        agent = AdaptiveHistoryAgent()

        obs_a = Observation(
            player_hand='Q', board=None, pot=[3, 3],
            current_player=0, current_round=1,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            raises_this_round=0,
            action_history=((0, "RAISE"), (1, "CALL")),
        )

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

        # History features (19:35) should differ
        assert not torch.allclose(enc_a[0, 19:], enc_b[0, 19:])

    def test_viewer_perspective_changes_history_encoding(self):
        """Same history encoded from P0's vs P1's perspective should differ."""
        agent = AdaptiveHistoryAgent()
        history = ((0, "RAISE"), (1, "CALL"))

        enc_p0 = agent._encode_action_history(history, viewer_id=0)
        enc_p1 = agent._encode_action_history(history, viewer_id=1)

        assert not torch.allclose(enc_p0, enc_p1)


# =============================================================================
# 5. Round splitting (inherited from HistoryValueAgent logic)
# =============================================================================

class TestRoundSplitting:

    def test_empty_history(self):
        agent = AdaptiveHistoryAgent()
        rounds = agent._split_into_rounds(())
        assert len(rounds) == 2
        assert rounds[0] == []
        assert rounds[1] == []

    def test_single_round_actions(self):
        agent = AdaptiveHistoryAgent()
        history = ((0, "CALL"), (1, "CALL"))
        rounds = agent._split_into_rounds(history)
        assert len(rounds[0]) == 2
        assert len(rounds[1]) == 0

    def test_two_rounds_of_actions(self):
        agent = AdaptiveHistoryAgent()
        history = (
            (0, "CALL"), (1, "CALL"),   # Round 0
            (0, "RAISE"), (1, "CALL"),  # Round 1
        )
        rounds = agent._split_into_rounds(history)
        assert len(rounds[0]) == 2
        assert len(rounds[1]) == 2

    def test_fold_terminates_splitting(self):
        agent = AdaptiveHistoryAgent()
        history = ((0, "FOLD"),)
        rounds = agent._split_into_rounds(history)
        assert len(rounds[0]) == 1
        assert rounds[0][0] == (0, "FOLD")


# =============================================================================
# 6. Combined carry-forward in get_action_evaluations()
# =============================================================================

class TestCarryForward:

    def test_evaluations_have_35d_encoded_states(self):
        agent = AdaptiveHistoryAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        evals = agent.get_action_evaluations(obs)
        assert len(evals) == len(obs.legal_actions)
        for e in evals:
            assert e["encoded"].shape == (1, 35)

    def test_evaluations_have_non_zero_history_features(self):
        """Simulated post-states should have non-trivial history features."""
        agent = AdaptiveHistoryAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        evals = agent.get_action_evaluations(obs)

        for e in evals:
            history_part = e["encoded"][0, 19:]
            assert not torch.all(history_part == 0.0), \
                f"History features should be non-zero after simulating {e['action'].name}"

    def test_evaluations_carry_opponent_stats(self):
        """When opponent_stats is provided, it should be carried into simulated states."""
        agent = AdaptiveHistoryAgent()
        stats = OpponentStats()
        for _ in range(10):
            stats.record_action("RAISE", was_facing_raise=False)
        stats.hands_observed = 5

        obs = Observation(
            player_hand='K', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            opponent_stats=stats,
            action_history=(),
        )
        evals = agent.get_action_evaluations(obs)

        for e in evals:
            stats_part = e["encoded"][0, 15:19]
            # Should NOT be default prior since we provided real stats
            default_prior = torch.tensor([0.5, 0.5, 0.5, 0.0])
            assert not torch.allclose(stats_part, default_prior), \
                f"Stats should be non-default for {e['action'].name}"

    def test_evaluation_keys(self):
        agent = AdaptiveHistoryAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        evals = agent.get_action_evaluations(obs)
        for e in evals:
            assert "action" in e
            assert "value" in e
            assert "is_terminal" in e
            assert "encoded" in e


# =============================================================================
# 7. Full game play
# =============================================================================

class TestGamePlay:

    def test_agent_plays_complete_game(self):
        agent = AdaptiveHistoryAgent()
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
        agent = AdaptiveHistoryAgent()
        for _ in range(10):
            game = LeducGame()
            game.reset()
            while not game.is_finished:
                obs = game.get_observation()
                action = agent.select_action(obs)
                game.step(action)
            assert game.is_finished

    def test_agent_plays_session_with_stats(self):
        """Agent should work within a PokerSession providing opponent stats."""
        agent = AdaptiveHistoryAgent()
        session = PokerSession()
        for _ in range(5):
            session.new_hand()
            while not session.is_finished:
                obs = session.get_observation()
                action = agent.select_action(obs)
                session.step(action)


# =============================================================================
# 8. Save/load roundtrip
# =============================================================================

class TestSaveLoad:

    def test_save_load_preserves_weights(self):
        agent = AdaptiveHistoryAgent()
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
            agent2 = AdaptiveHistoryAgent()
            agent2.load_model(path)

            with torch.no_grad():
                val_after = agent2.model(encoded).item()

            assert val_before == pytest.approx(val_after, abs=1e-6)
        finally:
            os.unlink(path)

    def test_save_load_architecture_matches(self):
        agent = AdaptiveHistoryAgent()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save_model(path)
            agent2 = AdaptiveHistoryAgent()
            agent2.load_model(path)

            for (n1, p1), (n2, p2) in zip(
                agent.model.named_parameters(),
                agent2.model.named_parameters()
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(path)


# =============================================================================
# 9. Trainer episode collection and loss
# =============================================================================

class TestAdaptiveHistoryTrainer:

    def test_collect_episode_returns_session_data(self):
        from src.training.adaptive_history_trainer import AdaptiveHistoryTrainer
        agent = AdaptiveHistoryAgent()
        trainer = AdaptiveHistoryTrainer(agent, hands_per_session=5)
        session_data = trainer.collect_episode()

        assert isinstance(session_data, list)
        assert len(session_data) == 5  # one per hand
        for chains, rewards in session_data:
            assert len(chains) == 2  # two players
            assert len(rewards) == 2

    def test_collected_encodings_are_35d(self):
        from src.training.adaptive_history_trainer import AdaptiveHistoryTrainer
        agent = AdaptiveHistoryAgent()
        trainer = AdaptiveHistoryTrainer(agent, hands_per_session=3)
        session_data = trainer.collect_episode()

        for chains, _ in session_data:
            for player_chain in chains:
                for encoded in player_chain:
                    assert encoded.shape == (1, 35)

    def test_update_model_produces_finite_loss(self):
        from src.training.adaptive_history_trainer import AdaptiveHistoryTrainer
        agent = AdaptiveHistoryAgent()
        agent.set_train_mode(True)
        trainer = AdaptiveHistoryTrainer(agent, hands_per_session=5)

        batch_data = []
        session_data = trainer.collect_episode()
        batch_data.extend(session_data)

        loss = trainer.update_model(batch_data)
        assert isinstance(loss, float)
        assert loss >= 0
        assert not (loss != loss)  # Not NaN

    def test_training_loop_completes(self):
        """Short training run should complete without error."""
        from src.training.adaptive_history_trainer import AdaptiveHistoryTrainer
        agent = AdaptiveHistoryAgent()
        trainer = AdaptiveHistoryTrainer(agent, learning_rate=1e-3, hands_per_session=3)

        events = []
        trainer.train(
            num_episodes=5,
            batch_size=10,
            callback=lambda d: events.append(d),
        )

        batch_events = [e for e in events if e["type"] == "batch_update"]
        assert len(batch_events) >= 1
        for e in batch_events:
            assert "loss" in e
            assert e["loss"] >= 0

    def test_update_params_changes_lr(self):
        from src.training.adaptive_history_trainer import AdaptiveHistoryTrainer
        agent = AdaptiveHistoryAgent()
        trainer = AdaptiveHistoryTrainer(agent, learning_rate=1e-4)

        trainer.update_params({"lr": 5e-3})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == pytest.approx(5e-3)

    def test_update_params_changes_hands_per_session(self):
        from src.training.adaptive_history_trainer import AdaptiveHistoryTrainer
        agent = AdaptiveHistoryAgent()
        trainer = AdaptiveHistoryTrainer(agent, hands_per_session=30)

        trainer.update_params({"hands_per_session": 50})
        assert trainer.hands_per_session == 50


# =============================================================================
# 10. Debug episode format
# =============================================================================

class TestDebugEpisode:

    def test_debug_episode_returns_expected_format(self):
        from src.training.adaptive_history_trainer import AdaptiveHistoryTrainer
        agent = AdaptiveHistoryAgent()
        trainer = AdaptiveHistoryTrainer(agent, hands_per_session=6)
        result = trainer.debug_episode()

        assert "trace" in result
        assert "final_rewards" in result
        assert "eval_type" in result
        assert "session_analytics" in result
        assert result["eval_type"] == "value"
        assert len(result["final_rewards"]) == 2

    def test_debug_episode_trace_has_step_info(self):
        from src.training.adaptive_history_trainer import AdaptiveHistoryTrainer
        agent = AdaptiveHistoryAgent()
        trainer = AdaptiveHistoryTrainer(agent, hands_per_session=6)
        result = trainer.debug_episode()

        assert len(result["trace"]) > 0
        step = result["trace"][0]
        assert "player_id" in step
        assert "observation" in step
        assert "evaluations" in step
        assert "selected_action" in step
        assert "true_value" in step
        assert "prediction_error" in step

    def test_debug_episode_has_opponent_stats(self):
        """Debug trace should include opponent_stats after warmup."""
        from src.training.adaptive_history_trainer import AdaptiveHistoryTrainer
        agent = AdaptiveHistoryAgent()
        trainer = AdaptiveHistoryTrainer(agent, hands_per_session=6)
        result = trainer.debug_episode()

        # After warmup hands, stats should be populated
        for step in result["trace"]:
            assert "opponent_stats" in step

    def test_debug_episode_has_action_history(self):
        """Debug trace should include action_history in observation."""
        from src.training.adaptive_history_trainer import AdaptiveHistoryTrainer
        agent = AdaptiveHistoryAgent()
        trainer = AdaptiveHistoryTrainer(agent, hands_per_session=6)
        result = trainer.debug_episode()

        for step in result["trace"]:
            assert "action_history" in step["observation"]

    def test_debug_episode_encoded_state_is_35d(self):
        from src.training.adaptive_history_trainer import AdaptiveHistoryTrainer
        agent = AdaptiveHistoryAgent()
        trainer = AdaptiveHistoryTrainer(agent, hands_per_session=6)
        result = trainer.debug_episode()

        for step in result["trace"]:
            assert len(step["encoded_state"]) == 35
            for e in step["evaluations"]:
                assert len(e["encoded_state"]) == 35


# =============================================================================
# 11. Registry integration
# =============================================================================

class TestRegistryIntegration:

    def test_registered_in_registry(self):
        from src.agents.registry import registry
        assert registry.is_registered("adaptive_history")

    def test_metadata_correct(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("adaptive_history")
        assert metadata is not None
        assert metadata.is_trainable is True
        assert metadata.category == "rl"
        assert metadata.requires_model_path is True
        assert metadata.display_name == "Adaptive History AI"

    def test_create_from_registry(self):
        from src.agents.registry import registry
        agent = registry.create("adaptive_history")
        assert isinstance(agent, AdaptiveHistoryAgent)
        assert isinstance(agent, AdaptiveValueAgent)
        assert isinstance(agent, ValueBasedAgent)
        assert agent.input_size == 35

    def test_trainer_class_is_correct(self):
        from src.agents.registry import registry
        from src.training.adaptive_history_trainer import AdaptiveHistoryTrainer
        metadata = registry.get_metadata("adaptive_history")
        assert metadata.trainer_class is AdaptiveHistoryTrainer


# =============================================================================
# 12. Edge cases
# =============================================================================

class TestEdgeCases:

    def test_encoding_with_empty_action_history_and_no_stats(self):
        """Both optional features missing should still produce valid 35d encoding."""
        agent = AdaptiveHistoryAgent()
        obs = Observation(
            player_hand='J', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            opponent_stats=None,
            action_history=None,
        )
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 35)
        assert not torch.any(torch.isnan(encoded))

    def test_encoding_with_finished_game(self):
        agent = AdaptiveHistoryAgent()
        obs = Observation(
            player_hand='K', board='K', pot=[5, 5],
            current_player=0, current_round=1,
            legal_actions=[],
            is_finished=True,
            action_history=((0, "CALL"), (1, "CALL"), (0, "RAISE"), (1, "CALL")),
        )
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 35)
        assert not torch.any(torch.isnan(encoded))

    def test_constants_match_design(self):
        assert AdaptiveHistoryAgent.FEATURES_PER_ROUND == 8
        assert AdaptiveHistoryAgent.NUM_ROUNDS == 2
        assert AdaptiveHistoryAgent.HISTORY_SIZE == 16
        assert AdaptiveHistoryAgent.STATS_SIZE == 4
        assert AdaptiveHistoryAgent.MAX_ACTIONS_PER_ROUND == 6

    def test_total_input_size_formula(self):
        """Total input = 15 base + 4 stats + 16 history = 35."""
        agent = AdaptiveHistoryAgent()
        expected = 15 + AdaptiveHistoryAgent.STATS_SIZE + AdaptiveHistoryAgent.HISTORY_SIZE
        assert agent.input_size == expected
        assert agent.input_size == 35
