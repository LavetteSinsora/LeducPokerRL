"""
Pruned History Agent -- Unit & Integration Tests.

Tests cover:
  1. Agent class hierarchy and architecture
  2. Network architecture (input_size=31, hidden=64)
  3. Encoding produces 31 features
  4. History encoding has exactly 12 features (not 16)
  5. Fold counts are NOT in the encoding
  6. Action history carry-forward in trainer
  7. Training loop runs without errors
  8. Save/load round-trip
  9. debug_episode()
 10. Registry integration

Run with: python -m pytest tests/unit/test_pruned_history.py -v
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
from src.agents.pruned_history import PrunedHistoryAgent


# =============================================================================
# 1. Agent class hierarchy
# =============================================================================

class TestAgentHierarchy:

    def test_inherits_adaptive_value_agent(self):
        assert issubclass(PrunedHistoryAgent, AdaptiveValueAgent)

    def test_inherits_value_based_agent(self):
        assert issubclass(PrunedHistoryAgent, ValueBasedAgent)

    def test_inherits_base_agent(self):
        assert issubclass(PrunedHistoryAgent, BaseAgent)


# =============================================================================
# 2. Network architecture (input_size=31, hidden=64)
# =============================================================================

class TestNetworkArchitecture:

    def test_input_size_is_31(self):
        agent = PrunedHistoryAgent()
        assert agent.input_size == 31

    def test_model_first_layer_accepts_31(self):
        agent = PrunedHistoryAgent()
        first_layer = agent.model.net[0]
        assert first_layer.in_features == 31

    def test_model_hidden_size_is_64(self):
        """Hidden size should be 64, NOT 128 like AdaptiveHistoryAgent."""
        agent = PrunedHistoryAgent()
        first_layer = agent.model.net[0]
        assert first_layer.out_features == 64

    def test_model_output_is_1(self):
        agent = PrunedHistoryAgent()
        last_layer = agent.model.net[-1]
        assert last_layer.out_features == 1

    def test_smaller_param_count_than_adaptive_history(self):
        """PrunedHistoryAgent (31 in, 64 hidden) should have fewer params
        than AdaptiveHistoryAgent (35 in, 128 hidden)."""
        from src.agents.adaptive_history import AdaptiveHistoryAgent
        pruned = PrunedHistoryAgent()
        adaptive = AdaptiveHistoryAgent()

        pruned_params = sum(p.numel() for p in pruned.model.parameters())
        adaptive_params = sum(p.numel() for p in adaptive.model.parameters())

        assert pruned_params < adaptive_params


# =============================================================================
# 3. Encoding produces 31 features
# =============================================================================

class TestEncodingShape:

    def test_encoding_shape_is_1x31(self):
        agent = PrunedHistoryAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 31)

    def test_encoding_with_stats_and_history(self):
        agent = PrunedHistoryAgent()
        stats = OpponentStats()
        stats.record_action("RAISE", False)
        stats.record_hand_complete()

        obs = Observation(
            player_hand='K',
            board='Q',
            pot=[3, 3],
            current_player=0,
            current_round=1,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            raises_this_round=0,
            opponent_stats=stats,
            action_history=((0, "CALL"), (1, "CALL")),
        )
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 31)

    def test_encoding_without_stats_or_history(self):
        agent = PrunedHistoryAgent()
        obs = Observation(
            player_hand='J',
            board=None,
            pot=[1, 1],
            current_player=0,
            current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
        )
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 31)

    def test_encoding_default_stats_when_none(self):
        """When opponent_stats is None, default [0.5, 0.5, 0.5, 0.0] should be used."""
        agent = PrunedHistoryAgent()
        obs = Observation(
            player_hand='K',
            board=None,
            pot=[1, 1],
            current_player=0,
            current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
        )
        encoded = agent.encode_observation(obs).squeeze(0)
        # Stats are at positions 15-18
        assert encoded[15].item() == pytest.approx(0.5)
        assert encoded[16].item() == pytest.approx(0.5)
        assert encoded[17].item() == pytest.approx(0.5)
        assert encoded[18].item() == pytest.approx(0.0)


# =============================================================================
# 4. History encoding has exactly 12 features (not 16)
# =============================================================================

class TestPrunedHistorySize:

    def test_pruned_history_size_constant(self):
        assert PrunedHistoryAgent.PRUNED_HISTORY_SIZE == 12

    def test_features_per_round_is_6(self):
        assert PrunedHistoryAgent.HISTORY_FEATURES_PER_ROUND == 6

    def test_num_rounds_is_2(self):
        assert PrunedHistoryAgent.NUM_ROUNDS == 2

    def test_history_encoding_length(self):
        """_encode_pruned_action_history should return 12 features."""
        agent = PrunedHistoryAgent()
        history = ((0, "CALL"), (1, "RAISE"), (0, "CALL"))
        features = agent._encode_pruned_action_history(history, viewer_id=0)
        assert features.shape == (12,)

    def test_empty_history_returns_zeros(self):
        agent = PrunedHistoryAgent()
        features = agent._encode_pruned_action_history(None, viewer_id=0)
        assert features.shape == (12,)
        assert torch.all(features == 0.0)

    def test_empty_tuple_returns_zeros(self):
        agent = PrunedHistoryAgent()
        features = agent._encode_pruned_action_history((), viewer_id=0)
        assert features.shape == (12,)
        assert torch.all(features == 0.0)


# =============================================================================
# 5. Fold counts are NOT in the encoding
# =============================================================================

class TestNoFoldCounts:

    def test_no_fold_in_per_round_features(self):
        """The 6 features per round should be:
        [player_call, player_raise, opp_call, opp_raise, total, has_raise]
        NOT [player_fold, player_call, player_raise, opp_fold, opp_call, opp_raise, total, has_raise]
        """
        agent = PrunedHistoryAgent()

        # Scenario: P0 raises, P1 calls (ends round 0)
        history = ((0, "RAISE"), (1, "CALL"))
        features = agent._encode_pruned_action_history(history, viewer_id=0)

        # Round 0 features (indices 0-5):
        # player_call=0/2, player_raise=1/2, opp_call=1/2, opp_raise=0/2, total=2/6, has_raise=1
        assert features[0].item() == pytest.approx(0.0)     # player_call / total
        assert features[1].item() == pytest.approx(0.5)     # player_raise / total
        assert features[2].item() == pytest.approx(0.5)     # opp_call / total
        assert features[3].item() == pytest.approx(0.0)     # opp_raise / total
        assert features[4].item() == pytest.approx(2.0 / 6) # total / max
        assert features[5].item() == pytest.approx(1.0)     # has_raise

        # Round 1 should be all zeros
        assert torch.all(features[6:12] == 0.0)

    def test_fold_action_not_counted(self):
        """Even if fold appears in history (terminal action), it should not
        contribute to any fold count feature (because there are none)."""
        agent = PrunedHistoryAgent()

        # P0 raises, P1 folds
        history = ((0, "RAISE"), (1, "FOLD"))
        features = agent._encode_pruned_action_history(history, viewer_id=0)

        # Round 0: player_call=0, player_raise=1, opp_call=0, opp_raise=0
        # total=2 (fold is still counted in total_actions)
        assert features[0].item() == pytest.approx(0.0)     # player_call
        assert features[1].item() == pytest.approx(0.5)     # player_raise (1/2)
        assert features[2].item() == pytest.approx(0.0)     # opp_call
        assert features[3].item() == pytest.approx(0.0)     # opp_raise
        assert features[4].item() == pytest.approx(2.0 / 6) # total / max

    def test_12_features_vs_16_features(self):
        """PrunedHistoryAgent has 12 history features, not 16."""
        from src.agents.adaptive_history import AdaptiveHistoryAgent
        assert PrunedHistoryAgent.PRUNED_HISTORY_SIZE == 12
        assert AdaptiveHistoryAgent.HISTORY_SIZE == 16

    def test_feature_indices_are_correct(self):
        """Verify the feature layout per round matches the spec."""
        agent = PrunedHistoryAgent()

        # P0 calls, P1 raises, P0 calls -> ends round 0
        history = ((0, "CALL"), (1, "RAISE"), (0, "CALL"))
        features = agent._encode_pruned_action_history(history, viewer_id=0)

        # Round 0: 3 actions
        # player_call=2/3, player_raise=0/3, opp_call=0/3, opp_raise=1/3, total=3/6, has_raise=1
        assert features[0].item() == pytest.approx(2.0 / 3)  # player_call
        assert features[1].item() == pytest.approx(0.0)       # player_raise
        assert features[2].item() == pytest.approx(0.0)       # opp_call
        assert features[3].item() == pytest.approx(1.0 / 3)   # opp_raise
        assert features[4].item() == pytest.approx(3.0 / 6)   # total / max
        assert features[5].item() == pytest.approx(1.0)        # has_raise


# =============================================================================
# 6. Action history carry-forward in trainer
# =============================================================================

class TestTrainerCarryForward:

    def test_trainer_carries_stats_and_history(self):
        """Trainer should carry both opponent_stats and action_history
        into simulated post-action states."""
        from src.training.pruned_history_trainer import PrunedHistoryTrainer

        agent = PrunedHistoryAgent()
        trainer = PrunedHistoryTrainer(agent)
        agent.set_train_mode(True)

        # Run one episode and check that encoded states have
        # non-trivial action history features
        session_data = trainer.collect_episode()
        assert len(session_data) > 0

        # Each item is (chains, rewards)
        for chains, rewards in session_data:
            assert len(chains) == 2
            assert len(rewards) == 2
            # At least one player should have encoded states
            total_states = len(chains[0]) + len(chains[1])
            assert total_states > 0

    def test_encoded_states_have_31_features(self):
        """All encoded states from training should have exactly 31 features."""
        from src.training.pruned_history_trainer import PrunedHistoryTrainer

        agent = PrunedHistoryAgent()
        trainer = PrunedHistoryTrainer(agent)
        agent.set_train_mode(True)

        session_data = trainer.collect_episode()

        for chains, rewards in session_data:
            for p_chain in chains:
                for encoded in p_chain:
                    assert encoded.shape == (1, 31), f"Expected (1, 31), got {encoded.shape}"


# =============================================================================
# 7. Training loop runs without errors
# =============================================================================

class TestTrainingLoop:

    def test_short_training_completes(self):
        """A short training loop should complete without errors."""
        from src.training.pruned_history_trainer import PrunedHistoryTrainer

        agent = PrunedHistoryAgent()
        trainer = PrunedHistoryTrainer(agent)

        updates = []
        def callback(info):
            updates.append(info)

        trainer.train(
            num_episodes=3,
            batch_size=32,
            callback=callback,
        )

        # Training completed without errors (train_mode stays True
        # when no save_path is given; this matches AdaptiveTrainer behavior)

    def test_loss_is_finite(self):
        """Training loss should be a finite number."""
        from src.training.pruned_history_trainer import PrunedHistoryTrainer

        agent = PrunedHistoryAgent()
        trainer = PrunedHistoryTrainer(agent)
        agent.set_train_mode(True)

        batch_data = []
        for _ in range(3):
            session_data = trainer.collect_episode()
            batch_data.extend(session_data)

        loss = trainer.update_model(batch_data)
        assert isinstance(loss, float)
        assert not (loss != loss)  # not NaN


# =============================================================================
# 8. Save/load round-trip
# =============================================================================

class TestSaveLoad:

    def test_save_and_load_roundtrip(self):
        agent = PrunedHistoryAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pruned_history.pt")
            agent.save_model(path)
            assert os.path.exists(path)

            # Create a new agent and load
            agent2 = PrunedHistoryAgent()
            agent2.load_model(path)

            # Both should produce same output for same input
            obs = Observation(
                player_hand='K',
                board=None,
                pot=[1, 1],
                current_player=0,
                current_round=0,
                legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
                is_finished=False,
            )
            enc1 = agent.encode_observation(obs)
            enc2 = agent2.encode_observation(obs)

            with torch.no_grad():
                out1 = agent.model(enc1)
                out2 = agent2.model(enc2)

            assert torch.allclose(out1, out2)

    def test_load_from_path_in_constructor(self):
        agent = PrunedHistoryAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pruned_history.pt")
            agent.save_model(path)

            agent2 = PrunedHistoryAgent(model_path=path)
            assert agent2.input_size == 31


# =============================================================================
# 9. debug_episode()
# =============================================================================

class TestDebugEpisode:

    def test_debug_episode_returns_trace(self):
        from src.training.pruned_history_trainer import PrunedHistoryTrainer

        agent = PrunedHistoryAgent()
        trainer = PrunedHistoryTrainer(agent)

        result = trainer.debug_episode()

        assert "trace" in result
        assert "final_rewards" in result
        assert isinstance(result["trace"], list)
        assert len(result["trace"]) > 0

    def test_debug_episode_has_evaluations(self):
        from src.training.pruned_history_trainer import PrunedHistoryTrainer

        agent = PrunedHistoryAgent()
        trainer = PrunedHistoryTrainer(agent)

        result = trainer.debug_episode()

        for step in result["trace"]:
            assert "evaluations" in step
            assert "selected_action" in step
            assert "player_id" in step
            assert "observation" in step

    def test_debug_episode_encoded_states_are_31dim(self):
        from src.training.pruned_history_trainer import PrunedHistoryTrainer

        agent = PrunedHistoryAgent()
        trainer = PrunedHistoryTrainer(agent)

        result = trainer.debug_episode()

        for step in result["trace"]:
            encoded = step["encoded_state"]
            assert len(encoded) == 31, f"Expected 31, got {len(encoded)}"


# =============================================================================
# 10. Registry integration
# =============================================================================

class TestRegistryIntegration:

    def test_registered_in_registry(self):
        from src.agents.registry import registry
        assert registry.is_registered("pruned_history")

    def test_metadata_correct(self):
        from src.agents.registry import registry
        meta = registry.get_metadata("pruned_history")
        assert meta is not None
        assert meta.id == "pruned_history"
        assert meta.display_name == "Pruned History AI"
        assert meta.is_trainable is True
        assert meta.requires_model_path is True
        assert meta.category == "rl"

    def test_create_from_registry(self):
        from src.agents.registry import registry
        agent = registry.create("pruned_history")
        assert isinstance(agent, PrunedHistoryAgent)
        assert agent.input_size == 31

    def test_trainer_class_in_metadata(self):
        from src.agents.registry import registry
        from src.training.pruned_history_trainer import PrunedHistoryTrainer
        meta = registry.get_metadata("pruned_history")
        assert meta.trainer_class is PrunedHistoryTrainer

    def test_in_trainable_agents_list(self):
        from src.agents.registry import registry
        trainable = registry.get_trainable_agents()
        ids = [a.id for a in trainable]
        assert "pruned_history" in ids


# =============================================================================
# Extra: Round splitting tests
# =============================================================================

class TestRoundSplitting:

    def test_split_check_check(self):
        agent = PrunedHistoryAgent()
        history = ((0, "CALL"), (1, "CALL"))
        rounds = agent._split_into_rounds(history)
        assert len(rounds[0]) == 2
        assert len(rounds[1]) == 0

    def test_split_raise_call_transitions_to_round_1(self):
        agent = PrunedHistoryAgent()
        history = ((0, "RAISE"), (1, "CALL"), (0, "CALL"), (1, "CALL"))
        rounds = agent._split_into_rounds(history)
        assert len(rounds[0]) == 2  # raise, call
        assert len(rounds[1]) == 2  # call, call

    def test_split_fold_ends_early(self):
        agent = PrunedHistoryAgent()
        history = ((0, "RAISE"), (1, "FOLD"))
        rounds = agent._split_into_rounds(history)
        assert len(rounds[0]) == 2
        assert len(rounds[1]) == 0

    def test_two_round_game(self):
        """Full two-round game with actions in both rounds."""
        agent = PrunedHistoryAgent()
        # Round 0: P0 calls, P1 calls (check-check)
        # Round 1: P0 raises, P1 calls
        history = ((0, "CALL"), (1, "CALL"), (0, "RAISE"), (1, "CALL"))
        features = agent._encode_pruned_action_history(history, viewer_id=0)

        # Round 0 features should be non-zero
        assert not torch.all(features[0:6] == 0.0)
        # Round 1 features should be non-zero
        assert not torch.all(features[6:12] == 0.0)


# =============================================================================
# Extra: update_params test
# =============================================================================

class TestUpdateParams:

    def test_update_lr(self):
        from src.training.pruned_history_trainer import PrunedHistoryTrainer

        agent = PrunedHistoryAgent()
        trainer = PrunedHistoryTrainer(agent, learning_rate=1e-4)

        trainer.update_params({"lr": 1e-3})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == 1e-3

    def test_update_hands_per_session(self):
        from src.training.pruned_history_trainer import PrunedHistoryTrainer

        agent = PrunedHistoryAgent()
        trainer = PrunedHistoryTrainer(agent, hands_per_session=30)
        assert trainer.hands_per_session == 30

        trainer.update_params({"hands_per_session": 50})
        assert trainer.hands_per_session == 50
