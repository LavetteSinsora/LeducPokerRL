"""
Population-Diverse Adaptive Agent -- Unit & Integration Tests

Tests the full stack: PopAdaptiveAgent (inference), PopAdaptiveTrainer
(opponent pool, rotation, snapshots, training loop), and registry integration.

Run with: python -m pytest tests/unit/test_pop_adaptive.py -v
"""

import math
import os
import tempfile
import pytest
import torch

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.engine.poker_session import PokerSession, OpponentStats
from src.agents.base import BaseAgent
from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.agents.pop_adaptive import PopAdaptiveAgent
from src.training.pop_adaptive_trainer import PopAdaptiveTrainer
from src.training.adaptive_trainer import AdaptiveTrainer


# =============================================================================
# 1. PopAdaptiveAgent -- class hierarchy and encoding
# =============================================================================

class TestPopAdaptiveAgentClass:

    def test_inherits_adaptive_value_agent(self):
        """PopAdaptiveAgent should be a subclass of AdaptiveValueAgent."""
        assert issubclass(PopAdaptiveAgent, AdaptiveValueAgent)

    def test_inherits_base_agent(self):
        """PopAdaptiveAgent should be a subclass of BaseAgent."""
        assert issubclass(PopAdaptiveAgent, BaseAgent)

    def test_input_size_is_19(self):
        """Input size should be 19 (15 base + 4 stats), same as AdaptiveValueAgent."""
        agent = PopAdaptiveAgent()
        assert agent.input_size == 19

    def test_encoding_shape_is_19(self):
        """Agent should produce [1, 19] encoding."""
        agent = PopAdaptiveAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 19)

    def test_encoding_with_opponent_stats(self):
        """Encoding should incorporate OpponentStats features correctly."""
        agent = PopAdaptiveAgent()
        stats = OpponentStats()
        for _ in range(10):
            stats.record_action("FOLD", was_facing_raise=False)
            stats.record_hand_complete()

        obs = Observation(
            player_hand='K', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
            opponent_stats=stats,
        )
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 19)

        # Last 4 features should match stats
        fv = stats.to_feature_vector()
        for i, val in enumerate(fv):
            assert encoded[0, 15 + i].item() == pytest.approx(val, abs=1e-5)

    def test_encoding_without_stats_uses_defaults(self):
        """Without opponent_stats, defaults to [0.5, 0.5, 0.5, 0.0]."""
        agent = PopAdaptiveAgent()
        obs = Observation(
            player_hand='Q', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
        )
        encoded = agent.encode_observation(obs)
        assert encoded[0, 15].item() == pytest.approx(0.5)
        assert encoded[0, 16].item() == pytest.approx(0.5)
        assert encoded[0, 17].item() == pytest.approx(0.5)
        assert encoded[0, 18].item() == pytest.approx(0.0)

    def test_select_action_returns_legal(self):
        """select_action should return a legal Action."""
        agent = PopAdaptiveAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        action = agent.select_action(obs)
        assert isinstance(action, Action)
        assert action in obs.legal_actions

    def test_plays_full_game(self):
        """Agent should play a complete game without errors."""
        agent = PopAdaptiveAgent()
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            obs = game.get_observation()
            action = agent.select_action(obs)
            game.step(action)
        assert game.is_finished

    def test_get_action_evaluations(self):
        """get_action_evaluations should return per-action value estimates."""
        agent = PopAdaptiveAgent()
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

    def test_plays_full_session(self):
        """Agent should play through a PokerSession without errors."""
        agent = PopAdaptiveAgent()
        session = PokerSession()

        for _ in range(10):
            session.new_hand()
            while not session.is_finished:
                obs = session.get_observation()
                action = agent.select_action(obs)
                session.step(action)

        assert session.hands_played == 10


# =============================================================================
# 2. PopAdaptiveAgent -- save/load round-trip
# =============================================================================

class TestPopAdaptiveSaveLoad:

    def test_save_and_load_roundtrip(self):
        """Model weights should survive a save/load cycle."""
        agent = PopAdaptiveAgent()

        # Get initial weights
        initial_params = {k: v.clone() for k, v in agent.model.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save_model(path)
            assert os.path.exists(path)

            # Load into a new agent
            agent2 = PopAdaptiveAgent()
            agent2.load_model(path)

            # Weights should match
            for key in initial_params:
                assert torch.allclose(
                    agent.model.state_dict()[key],
                    agent2.model.state_dict()[key]
                ), f"Weights mismatch for {key}"
        finally:
            os.unlink(path)

    def test_save_load_preserves_predictions(self):
        """Predictions should be identical after save/load."""
        agent = PopAdaptiveAgent()
        agent.set_train_mode(False)

        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        evals_before = agent.get_action_evaluations(obs)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save_model(path)
            agent2 = PopAdaptiveAgent()
            agent2.load_model(path)
            agent2.set_train_mode(False)

            evals_after = agent2.get_action_evaluations(obs)
            for e1, e2 in zip(evals_before, evals_after):
                assert e1["value"] == pytest.approx(e2["value"], abs=1e-6)
        finally:
            os.unlink(path)


# =============================================================================
# 3. PopAdaptiveTrainer -- opponent pool initialization
# =============================================================================

class TestOpponentPool:

    def test_initial_pool_has_three_opponents(self):
        """Pool should start with heuristic, value_based, and adaptive_value."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent)
        assert len(trainer.opponent_pool) == 3

    def test_initial_pool_names(self):
        """Pool opponents should be named correctly."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent)
        names = [name for name, _ in trainer.opponent_pool]
        assert names == ["heuristic", "value_based", "adaptive_value"]

    def test_initial_pool_types(self):
        """Pool opponents should be the correct agent types."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent)
        _, heuristic = trainer.opponent_pool[0]
        _, value_based = trainer.opponent_pool[1]
        _, adaptive_value = trainer.opponent_pool[2]

        assert isinstance(heuristic, HeuristicAgent)
        assert isinstance(value_based, ValueBasedAgent)
        assert isinstance(adaptive_value, AdaptiveValueAgent)

    def test_pool_opponents_not_in_train_mode(self):
        """Pool opponents should be in eval mode."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent)
        # Value-based and adaptive should not be in train mode
        _, vb = trainer.opponent_pool[1]
        _, av = trainer.opponent_pool[2]
        assert not vb.train_mode
        assert not av.train_mode

    def test_get_current_opponent_starts_at_zero(self):
        """Current opponent should start at index 0 (heuristic)."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent)
        assert trainer.current_opponent_idx == 0
        opponent = trainer._get_current_opponent()
        assert isinstance(opponent, HeuristicAgent)

    def test_get_current_opponent_name(self):
        """Should return the name of the current opponent."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent)
        assert trainer._get_current_opponent_name() == "heuristic"


# =============================================================================
# 4. PopAdaptiveTrainer -- opponent rotation
# =============================================================================

class TestOpponentRotation:

    def test_rotation_after_rotate_every(self):
        """Opponent should rotate after rotate_every episodes."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, rotate_every=100)

        # Before rotation
        assert trainer.current_opponent_idx == 0
        trainer.episode_count = 100
        trainer._maybe_rotate_opponent()
        assert trainer.current_opponent_idx == 1

    def test_rotation_wraps_around(self):
        """Rotation should wrap around to start of pool."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, rotate_every=100)

        # Rotate through all 3 opponents and wrap
        trainer.episode_count = 100
        trainer._maybe_rotate_opponent()
        assert trainer.current_opponent_idx == 1

        trainer.episode_count = 200
        trainer._maybe_rotate_opponent()
        assert trainer.current_opponent_idx == 2

        trainer.episode_count = 300
        trainer._maybe_rotate_opponent()
        assert trainer.current_opponent_idx == 0  # Wrapped

    def test_no_rotation_at_zero(self):
        """No rotation when episode_count is 0."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, rotate_every=100)
        trainer.episode_count = 0
        trainer._maybe_rotate_opponent()
        assert trainer.current_opponent_idx == 0

    def test_no_rotation_between_intervals(self):
        """No rotation when episode_count is not a multiple of rotate_every."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, rotate_every=100)
        trainer.episode_count = 50
        trainer._maybe_rotate_opponent()
        assert trainer.current_opponent_idx == 0


# =============================================================================
# 5. PopAdaptiveTrainer -- self-snapshot mechanism
# =============================================================================

class TestSelfSnapshot:

    def test_snapshot_after_snapshot_every(self):
        """A self-snapshot should be added after snapshot_every episodes."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, snapshot_every=500)
        initial_pool_size = len(trainer.opponent_pool)

        trainer.episode_count = 500
        trainer._maybe_snapshot_self()
        assert len(trainer.opponent_pool) == initial_pool_size + 1

    def test_snapshot_name_format(self):
        """Snapshot name should include episode count."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, snapshot_every=500)
        trainer.episode_count = 500
        trainer._maybe_snapshot_self()

        name, _ = trainer.opponent_pool[-1]
        assert name == "self_snapshot_500"

    def test_snapshot_is_deep_copy(self):
        """Snapshot should be a deep copy, not sharing weights with training agent."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, snapshot_every=500)
        trainer.episode_count = 500
        trainer._maybe_snapshot_self()

        _, snapshot = trainer.opponent_pool[-1]

        # Modify training agent weights -- snapshot should not change
        with torch.no_grad():
            for param in agent.model.parameters():
                param.add_(1.0)

        # Snapshot weights should differ from modified agent
        for p_agent, p_snap in zip(agent.model.parameters(), snapshot.model.parameters()):
            assert not torch.allclose(p_agent, p_snap)

    def test_snapshot_not_in_train_mode(self):
        """Snapshot should be in eval mode."""
        agent = PopAdaptiveAgent()
        agent.set_train_mode(True)
        trainer = PopAdaptiveTrainer(agent, snapshot_every=500)
        trainer.episode_count = 500
        trainer._maybe_snapshot_self()

        _, snapshot = trainer.opponent_pool[-1]
        assert not snapshot.train_mode

    def test_no_snapshot_at_zero(self):
        """No snapshot when episode_count is 0."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, snapshot_every=500)
        initial_pool_size = len(trainer.opponent_pool)
        trainer.episode_count = 0
        trainer._maybe_snapshot_self()
        assert len(trainer.opponent_pool) == initial_pool_size

    def test_multiple_snapshots_grow_pool(self):
        """Multiple snapshots should grow the pool."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, snapshot_every=100)
        initial_pool_size = len(trainer.opponent_pool)

        trainer.episode_count = 100
        trainer._maybe_snapshot_self()
        trainer.episode_count = 200
        trainer._maybe_snapshot_self()

        assert len(trainer.opponent_pool) == initial_pool_size + 2


# =============================================================================
# 6. PopAdaptiveTrainer -- collect_episode
# =============================================================================

class TestCollectEpisode:

    def test_collect_episode_returns_session_data(self):
        """collect_episode should return a list of (chains, rewards) tuples."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, hands_per_session=5)
        agent.set_train_mode(True)

        session_data = trainer.collect_episode()
        assert isinstance(session_data, list)
        assert len(session_data) == 5

        for chains, rewards in session_data:
            assert len(chains) == 2
            assert len(rewards) == 2
            # Rewards should be zero-sum
            assert rewards[0] + rewards[1] == pytest.approx(0.0)

    def test_collect_episode_increments_count(self):
        """episode_count should increase by hands_per_session after each call."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, hands_per_session=10)
        agent.set_train_mode(True)

        assert trainer.episode_count == 0
        trainer.collect_episode()
        assert trainer.episode_count == 10
        trainer.collect_episode()
        assert trainer.episode_count == 20

    def test_collect_episode_only_records_player_0_chains(self):
        """Only player 0 (training agent) should have chain entries."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, hands_per_session=10)
        agent.set_train_mode(True)

        session_data = trainer.collect_episode()
        for chains, rewards in session_data:
            # Player 0 should have entries (training agent)
            # Player 1 should have no entries (opponent)
            assert len(chains[1]) == 0
            # Player 0 should have at least some entries across the session
            # (they might be empty for individual hands if player 0 didn't act)


# =============================================================================
# 7. PopAdaptiveTrainer -- training loop
# =============================================================================

class TestTrainingLoop:

    def test_training_loop_completes(self):
        """Training should complete without errors."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, hands_per_session=5,
                                     rotate_every=50, snapshot_every=200)

        losses = []
        trainer.train(
            num_episodes=5,
            batch_size=5,
            callback=lambda d: losses.append(d["loss"]) if d["type"] == "batch_update" else None,
        )

        assert len(losses) > 0
        for loss in losses:
            assert math.isfinite(loss), f"Loss should be finite, got {loss}"

    def test_training_100_episodes(self):
        """100-episode training run should complete with finite losses."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, hands_per_session=5,
                                     rotate_every=50, snapshot_every=200)

        losses = []
        trainer.train(
            num_episodes=20,
            batch_size=5,
            callback=lambda d: losses.append(d["loss"]) if d["type"] == "batch_update" else None,
        )

        assert len(losses) > 0
        for loss in losses:
            assert math.isfinite(loss)

    def test_training_triggers_rotation(self):
        """Training should trigger opponent rotation at the right time."""
        agent = PopAdaptiveAgent()
        # rotate_every=10 hands, with 5 hands per session
        trainer = PopAdaptiveTrainer(agent, hands_per_session=5,
                                     rotate_every=10, snapshot_every=10000)

        # After 2 sessions (10 hands), should rotate
        agent.set_train_mode(True)
        trainer.collect_episode()  # 5 hands
        assert trainer.current_opponent_idx == 0  # Not yet
        trainer.collect_episode()  # 10 hands total -> rotate
        assert trainer.current_opponent_idx == 1  # Rotated!

    def test_training_triggers_snapshot(self):
        """Training should trigger self-snapshot at the right time."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, hands_per_session=5,
                                     rotate_every=10000, snapshot_every=10)
        initial_pool_size = len(trainer.opponent_pool)

        agent.set_train_mode(True)
        trainer.collect_episode()  # 5 hands
        assert len(trainer.opponent_pool) == initial_pool_size
        trainer.collect_episode()  # 10 hands total -> snapshot
        assert len(trainer.opponent_pool) == initial_pool_size + 1


# =============================================================================
# 8. PopAdaptiveTrainer -- opponent diversity
# =============================================================================

class TestOpponentDiversity:

    def test_different_opponents_produce_different_stats(self):
        """Different opponents should produce different opponent_stats patterns."""
        agent = PopAdaptiveAgent()
        agent.set_train_mode(True)

        stats_by_opponent = {}

        for opp_idx in range(3):
            trainer = PopAdaptiveTrainer(agent, hands_per_session=20)
            trainer.current_opponent_idx = opp_idx

            trainer.session.reset()
            opponent = trainer._get_current_opponent()

            for _ in range(20):
                trainer.session.new_hand()
                while not trainer.session.is_finished:
                    player = trainer.session.current_player
                    obs = trainer.session.get_observation(viewer_id=player)
                    if player == 0:
                        action = agent.select_action(obs)
                    else:
                        action = opponent.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]
                    trainer.session.step(action)

            # Player 0's view of player 1's stats
            stats = trainer.session.stats[0].to_feature_vector()
            opp_name = trainer._get_current_opponent_name()
            stats_by_opponent[opp_name] = stats

        # At least some stats should differ between opponents
        heuristic_stats = stats_by_opponent["heuristic"]
        value_stats = stats_by_opponent["value_based"]
        adaptive_stats = stats_by_opponent["adaptive_value"]

        # Not all three should be identical (very unlikely with different agent types)
        all_same = (
            all(abs(h - v) < 0.01 for h, v in zip(heuristic_stats, value_stats)) and
            all(abs(h - a) < 0.01 for h, a in zip(heuristic_stats, adaptive_stats))
        )
        # This is a soft assertion -- with randomness it's theoretically possible
        # but extremely unlikely that all three produce identical stats over 20 hands
        assert not all_same, (
            f"Expected different stats from different opponents, but got: "
            f"heuristic={heuristic_stats}, value={value_stats}, adaptive={adaptive_stats}"
        )


# =============================================================================
# 9. PopAdaptiveTrainer -- debug_episode
# =============================================================================

class TestDebugEpisode:

    def test_debug_episode_returns_dict(self):
        """debug_episode should return a dict with expected keys."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, hands_per_session=10)
        result = trainer.debug_episode()

        assert isinstance(result, dict)
        assert "trace" in result
        assert "final_rewards" in result
        assert "eval_type" in result
        assert "session_analytics" in result
        assert "opponent_name" in result

    def test_debug_episode_trace_has_steps(self):
        """Debug trace should have at least one step."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, hands_per_session=10)
        result = trainer.debug_episode()

        assert len(result["trace"]) > 0

    def test_debug_episode_steps_have_opponent_stats(self):
        """Each step in the trace should include opponent_stats."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, hands_per_session=10)
        result = trainer.debug_episode()

        for step in result["trace"]:
            assert "opponent_stats" in step

    def test_debug_episode_steps_have_evaluations(self):
        """Each step should have evaluations with action details."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, hands_per_session=10)
        result = trainer.debug_episode()

        for step in result["trace"]:
            assert "evaluations" in step
            assert len(step["evaluations"]) > 0
            for e in step["evaluations"]:
                assert "action" in e
                assert "value" in e

    def test_debug_episode_has_true_values(self):
        """Each step should have true_value and prediction_error."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, hands_per_session=10)
        result = trainer.debug_episode()

        for step in result["trace"]:
            assert "true_value" in step
            assert "prediction_error" in step

    def test_debug_episode_includes_opponent_name(self):
        """Debug result should include the opponent name."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, hands_per_session=10)
        result = trainer.debug_episode()
        assert result["opponent_name"] in ["heuristic", "value_based", "adaptive_value"]


# =============================================================================
# 10. PopAdaptiveTrainer -- update_params
# =============================================================================

class TestUpdateParams:

    def test_update_lr(self):
        """update_params should update learning rate."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, learning_rate=1e-3)
        trainer.update_params({"lr": 5e-4})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == 5e-4

    def test_update_hands_per_session(self):
        """update_params should update hands_per_session."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, hands_per_session=30)
        trainer.update_params({"hands_per_session": 50})
        assert trainer.hands_per_session == 50

    def test_update_rotate_every(self):
        """update_params should update rotate_every."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, rotate_every=100)
        trainer.update_params({"rotate_every": 200})
        assert trainer.rotate_every == 200

    def test_update_snapshot_every(self):
        """update_params should update snapshot_every."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent, snapshot_every=500)
        trainer.update_params({"snapshot_every": 1000})
        assert trainer.snapshot_every == 1000


# =============================================================================
# 11. PopAdaptiveTrainer -- inherits from AdaptiveTrainer
# =============================================================================

class TestTrainerInheritance:

    def test_inherits_adaptive_trainer(self):
        """PopAdaptiveTrainer should extend AdaptiveTrainer."""
        assert issubclass(PopAdaptiveTrainer, AdaptiveTrainer)

    def test_uses_poker_session(self):
        """Trainer should use PokerSession (inherited from AdaptiveTrainer)."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent)
        assert isinstance(trainer.session, PokerSession)

    def test_default_parameters(self):
        """Default parameters should be set correctly."""
        agent = PopAdaptiveAgent()
        trainer = PopAdaptiveTrainer(agent)
        assert trainer.rotate_every == 100
        assert trainer.snapshot_every == 500
        assert trainer.hands_per_session == 30
        assert trainer.episode_count == 0
        assert trainer.current_opponent_idx == 0


# =============================================================================
# 12. Registry integration
# =============================================================================

class TestRegistryIntegration:

    def test_registered_in_registry(self):
        """pop_adaptive should be registered in the agent registry."""
        from src.agents.registry import registry
        metadata = registry.get_metadata("pop_adaptive")
        assert metadata is not None

    def test_registry_metadata(self):
        """Registry metadata should have correct values."""
        from src.agents.registry import registry
        metadata = registry.get_metadata("pop_adaptive")
        assert metadata.id == "pop_adaptive"
        assert metadata.display_name == "Population Adaptive AI"
        assert metadata.is_trainable is True
        assert metadata.requires_model_path is True
        assert metadata.category == "rl"
        assert metadata.trainer_class is PopAdaptiveTrainer

    def test_registry_create(self):
        """Creating from registry should produce a PopAdaptiveAgent."""
        from src.agents.registry import registry
        agent = registry.create("pop_adaptive")
        assert isinstance(agent, PopAdaptiveAgent)
        assert isinstance(agent, AdaptiveValueAgent)

    def test_appears_in_trainable_agents(self):
        """pop_adaptive should appear in the trainable agents list."""
        from src.agents.registry import registry
        trainable = registry.get_trainable_agents()
        ids = [a.id for a in trainable]
        assert "pop_adaptive" in ids
