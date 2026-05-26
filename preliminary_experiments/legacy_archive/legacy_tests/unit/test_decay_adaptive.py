"""
Decay-Weighted Adaptive Agent — Unit & Integration Tests

Tests the full stack: DecayOpponentStats (EMA math), DecayPokerSession
(game wrapper), DecayAdaptiveAgent (inference), DecayAdaptiveTrainer
(training loop), and registry integration.

Run with: python -m pytest tests/unit/test_decay_adaptive.py -v
"""

import math
import pytest
import torch

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.engine.decay_stats import DecayOpponentStats
from src.engine.decay_session import DecayPokerSession
from src.engine.poker_session import PokerSession
from src.agents.base import BaseAgent
from src.agents.adaptive_value import AdaptiveValueAgent
from src.agents.decay_adaptive import DecayAdaptiveAgent
from src.training.decay_adaptive_trainer import DecayAdaptiveTrainer


# =============================================================================
# 1. DecayOpponentStats — EMA correctness
# =============================================================================

class TestDecayOpponentStats:

    def test_initial_rates_are_half(self):
        """With no data, all rates should be 0.5 (maximum entropy)."""
        stats = DecayOpponentStats()
        assert stats.fold_rate == 0.5
        assert stats.raise_rate == 0.5
        assert stats.fold_to_raise_rate == 0.5

    def test_initial_feature_vector(self):
        """Feature vector starts at [0.5, 0.5, 0.5, 0.0] — uninformative prior."""
        stats = DecayOpponentStats()
        fv = stats.to_feature_vector()
        assert len(fv) == 4
        assert fv == [0.5, 0.5, 0.5, 0.0]

    def test_single_fold_ema_update(self):
        """After one FOLD with alpha=0.1: fold_rate = 0.1*1 + 0.9*0.5 = 0.55."""
        stats = DecayOpponentStats(alpha=0.1)
        stats.record_action("FOLD", was_facing_raise=False)
        assert stats.fold_rate == pytest.approx(0.55)
        assert stats.raise_rate == pytest.approx(0.45)  # 0.1*0 + 0.9*0.5
        assert stats.total_actions == 1

    def test_single_raise_ema_update(self):
        """After one RAISE with alpha=0.1: raise_rate = 0.1*1 + 0.9*0.5 = 0.55."""
        stats = DecayOpponentStats(alpha=0.1)
        stats.record_action("RAISE", was_facing_raise=False)
        assert stats.raise_rate == pytest.approx(0.55)
        assert stats.fold_rate == pytest.approx(0.45)

    def test_single_call_ema_update(self):
        """After one CALL with alpha=0.1: call_rate_ema updates, fold and raise decay."""
        stats = DecayOpponentStats(alpha=0.1)
        stats.record_action("CALL", was_facing_raise=False)
        assert stats.call_rate_ema == pytest.approx(0.55)
        assert stats.fold_rate == pytest.approx(0.45)
        assert stats.raise_rate == pytest.approx(0.45)

    def test_repeated_folds_converge_toward_one(self):
        """Many consecutive folds should push fold_rate toward 1.0."""
        stats = DecayOpponentStats(alpha=0.1)
        for _ in range(100):
            stats.record_action("FOLD", was_facing_raise=False)
        assert stats.fold_rate > 0.99
        assert stats.raise_rate < 0.01

    def test_repeated_raises_converge_toward_one(self):
        """Many consecutive raises should push raise_rate toward 1.0."""
        stats = DecayOpponentStats(alpha=0.1)
        for _ in range(100):
            stats.record_action("RAISE", was_facing_raise=False)
        assert stats.raise_rate > 0.99
        assert stats.fold_rate < 0.01

    def test_ema_more_recent_weighted_than_uniform(self):
        """EMA should give more weight to recent actions than uniform averaging.

        Scenario: 10 CALLs then 10 FOLDs.
        - Uniform fold_rate = 10/20 = 0.5
        - EMA fold_rate should be > 0.5 (recency bias toward FOLDs)
        """
        stats = DecayOpponentStats(alpha=0.1)
        for _ in range(10):
            stats.record_action("CALL", was_facing_raise=False)
        for _ in range(10):
            stats.record_action("FOLD", was_facing_raise=False)
        # EMA should be biased toward recent FOLDs
        assert stats.fold_rate > 0.5, (
            f"EMA fold_rate ({stats.fold_rate:.4f}) should be > 0.5 after recent folds"
        )

    def test_ema_vs_uniform_with_strategy_shift(self):
        """Compare EMA vs uniform on a strategy-shifting opponent.

        Opponent plays 50 CALLs then 50 FOLDs.
        Uniform: fold_rate = 50/100 = 0.5
        EMA: fold_rate >> 0.5 (recent FOLDs dominate)
        """
        from src.engine.poker_session import OpponentStats

        ema_stats = DecayOpponentStats(alpha=0.1)
        uniform_stats = OpponentStats()

        for _ in range(50):
            ema_stats.record_action("CALL", was_facing_raise=False)
            uniform_stats.record_action("CALL", was_facing_raise=False)
        for _ in range(50):
            ema_stats.record_action("FOLD", was_facing_raise=False)
            uniform_stats.record_action("FOLD", was_facing_raise=False)

        assert uniform_stats.fold_rate == 0.5
        assert ema_stats.fold_rate > uniform_stats.fold_rate, (
            f"EMA ({ema_stats.fold_rate:.4f}) should > uniform ({uniform_stats.fold_rate:.4f})"
        )

    def test_fold_to_raise_only_updates_when_facing_raise(self):
        """fold_to_raise_rate should only update when was_facing_raise=True."""
        stats = DecayOpponentStats(alpha=0.5)
        initial_ftr = stats.fold_to_raise_rate

        # FOLD but NOT facing raise — fold_to_raise should NOT change
        stats.record_action("FOLD", was_facing_raise=False)
        assert stats.fold_to_raise_rate == initial_ftr

        # FOLD while facing raise — fold_to_raise SHOULD change
        stats.record_action("FOLD", was_facing_raise=True)
        assert stats.fold_to_raise_rate != initial_ftr
        assert stats.fold_to_raise_rate == pytest.approx(0.75)  # 0.5*1 + 0.5*0.5

    def test_record_hand_complete(self):
        stats = DecayOpponentStats()
        assert stats.hands_observed == 0
        stats.record_hand_complete()
        assert stats.hands_observed == 1
        stats.record_hand_complete()
        assert stats.hands_observed == 2

    def test_confidence_signal_ramps(self):
        """Confidence = min(hands_observed / 50, 1.0)."""
        stats = DecayOpponentStats()
        assert stats.to_feature_vector()[3] == 0.0

        for _ in range(25):
            stats.record_hand_complete()
        assert stats.to_feature_vector()[3] == pytest.approx(0.5)

        for _ in range(25):
            stats.record_hand_complete()
        assert stats.to_feature_vector()[3] == pytest.approx(1.0)

        # Beyond 50 stays at 1.0
        stats.record_hand_complete()
        assert stats.to_feature_vector()[3] == pytest.approx(1.0)

    def test_reset_clears_state(self):
        stats = DecayOpponentStats(alpha=0.2)
        for _ in range(20):
            stats.record_action("FOLD", was_facing_raise=True)
            stats.record_hand_complete()

        stats.reset()
        assert stats.fold_rate == 0.5
        assert stats.raise_rate == 0.5
        assert stats.fold_to_raise_rate == 0.5
        assert stats.hands_observed == 0
        assert stats.total_actions == 0

    def test_high_alpha_adapts_faster(self):
        """Higher alpha should mean faster adaptation to new behavior."""
        slow = DecayOpponentStats(alpha=0.05)
        fast = DecayOpponentStats(alpha=0.3)

        # Both start the same
        for _ in range(20):
            slow.record_action("CALL", was_facing_raise=False)
            fast.record_action("CALL", was_facing_raise=False)

        # Strategy shifts to FOLD
        for _ in range(5):
            slow.record_action("FOLD", was_facing_raise=False)
            fast.record_action("FOLD", was_facing_raise=False)

        # Fast alpha should have higher fold_rate (adapted more quickly)
        assert fast.fold_rate > slow.fold_rate, (
            f"fast alpha ({fast.fold_rate:.4f}) should > slow ({slow.fold_rate:.4f})"
        )

    def test_to_feature_vector_length(self):
        stats = DecayOpponentStats()
        fv = stats.to_feature_vector()
        assert len(fv) == 4

    def test_alpha_preserved_after_init(self):
        stats = DecayOpponentStats(alpha=0.25)
        assert stats.alpha == 0.25


# =============================================================================
# 2. DecayPokerSession
# =============================================================================

class TestDecayPokerSession:

    def test_inherits_poker_session(self):
        session = DecayPokerSession()
        assert isinstance(session, PokerSession)

    def test_uses_decay_stats(self):
        session = DecayPokerSession(alpha=0.2)
        assert isinstance(session.stats[0], DecayOpponentStats)
        assert isinstance(session.stats[1], DecayOpponentStats)
        assert session.stats[0].alpha == 0.2
        assert session.stats[1].alpha == 0.2

    def test_wraps_game_correctly(self):
        session = DecayPokerSession()
        assert isinstance(session.game, LeducGame)

    def test_new_hand_and_step(self):
        """Play a hand through DecayPokerSession."""
        session = DecayPokerSession()
        session.new_hand()
        assert not session.is_finished

        while not session.is_finished:
            obs = session.get_observation()
            action = obs.legal_actions[0]  # Pick first legal action
            session.step(action)

        assert session.is_finished
        assert session.hands_played == 1

    def test_observation_has_decay_stats(self):
        """get_observation should attach DecayOpponentStats."""
        session = DecayPokerSession()
        session.new_hand()
        obs = session.get_observation()
        assert obs.opponent_stats is not None
        assert isinstance(obs.opponent_stats, DecayOpponentStats)

    def test_stats_accumulate_across_hands(self):
        """Stats should persist across hands within a session."""
        session = DecayPokerSession()

        for _ in range(5):
            session.new_hand()
            while not session.is_finished:
                obs = session.get_observation()
                # Use CALL to ensure both players act (FOLD ends hand immediately)
                action = Action.CALL if Action.CALL in obs.legal_actions else obs.legal_actions[0]
                session.step(action)

        assert session.hands_played == 5
        # Both players acted, so both stat trackers should have data
        total_actions = session.stats[0].total_actions + session.stats[1].total_actions
        assert total_actions > 0
        assert session.stats[0].hands_observed == 5
        assert session.stats[1].hands_observed == 5

    def test_reset_creates_fresh_decay_stats(self):
        session = DecayPokerSession(alpha=0.15)
        session.new_hand()
        while not session.is_finished:
            obs = session.get_observation()
            session.step(obs.legal_actions[0])

        session.reset()
        assert session.stats[0].total_actions == 0
        assert isinstance(session.stats[0], DecayOpponentStats)
        assert session.stats[0].alpha == 0.15
        assert session.hands_played == 0

    def test_alpha_preserved(self):
        session = DecayPokerSession(alpha=0.3)
        assert session.alpha == 0.3


# =============================================================================
# 3. DecayAdaptiveAgent
# =============================================================================

class TestDecayAdaptiveAgent:

    def test_inherits_adaptive_value_agent(self):
        assert issubclass(DecayAdaptiveAgent, AdaptiveValueAgent)

    def test_inherits_base_agent(self):
        assert issubclass(DecayAdaptiveAgent, BaseAgent)

    def test_encoding_shape_is_19(self):
        """Agent should produce [1, 19] encoding (15 base + 4 stats)."""
        agent = DecayAdaptiveAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 19)

    def test_encoding_with_decay_stats(self):
        """Encoding should incorporate DecayOpponentStats features."""
        agent = DecayAdaptiveAgent()
        stats = DecayOpponentStats(alpha=0.1)
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

        # Last 4 features should be from stats
        fv = stats.to_feature_vector()
        for i, val in enumerate(fv):
            assert encoded[0, 15 + i].item() == pytest.approx(val, abs=1e-5)

    def test_encoding_without_stats_uses_defaults(self):
        """Without opponent_stats, defaults to [0.5, 0.5, 0.5, 0.0]."""
        agent = DecayAdaptiveAgent()
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
        agent = DecayAdaptiveAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        action = agent.select_action(obs)
        assert isinstance(action, Action)
        assert action in obs.legal_actions

    def test_plays_full_game(self):
        """Agent should play a complete game without errors."""
        agent = DecayAdaptiveAgent()
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            obs = game.get_observation()
            action = agent.select_action(obs)
            game.step(action)
        assert game.is_finished

    def test_plays_full_session(self):
        """Agent should play through a DecayPokerSession without errors."""
        agent = DecayAdaptiveAgent()
        session = DecayPokerSession()

        for _ in range(10):
            session.new_hand()
            while not session.is_finished:
                obs = session.get_observation()
                action = agent.select_action(obs)
                session.step(action)

        assert session.hands_played == 10

    def test_get_action_evaluations(self):
        agent = DecayAdaptiveAgent()
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

    def test_input_size_is_19(self):
        agent = DecayAdaptiveAgent()
        assert agent.input_size == 19


# =============================================================================
# 4. DecayAdaptiveTrainer
# =============================================================================

class TestDecayAdaptiveTrainer:

    def test_uses_decay_session(self):
        agent = DecayAdaptiveAgent()
        trainer = DecayAdaptiveTrainer(agent, alpha=0.15)
        assert isinstance(trainer.session, DecayPokerSession)
        assert trainer.session.alpha == 0.15

    def test_collect_episode_returns_session_data(self):
        """collect_episode should return a list of (chains, rewards) tuples."""
        agent = DecayAdaptiveAgent()
        trainer = DecayAdaptiveTrainer(agent, hands_per_session=5)
        agent.set_train_mode(True)

        session_data = trainer.collect_episode()
        assert isinstance(session_data, list)
        assert len(session_data) == 5  # One per hand

        for chains, rewards in session_data:
            assert len(chains) == 2  # Two players
            assert len(rewards) == 2
            # Rewards should be zero-sum
            assert rewards[0] + rewards[1] == pytest.approx(0.0)

    def test_training_loop_completes(self):
        """5 sessions of training should complete with finite loss."""
        agent = DecayAdaptiveAgent()
        trainer = DecayAdaptiveTrainer(agent, hands_per_session=10, alpha=0.1)

        losses = []
        trainer.train(
            num_episodes=5,
            batch_size=10,
            callback=lambda d: losses.append(d["loss"]) if d["type"] == "batch_update" else None,
        )

        assert len(losses) > 0
        for loss in losses:
            assert math.isfinite(loss), f"Loss should be finite, got {loss}"

    def test_update_params_alpha(self):
        agent = DecayAdaptiveAgent()
        trainer = DecayAdaptiveTrainer(agent, alpha=0.1)
        assert trainer.alpha == 0.1

        trainer.update_params({"alpha": 0.3})
        assert trainer.alpha == 0.3
        assert isinstance(trainer.session, DecayPokerSession)
        assert trainer.session.alpha == 0.3

    def test_update_params_lr(self):
        agent = DecayAdaptiveAgent()
        trainer = DecayAdaptiveTrainer(agent, learning_rate=1e-3)
        trainer.update_params({"lr": 5e-4})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == 5e-4

    def test_update_params_hands_per_session(self):
        agent = DecayAdaptiveAgent()
        trainer = DecayAdaptiveTrainer(agent, hands_per_session=30)
        trainer.update_params({"hands_per_session": 50})
        assert trainer.hands_per_session == 50

    def test_debug_episode(self):
        """debug_episode should return a dict with trace and session analytics."""
        agent = DecayAdaptiveAgent()
        trainer = DecayAdaptiveTrainer(agent, hands_per_session=10)
        result = trainer.debug_episode()

        assert "trace" in result
        assert "final_rewards" in result
        assert "session_analytics" in result
        assert len(result["trace"]) > 0

        # Each step should have opponent_stats
        for step in result["trace"]:
            assert "opponent_stats" in step


# =============================================================================
# 5. Registry integration
# =============================================================================

class TestRegistryIntegration:

    def test_registered_in_registry(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("decay_adaptive")
        assert metadata is not None

    def test_registry_metadata(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("decay_adaptive")
        assert metadata.id == "decay_adaptive"
        assert metadata.is_trainable is True
        assert metadata.requires_model_path is True
        assert metadata.category == "rl"
        assert metadata.trainer_class is DecayAdaptiveTrainer

    def test_registry_create(self):
        from src.agents.registry import registry
        agent = registry.create("decay_adaptive")
        assert isinstance(agent, DecayAdaptiveAgent)
        assert isinstance(agent, AdaptiveValueAgent)

    def test_appears_in_trainable_agents(self):
        from src.agents.registry import registry
        trainable = registry.get_trainable_agents()
        ids = [a.id for a in trainable]
        assert "decay_adaptive" in ids
