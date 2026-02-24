"""
CFR Unit & Integration Tests

Tests written BEFORE implementation to define expected behavior.
Run with: python -m pytest tests/unit/test_cfr.py -v
"""

import pytest
import numpy as np
import os
import tempfile

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.base import BaseAgent


# =============================================================================
# 1. Observation action_history extension
# =============================================================================

class TestObservationActionHistory:
    """Verify that action_history flows through the game engine."""

    def test_initial_observation_has_empty_history(self):
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        assert obs.action_history is not None
        assert len(obs.action_history) == 0

    def test_history_grows_after_action(self):
        game = LeducGame()
        game.reset()
        game.step(Action.CALL)
        obs = game.get_observation()
        assert len(obs.action_history) == 1
        assert obs.action_history[0] == (0, "CALL")

    def test_history_tracks_multiple_actions(self):
        game = LeducGame()
        game.reset()
        game.step(Action.CALL)   # P0 checks
        game.step(Action.RAISE)  # P1 raises
        obs = game.get_observation()
        assert len(obs.action_history) == 2
        assert obs.action_history[0] == (0, "CALL")
        assert obs.action_history[1] == (1, "RAISE")

    def test_action_history_is_tuple(self):
        """Frozen dataclass requires immutable fields."""
        game = LeducGame()
        game.reset()
        game.step(Action.RAISE)
        obs = game.get_observation()
        assert isinstance(obs.action_history, tuple)

    def test_observation_without_history_defaults_none(self):
        """Backward compat: Observations without action_history default to None."""
        obs = Observation(
            player_hand='K', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False,
        )
        assert obs.action_history is None

    def test_history_flows_through_poker_session(self):
        """PokerSession.get_observation() should preserve action_history."""
        from src.engine.poker_session import PokerSession
        session = PokerSession()
        session.new_hand()
        session.step(Action.CALL)
        obs = session.get_observation()
        assert obs.action_history is not None
        assert len(obs.action_history) == 1


# =============================================================================
# 2. Strategy storage
# =============================================================================

class TestInfoSetData:
    def test_uniform_with_zero_regrets(self):
        from src.cfr.strategy import InfoSetData
        info = InfoSetData()
        strategy = info.get_current_strategy()
        np.testing.assert_allclose(strategy, [1/3, 1/3, 1/3])

    def test_positive_regrets_normalize(self):
        from src.cfr.strategy import InfoSetData
        info = InfoSetData()
        info.regret_sum = np.array([0.0, 2.0, 8.0])
        strategy = info.get_current_strategy()
        np.testing.assert_allclose(strategy, [0.0, 0.2, 0.8])

    def test_negative_regrets_give_uniform(self):
        from src.cfr.strategy import InfoSetData
        info = InfoSetData()
        info.regret_sum = np.array([-5.0, -3.0, -1.0])
        strategy = info.get_current_strategy()
        np.testing.assert_allclose(strategy, [1/3, 1/3, 1/3])

    def test_average_strategy_with_zero_sum(self):
        from src.cfr.strategy import InfoSetData
        info = InfoSetData()
        avg = info.get_average_strategy()
        np.testing.assert_allclose(avg, [1/3, 1/3, 1/3])

    def test_average_strategy_normalizes(self):
        from src.cfr.strategy import InfoSetData
        info = InfoSetData()
        info.strategy_sum = np.array([100.0, 200.0, 300.0])
        avg = info.get_average_strategy()
        np.testing.assert_allclose(avg, [1/6, 2/6, 3/6])


class TestTabularStrategyStore:
    def test_get_or_create(self):
        from src.cfr.strategy import TabularStrategyStore
        store = TabularStrategyStore()
        info = store.get_info_set("Q:cr")
        assert info is not None
        # Same key returns same object
        assert store.get_info_set("Q:cr") is info

    def test_save_load_roundtrip(self):
        from src.cfr.strategy import TabularStrategyStore
        store = TabularStrategyStore()
        info = store.get_info_set("Q:cr")
        info.regret_sum = np.array([1.0, 2.0, 3.0])
        info.strategy_sum = np.array([100.0, 200.0, 300.0])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store.save(path)
            store2 = TabularStrategyStore()
            store2.load(path)
            info2 = store2.get_info_set("Q:cr")
            np.testing.assert_allclose(info2.regret_sum, [1, 2, 3])
            np.testing.assert_allclose(info2.strategy_sum, [100, 200, 300])
        finally:
            os.unlink(path)

    def test_num_info_sets(self):
        from src.cfr.strategy import TabularStrategyStore
        store = TabularStrategyStore()
        assert store.num_info_sets() == 0
        store.get_info_set("a")
        store.get_info_set("b")
        assert store.num_info_sets() == 2

    def test_legal_action_masking(self):
        from src.cfr.strategy import TabularStrategyStore
        store = TabularStrategyStore()
        info = store.get_info_set("test")
        info.regret_sum = np.array([10.0, 5.0, 0.0])  # FOLD=10, CALL=5, RAISE=0
        # Only FOLD and CALL are legal
        strategy = store.get_strategy("test", [Action.FOLD, Action.CALL])
        assert strategy[Action.RAISE.value] == 0.0
        assert strategy[Action.FOLD.value] > 0
        assert strategy[Action.CALL.value] > 0
        np.testing.assert_allclose(sum(strategy), 1.0)


# =============================================================================
# 3. CFR solver
# =============================================================================

class TestCFRSolver:
    def test_deals_cover_all_combinations(self):
        from src.cfr.solver import LeducCFRSolver
        from src.cfr.strategy import TabularStrategyStore
        solver = LeducCFRSolver(TabularStrategyStore())
        weights = sum(w for _, _, _, w in solver.deals)
        # Weights should sum to 1.0 (probability distribution)
        np.testing.assert_allclose(weights, 1.0, atol=1e-10)

    def test_single_iteration_creates_infosets(self):
        from src.cfr.solver import LeducCFRSolver
        from src.cfr.strategy import TabularStrategyStore
        store = TabularStrategyStore()
        solver = LeducCFRSolver(store)
        solver.run_iteration(1)
        # Should have discovered many infosets
        assert store.num_info_sets() > 50

    def test_exploitability_decreases(self):
        from src.cfr.solver import LeducCFRSolver
        from src.cfr.strategy import TabularStrategyStore
        store = TabularStrategyStore()
        solver = LeducCFRSolver(store)

        # Run 100 iterations, measure exploitability
        for i in range(1, 101):
            solver.run_iteration(i)
        early_exploit = solver.compute_exploitability()

        # Run 900 more
        for i in range(101, 1001):
            solver.run_iteration(i)
        late_exploit = solver.compute_exploitability()

        assert late_exploit < early_exploit, (
            f"Exploitability should decrease: {early_exploit:.4f} -> {late_exploit:.4f}"
        )

    def test_exploitability_near_zero_at_10k(self):
        from src.cfr.solver import LeducCFRSolver
        from src.cfr.strategy import TabularStrategyStore
        store = TabularStrategyStore()
        solver = LeducCFRSolver(store)
        for i in range(1, 10001):
            solver.run_iteration(i)
        exploit = solver.compute_exploitability()
        assert exploit < 0.01, f"Exploitability after 10K iterations should be < 0.01, got {exploit:.6f}"


# =============================================================================
# 4. CFR Agent (BaseAgent contract)
# =============================================================================

class TestCFRAgent:
    def test_inherits_base_agent(self):
        from src.agents.cfr_agent import CFRAgent
        assert issubclass(CFRAgent, BaseAgent)

    def test_select_action_returns_legal(self):
        from src.agents.cfr_agent import CFRAgent
        agent = CFRAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        action = agent.select_action(obs)
        assert isinstance(action, Action)
        assert action in obs.legal_actions

    def test_plays_full_game(self):
        from src.agents.cfr_agent import CFRAgent
        agent = CFRAgent()
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            obs = game.get_observation()
            action = agent.select_action(obs)
            game.step(action)
        assert game.is_finished

    def test_save_load_roundtrip(self):
        from src.agents.cfr_agent import CFRAgent
        from src.cfr.solver import LeducCFRSolver
        agent = CFRAgent()
        solver = LeducCFRSolver(agent.strategy_store)
        # Run a few iterations to populate strategy
        for i in range(1, 101):
            solver.run_iteration(i)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            agent.save_model(path)
            agent2 = CFRAgent()
            agent2.load_model(path)
            assert agent2.strategy_store.num_info_sets() == agent.strategy_store.num_info_sets()
        finally:
            os.unlink(path)

    def test_get_action_evaluations(self):
        from src.agents.cfr_agent import CFRAgent
        agent = CFRAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        evals = agent.get_action_evaluations(obs)
        assert len(evals) == len(obs.legal_actions)
        for e in evals:
            assert "action" in e
            assert "probability" in e

    def test_registered_in_registry(self):
        from src.agents.registry import registry
        metadata = registry.get_metadata("cfr")
        assert metadata is not None
        assert metadata.is_trainable
        assert metadata.category == "game_theory"


# =============================================================================
# 5. CFR Trainer integration
# =============================================================================

class TestCFRTrainer:
    def test_callback_protocol(self):
        from src.agents.cfr_agent import CFRAgent
        from src.training.cfr_trainer import CFRTrainer
        agent = CFRAgent()
        trainer = CFRTrainer(agent)
        events = []
        trainer.train(num_episodes=200, batch_size=50,
                      callback=lambda d: events.append(d))

        batch_events = [e for e in events if e["type"] == "batch_update"]
        eval_events = [e for e in events if e["type"] == "evaluation"]
        assert len(batch_events) > 0
        assert all("loss" in e for e in batch_events)
        assert len(eval_events) > 0
        assert all("avg_chips_per_round" in e for e in eval_events)

    def test_exploitability_is_loss(self):
        """Loss field should be exploitability (should decrease)."""
        from src.agents.cfr_agent import CFRAgent
        from src.training.cfr_trainer import CFRTrainer
        agent = CFRAgent()
        trainer = CFRTrainer(agent)
        losses = []
        trainer.train(
            num_episodes=500, batch_size=100,
            callback=lambda d: losses.append(d["loss"]) if d["type"] == "batch_update" else None,
        )
        assert losses[-1] < losses[0], f"Exploitability should decrease: {losses}"

    def test_stop_requested(self):
        import threading, time
        from src.agents.cfr_agent import CFRAgent
        from src.training.cfr_trainer import CFRTrainer
        agent = CFRAgent()
        trainer = CFRTrainer(agent)

        def stop_soon():
            time.sleep(0.05)
            trainer.request_stop()

        threading.Thread(target=stop_soon, daemon=True).start()
        trainer.train(num_episodes=1_000_000, batch_size=1000)
        # Should have stopped early — check infosets exist but far fewer iterations ran
        assert agent.strategy_store.num_info_sets() > 0

    def test_trained_cfr_beats_heuristic(self):
        """Nash equilibrium should not lose to any fixed strategy."""
        from src.agents.cfr_agent import CFRAgent
        from src.agents.heuristic import HeuristicAgent
        from src.training.cfr_trainer import CFRTrainer
        from src.training.evaluation import quick_evaluate

        agent = CFRAgent()
        trainer = CFRTrainer(agent)
        trainer.train(num_episodes=5000, batch_size=500)

        avg_chips = quick_evaluate(agent, HeuristicAgent(), num_rounds=2000)
        assert avg_chips > -0.15, f"CFR should not lose to heuristic: {avg_chips:.3f}"
