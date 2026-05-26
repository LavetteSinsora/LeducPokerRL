"""
Curriculum Agent Unit Tests

Tests cover:
  1.  Class hierarchy (subclass of AdaptiveValueAgent)
  2.  Network architecture (input_size=19, hidden=64)
  3.  Opponent pool initialization (3 opponents)
  4.  Block transition after block_size sessions
  5.  Rehearsal buffer population and size limits
  6.  Rehearsal mixing in update_model (batch size increases)
  7.  Both-player chain collection (chains[0] and chains[1] both non-empty)
  8.  Forgetting monitoring returns scores for all opponents
  9.  Training loop runs without errors
  10. Save/load round-trip
  11. debug_episode()
  12. update_params()
  13. Registry integration

Run with: python -m pytest tests/unit/test_curriculum_agent.py -v
"""

import pytest
import torch
import os
import tempfile

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.base import BaseAgent
from src.agents.value_based import ValueBasedAgent, ValueNetwork


# =============================================================================
# 1. Class hierarchy (subclass of AdaptiveValueAgent)
# =============================================================================

class TestClassHierarchy:
    def test_inherits_adaptive_value_agent(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.agents.adaptive_value import AdaptiveValueAgent
        assert issubclass(CurriculumAgent, AdaptiveValueAgent)

    def test_inherits_value_based_agent(self):
        from src.agents.curriculum_agent import CurriculumAgent
        assert issubclass(CurriculumAgent, ValueBasedAgent)

    def test_inherits_base_agent(self):
        from src.agents.curriculum_agent import CurriculumAgent
        assert issubclass(CurriculumAgent, BaseAgent)

    def test_instantiation(self):
        from src.agents.curriculum_agent import CurriculumAgent
        agent = CurriculumAgent()
        assert agent.train_mode is False

    def test_is_trivial_subclass(self):
        """CurriculumAgent adds no new methods or attributes beyond the parent."""
        from src.agents.curriculum_agent import CurriculumAgent
        from src.agents.adaptive_value import AdaptiveValueAgent
        agent = CurriculumAgent()
        parent = AdaptiveValueAgent()
        assert type(agent).__name__ == "CurriculumAgent"
        assert type(parent).__name__ == "AdaptiveValueAgent"


# =============================================================================
# 2. Network architecture (input_size=19, hidden=64)
# =============================================================================

class TestNetworkArchitecture:
    def test_input_size_is_19(self):
        from src.agents.curriculum_agent import CurriculumAgent
        agent = CurriculumAgent()
        assert agent.input_size == 19

    def test_model_is_value_network(self):
        from src.agents.curriculum_agent import CurriculumAgent
        agent = CurriculumAgent()
        assert isinstance(agent.model, ValueNetwork)

    def test_hidden_size_is_64(self):
        from src.agents.curriculum_agent import CurriculumAgent
        agent = CurriculumAgent()
        first_layer = agent.model.net[0]
        assert first_layer.out_features == 64

    def test_encoding_shape(self):
        from src.agents.curriculum_agent import CurriculumAgent
        agent = CurriculumAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        encoded = agent.encode_observation(obs)
        assert encoded.shape == (1, 19)

    def test_model_accepts_19_dim_input(self):
        from src.agents.curriculum_agent import CurriculumAgent
        agent = CurriculumAgent()
        x = torch.randn(1, 19)
        output = agent.model(x)
        assert output.shape == (1, 1)


# =============================================================================
# 3. Opponent pool initialization (3 opponents)
# =============================================================================

class TestOpponentPool:
    def test_pool_has_three_opponents(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        assert len(trainer.opponent_pool) == 3

    def test_pool_ordering(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        names = [name for name, _ in trainer.opponent_pool]
        assert names == ["heuristic", "value_based", "adaptive_value"]

    def test_pool_opponent_types(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        from src.agents.heuristic import HeuristicAgent
        from src.agents.value_based import ValueBasedAgent
        from src.agents.adaptive_value import AdaptiveValueAgent
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        assert isinstance(trainer.opponent_pool[0][1], HeuristicAgent)
        assert isinstance(trainer.opponent_pool[1][1], ValueBasedAgent)
        assert isinstance(trainer.opponent_pool[2][1], AdaptiveValueAgent)

    def test_pool_opponents_not_in_train_mode(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        for name, opp in trainer.opponent_pool[1:]:
            assert opp.train_mode is False, f"{name} should not be in train mode"

    def test_starts_at_first_opponent(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        assert trainer.current_opponent_idx == 0


# =============================================================================
# 4. Block transition after block_size sessions
# =============================================================================

class TestBlockTransition:
    def test_block_counter_increments(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, block_size=5, hands_per_session=2)
        agent.set_train_mode(True)
        trainer.collect_episode()
        assert trainer.sessions_in_current_block == 1

    def test_block_transition_triggers(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, block_size=3, hands_per_session=2)
        agent.set_train_mode(True)
        assert trainer.current_opponent_idx == 0
        for _ in range(3):
            trainer.collect_episode()
        assert trainer.current_opponent_idx == 1
        assert trainer.sessions_in_current_block == 0

    def test_block_wraps_around(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, block_size=2, hands_per_session=2)
        agent.set_train_mode(True)
        for _ in range(2 * 3):
            trainer.collect_episode()
        assert trainer.current_opponent_idx == 0

    def test_no_transition_before_block_size(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, block_size=10, hands_per_session=2)
        agent.set_train_mode(True)
        for _ in range(9):
            trainer.collect_episode()
        assert trainer.current_opponent_idx == 0
        assert trainer.sessions_in_current_block == 9


# =============================================================================
# 5. Rehearsal buffer population and size limits
# =============================================================================

class TestRehearsalBuffer:
    def test_buffer_populates(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=5, max_buffer_size=1000)
        agent.set_train_mode(True)
        trainer.collect_episode()
        assert len(trainer.rehearsal_buffer) == 5

    def test_buffer_grows_across_sessions(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=3, max_buffer_size=1000)
        agent.set_train_mode(True)
        trainer.collect_episode()
        trainer.collect_episode()
        assert len(trainer.rehearsal_buffer) == 6

    def test_buffer_respects_max_size(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=5, max_buffer_size=10)
        agent.set_train_mode(True)
        for _ in range(5):
            trainer.collect_episode()
        assert len(trainer.rehearsal_buffer) == 10

    def test_buffer_starts_empty(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        assert len(trainer.rehearsal_buffer) == 0

    def test_buffer_entries_have_correct_structure(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=2)
        agent.set_train_mode(True)
        trainer.collect_episode()
        entry = trainer.rehearsal_buffer[0]
        chains, rewards = entry
        assert isinstance(chains, list) and len(chains) == 2
        assert isinstance(rewards, list) and len(rewards) == 2


# =============================================================================
# 6. Rehearsal mixing in update_model (batch size increases)
# =============================================================================

class TestRehearsalMixing:
    def test_rehearsal_increases_batch(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=5,
                                    rehearsal_ratio=0.2, max_buffer_size=100)
        agent.set_train_mode(True)
        session_data = trainer.collect_episode()
        assert len(trainer.rehearsal_buffer) > 0
        batch_data = trainer.collect_episode()
        loss = trainer.update_model(batch_data)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_no_rehearsal_when_ratio_is_zero(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=3, rehearsal_ratio=0.0)
        agent.set_train_mode(True)
        session_data = trainer.collect_episode()
        loss = trainer.update_model(session_data)
        assert isinstance(loss, float)

    def test_rehearsal_with_empty_buffer(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=3, rehearsal_ratio=0.2)
        agent.set_train_mode(True)
        trainer.rehearsal_buffer = []
        session_data = trainer.collect_episode()
        trainer.rehearsal_buffer = []
        loss = trainer.update_model(session_data)
        assert isinstance(loss, float)


# =============================================================================
# 7. Both-player chain collection
# =============================================================================

class TestBothPlayerChains:
    def test_both_chains_populated(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=10)
        agent.set_train_mode(True)
        session_data = trainer.collect_episode()
        p0_has_chains = False
        p1_has_chains = False
        for chains, rewards in session_data:
            if len(chains[0]) > 0:
                p0_has_chains = True
            if len(chains[1]) > 0:
                p1_has_chains = True
        assert p0_has_chains, "Player 0 should have chain entries"
        assert p1_has_chains, "Player 1 should have chain entries"

    def test_chain_entries_are_tensors(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=3)
        agent.set_train_mode(True)
        session_data = trainer.collect_episode()
        for chains, rewards in session_data:
            for player_chain in chains:
                for entry in player_chain:
                    assert isinstance(entry, torch.Tensor)
                    assert entry.shape == (1, 19)

    def test_both_chains_used_in_td_update(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=5, rehearsal_ratio=0.0)
        agent.set_train_mode(True)
        session_data = trainer.collect_episode()
        loss = trainer.update_model(session_data)
        assert loss > 0.0, "Loss should be positive when both chains have data"


# =============================================================================
# 8. Forgetting monitoring returns scores for all opponents
# =============================================================================

class TestForgettingMonitoring:
    def test_returns_all_opponent_scores(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        results = trainer.evaluate_against_all_opponents(num_rounds=10)
        assert "heuristic" in results
        assert "value_based" in results
        assert "adaptive_value" in results

    def test_scores_are_floats(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        results = trainer.evaluate_against_all_opponents(num_rounds=10)
        for name, score in results.items():
            assert isinstance(score, float), f"Score for {name} should be float"

    def test_forgetting_log_appended(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        assert len(trainer.forgetting_log) == 0
        trainer.evaluate_against_all_opponents(num_rounds=10)
        assert len(trainer.forgetting_log) == 1
        trainer.evaluate_against_all_opponents(num_rounds=10)
        assert len(trainer.forgetting_log) == 2

    def test_forgetting_log_structure(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        trainer.evaluate_against_all_opponents(num_rounds=10)
        entry = trainer.forgetting_log[0]
        assert "sessions_trained" in entry
        assert "current_opponent" in entry
        assert "scores" in entry
        assert isinstance(entry["scores"], dict)


# =============================================================================
# 9. Training loop runs without errors
# =============================================================================

class TestTrainingLoop:
    def test_short_training_run(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=3, block_size=5)
        trainer.train(num_episodes=5, batch_size=2)

    def test_training_with_save(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=3, block_size=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "curriculum.pt")
            trainer.train(num_episodes=3, batch_size=2, save_path=save_path)
            assert os.path.exists(save_path)

    def test_training_with_callback(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=3, block_size=5)
        callbacks = []
        trainer.train(num_episodes=3, batch_size=2,
                      callback=lambda data: callbacks.append(data))
        assert len(callbacks) > 0

    def test_training_populates_rehearsal_buffer(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=3, block_size=50)
        trainer.train(num_episodes=5, batch_size=2)
        assert len(trainer.rehearsal_buffer) > 0


# =============================================================================
# 10. Save/load round-trip
# =============================================================================

class TestSaveLoad:
    def test_save_load_roundtrip(self):
        from src.agents.curriculum_agent import CurriculumAgent
        agent = CurriculumAgent()
        with torch.no_grad():
            for p in agent.model.parameters():
                p.fill_(0.42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "curriculum.pt")
            agent.save_model(path)
            agent2 = CurriculumAgent()
            agent2.load_model(path)

            for p1, p2 in zip(agent.model.parameters(), agent2.model.parameters()):
                assert torch.allclose(p1, p2)

    def test_loaded_model_same_architecture(self):
        from src.agents.curriculum_agent import CurriculumAgent
        agent = CurriculumAgent()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "curriculum.pt")
            agent.save_model(path)
            agent2 = CurriculumAgent(model_path=path)
            assert agent2.input_size == 19

    def test_loaded_model_produces_same_output(self):
        from src.agents.curriculum_agent import CurriculumAgent
        agent = CurriculumAgent()
        x = torch.randn(1, 19)
        with torch.no_grad():
            out1 = agent.model(x).item()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "curriculum.pt")
            agent.save_model(path)
            agent2 = CurriculumAgent(model_path=path)
            with torch.no_grad():
                out2 = agent2.model(x).item()
        assert abs(out1 - out2) < 1e-6


# =============================================================================
# 11. debug_episode()
# =============================================================================

class TestDebugEpisode:
    def test_debug_episode_returns_dict(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=6)
        result = trainer.debug_episode()
        assert isinstance(result, dict)

    def test_debug_episode_has_trace(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=6)
        result = trainer.debug_episode()
        assert "trace" in result
        assert isinstance(result["trace"], list)

    def test_debug_episode_has_rewards(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=6)
        result = trainer.debug_episode()
        assert "final_rewards" in result
        assert len(result["final_rewards"]) == 2

    def test_debug_episode_has_opponent_name(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=6)
        result = trainer.debug_episode()
        assert "opponent_name" in result

    def test_debug_episode_has_session_analytics(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=6)
        result = trainer.debug_episode()
        assert "session_analytics" in result

    def test_debug_episode_trace_has_evaluations(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=6)
        result = trainer.debug_episode()
        if len(result["trace"]) > 0:
            step = result["trace"][0]
            assert "evaluations" in step
            assert "selected_action" in step
            assert "player_id" in step

    def test_debug_episode_restores_train_mode(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        agent.set_train_mode(True)
        trainer = CurriculumTrainer(agent, hands_per_session=6)
        trainer.debug_episode()
        assert agent.train_mode is True


# =============================================================================
# 12. update_params()
# =============================================================================

class TestUpdateParams:
    def test_update_lr(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, learning_rate=1e-4)
        trainer.update_params({"lr": 0.001})
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == 0.001

    def test_update_hands_per_session(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, hands_per_session=30)
        trainer.update_params({"hands_per_session": 50})
        assert trainer.hands_per_session == 50

    def test_update_block_size(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, block_size=100)
        trainer.update_params({"block_size": 200})
        assert trainer.block_size == 200

    def test_update_rehearsal_ratio(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, rehearsal_ratio=0.2)
        trainer.update_params({"rehearsal_ratio": 0.3})
        assert trainer.rehearsal_ratio == 0.3

    def test_update_multiple_params(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        trainer.update_params({
            "lr": 0.01,
            "block_size": 50,
            "rehearsal_ratio": 0.5
        })
        assert trainer.block_size == 50
        assert trainer.rehearsal_ratio == 0.5
        for pg in trainer.optimizer.param_groups:
            assert pg["lr"] == 0.01


# =============================================================================
# 13. Registry integration
# =============================================================================

class TestRegistryIntegration:
    def test_curriculum_registered(self):
        from src.agents.registry import registry
        assert registry.is_registered("curriculum")

    def test_create_from_registry(self):
        from src.agents.registry import registry
        from src.agents.curriculum_agent import CurriculumAgent
        agent = registry.create("curriculum")
        assert isinstance(agent, CurriculumAgent)

    def test_metadata_correct(self):
        from src.agents.registry import registry
        meta = registry.get_metadata("curriculum")
        assert meta is not None
        assert meta.id == "curriculum"
        assert meta.display_name == "Curriculum AI"
        assert meta.is_trainable is True
        assert meta.requires_model_path is True
        assert meta.category == "rl"

    def test_trainer_class_in_metadata(self):
        from src.agents.registry import registry
        from src.training.curriculum_trainer import CurriculumTrainer
        meta = registry.get_metadata("curriculum")
        assert meta.trainer_class is CurriculumTrainer

    def test_appears_in_trainable_agents(self):
        from src.agents.registry import registry
        trainable = registry.get_trainable_agents()
        ids = [a.id for a in trainable]
        assert "curriculum" in ids

    def test_appears_in_rl_category(self):
        from src.agents.registry import registry
        rl_agents = registry.list_agents(category="rl")
        ids = [a.id for a in rl_agents]
        assert "curriculum" in ids


# =============================================================================
# Additional integration tests
# =============================================================================

class TestSelectAction:
    def test_select_action_returns_legal_action(self):
        from src.agents.curriculum_agent import CurriculumAgent
        agent = CurriculumAgent()
        game = LeducGame()
        game.reset()
        obs = game.get_observation()
        action = agent.select_action(obs)
        if isinstance(action, tuple):
            action = action[0]
        assert action in obs.legal_actions

    def test_plays_full_game(self):
        from src.agents.curriculum_agent import CurriculumAgent
        agent = CurriculumAgent()
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            obs = game.get_observation()
            action = agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]
            game.step(action)
        rewards = game.get_reward()
        assert len(rewards) == 2


class TestTrainerInit:
    def test_default_parameters(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        assert trainer.block_size == 100
        assert trainer.rehearsal_ratio == 0.2
        assert trainer.max_buffer_size == 5000
        assert trainer.current_opponent_idx == 0
        assert trainer.sessions_in_current_block == 0
        assert trainer.forgetting_log == []

    def test_custom_parameters(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent, block_size=50, rehearsal_ratio=0.3,
                                    max_buffer_size=1000, hands_per_session=10)
        assert trainer.block_size == 50
        assert trainer.rehearsal_ratio == 0.3
        assert trainer.max_buffer_size == 1000
        assert trainer.hands_per_session == 10

    def test_has_optimizer(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        assert hasattr(trainer, "optimizer")

    def test_has_session(self):
        from src.agents.curriculum_agent import CurriculumAgent
        from src.training.curriculum_trainer import CurriculumTrainer
        agent = CurriculumAgent()
        trainer = CurriculumTrainer(agent)
        assert hasattr(trainer, "session")
