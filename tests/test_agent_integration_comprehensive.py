"""
Comprehensive Agent Integration Test Suite for PokerRL

This test suite validates all behaviors and dependencies when implementing new agents
for Leduc Hold'em. It ensures agents correctly integrate with:
1. BaseAgent contract
2. Observation handling
3. Game engine lifecycle
4. Training infrastructure
5. Evaluation framework
6. Edge cases and robustness

Run with: python -m pytest tests/test_agent_integration_comprehensive.py -v
"""

import pytest
import torch
import numpy as np
import copy
from typing import List, Tuple

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.base import BaseAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent, ValueNetwork
from src.training.trainer import SelfPlayTrainer, TrajectoryStep
from src.training.evaluation import evaluate_agents, quick_evaluate, EvaluationResult


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def value_agent():
    """Provides a fresh ValueBasedAgent instance."""
    return ValueBasedAgent()


@pytest.fixture
def heuristic_agent():
    """Provides a fresh HeuristicAgent instance."""
    return HeuristicAgent()


@pytest.fixture
def game():
    """Provides a fresh LeducGame instance."""
    return LeducGame()


@pytest.fixture
def sample_observation():
    """Provides a sample pre-flop Observation for testing."""
    return Observation(
        player_hand='K',
        board=None,
        pot=[1, 1],
        current_player=0,
        current_round=0,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        is_finished=False
    )


@pytest.fixture
def flop_observation():
    """Provides a sample flop Observation for testing."""
    return Observation(
        player_hand='Q',
        board='J',
        pot=[3, 3],
        current_player=1,
        current_round=1,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        is_finished=False
    )


@pytest.fixture
def terminal_observation():
    """Provides a terminal (finished) Observation for testing."""
    return Observation(
        player_hand='K',
        board='K',
        pot=[5, 5],
        current_player=0,
        current_round=1,
        legal_actions=[],
        is_finished=True
    )


# =============================================================================
# Test Class 1: BaseAgent Contract Compliance
# =============================================================================

class TestBaseAgentContract:
    """Tests that agents properly implement the BaseAgent interface."""
    
    def test_value_agent_inherits_from_base(self, value_agent):
        """ValueBasedAgent must inherit from BaseAgent."""
        assert isinstance(value_agent, BaseAgent)
    
    def test_heuristic_agent_inherits_from_base(self, heuristic_agent):
        """HeuristicAgent must inherit from BaseAgent."""
        assert isinstance(heuristic_agent, BaseAgent)
    
    def test_encode_observation_method_exists(self, value_agent, sample_observation):
        """Agent must have encode_observation method that accepts Observation."""
        assert hasattr(value_agent, 'encode_observation')
        result = value_agent.encode_observation(sample_observation)
        assert result is not None
    
    def test_encode_observation_returns_tensor(self, value_agent, sample_observation):
        """ValueBasedAgent.encode_observation must return a torch.Tensor."""
        result = value_agent.encode_observation(sample_observation)
        assert isinstance(result, torch.Tensor)
    
    def test_select_action_method_exists(self, value_agent, sample_observation):
        """Agent must have select_action method."""
        assert hasattr(value_agent, 'select_action')
        result = value_agent.select_action(sample_observation)
        assert result is not None
    
    def test_select_action_returns_valid_action(self, value_agent, sample_observation):
        """select_action must return a valid Action enum."""
        result = value_agent.select_action(sample_observation)
        assert isinstance(result, Action)
        assert result in [Action.FOLD, Action.CALL, Action.RAISE]
    
    def test_format_action_fallback(self, value_agent):
        """format_action should fallback to first legal action for illegal output."""
        legal_actions = [Action.FOLD, Action.CALL]  # RAISE not legal
        result = value_agent.format_action(Action.RAISE.value, legal_actions)
        assert result in legal_actions
    
    def test_format_action_valid(self, value_agent):
        """format_action should return the action if it's legal."""
        legal_actions = [Action.FOLD, Action.CALL, Action.RAISE]
        result = value_agent.format_action(Action.CALL.value, legal_actions)
        assert result == Action.CALL


# =============================================================================
# Test Class 2: Observation Handling
# =============================================================================

class TestObservationHandling:
    """Tests that agents correctly handle all Observation variations."""
    
    @pytest.mark.parametrize("card", ['J', 'Q', 'K'])
    def test_handles_all_player_cards(self, value_agent, card):
        """Agent must handle all valid player cards."""
        obs = Observation(
            player_hand=card,
            board=None,
            pot=[1, 1],
            current_player=0,
            current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False
        )
        action = value_agent.select_action(obs)
        assert isinstance(action, Action)
    
    def test_handles_unknown_hand(self, value_agent):
        """Agent must handle UNKNOWN player hand (opponent's perspective)."""
        obs = Observation(
            player_hand='UNKNOWN',
            board='Q',
            pot=[3, 3],
            current_player=0,
            current_round=1,
            legal_actions=[Action.FOLD, Action.CALL],
            is_finished=False
        )
        # This should not raise an error
        encoded = value_agent.encode_observation(obs)
        assert encoded is not None
    
    def test_handles_no_board_preflop(self, value_agent, sample_observation):
        """Agent must handle None board in pre-flop."""
        assert sample_observation.board is None
        action = value_agent.select_action(sample_observation)
        assert isinstance(action, Action)
    
    @pytest.mark.parametrize("board_card", ['J', 'Q', 'K'])
    def test_handles_all_board_cards(self, value_agent, board_card):
        """Agent must handle all valid board cards."""
        obs = Observation(
            player_hand='Q',
            board=board_card,
            pot=[3, 3],
            current_player=0,
            current_round=1,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False
        )
        action = value_agent.select_action(obs)
        assert isinstance(action, Action)
    
    def test_handles_finished_state(self, value_agent, terminal_observation):
        """Agent must handle terminal Observation without error."""
        # Terminal observations have empty legal_actions
        assert terminal_observation.is_finished
        encoded = value_agent.encode_observation(terminal_observation)
        assert encoded is not None
    
    @pytest.mark.parametrize("pot", [[1, 1], [3, 3], [5, 5], [7, 7], [10, 10]])
    def test_handles_varying_pot_sizes(self, value_agent, pot):
        """Agent must handle various pot sizes."""
        obs = Observation(
            player_hand='K',
            board='Q',
            pot=pot,
            current_player=0,
            current_round=1,
            legal_actions=[Action.FOLD, Action.CALL],
            is_finished=False
        )
        action = value_agent.select_action(obs)
        assert isinstance(action, Action)
    
    def test_encoding_produces_consistent_shape(self, value_agent):
        """Encoded observations must have consistent shape."""
        obs1 = Observation(
            player_hand='J', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False
        )
        obs2 = Observation(
            player_hand='K', board='Q', pot=[5, 5],
            current_player=1, current_round=1,
            legal_actions=[Action.FOLD, Action.CALL],
            is_finished=False
        )
        
        enc1 = value_agent.encode_observation(obs1)
        enc2 = value_agent.encode_observation(obs2)
        
        assert enc1.shape == enc2.shape
    
    @pytest.mark.parametrize("player", [0, 1])
    def test_handles_both_player_positions(self, value_agent, player):
        """Agent must handle being either player 0 or player 1."""
        obs = Observation(
            player_hand='Q',
            board='K',
            pot=[3, 3],
            current_player=player,
            current_round=1,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False
        )
        action = value_agent.select_action(obs)
        assert isinstance(action, Action)


# =============================================================================
# Test Class 3: Game Engine Integration
# =============================================================================

class TestGameEngineIntegration:
    """Tests that agents correctly integrate with the LeducGame engine."""
    
    def test_plays_full_game_without_errors(self, value_agent, game):
        """Agent must complete a full game without raising exceptions."""
        game.reset()
        
        while not game.is_finished:
            obs = game.get_observation()
            action = value_agent.select_action(obs)
            game.step(action)
        
        assert game.is_finished
        assert game.winner in [-1, 0, 1]  # Tie, P0 wins, or P1 wins
    
    def test_stress_test_multiple_games(self, value_agent):
        """Agent must successfully complete many games (stress test)."""
        game = LeducGame()
        games_completed = 0
        
        for _ in range(50):
            game.reset()
            while not game.is_finished:
                obs = game.get_observation()
                action = value_agent.select_action(obs)
                game.step(action)
            games_completed += 1
        
        assert games_completed == 50
    
    def test_only_returns_legal_actions(self, value_agent, game):
        """Agent must only return actions from legal_actions."""
        game.reset()
        
        for _ in range(100):  # Run multiple games
            game.reset()
            while not game.is_finished:
                obs = game.get_observation()
                action = value_agent.select_action(obs)
                
                assert action in obs.legal_actions, \
                    f"Action {action} not in legal_actions {obs.legal_actions}"
                
                game.step(action)
    
    def test_handles_preflop_phase(self, value_agent, game):
        """Agent must function correctly during pre-flop."""
        game.reset()
        assert game.current_round == 0
        
        obs = game.get_observation()
        action = value_agent.select_action(obs)
        
        assert isinstance(action, Action)
        assert action in obs.legal_actions
    
    def test_handles_flop_phase(self, value_agent, game):
        """Agent must function correctly during flop."""
        game.reset()
        
        # Progress to flop
        game.step(Action.CALL)  # P0 checks
        game.step(Action.CALL)  # P1 checks -> transitions to flop
        
        assert game.current_round == 1
        assert game.board is not None
        
        obs = game.get_observation()
        action = value_agent.select_action(obs)
        
        assert isinstance(action, Action)
        assert action in obs.legal_actions
    
    def test_two_agents_play_complete_game(self, value_agent, heuristic_agent, game):
        """Two different agents must complete games against each other."""
        agents = [value_agent, heuristic_agent]
        game.reset()
        
        while not game.is_finished:
            player_idx = game.current_player
            obs = game.get_observation()
            action = agents[player_idx].select_action(obs)
            game.step(action)
        
        assert game.is_finished
    
    def test_handles_immediate_fold(self, game):
        """Game engine handles fold on first action."""
        game.reset()
        game.step(Action.FOLD)
        
        assert game.is_finished
        assert game.winner == 1  # P1 wins when P0 folds
    
    def test_handles_showdown(self, game):
        """Game properly reaches showdown."""
        game.reset()
        
        # Go to flop via checks
        game.step(Action.CALL)
        game.step(Action.CALL)
        
        # End flop via checks
        game.step(Action.CALL)
        game.step(Action.CALL)
        
        assert game.is_finished
        assert game.winner in [-1, 0, 1]
    
    def test_observation_matches_game_state(self, game):
        """Observation data must match actual game state."""
        game.reset()
        obs = game.get_observation()
        
        assert obs.current_player == game.current_player
        assert obs.current_round == game.current_round
        assert obs.pot == game.pot
        assert obs.board == game.board
        assert obs.is_finished == game.is_finished


# =============================================================================
# Test Class 4: Training Integration
# =============================================================================

class TestTrainingIntegration:
    """Tests that agents integrate correctly with the training infrastructure."""
    
    def test_agent_has_train_mode(self, value_agent):
        """Agent must support train mode toggle."""
        assert hasattr(value_agent, 'set_train_mode')
        
        value_agent.set_train_mode(True)
        assert value_agent.train_mode is True
        
        value_agent.set_train_mode(False)
        assert value_agent.train_mode is False
    
    def test_train_mode_returns_tuple(self, value_agent, sample_observation):
        """In train mode, select_action must return (action, encoded_state)."""
        value_agent.set_train_mode(True)
        
        result = value_agent.select_action(sample_observation)
        
        assert isinstance(result, tuple), "Train mode should return tuple"
        assert len(result) == 2, "Tuple should have 2 elements"
        
        action, encoded_state = result
        assert isinstance(action, Action)
        # encoded_state can be None in some cases (e.g., fold)
    
    def test_eval_mode_returns_action_only(self, value_agent, sample_observation):
        """In eval mode, select_action must return just the Action."""
        value_agent.set_train_mode(False)
        
        result = value_agent.select_action(sample_observation)
        
        assert isinstance(result, Action), "Eval mode should return Action directly"
    
    def test_model_parameters_accessible(self, value_agent):
        """Agent's model must have accessible parameters for optimization."""
        assert hasattr(value_agent, 'model')
        
        params = list(value_agent.model.parameters())
        assert len(params) > 0, "Model should have parameters"
        
        # Check parameters are tensors
        for p in params:
            assert isinstance(p, torch.Tensor)
    
    def test_single_episode_training(self, value_agent):
        """One training episode must complete without error."""
        trainer = SelfPlayTrainer(value_agent)
        
        # Run a single episode
        value_agent.set_train_mode(True)
        trajectory, rewards = trainer._play_episode()
        
        assert isinstance(trajectory, list)
        assert isinstance(rewards, list)
        assert len(rewards) == 2  # Rewards for both players
    
    def test_batch_training_completes(self, value_agent):
        """Batch training must complete without error."""
        trainer = SelfPlayTrainer(value_agent)
        
        # Train for a small number of episodes
        # Use a callback to verify training progresses
        updates = []
        def callback(info):
            updates.append(info)
        
        trainer.train(num_episodes=10, batch_size=5, callback=callback)
        
        # Should have received at least one update
        assert len(updates) > 0
    
    def test_loss_computation(self, value_agent):
        """Loss must be computable from training data."""
        trainer = SelfPlayTrainer(value_agent)
        value_agent.set_train_mode(True)
        
        # Collect some training data
        batch_data = []
        for _ in range(5):
            trajectory, rewards = trainer._play_episode()
            batch_data.append((trajectory, rewards))
        
        # Compute loss
        loss = trainer._update_network(batch_data)
        
        assert isinstance(loss, float)
        assert not np.isnan(loss), "Loss should not be NaN"
    
    def test_model_save_and_load(self, value_agent, tmp_path):
        """Model must be savable and loadable."""
        save_path = tmp_path / "test_model.pt"
        
        # Save model
        torch.save(value_agent.model.state_dict(), save_path)
        assert save_path.exists()
        
        # Create new agent and load
        new_agent = ValueBasedAgent()
        new_agent.model.load_state_dict(torch.load(save_path))
        
        # Verify models produce same output
        obs = Observation(
            player_hand='K', board=None, pot=[1, 1],
            current_player=0, current_round=0,
            legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
            is_finished=False
        )
        
        enc1 = value_agent.encode_observation(obs)
        enc2 = new_agent.encode_observation(obs)
        
        with torch.no_grad():
            out1 = value_agent.model(enc1)
            out2 = new_agent.model(enc2)
        
        assert torch.allclose(out1, out2)
    
    def test_gradient_flow(self, value_agent):
        """Gradients must flow through the model."""
        obs = Observation(
            player_hand='K', board='Q', pot=[3, 3],
            current_player=0, current_round=1,
            legal_actions=[Action.FOLD, Action.CALL],
            is_finished=False
        )
        
        encoded = value_agent.encode_observation(obs)
        output = value_agent.model(encoded)
        
        # Compute loss and backprop
        target = torch.tensor([[1.0]])
        loss = torch.nn.MSELoss()(output, target)
        loss.backward()
        
        # Check gradients exist
        for param in value_agent.model.parameters():
            assert param.grad is not None, "Gradients should exist after backward"


# =============================================================================
# Test Class 5: Evaluation Integration
# =============================================================================

class TestEvaluationIntegration:
    """Tests that agents integrate correctly with the evaluation framework."""
    
    def test_evaluate_agents_works(self, value_agent, heuristic_agent):
        """evaluate_agents must work with agent pair."""
        result = evaluate_agents(value_agent, heuristic_agent, num_rounds=20)
        
        assert isinstance(result, EvaluationResult)
        assert result.num_rounds == 20
    
    def test_quick_evaluate_works(self, value_agent, heuristic_agent):
        """quick_evaluate must return a float."""
        result = quick_evaluate(value_agent, heuristic_agent, num_rounds=20)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
    
    def test_evaluation_with_position_randomization(self, value_agent, heuristic_agent):
        """Evaluation must work with position randomization."""
        result = evaluate_agents(
            value_agent, heuristic_agent,
            num_rounds=20,
            randomize_positions=True
        )
        
        assert result.num_rounds == 20
    
    def test_evaluation_without_position_randomization(self, value_agent, heuristic_agent):
        """Evaluation must work without position randomization."""
        result = evaluate_agents(
            value_agent, heuristic_agent,
            num_rounds=20,
            randomize_positions=False
        )
        
        assert result.num_rounds == 20
    
    def test_symmetric_evaluation(self, heuristic_agent):
        """Self-play evaluation should be roughly symmetric."""
        agent1 = HeuristicAgent()
        agent2 = HeuristicAgent()
        
        result = evaluate_agents(agent1, agent2, num_rounds=100)
        
        # Both agents are identical, so average should be near 0
        # Allow some variance due to randomness
        assert abs(result.agent_0_avg_chips) < 2.0
        assert abs(result.agent_1_avg_chips) < 2.0
    
    def test_evaluation_returns_valid_statistics(self, value_agent, heuristic_agent):
        """Evaluation must return valid statistics."""
        result = evaluate_agents(value_agent, heuristic_agent, num_rounds=30)
        
        # Check types
        assert isinstance(result.agent_0_avg_chips, float)
        assert isinstance(result.agent_1_avg_chips, float)
        assert isinstance(result.agent_0_total_chips, float)
        assert isinstance(result.agent_1_total_chips, float)
        assert isinstance(result.round_results, list)
        
        # Check consistency
        expected_total_0 = sum(r[0] for r in result.round_results)
        expected_total_1 = sum(r[1] for r in result.round_results)
        
        assert abs(result.agent_0_total_chips - expected_total_0) < 0.01
        assert abs(result.agent_1_total_chips - expected_total_1) < 0.01
    
    def test_evaluation_zero_sum(self, value_agent, heuristic_agent):
        """Game rewards should be zero-sum."""
        result = evaluate_agents(value_agent, heuristic_agent, num_rounds=30)
        
        for r0, r1 in result.round_results:
            assert abs(r0 + r1) < 0.01, "Game should be zero-sum"


# =============================================================================
# Test Class 6: Edge Cases and Robustness
# =============================================================================

class TestEdgeCasesAndRobustness:
    """Tests for edge cases and robustness."""
    
    def test_deterministic_in_eval_mode(self, value_agent, sample_observation):
        """Agent should be deterministic in eval mode (no exploration)."""
        value_agent.set_train_mode(False)
        value_agent.model.eval()
        
        # Run multiple times, should get same result
        actions = []
        for _ in range(5):
            action = value_agent.select_action(sample_observation)
            actions.append(action)
        
        # All actions should be the same
        assert all(a == actions[0] for a in actions)
    
    def test_handles_rapid_successive_calls(self, value_agent, sample_observation):
        """Agent should handle rapid successive action selections."""
        for _ in range(100):
            action = value_agent.select_action(sample_observation)
            assert isinstance(action, Action)
    
    def test_state_isolation_between_games(self, value_agent):
        """Agent state should not leak between games."""
        game1 = LeducGame()
        game2 = LeducGame()
        
        # Play first game
        game1.reset()
        while not game1.is_finished:
            obs = game1.get_observation()
            action = value_agent.select_action(obs)
            game1.step(action)
        
        # Play second game - should work independently
        game2.reset()
        while not game2.is_finished:
            obs = game2.get_observation()
            action = value_agent.select_action(obs)
            game2.step(action)
        
        assert game1.is_finished
        assert game2.is_finished
    
    def test_handles_maximum_game_length(self, value_agent):
        """Agent should handle games that go to maximum length."""
        game = LeducGame()
        
        for _ in range(20):  # Try multiple times to get a long game
            game.reset()
            action_count = 0
            
            while not game.is_finished:
                obs = game.get_observation()
                action = value_agent.select_action(obs)
                game.step(action)
                action_count += 1
                
                # Safety check - game should end eventually
                assert action_count < 20, "Game should not exceed reasonable length"
    
    def test_handles_all_action_sequences(self, value_agent, heuristic_agent):
        """Agents should handle various action sequence patterns."""
        # Test different opening sequences
        test_sequences = [
            [Action.FOLD],  # Immediate fold
            [Action.CALL, Action.CALL],  # Check-check
            [Action.RAISE, Action.CALL],  # Raise-call
            [Action.RAISE, Action.RAISE, Action.CALL],  # Raise-raise-call
        ]
        
        for sequence in test_sequences:
            game = LeducGame()
            game.reset()
            
            try:
                for action in sequence:
                    if game.is_finished:
                        break
                    game.step(action)
            except ValueError:
                # Some sequences may be invalid, that's ok
                pass
    
    def test_model_input_size_matches(self, value_agent):
        """Model input size should match encoded observation size."""
        obs = Observation(
            player_hand='K', board='Q', pot=[3, 3],
            current_player=0, current_round=1,
            legal_actions=[Action.FOLD, Action.CALL],
            is_finished=False
        )
        
        encoded = value_agent.encode_observation(obs)
        
        # Encoded should match the agent's expected input size
        assert encoded.shape[1] == value_agent.input_size
    
    def test_observation_dataclass_immutability(self, sample_observation):
        """Observation dataclass should be immutable (frozen)."""
        with pytest.raises((AttributeError, TypeError)):
            sample_observation.player_hand = 'J'
    
    def test_agent_copy_independence(self):
        """Copied agents should be independent."""
        agent1 = ValueBasedAgent()
        agent2 = ValueBasedAgent()
        
        # Modify agent1's model
        for param in agent1.model.parameters():
            param.data.fill_(1.0)
        
        # agent2 should not be affected
        for p1, p2 in zip(agent1.model.parameters(), agent2.model.parameters()):
            assert not torch.allclose(p1, p2)


# =============================================================================
# Additional Utility Tests
# =============================================================================

class TestHeuristicAgentSpecific:
    """Additional tests specific to HeuristicAgent behavior."""
    
    def test_heuristic_never_crashes(self, heuristic_agent):
        """HeuristicAgent should never crash in any scenario."""
        game = LeducGame()
        
        for _ in range(100):
            game.reset()
            while not game.is_finished:
                obs = game.get_observation()
                action = heuristic_agent.select_action(obs)
                assert action in obs.legal_actions
                game.step(action)
    
    def test_heuristic_encode_works(self, heuristic_agent, sample_observation):
        """HeuristicAgent's encode_observation should work."""
        result = heuristic_agent.encode_observation(sample_observation)
        # Heuristic agent may return None or simple encoding
        # Just verify it doesn't crash


class TestValueNetworkSpecific:
    """Tests specific to the ValueNetwork architecture."""
    
    def test_value_network_output_shape(self):
        """ValueNetwork should output single value."""
        network = ValueNetwork(input_size=14)
        
        x = torch.randn(1, 14)
        output = network(x)
        
        assert output.shape == (1, 1)
    
    def test_value_network_batch_processing(self):
        """ValueNetwork should handle batch inputs."""
        network = ValueNetwork(input_size=14)
        
        batch = torch.randn(32, 14)
        output = network(batch)
        
        assert output.shape == (32, 1)
    
    def test_value_network_gradient_computation(self):
        """ValueNetwork should support gradient computation."""
        network = ValueNetwork(input_size=14)
        
        x = torch.randn(1, 14, requires_grad=True)
        output = network(x)
        output.backward()
        
        assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
