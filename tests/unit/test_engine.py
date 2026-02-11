import unittest
from src.engine.leduc_game import Action, LeducGame

class TestLeducGame(unittest.TestCase):
    def setUp(self):
        self.game = LeducGame()

    def test_initial_state(self):
        obs = self.game.reset()
        self.assertEqual(len(self.game.deck), 4)
        self.assertEqual(self.game.pot, [1, 1])
        self.assertEqual(self.game.current_round, 0)
        self.assertIn(self.game.player_hands[0], ['J', 'Q', 'K'])
        self.assertEqual(len(obs.legal_actions), 3) # Fold, Call, Raise

    def test_fold(self):
        self.game.reset()
        self.game.step(Action.FOLD)
        self.assertTrue(self.game.is_finished)
        self.assertEqual(self.game.winner, 1)
        self.assertEqual(self.game.get_reward(), [-1, 1])

    def test_check_check_transition(self):
        self.game.reset()
        self.game.step(Action.CALL) # P0 checks
        self.game.step(Action.CALL) # P1 checks
        self.assertEqual(self.game.current_round, 1)
        self.assertIsNotNone(self.game.board)
        self.assertEqual(self.game.pot, [1, 1])

    def test_raise_call_transition(self):
        self.game.reset()
        self.game.step(Action.RAISE) # P0 raises to 1+2=3
        self.game.step(Action.CALL)  # P1 calls (pot becomes 3,3)
        self.assertEqual(self.game.current_round, 1)
        self.assertEqual(self.game.pot, [3, 3])

    def test_max_raises(self):
        self.game.reset()
        self.game.step(Action.RAISE) # R1
        self.game.step(Action.RAISE) # R2
        obs, _, _, _ = self.game.step(Action.CALL) # Match R2
        self.assertEqual(self.game.current_round, 1)
        # Check that we can raise again in Flop
        self.assertIn(Action.RAISE, obs.legal_actions)

    def test_showdown_pair(self):
        # Force a state
        self.game.reset()
        self.game.player_hands = ['J', 'Q']
        self.game.board = 'J'
        self.game.current_round = 1
        self.game.pot = [5, 5]
        self.game._showdown()
        self.assertEqual(self.game.winner, 0) # P0 has pair of J

    def test_showdown_high_card(self):
        self.game.reset()
        self.game.player_hands = ['Q', 'J']
        self.game.board = 'K'
        self.game.current_round = 1
        self.game._showdown()
        self.assertEqual(self.game.winner, 0) # Q > J

    def test_showdown_tie(self):
        self.game.reset()
        self.game.player_hands = ['J', 'J']
        self.game.board = 'K'
        self.game.current_round = 1
        self.game._showdown()
        self.assertEqual(self.game.winner, -1)
        self.assertEqual(self.game.get_reward(), [0, 0])

if __name__ == "__main__":
    unittest.main()
