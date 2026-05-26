from engine.leduc_game import Action, LeducGame


def test_game_reset_starts_in_preflop():
    game = LeducGame()
    assert game.current_round == 0
    assert game.current_player == 0
    assert game.is_finished is False


def test_game_step_records_history():
    game = LeducGame()
    action = game.get_legal_actions()[0]
    game.step(action)
    assert len(game.history) == 1
    assert game.history[0][1] == Action(action).name
