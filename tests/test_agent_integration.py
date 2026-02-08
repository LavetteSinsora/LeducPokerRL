from src.engine.leduc_game import LeducGame
from src.agents import ValueBasedAgent

def test_integration():
    game = LeducGame()
    obs = game.reset()
    
    # Initialize two agents
    agents = [ValueBasedAgent(), ValueBasedAgent()]
    
    print("Starting integration test...")
    print(f"Initial State: {game}")
    
    while not game.is_finished:
        player_idx = game.current_player
        agent = agents[player_idx]
        
        # Current observation for the acting player
        obs = game.get_observation()
        
        # Agent selects an action
        action = agent.select_action(obs)
        
        print(f"Player {player_idx} ({obs.player_hand}) chooses {action.name}")
        
        # Execute action in engine
        obs, reward, done, _ = game.step(action)
        
        if game.board:
            print(f"Board: {game.board}")
            
    print(f"Game Finished! Winner: {game.winner}")
    print(f"History: {game.history}")
    print(f"Pot: {game.pot}")
    print(f"Result: {game._get_reward()}")

if __name__ == "__main__":
    test_integration()
