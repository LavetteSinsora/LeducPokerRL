
import torch
from src.agents.value_based import ValueBasedAgent
from src.training.value_based_trainer import SelfPlayTrainer
from src.engine.leduc_game import LeducGame, Action

def verify_sign_error():
    agent = ValueBasedAgent()
    trainer = SelfPlayTrainer(agent)
    
    # Simulate one episode manually to control the transitions
    # Player 0 moves, reaches state s' where it is Player 1's turn
    game = LeducGame()
    game.reset()
    
    p0_hand = game.player_hands[0]
    p1_hand = game.player_hands[1]
    
    print(f"P0 Hand: {p0_hand}, P1 Hand: {p1_hand}")
    
    obs = game.get_observation(viewer_id=0)
    # Force an action (e.g., CALL)
    action = Action.CALL
    
    # In trainer._play_episode:
    # action, encoded_state = self.agent.select_action(obs)
    # results = self.agent.get_action_evaluations(obs)
    # selected_encoded = results[idx]["encoded"]
    
    results = agent.get_action_evaluations(obs)
    call_eval = next(r for r in results if r["action"] == Action.CALL)
    encoded_state = call_eval["encoded"]
    
    # Check whose turn it is in encoded_state
    # encoded_state is a tensor. We can't easily check current_player from it without decoding,
    # but we can look at the LeducGame state after the step.
    
    game.step(Action.CALL)
    print(f"Current player after P0 CALL: {game.current_player}")
    
    # Trajectory would contain:
    trajectory = [TrajectoryStep(encoded_state=encoded_state, player_id=0)]
    
    # Let's say the game ends now (e.g., after more steps) and rewards are [10, -10]
    rewards = [10.0, -10.0]
    batch_data = [(trajectory, rewards)]
    
    # In _update_network:
    # target = torch.FloatTensor([rewards[step.player_id]])  # rewards[0] = 10
    # prediction = self.agent.model(step.encoded_state)      # V(s', player=1)
    
    target = rewards[0]
    print(f"Target for V(s', player=1): {target}")
    print(f"Reward for Player 1 (whose turn it is at s'): {rewards[1]}")
    
    if target != rewards[1]:
        print("!!! SIGN ERROR DETECTED !!!")
        print(f"V(s', player=1) is being trained to predict P0's reward ({target}) instead of P1's reward ({rewards[1]}).")
    else:
        print("No sign error detected.")

if __name__ == "__main__":
    verify_sign_error()
