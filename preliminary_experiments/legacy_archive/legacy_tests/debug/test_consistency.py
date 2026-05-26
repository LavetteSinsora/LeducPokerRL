
import torch
import numpy as np
from src.agents.value_based import ValueBasedAgent
from src.engine.leduc_game import Action, LeducGame, Observation

def test_inference_consistency():
    # User Design: model(state_after_action) predicts value for the ORIGINAL player.
    # Case: Agent is P0. Agent takes action leading to state where it's P1's turn.
    
    agent = ValueBasedAgent()
    
    # Mock the model to return a high value (e.g. 10.0)
    # This represents a "Good" result for P0.
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.val = 10.0
        def forward(self, x):
            return torch.tensor([[self.val]])
            
    agent.model = MockModel()
    
    # Create an observation where it's P0's turn
    obs = Observation(
        player_hand='J',
        board='Q',
        pot=[1, 1],
        current_player=0,
        current_round=0,
        legal_actions=[Action.CALL, Action.RAISE],
        is_finished=False
    )
    
    print("Testing consistency between Training and Selection...")
    print(f"User Design: Model returns Value for the acting player. V=10 means GOOD for P0.")
    
    # 1. Evaluate actions
    results = agent.get_action_evaluations(obs)
    
    for r in results:
        action_name = r["action"].name
        val_for_selection = r["value"]
        
        # Check if the turn switched in the simulated next_obs
        # We simulate the step manually to know
        sim = LeducGame()
        sim.set_state(obs)
        next_obs, reward, done, _ = sim.step(r["action"])
        
        turn_switched = next_obs.current_player != obs.current_player
        
        print(f"\nAction: {action_name}")
        print(f"  Done: {done}")
        print(f"  Turn Switched: {turn_switched}")
        print(f"  Model Output (v_model): 10.0")
        print(f"  Value used for Selection (val): {val_for_selection}")
        
        if turn_switched and not done:
            if val_for_selection == -10.0:
                print("  => Result: Value was NEGATED because turn switched.")
                print("  => CONTRADICTION: Agent thinks a state with V=10 (Good for P0) is -10 (Bad for P0).")
            else:
                print("  => Result: Value was NOT negated.")
        
        if done:
            print(f"  => Result: Terminal state value is {val_for_selection}")

if __name__ == "__main__":
    test_inference_consistency()
