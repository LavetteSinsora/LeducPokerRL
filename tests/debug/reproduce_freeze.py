
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.agents.value_based import ValueBasedAgent

def reproduce_freeze():
    print("Starting reproduction script...")
    
    # 1. Setup Agent
    agent = ValueBasedAgent()
    
    # 2. Construct a Flop Observation
    # Scenario: Agent is Player 1 (second to act), Round is Flop (1)
    # Board card is present.
    # Check if this causes a loop or crash in select_action
    
    obs = Observation(
        player_hand='Q',
        board='K',
        pot=[2, 2], # 4 chips each after pre-flop
        current_player=1,
        current_round=1, # Flop
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        is_finished=False
    )
    
    print(f"Testing select_action with observation: {obs}")
    
    try:
        action = agent.select_action(obs)
        print(f"Agent selected action (Flop Start): {action}")
    except Exception as e:
        print(f"Caught exception in Flop Start: {e}")
        import traceback
        traceback.print_exc()

    # Case 2: Flop, Player 0 checked. Agent (Player 1) can Check (Call) to Showdown.
    obs_showdown = Observation(
        player_hand='J',
        board='Q',
        pot=[4, 4], 
        current_player=1,
        current_round=1,
        legal_actions=[Action.CALL, Action.RAISE], # Call here means Check -> Showdown
        is_finished=False
    )
    print(f"\nTesting Showdown Simulation with obs: {obs_showdown}")
    try:
        action = agent.select_action(obs_showdown)
        print(f"Agent selected action (Potential Showdown): {action}")
    except Exception as e:
        print(f"Caught exception in Showdown Test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce_freeze()
