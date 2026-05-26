import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine.leduc_game import Action
from src.engine.observation import Observation
from src.agents.value_based import ValueBasedAgent

def verify_agent(model_path):
    print("=" * 70)
    print("TD(0) AGENT VERIFICATION REPORT")
    print("=" * 70)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    agent = ValueBasedAgent(model_path=model_path)
    agent.set_train_mode(False)
    
    print(f"Checking Hand Strength Ordering (Pre-flop, Pot=[1,1])")
    for pos, pos_label in [(0, "P0/Button"), (1, "P1/BigBlind")]:
        results = []
        for card in ['J', 'Q', 'K']:
            obs = Observation(
                player_hand=card,
                board=None,
                pot=[1, 1],
                current_player=pos,
                current_round=0,
                legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
                is_finished=False
            )
            v, _ = agent._evaluate_state(obs, viewer_id=pos)
            results.append((card, v))
        
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"  {pos_label}: {' > '.join([f'{c}({v:+.2f})' for c, v in results])}")
        
    print("\nChecking Positional Advantage (Holding K, Pre-flop, Pot=[1,1])")
    obs0 = Observation(player_hand='K', board=None, pot=[1,1], current_player=0, current_round=0, legal_actions=[], is_finished=False)
    obs1 = Observation(player_hand='K', board=None, pot=[1,1], current_player=1, current_round=0, legal_actions=[], is_finished=False)
    v0, _ = agent._evaluate_state(obs0, viewer_id=0)
    v1, _ = agent._evaluate_state(obs1, viewer_id=1)
    print(f"  V(K) for P0: {v0:+.2f}")
    print(f"  V(K) for P1: {v1:+.2f}")
    print(f"  P1 Advantage: {v1 - v0:+.2f} (Expecting positive, as P1 acts second)")

    print("\nChecking Pair Mechanics (Round 1, Pot=[3,3])")
    for board in ['J', 'Q', 'K']:
        results = []
        for card in ['J', 'Q', 'K']:
            obs = Observation(
                player_hand=card,
                board=board,
                pot=[3, 3],
                current_player=0,
                current_round=1,
                legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
                is_finished=False
            )
            v, _ = agent._evaluate_state(obs, viewer_id=0)
            results.append((card, v, card == board))
        
        results.sort(key=lambda x: x[1], reverse=True)
        pair_str = lambda p: "*PAIR" if p else ""
        print(f"  Board={board}: {' > '.join([f'{c}({v:+.2f}{pair_str(p)})' for c, v, p in results])}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    verify_agent("models/value_agent.pt")
