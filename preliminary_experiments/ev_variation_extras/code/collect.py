"""
collect.py — state enumeration and Monte Carlo EV data collection.

Enumerates all 486 fully-specified Leduc Hold'em game states (both player
hands fixed), runs MC rollouts of a trained ValueBasedAgent against 8
opponent agents, and writes the results to data.json.

Run from project root:
    python -m preliminary_experiments.ev_variation_extras.code.collect
    python -m preliminary_experiments.ev_variation_extras.code.collect --smoke
"""

import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

from agents.cfr.agent import CFRAgent
from agents.value_based.agent import ValueBasedAgent
from agents.rule_based import ALL_AGENTS
from preliminary_experiments.ev_variation_extras.code.sim_engine import FixedStateSimulator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
N_ROLLOUTS = 200

VALUE_CKPT  = "agents/value_based/checkpoint.pt"
CFR_CKPT    = "agents/cfr/checkpoint.pt"
OUTPUT_PATH = "OpponentModeling/EV_variation_analysis/data.json"

RANKS = ['J', 'Q', 'K']
HAND_PAIRS = [(h0, h1) for h0 in RANKS for h1 in RANKS]

OPPONENT_KEYS = [
    'tight_passive',
    'tight_aggressive',
    'loose_passive',
    'loose_aggressive',
    'maniac',
    'random',
    'value_based',
    'cfr',
]

# ---------------------------------------------------------------------------
# Pre-flop action configs: (pot, current_player, raises)
#
# Derived from Leduc Hold'em mechanics (antes=1 each, pre-flop bet=2):
#   Initial: pot=[1,1], cp=0, r=0
#   After P0 checks (calls with equal pots): pot=[1,1], cp=1, r=0
#   After P0 raises (pot[0] = pot[1]+2=3): pot=[3,1], cp=1, r=1
#   After P0 raises, P1 re-raises (pot[1] = pot[0]+2=5): pot=[3,5], cp=0, r=2
#   After P0 checks, P1 raises (pot[1] = pot[0]+2=3): pot=[1,3], cp=0, r=1
#   After P0 checks, P1 raises, P0 re-raises (pot[0] = pot[1]+2=5): pot=[5,3], cp=1, r=2
# ---------------------------------------------------------------------------

PREFLOP_CONFIGS = [
    ([1, 1], 0, 0),   # initial
    ([1, 1], 1, 0),   # P0 checked
    ([3, 1], 1, 1),   # P0 raised
    ([3, 5], 0, 2),   # P0 raised, P1 re-raised
    ([1, 3], 0, 1),   # P0 checked, P1 raised
    ([5, 3], 1, 2),   # P0 checked, P1 raised, P0 re-raised
]

# Flop entry pots: possible [X,X] values at the start of the flop round
# (from pre-flop: both check=[1,1], one raise+call=[3,3], both raise=[5,5])
FLOP_ENTRY_POTS = [1, 3, 5]


def _flop_configs(X: int) -> list:
    """
    6 mid-flop action states for flop entry pot [X, X] (flop bet = 4).
    Mirrors the structure of PREFLOP_CONFIGS but with bet_amount=4.
    """
    return [
        ([X,     X    ], 0, 0),   # P0 acts first
        ([X,     X    ], 1, 0),   # P0 checked
        ([X+4,   X    ], 1, 1),   # P0 raised  (pot[0] = X+4)
        ([X+4,   X+8  ], 0, 2),   # P0 raised, P1 re-raised  (pot[1] = X+4+4)
        ([X,     X+4  ], 0, 1),   # P0 checked, P1 raised  (pot[1] = X+4)
        ([X+8,   X+4  ], 1, 2),   # P0 checked, P1 raised, P0 re-raised  (pot[0] = X+4+4)
    ]


def _valid_boards(hand0: str, hand1: str) -> list:
    """
    Returns the sorted list of card ranks that can legally appear as the
    board card given that hand0 and hand1 have already been dealt.

    A rank is valid if not all copies of it appear in {hand0, hand1}.
    Leduc deck: 2×J, 2×Q, 2×K.
    """
    remaining = ['J', 'J', 'Q', 'Q', 'K', 'K']
    remaining.remove(hand0)
    remaining.remove(hand1)
    return sorted(set(remaining))


def _make_state_id(rnd, hand0, hand1, board, pot, cp, raises) -> str:
    """
    Human-readable state identifier.
    Pre-flop:  "pf_{h0}{h1}_p{p0}-{p1}_cp{cp}_r{r}"
    Flop:      "fl_{h0}{h1}{board}_p{p0}-{p1}_cp{cp}_r{r}"
    """
    p = f"p{pot[0]}-{pot[1]}"
    if rnd == 0:
        return f"pf_{hand0}{hand1}_{p}_cp{cp}_r{raises}"
    return f"fl_{hand0}{hand1}{board}_{p}_cp{cp}_r{raises}"


# ---------------------------------------------------------------------------
# State enumeration
# ---------------------------------------------------------------------------

def enumerate_states() -> list:
    """
    Returns all 486 fully-specified game state descriptors.

    Pre-flop: 9 hand pairs × 6 action configs = 54
    Flop:    24 (hand0, hand1, board) triples × 3 entry pots × 6 action configs = 432

    Each descriptor is a dict with keys:
        state_id, round, hand0, hand1, board, pot, current_player, raises
    """
    states = []

    # -- Pre-flop (round=0) --
    for hand0, hand1 in HAND_PAIRS:
        for pot, cp, raises in PREFLOP_CONFIGS:
            sid = _make_state_id(0, hand0, hand1, None, pot, cp, raises)
            states.append({
                "state_id":       sid,
                "round":          0,
                "hand0":          hand0,
                "hand1":          hand1,
                "board":          None,
                "pot":            list(pot),
                "current_player": cp,
                "raises":         raises,
            })

    # -- Flop (round=1) --
    for hand0, hand1 in HAND_PAIRS:
        for board in _valid_boards(hand0, hand1):
            for X in FLOP_ENTRY_POTS:
                for pot, cp, raises in _flop_configs(X):
                    sid = _make_state_id(1, hand0, hand1, board, pot, cp, raises)
                    states.append({
                        "state_id":       sid,
                        "round":          1,
                        "hand0":          hand0,
                        "hand1":          hand1,
                        "board":          board,
                        "pot":            list(pot),
                        "current_player": cp,
                        "raises":         raises,
                    })

    return states


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(n_rollouts: int = N_ROLLOUTS, max_states: int = None):
    """
    Enumerate states, run MC rollouts, write data.json.

    Args:
        n_rollouts: Rollouts per (state, opponent) bucket.
        max_states: If set, only process the first N states (for smoke tests).
    """
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading agents...")
    value_agent = ValueBasedAgent(model_path=VALUE_CKPT)
    value_agent.set_train_mode(False)

    opponents = {}
    for key, cls in ALL_AGENTS.items():
        opponents[key] = cls()
    opponents['value_based'] = ValueBasedAgent(model_path=VALUE_CKPT)
    opponents['value_based'].set_train_mode(False)
    opponents['cfr'] = CFRAgent(model_path=CFR_CKPT)

    states = enumerate_states()
    if max_states is not None:
        states = states[:max_states]

    n_states = len(states)
    n_opps   = len(OPPONENT_KEYS)
    print(f"States:    {n_states}")
    print(f"Opponents: {OPPONENT_KEYS}")
    print(f"Rollouts per bucket: ~{n_rollouts}")
    print(f"Total (state, opponent) pairs: {n_states * n_opps:,}")
    print()

    records = []
    total_pairs = n_states * n_opps
    done = 0
    t0 = time.time()

    for state in states:
        sim = FixedStateSimulator(
            hand0=state['hand0'],
            hand1=state['hand1'],
            pot=state['pot'],
            current_player=state['current_player'],
            rnd=state['round'],
            raises=state['raises'],
            board=state['board'],
        )

        for opp_key in OPPONENT_KEYS:
            opp_agent = opponents[opp_key]
            ev, ev_std, n = sim.rollout(value_agent, opp_agent, n_rollouts)

            records.append({
                "state_id":       state['state_id'],
                "round":          state['round'],
                "hand0":          state['hand0'],
                "hand1":          state['hand1'],
                "board":          state['board'],
                "pot":            state['pot'],
                "current_player": state['current_player'],
                "raises":         state['raises'],
                "opponent":       opp_key,
                "ev":             round(ev, 5),
                "ev_std":         round(ev_std, 5),
                "n":              n,
            })

            done += 1
            if done % 200 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total_pairs - done) / rate
                print(f"  {done}/{total_pairs} ({100*done/total_pairs:.1f}%) "
                      f"— {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    # Build output
    output = {
        "metadata": {
            "date":                         datetime.now().strftime("%Y-%m-%d"),
            "n_states":                     n_states,
            "n_opponents":                  n_opps,
            "n_rollouts_per_bucket":        n_rollouts,
            "total_records":                len(records),
            "evaluating_agent":             "value_based",
            "evaluating_agent_checkpoint":  VALUE_CKPT,
            "cfr_checkpoint":               CFR_CKPT,
            "opponent_keys":                OPPONENT_KEYS,
            "seed":                         SEED,
            "board_sampling":               "proportional_card_removal_preflop_only",
            "note": (
                "EV is from the perspective of current_player (value agent seat). "
                "Pre-flop rollouts are distributed across board cards by card-removal "
                "probability to remove board-card randomness from the EV estimate."
            ),
        },
        "records": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {len(records)} records → {OUTPUT_PATH}")
    print(f"Total time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke:
        print("=== SMOKE TEST (5 rollouts, 5 states) ===")
        collect(n_rollouts=5, max_states=5)
    else:
        collect()
