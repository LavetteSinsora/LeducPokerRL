"""
compute_predictions.py — Network-predicted EVs for the modulated_value agent.

For each of the 486 states in data.json × each of the 6 rule-based opponents:
  1. Builds a mock Observation with game-state fields from data.json
  2. Attaches a FrozenStats object carrying the pre-collected 4-dim stats
  3. Calls ModulatedValueAgent._get_value(obs, viewer_id) to get a prediction

Also records the value_based baseline prediction (no opponent stats) for each
state, enabling a direct apples-to-apples comparison.

Input:
    OpponentModeling/EV_variation_analysis/data.json          (ground truth)
    modulated_value_agent_analysis/opponent_stats.json        (from collect_stats.py)

Output:
    modulated_value_agent_analysis/predictions.json
    {
      "metadata": {...},
      "records": [
        {
          "state_id": "pf_JQ_p1-1_cp0_r0",
          "opponent": "tight_passive",
          "modulated_pred": 0.23,
          "value_based_pred": 0.18
        }, ...
      ]
    }

Run from project root:
    python -m preliminary_experiments.ev_variation_extras.modulated_value_agent_analysis.compute_predictions
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from engine.observation import Observation
from engine.leduc_game import Action
from preliminary_experiments.promoted_registry.modulated_value.agent import ModulatedValueAgent
from agents.value_based.agent import ValueBasedAgent

DATA_PATH        = os.path.join(ROOT, "paper", "ev_analysis", "data.json")
STATS_PATH       = os.path.join(HERE, "opponent_stats.json")
OUTPUT_PATH      = os.path.join(HERE, "predictions.json")
MOD_CKPT         = os.path.join(ROOT, "agents", "modulated_value", "checkpoint.pt")
VB_CKPT          = os.path.join(ROOT, "agents", "value_based", "checkpoint.pt")

RULE_BASED_KEYS  = [
    "tight_passive",
    "tight_aggressive",
    "loose_passive",
    "loose_aggressive",
    "maniac",
    "random",
]


class FrozenStats:
    """
    Minimal stand-in for OpponentStats that returns a pre-computed vector.
    Compatible with ModulatedValueAgent._encode_stats() which calls
    obs.opponent_stats.to_feature_vector().
    """
    def __init__(self, vec: list):
        self._vec = list(vec)

    def to_feature_vector(self) -> list:
        return self._vec


def _build_obs(rec: dict, stats_obj) -> Observation:
    """
    Construct an Observation from a data.json state record.

    We set is_finished=False and legal_actions=[] because _get_value() only
    needs the state encoding, not action enumeration.
    The hand from the acting player's perspective is rec['hand0']
    (the value agent always plays as player 0 in the original collection).
    """
    return Observation(
        player_hand=rec["hand0"],
        board=rec["board"],           # None for pre-flop
        pot=list(rec["pot"]),
        current_player=rec["current_player"],
        current_round=rec["round"],
        legal_actions=[],             # not needed for value query
        is_finished=False,
        raises_this_round=rec["raises"],
        opponent_stats=stats_obj,
    )


def main():
    # --- Load inputs ---
    print(f"Loading data from {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)

    # De-duplicate records by state_id (each state_id appears once per opponent;
    # we need the state metadata, not the opponent-specific EV)
    state_meta = {}
    for rec in data["records"]:
        sid = rec["state_id"]
        if sid not in state_meta:
            state_meta[sid] = {
                "hand0":          rec["hand0"],
                "hand1":          rec["hand1"],
                "board":          rec["board"],
                "pot":            rec["pot"],
                "current_player": rec["current_player"],
                "round":          rec["round"],
                "raises":         rec["raises"],
            }
    state_ids = list(state_meta.keys())
    print(f"  {len(state_ids)} unique states loaded.")

    print(f"Loading opponent stats from {STATS_PATH}")
    with open(STATS_PATH) as f:
        opp_stats_raw = json.load(f)

    # --- Load models ---
    print(f"Loading ModulatedValueAgent from {MOD_CKPT}")
    mod_agent = ModulatedValueAgent(model_path=MOD_CKPT)
    mod_agent.set_train_mode(False)

    print(f"Loading ValueBasedAgent from {VB_CKPT}")
    vb_agent = ValueBasedAgent(model_path=VB_CKPT)
    vb_agent.set_train_mode(False)

    # --- Compute predictions ---
    records = []
    n_total = len(state_ids) * len(RULE_BASED_KEYS)
    print(f"\nComputing predictions for {len(state_ids)} states × {len(RULE_BASED_KEYS)} opponents "
          f"= {n_total} entries ...")

    for sid in state_ids:
        meta = state_meta[sid]
        cp   = meta["current_player"]

        # Value-based baseline (no opponent stats) — same for all opponents
        base_obs = _build_obs(meta, stats_obj=None)
        vb_pred  = vb_agent._get_value(base_obs, viewer_id=cp)

        for opp_key in RULE_BASED_KEYS:
            stats_vec  = opp_stats_raw[opp_key]
            frozen     = FrozenStats(stats_vec)
            mod_obs    = _build_obs(meta, stats_obj=frozen)
            mod_pred   = mod_agent._get_value(mod_obs, viewer_id=cp)

            records.append({
                "state_id":       sid,
                "opponent":       opp_key,
                "modulated_pred": round(float(mod_pred), 6),
                "value_based_pred": round(float(vb_pred), 6),
            })

    print(f"  Done. {len(records)} records generated.")

    output = {
        "metadata": {
            "description": (
                "Per-state network predictions from ModulatedValueAgent and ValueBasedAgent. "
                "modulated_pred = V_base + gate * Delta conditioned on each opponent's 4-dim stats. "
                "value_based_pred = V_base alone (no opponent conditioning)."
            ),
            "n_states":            len(state_ids),
            "n_opponents":         len(RULE_BASED_KEYS),
            "n_records":           len(records),
            "opponent_keys":       RULE_BASED_KEYS,
            "modulated_checkpoint": MOD_CKPT,
            "value_based_checkpoint": VB_CKPT,
            "stats_source":        STATS_PATH,
        },
        "records": records,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved predictions → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
