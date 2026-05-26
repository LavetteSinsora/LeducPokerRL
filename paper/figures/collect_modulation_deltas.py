"""
figures/collect_modulation_deltas.py
======================================
For each state in EV_variation_analysis/data.json × {tight_passive, loose_aggressive}:
  - V_base(s)          : from the frozen base inside FullModulationAgent (= value_based)
  - V_mod(s, stats_j)  : from full_modulation with 7-dim prototype stats, averaged over seeds
  - Δ(s, j)            : V_mod - V_base

Also records the ground-truth MC EV from data.json for each (state, opponent).

Output: figures/modulation_deltas.json
"""

import json
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from engine.observation import Observation
from paper.agents.full_modulation.agent import FullModulationAgent

DATA_PATH = os.path.join(ROOT, "paper", "ev_analysis", "data.json")
OUT_PATH  = os.path.join(HERE, "modulation_deltas.json")

SEEDS      = [0, 1, 2]
OPPONENTS  = ["tight_passive", "loose_aggressive", "maniac"]

# 7-dim prototype stats: [pf_fold, pf_raise, fl_raise, pf_fold_to_raise,
#                         fl_fold_to_raise, raise_after_raise, confidence]
PROTOTYPE_STATS = {
    "tight_passive":    [0.70, 0.10, 0.10, 0.70, 0.70, 0.05, 0.90],
    "loose_aggressive": [0.10, 0.70, 0.65, 0.15, 0.15, 0.65, 0.90],
    "maniac":           [0.05, 0.90, 0.90, 0.10, 0.10, 0.80, 0.90],
}


def load_agents():
    agents = []
    for seed in SEEDS:
        ckpt = os.path.join(
            ROOT, "paper", "agents", "full_modulation",
            "outputs", f"seed_{seed}", "checkpoint_final.pt",
        )
        if not os.path.isfile(ckpt):
            print(f"  WARNING: missing {ckpt}")
            continue
        a = FullModulationAgent()
        a.load_model(ckpt)
        a.set_train_mode(False)
        agents.append(a)
    return agents


def build_obs(rec: dict) -> Observation:
    """Reconstruct Observation from a data.json state record."""
    return Observation(
        player_hand=rec["hand0"],
        board=rec["board"],
        pot=list(rec["pot"]),
        current_player=rec["current_player"],
        current_round=rec["round"],
        legal_actions=[],
        is_finished=False,
        raises_this_round=rec["raises"],
        opponent_stats=None,
    )


def main():
    print(f"Loading {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)

    # ── ground truth EV: {state_id: {opponent: ev}} ───────────────────────────
    gt_ev = {}
    for rec in data["records"]:
        sid = rec["state_id"]
        opp = rec["opponent"]
        if opp not in OPPONENTS:
            continue
        gt_ev.setdefault(sid, {})[opp] = rec["ev"]

    # ── unique state metadata ──────────────────────────────────────────────────
    state_meta = {}
    for rec in data["records"]:
        sid = rec["state_id"]
        if sid not in state_meta:
            state_meta[sid] = rec
    state_ids = [sid for sid in state_meta if all(opp in gt_ev.get(sid, {}) for opp in OPPONENTS)]
    print(f"  {len(state_ids)} states with GT EV for all opponents")

    # ── load models ───────────────────────────────────────────────────────────
    agents = load_agents()
    print(f"  Loaded {len(agents)} full_modulation seeds")

    # ── compute Δ per state × opponent × seed ─────────────────────────────────
    records = []
    for sid in state_ids:
        meta = state_meta[sid]
        obs  = build_obs(meta)
        cp   = meta["current_player"]

        game_enc = agents[0]._encode_game(obs, viewer_id=cp).unsqueeze(0)  # (1,15)

        # V_base is the same across seeds (frozen)
        with torch.no_grad():
            v_base = agents[0].base(game_enc).item()

        for opp in OPPONENTS:
            stats_vec = torch.tensor(PROTOTYPE_STATS[opp], dtype=torch.float32).unsqueeze(0)

            deltas = []
            for agent in agents:
                mod_inp = torch.cat([game_enc, stats_vec], dim=1)
                with torch.no_grad():
                    delta = agent.mod(mod_inp).item()
                deltas.append(delta)

            records.append({
                "state_id":     sid,
                "hand":         meta["hand0"],
                "board":        meta["board"],
                "round":        meta["round"],
                "pot":          meta["pot"],
                "raises":       meta["raises"],
                "opponent":     opp,
                "gt_ev":        gt_ev[sid][opp],
                "gt_ev_spread": round(max(gt_ev[sid].values()) - min(gt_ev[sid].values()), 6),
                "v_base":       round(v_base, 6),
                "delta_mean":   round(float(np.mean(deltas)), 6),
                "delta_std":    round(float(np.std(deltas, ddof=0)), 6),
                "delta_seeds":  [round(d, 6) for d in deltas],
            })

    output = {
        "metadata": {
            "n_states": len(state_ids),
            "opponents": OPPONENTS,
            "seeds": SEEDS,
            "n_records": len(records),
            "prototype_stats": PROTOTYPE_STATS,
        },
        "records": records,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {OUT_PATH}  ({len(records)} records)")


if __name__ == "__main__":
    main()
