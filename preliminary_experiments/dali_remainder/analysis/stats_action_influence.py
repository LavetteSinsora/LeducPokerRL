"""
Opponent-stats action influence analysis.

For every unique Leduc decision state (where the player has ≥2 legal actions):
  1. Query full_modulation with each of 6 prototype opponent-stat vectors.
  2. Compare to the action chosen with neutral stats (all 0.5) — the baseline.
  3. Record every "flip": a state+opponent pair where stats change the action.
  4. For each flip, check whether the stats-influenced action agrees better
     with the CFR Nash strategy (ground truth).

Outputs: analysis/results/stats_action_influence.json
"""

import json
import os
import sys
import random
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from engine.leduc_game import Action, LeducGame
from paper.agents.full_modulation.agent import FullModulationAgent
from agents.cfr.agent import CFRAgent

# ── config ─────────────────────────────────────────────────────────────────────
N_HANDS      = 5_000   # hands to simulate for state collection
SEEDS        = [0, 1, 2]

NEUTRAL_STATS = [0.5] * 7

PROTOTYPE_STATS = {
    "tight_passive":    [0.70, 0.10, 0.10, 0.70, 0.70, 0.05, 0.90],
    "tight_aggressive": [0.20, 0.65, 0.55, 0.30, 0.30, 0.55, 0.90],
    "loose_passive":    [0.15, 0.20, 0.15, 0.25, 0.20, 0.10, 0.90],
    "loose_aggressive": [0.10, 0.70, 0.65, 0.15, 0.15, 0.65, 0.90],
    "maniac":           [0.05, 0.90, 0.90, 0.10, 0.10, 0.80, 0.90],
    "random":           [0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.90],
}

OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def _hand_strength(card: str) -> str:
    return {"J": "weak", "Q": "medium", "K": "strong"}.get(card, "?")

def _has_pair(obs) -> bool:
    return obs.board is not None and obs.player_hand == obs.board

def _state_label(obs) -> str:
    """Human-readable state descriptor."""
    round_str = "preflop" if obs.current_round == 0 else "postflop"
    board_str = obs.board if obs.board else "none"
    pot_sum   = obs.pot[0] + obs.pot[1]
    pair_str  = "PAIR" if _has_pair(obs) else "no-pair"
    legal_str = "/".join(a.name for a in obs.legal_actions)
    return (f"{obs.player_hand}|board={board_str}|{round_str}|"
            f"pot={pot_sum}|{pair_str}|legal={legal_str}")

def _state_key(obs) -> tuple:
    """Hashable key for deduplication."""
    return (obs.player_hand, obs.board, obs.current_round,
            tuple(obs.pot), obs.raises_this_round, obs.current_player,
            tuple(a.value for a in obs.legal_actions))

def _greedy_action(agent: FullModulationAgent, obs, stats: list) -> Action:
    """Greedy (no temperature) action from full_modulation given stats."""
    results = agent._get_action_values(obs, np.array(stats, dtype=np.float32))
    return max(results, key=lambda r: r["value"])["action"]

def _cfr_best_action(cfr: CFRAgent, obs) -> Action:
    """CFR's highest-probability action (deterministic argmax of average strategy)."""
    evals = cfr.get_action_evaluations(obs)
    return max(evals, key=lambda e: e["probability"])["action"]

def _cfr_strategy(cfr: CFRAgent, obs) -> dict:
    """Returns {action: probability} dict."""
    evals = cfr.get_action_evaluations(obs)
    return {e["action"]: e["probability"] for e in evals}

# ── collect unique decision states ─────────────────────────────────────────────

def collect_states(n_hands: int) -> list:
    """
    Simulate n_hands hands (both players random), collect every unique
    decision state (obs where legal_actions ≥ 2) from both players' perspectives.
    Returns list of Observation objects (one per unique state key).
    """
    seen = {}
    for _ in range(n_hands):
        game = LeducGame()
        game.reset()
        while not game.is_finished:
            cp  = game.current_player
            obs = game.get_observation(viewer_id=cp)
            if len(obs.legal_actions) >= 2:
                key = _state_key(obs)
                if key not in seen:
                    seen[key] = obs
            # Random play to explore state space
            game.step(random.choice(obs.legal_actions))
    return list(seen.values())


# ── main analysis ──────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    torch.manual_seed(42)

    # Load CFR
    cfr = CFRAgent(model_path=os.path.join(ROOT, "agents", "cfr", "checkpoint.pt"))
    print(f"CFR loaded.")

    # Load full_modulation across all seeds
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
        agents.append((seed, a))
    print(f"Loaded {len(agents)} full_modulation seeds.")

    # Collect states
    print(f"Collecting unique decision states from {N_HANDS:,} random hands...")
    states = collect_states(N_HANDS)
    print(f"Found {len(states)} unique decision states.")

    # ── per-state analysis ─────────────────────────────────────────────────────
    # For each state:
    #   - neutral action (stats=0.5) for each seed
    #   - per-prototype action for each seed
    #   - CFR best action + full strategy

    flips     = []   # all cases where stats changed the action
    all_states_summary = []

    for obs in states:
        cfr_strat   = _cfr_strategy(cfr, obs)
        cfr_best    = max(cfr_strat, key=cfr_strat.get)
        label       = _state_label(obs)
        round_name  = "preflop" if obs.current_round == 0 else "postflop"
        strength    = _hand_strength(obs.player_hand)
        has_pair    = _has_pair(obs)

        state_record = {
            "label":    label,
            "hand":     obs.player_hand,
            "board":    obs.board,
            "round":    round_name,
            "strength": strength,
            "has_pair": has_pair,
            "pot":      list(obs.pot),
            "raises":   obs.raises_this_round,
            "legal":    [a.name for a in obs.legal_actions],
            "cfr_best": cfr_best.name,
            "cfr_probs": {a.name: round(p, 4) for a, p in cfr_strat.items()},
            "profiles": {},
            "flip_count": 0,
            "flip_toward_cfr": 0,
            "flip_away_cfr": 0,
        }

        for prof_name, prof_stats in PROTOTYPE_STATS.items():
            prof_record = {"flipped_seeds": [], "neutral_actions": [], "stats_actions": []}

            for seed, agent in agents:
                neutral_action = _greedy_action(agent, obs, NEUTRAL_STATS)
                stats_action   = _greedy_action(agent, obs, prof_stats)

                prof_record["neutral_actions"].append(neutral_action.name)
                prof_record["stats_actions"].append(stats_action.name)

                if stats_action != neutral_action:
                    prof_record["flipped_seeds"].append(seed)
                    state_record["flip_count"] += 1

                    # Did the flip move toward or away from CFR?
                    neutral_cfr_prob = cfr_strat.get(neutral_action, 0.0)
                    stats_cfr_prob   = cfr_strat.get(stats_action, 0.0)

                    toward_cfr = stats_cfr_prob > neutral_cfr_prob

                    flip_entry = {
                        "state":          label,
                        "hand":           obs.player_hand,
                        "board":          obs.board,
                        "round":          round_name,
                        "strength":       strength,
                        "has_pair":       has_pair,
                        "pot_sum":        obs.pot[0] + obs.pot[1],
                        "raises":         obs.raises_this_round,
                        "opponent":       prof_name,
                        "seed":           seed,
                        "neutral_action": neutral_action.name,
                        "stats_action":   stats_action.name,
                        "cfr_best":       cfr_best.name,
                        "neutral_cfr_prob": round(neutral_cfr_prob, 4),
                        "stats_cfr_prob":   round(stats_cfr_prob, 4),
                        "toward_cfr":     toward_cfr,
                    }
                    flips.append(flip_entry)

                    if toward_cfr:
                        state_record["flip_toward_cfr"] += 1
                    else:
                        state_record["flip_away_cfr"] += 1

            # Majority neutral and stats actions across seeds
            from collections import Counter
            prof_record["majority_neutral"] = Counter(prof_record["neutral_actions"]).most_common(1)[0][0]
            prof_record["majority_stats"]   = Counter(prof_record["stats_actions"]).most_common(1)[0][0]
            prof_record["any_flip"]         = len(prof_record["flipped_seeds"]) > 0

            state_record["profiles"][prof_name] = prof_record

        all_states_summary.append(state_record)

    # ── aggregate statistics ───────────────────────────────────────────────────
    total_decisions    = len(states) * len(PROTOTYPE_STATS) * len(agents)
    total_flips        = len(flips)
    flip_rate          = total_flips / total_decisions if total_decisions else 0

    toward_cfr_count   = sum(1 for f in flips if f["toward_cfr"])
    away_cfr_count     = total_flips - toward_cfr_count
    toward_cfr_rate    = toward_cfr_count / total_flips if total_flips else 0

    # By opponent profile
    by_opponent = defaultdict(lambda: {"flips": 0, "toward_cfr": 0})
    for f in flips:
        by_opponent[f["opponent"]]["flips"] += 1
        if f["toward_cfr"]:
            by_opponent[f["opponent"]]["toward_cfr"] += 1

    # By round
    by_round = defaultdict(lambda: {"flips": 0, "toward_cfr": 0})
    for f in flips:
        by_round[f["round"]]["flips"] += 1
        if f["toward_cfr"]:
            by_round[f["round"]]["toward_cfr"] += 1

    # By hand strength
    by_strength = defaultdict(lambda: {"flips": 0, "toward_cfr": 0})
    for f in flips:
        by_strength[f["strength"]]["flips"] += 1
        if f["toward_cfr"]:
            by_strength[f["strength"]]["toward_cfr"] += 1

    # By flip direction (neutral_action → stats_action)
    by_transition = defaultdict(int)
    for f in flips:
        by_transition[f"{f['neutral_action']}→{f['stats_action']}"] += 1

    # Most frequently flipped states
    states_by_flips = sorted(all_states_summary,
                             key=lambda s: s["flip_count"], reverse=True)[:20]

    # ── print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  OPPONENT STATS ACTION INFLUENCE — SUMMARY")
    print(f"{'='*65}")
    print(f"  Unique decision states:  {len(states)}")
    print(f"  Total state×opp×seed:   {total_decisions:,}")
    print(f"  Flips (stats≠neutral):  {total_flips:,}  ({flip_rate:.1%})")
    print(f"  Toward CFR:  {toward_cfr_count:,} ({toward_cfr_rate:.1%})")
    print(f"  Away from CFR: {away_cfr_count:,} ({1-toward_cfr_rate:.1%})")

    print(f"\n  By opponent profile:")
    for opp, d in sorted(by_opponent.items(), key=lambda x: -x[1]["flips"]):
        t = d["toward_cfr"]; n = d["flips"]
        print(f"    {opp:<22}  {n:4d} flips  toward_cfr={t/n:.1%}")

    print(f"\n  By round:")
    for rnd, d in by_round.items():
        t = d["toward_cfr"]; n = d["flips"]
        print(f"    {rnd:<12}  {n:4d} flips  toward_cfr={t/n:.1%}")

    print(f"\n  By hand strength:")
    for s, d in sorted(by_strength.items(), key=lambda x: -x[1]["flips"]):
        t = d["toward_cfr"]; n = d["flips"]
        print(f"    {s:<10}  {n:4d} flips  toward_cfr={t/n:.1%}")

    print(f"\n  Action transitions:")
    for trans, cnt in sorted(by_transition.items(), key=lambda x: -x[1]):
        print(f"    {trans:<20}  {cnt:4d}")

    print(f"\n  Top 10 most stats-sensitive states:")
    for s in states_by_flips[:10]:
        frac = f"{s['flip_toward_cfr']}/{s['flip_count']} toward_cfr"
        print(f"    [{s['flip_count']:2d} flips | {frac}] {s['label']}")

    # ── save results ───────────────────────────────────────────────────────────
    results = {
        "n_states":          len(states),
        "n_hands_simulated": N_HANDS,
        "seeds":             SEEDS,
        "total_decisions":   total_decisions,
        "total_flips":       total_flips,
        "flip_rate":         round(flip_rate, 4),
        "toward_cfr_count":  toward_cfr_count,
        "toward_cfr_rate":   round(toward_cfr_rate, 4),
        "by_opponent":       {k: dict(v) for k, v in by_opponent.items()},
        "by_round":          {k: dict(v) for k, v in by_round.items()},
        "by_strength":       {k: dict(v) for k, v in by_strength.items()},
        "by_transition":     dict(by_transition),
        "top_flipped_states": states_by_flips[:20],
        "all_flips":         flips,
    }

    # Convert numpy bools to Python bools for JSON serialization
    def _jsonify(obj):
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    out_path = os.path.join(OUT_DIR, "stats_action_influence.json")
    with open(out_path, "w") as f:
        json.dump(_jsonify(results), f, indent=2)
    print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    main()
