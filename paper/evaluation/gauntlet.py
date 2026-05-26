"""
paper — Gauntlet Evaluation
=======================================
Runs all study agents × pool agents matchups.

Study agents (17 instances total):
  value_based            (1 instance — baseline, no seed)
  full_modulation        seeds [0, 1, 2]
  state_only             seeds [0, 1, 2]
  finetuned_base         seeds [0, 1, 2]
  gated_modulation       seed  [0]
  value_based_deep       seeds [0, 1, 2]
  full_modulation_deep   seeds [0, 1, 2]
  value_based_pool       seeds [0, 1, 2]

Pool agents (13):
  cfr, heuristic, tight_passive, tight_aggressive,
  loose_passive, loose_aggressive, maniac, random,
  adaptive_value, opp_encoder_v1,
  reinforce, actor_critic, dqn

Protocol:
  - 10,000 rounds per matchup, position-alternated
  - session_length=100 hands; stats tracker reset at each session boundary
  - Per-hand JSONL: {hand_id, reward, cumulative, position, session,
                      hand_in_session, confidence}
  - Summary JSON per matchup: chips_per_round, ci_95_low, ci_95_high,
                               cold_mean, warm_mean, n_hands

Usage:
  # Single matchup:
  python -m paper.evaluation.gauntlet \\
      --study full_modulation --seed 0 --pool cfr

  # All matchups (sequential):
  python -m paper.evaluation.gauntlet --all

  # All matchups with N parallel workers:
  python -m paper.evaluation.gauntlet --all --workers 8
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from engine.leduc_game import LeducGame, Action
from paper.evaluation.shared.stats_tracker import (
    OpponentStatsTracker,
)
from paper.evaluation.pool import build_pool, STATS_INJECTED, POOL_AGENT_KEYS

# ── constants ────────────────────────────────────────────────────────────────

SESSION_LENGTH  = 100
PRIOR_STRENGTH  = 20.0
DEFAULT_ROUNDS  = 10_000
RESULTS_DIR     = os.path.join(HERE, "results")

# ── study agent registry ──────────────────────────────────────────────────────
# Each entry: (agent_name, seed_or_None, checkpoint_path, is_stat_aware)
#   is_stat_aware=True  → select_action(obs, opp_stats=stats_7dim)
#   is_stat_aware=False → select_action(obs)

def _study_agent_list(root: str) -> list[tuple]:
    """Return list of (agent_name, seed, ckpt_path, is_stat_aware) tuples."""
    entries = []

    # cfr: Nash equilibrium tabular agent — upper-bound reference
    cfr_path = os.path.join(root, "agents", "cfr", "checkpoint.pt")
    entries.append(("cfr", None, cfr_path, False))

    # value_based: canonical baseline, standard interface
    vb_path = os.path.join(root, "agents", "value_based", "checkpoint.pt")
    entries.append(("value_based", None, vb_path, False))

    # full_modulation: frozen base + ungated mod head
    for seed in [0, 1, 2]:
        ckpt = os.path.join(
            root, "paper", "agents", "full_modulation",
            "outputs", f"seed_{seed}", "checkpoint_final.pt",
        )
        entries.append(("full_modulation", seed, ckpt, True))

    # state_only: frozen base + state-only residual (ignores stats but accepts kwarg)
    for seed in [0, 1, 2]:
        ckpt = os.path.join(
            root, "paper", "agents", "ablations", "state_only",
            "outputs", f"seed_{seed}", "checkpoint_final.pt",
        )
        entries.append(("state_only", seed, ckpt, True))

    # finetuned_base: unfrozen base + mod head
    for seed in [0, 1, 2]:
        ckpt = os.path.join(
            root, "paper", "agents", "ablations", "finetuned_base",
            "outputs", f"seed_{seed}", "checkpoint_final.pt",
        )
        entries.append(("finetuned_base", seed, ckpt, True))

    # gated_modulation: frozen base + state-gated mod head
    ckpt = os.path.join(
        root, "preliminary_experiments", "dali_variants", "gated_modulation",
        "outputs", "seed_0", "checkpoint_final.pt",
    )
    entries.append(("gated_modulation", 0, ckpt, True))

    # value_based_deep: 15→64→64→64→1, depth ablation (standard interface)
    for seed in [0, 1, 2]:
        ckpt = os.path.join(
            root, "preliminary_experiments", "dali_variants", "value_based_deep",
            "outputs", f"seed_{seed}", "checkpoint_final.pt",
        )
        entries.append(("value_based_deep", seed, ckpt, False))

    # full_modulation_deep: frozen deep base + mod head (stat-aware)
    for seed in [0, 1, 2]:
        ckpt = os.path.join(
            root, "preliminary_experiments", "dali_variants", "full_modulation_deep",
            "outputs", f"seed_{seed}", "checkpoint_final.pt",
        )
        entries.append(("full_modulation_deep", seed, ckpt, True))

    # value_based_pool: shallow arch (15→64→64→1) + weighted pool recipe
    # Isolates training recipe effect from depth (compare vs value_based and value_based_deep)
    for seed in [0, 1, 2]:
        ckpt = os.path.join(
            root, "paper", "agents", "value_based_pool",
            "outputs", f"seed_{seed}", "checkpoint_final.pt",
        )
        entries.append(("value_based_pool", seed, ckpt, False))

    # scratch_joint: random-init base + mod head, jointly trained (no pretrained base)
    for seed in [0, 1, 2]:
        ckpt = os.path.join(
            root, "paper", "agents", "ablations", "scratch_joint",
            "outputs", f"seed_{seed}", "checkpoint_final.pt",
        )
        entries.append(("scratch_joint", seed, ckpt, True))

    # finetuned_base_only: extract ONLY the base network from finetuned_base checkpoints.
    # Answers: what has the base learned after drifting 0.41 chips/state from pretraining?
    # If it performs worse than value_based: base was corrupted by opponent-specific gradient noise.
    # If better: base absorbed a useful "pool-average" value function.
    for seed in [0, 1, 2]:
        ckpt = os.path.join(
            root, "paper", "agents", "ablations", "finetuned_base",
            "outputs", f"seed_{seed}", "checkpoint_final.pt",
        )
        entries.append(("finetuned_base_only", seed, ckpt, False))

    # Pool-trained RL baselines — evaluated as study agents for the comparison figure.
    # Strongest seeds selected by prior eval vs value_based (see pool.py comment).
    reinforce_v2_ckpt = os.path.join(
        root, "paper", "baselines", "reinforce",
        "outputs_v2", "seed_0", "checkpoint_final.pt",
    )
    if os.path.isfile(reinforce_v2_ckpt):
        entries.append(("reinforce_v2", 0, reinforce_v2_ckpt, False))

    actor_critic_v2_ckpt = os.path.join(
        root, "paper", "baselines", "actor_critic",
        "outputs_v2", "seed_1", "checkpoint_final.pt",
    )
    if os.path.isfile(actor_critic_v2_ckpt):
        entries.append(("actor_critic_v2", 1, actor_critic_v2_ckpt, False))

    dqn_v2_ckpt = os.path.join(
        root, "paper", "baselines", "dqn",
        "outputs_v2", "seed_2", "checkpoint_final.pt",
    )
    if os.path.isfile(dqn_v2_ckpt):
        entries.append(("dqn_v2", 2, dqn_v2_ckpt, False))

    # v3: pool-trained at 200K episodes — matched training budget to study agents
    reinforce_v3_ckpt = os.path.join(
        root, "paper", "baselines", "reinforce",
        "outputs_v3", "seed_0", "checkpoint_final.pt",
    )
    if os.path.isfile(reinforce_v3_ckpt):
        entries.append(("reinforce_v3", 0, reinforce_v3_ckpt, False))

    actor_critic_v3_ckpt = os.path.join(
        root, "paper", "baselines", "actor_critic",
        "outputs_v3", "seed_0", "checkpoint_final.pt",
    )
    if os.path.isfile(actor_critic_v3_ckpt):
        entries.append(("actor_critic_v3", 0, actor_critic_v3_ckpt, False))

    dqn_v3_ckpt = os.path.join(
        root, "paper", "baselines", "dqn",
        "outputs_v3", "seed_0", "checkpoint_final.pt",
    )
    if os.path.isfile(dqn_v3_ckpt):
        entries.append(("dqn_v3", 0, dqn_v3_ckpt, False))

    return entries


def _load_study_agent(agent_name: str, ckpt_path: str):
    """Load and return a study agent in eval mode."""
    if agent_name == "cfr":
        from agents.cfr.agent import CFRAgent
        agent = CFRAgent(model_path=ckpt_path)
    elif agent_name == "value_based":
        from agents.value_based.agent import ValueBasedAgent
        agent = ValueBasedAgent(model_path=ckpt_path)
    elif agent_name == "full_modulation":
        from paper.agents.full_modulation.agent import FullModulationAgent
        agent = FullModulationAgent()
        agent.load_model(ckpt_path)
    elif agent_name == "state_only":
        from paper.agents.ablations.state_only.agent import StateOnlyAgent
        agent = StateOnlyAgent()
        agent.load_model(ckpt_path)
    elif agent_name == "finetuned_base":
        from paper.agents.ablations.finetuned_base.agent import FinetunedBaseAgent
        agent = FinetunedBaseAgent()
        agent.load_model(ckpt_path)
    elif agent_name == "gated_modulation":
        from preliminary_experiments.dali_variants.gated_modulation.agent import GatedModulationAgent
        agent = GatedModulationAgent()
        agent.load_model(ckpt_path)
    elif agent_name == "value_based_deep":
        from preliminary_experiments.dali_variants.value_based_deep.agent import ValueDeepAgent
        agent = ValueDeepAgent(model_path=ckpt_path)
    elif agent_name == "full_modulation_deep":
        from preliminary_experiments.dali_variants.full_modulation_deep.agent import FullModulationDeepAgent
        # base_ckpt is the seed-matched deep base; mod_ckpt is loaded via load_model below
        import re
        seed_match = re.search(r"seed_(\d+)", ckpt_path)
        seed_n = int(seed_match.group(1)) if seed_match else 0
        base_ckpt = os.path.join(
            ROOT, "preliminary_experiments", "dali_variants", "value_based_deep",
            "outputs", f"seed_{seed_n}", "checkpoint_final.pt",
        )
        agent = FullModulationDeepAgent(base_ckpt=base_ckpt)
        agent.load_model(ckpt_path)
    elif agent_name == "value_based_pool":
        from paper.agents.value_based_pool.train import ValuePoolAgent
        agent = ValuePoolAgent(model_path=ckpt_path)
    elif agent_name == "scratch_joint":
        from paper.agents.ablations.scratch_joint.agent import ScratchJointAgent
        agent = ScratchJointAgent()
        agent.load_model(ckpt_path)
    elif agent_name == "finetuned_base_only":
        # Load the finetuned_base checkpoint ({"base": ..., "mod": ...})
        # but use ONLY the base network — no mod head.
        # Wraps it in the ValuePoolAgent interface (same 15→64→64→1 arch, same encoding).
        import torch
        from paper.agents.value_based_pool.train import ValuePoolAgent, ValueShallowNet
        agent = ValuePoolAgent()   # creates empty shallow net
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # finetuned_base checkpoint stores {"base": state_dict, "mod": state_dict}
        agent.net.load_state_dict(state["base"])
    elif agent_name in ("reinforce_v2", "reinforce_v3"):
        from paper.baselines.reinforce.agent import REINFORCEAgent
        agent = REINFORCEAgent(model_path=ckpt_path)
    elif agent_name in ("actor_critic_v2", "actor_critic_v3"):
        from paper.baselines.actor_critic.agent import ActorCriticAgent
        agent = ActorCriticAgent(model_path=ckpt_path)
    elif agent_name in ("dqn_v2", "dqn_v3"):
        from paper.baselines.dqn.agent import DQNAgent
        agent = DQNAgent(model_path=ckpt_path)
    else:
        raise ValueError(f"Unknown study agent: {agent_name}")
    agent.set_train_mode(False)
    return agent


def _load_pool_priors(root: str) -> dict:
    """Load pool prior means for the stats tracker."""
    priors_path = os.path.join(
        root, "preliminary_experiments", "opp_stats_input_aug",
        "outputs", "pool_random", "pool_priors.json",
    )
    if os.path.exists(priors_path):
        with open(priors_path) as f:
            return json.load(f)
    # Fallback uninformative
    return {
        "preflop_fold_rate": 0.5,
        "preflop_raise_rate": 0.5,
        "flop_raise_rate": 0.5,
        "preflop_fold_to_raise": 0.5,
        "flop_fold_to_raise": 0.5,
        "raise_after_raise_rate": 0.5,
    }


# ── stats injection helper ────────────────────────────────────────────────────

class _StatsWrapper:
    """Minimal object compatible with obs.opponent_stats interface (4-dim)."""
    def __init__(self, features: list):
        self._features = features

    def to_feature_vector(self):
        return self._features


# ── core play loop ────────────────────────────────────────────────────────────

def play_matchup(
    study_agent,
    pool_agent,
    pool_agent_key: str,
    is_stat_aware: bool,
    pool_is_stats_injected: bool,
    pool_means: dict,
    num_rounds: int = DEFAULT_ROUNDS,
    session_length: int = SESSION_LENGTH,
    prior_strength: float = PRIOR_STRENGTH,
) -> list[dict]:
    """
    Play `num_rounds` hands: study_agent vs pool_agent.
    Positions alternate every hand.

    Returns list of per-hand dicts:
      {hand_id, reward, cumulative, position, session, hand_in_session, confidence}
    """
    # tracker_for_study: tracks pool agent's actions → study agent reads these
    # tracker_for_pool:  tracks study agent's actions → pool agent reads these
    tracker_for_study = OpponentStatsTracker(pool_means, prior_strength, session_length)
    tracker_for_pool  = OpponentStatsTracker(pool_means, prior_strength, session_length)

    hands_since_reset = 0
    cumulative = 0.0
    records = []

    for i in range(num_rounds):
        # Alternate seats every hand
        if i % 2 == 0:
            study_seat, pool_seat = 0, 1
        else:
            study_seat, pool_seat = 1, 0

        game = LeducGame()
        game.reset()

        prev_raise = False
        prev_round = -1

        while not game.is_finished:
            cp = game.current_player
            obs_cp = game.get_observation(viewer_id=cp)

            if obs_cp.current_round != prev_round:
                prev_raise = False
                prev_round = obs_cp.current_round

            if cp == study_seat:
                # ── study agent acts ──────────────────────────────────────────
                if is_stat_aware:
                    stats = tracker_for_study.get_features()  # (7,) numpy
                    action = study_agent.select_action(obs_cp, opp_stats=stats)
                else:
                    action = study_agent.select_action(obs_cp)

                # pool agent observes study agent's action
                tracker_for_pool.update_action(
                    action, obs_cp.current_round, prev_raise, obs_cp.legal_actions
                )

            else:
                # ── pool agent acts ───────────────────────────────────────────
                if pool_is_stats_injected:
                    feats_4 = tracker_for_pool.get_features()[:4]
                    feats_list = feats_4.tolist() if hasattr(feats_4, "tolist") else list(feats_4)
                    obs_injected = dataclasses.replace(
                        obs_cp,
                        opponent_stats=_StatsWrapper(feats_list),
                    )
                    action = pool_agent.select_action(obs_injected)
                else:
                    action = pool_agent.select_action(obs_cp)

                # study agent observes pool agent's action
                tracker_for_study.update_action(
                    action, obs_cp.current_round, prev_raise, obs_cp.legal_actions
                )

            prev_raise = (action == Action.RAISE)
            game.step(action)

        tracker_for_study.update_hand_end()
        tracker_for_pool.update_hand_end()

        rewards = game.get_reward()
        reward  = rewards[study_seat]
        cumulative += reward

        # confidence = tracker_for_study.get_features()[6]
        conf_features = tracker_for_study.get_features()
        confidence = float(conf_features[6]) if len(conf_features) > 6 else 0.0

        session_idx     = i // session_length
        hand_in_session = i % session_length

        records.append({
            "hand_id":         i,
            "reward":          float(reward),
            "cumulative":      float(cumulative),
            "position":        int(study_seat),
            "session":         int(session_idx),
            "hand_in_session": int(hand_in_session),
            "confidence":      confidence,
        })

        # Session reset
        hands_since_reset += 1
        if hands_since_reset >= session_length:
            tracker_for_study.reset()
            tracker_for_pool.reset()
            hands_since_reset = 0

    return records


# ── summary statistics ────────────────────────────────────────────────────────

def _compute_summary(records: list[dict], n_bootstrap: int = 2000) -> dict:
    """
    Compute per-matchup summary statistics from per-hand records.

    Returns:
      chips_per_round  : mean reward across all hands
      ci_95_low/high   : bootstrap 95% CI of chips_per_round
      cold_mean        : mean reward for hands 0–19 within each session (cold-start)
      warm_mean        : mean reward for hands 20–99 within each session
      n_hands          : total hands played
    """
    rewards = np.array([r["reward"] for r in records], dtype=np.float64)
    n = len(rewards)

    # Chips per round
    chips = float(np.mean(rewards))

    # Bootstrap 95% CI
    rng = np.random.default_rng(42)
    boot_means = [
        float(np.mean(rng.choice(rewards, size=n, replace=True)))
        for _ in range(n_bootstrap)
    ]
    ci_low  = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    # Cold-start vs warm split
    cold_rewards = [r["reward"] for r in records if r["hand_in_session"] < 20]
    warm_rewards = [r["reward"] for r in records if 20 <= r["hand_in_session"] < 100]
    cold_mean = float(np.mean(cold_rewards)) if cold_rewards else float("nan")
    warm_mean = float(np.mean(warm_rewards)) if warm_rewards else float("nan")

    return {
        "chips_per_round": chips,
        "ci_95_low":       ci_low,
        "ci_95_high":      ci_high,
        "cold_mean":       cold_mean,
        "warm_mean":       warm_mean,
        "n_hands":         n,
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _result_dir(agent_name: str, seed) -> str:
    if seed is None:
        subdir = agent_name
    else:
        subdir = f"{agent_name}_seed{seed}"
    return os.path.join(RESULTS_DIR, subdir)


def _already_done(agent_name: str, seed, pool_key: str) -> bool:
    """Return True if this matchup already has a summary JSON."""
    d = _result_dir(agent_name, seed)
    summary_path = os.path.join(d, f"vs_{pool_key}.json")
    return os.path.isfile(summary_path)


def _save_matchup(
    agent_name: str,
    seed,
    pool_key: str,
    records: list[dict],
    summary: dict,
):
    d = _result_dir(agent_name, seed)
    os.makedirs(d, exist_ok=True)

    # Per-hand JSONL
    jsonl_path = os.path.join(d, f"vs_{pool_key}.jsonl")
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Summary JSON
    summary_full = {
        "study_agent": agent_name,
        "seed":        seed,
        "pool_agent":  pool_key,
        **summary,
    }
    summary_path = os.path.join(d, f"vs_{pool_key}.json")
    with open(summary_path, "w") as f:
        json.dump(summary_full, f, indent=2)


# ── single matchup runner ─────────────────────────────────────────────────────

def run_single_matchup(
    agent_name: str,
    seed,
    ckpt_path: str,
    is_stat_aware: bool,
    pool_key: str,
    pool: dict,
    pool_means: dict,
    num_rounds: int = DEFAULT_ROUNDS,
    skip_if_done: bool = True,
    verbose: bool = True,
) -> dict:
    """Run one (study_agent, pool_agent) matchup and save results."""
    if skip_if_done and _already_done(agent_name, seed, pool_key):
        if verbose:
            seed_str = f"seed{seed}" if seed is not None else "noseed"
            print(f"  [skip] {agent_name}_{seed_str} vs {pool_key} — already done")
        return {}

    seed_str = f"seed{seed}" if seed is not None else "noseed"
    if verbose:
        print(f"  {agent_name}_{seed_str:8s} vs {pool_key:20s} ...", end="", flush=True)

    t0 = time.time()
    study_agent = _load_study_agent(agent_name, ckpt_path)
    pool_agent  = pool[pool_key]
    pool_is_stats_injected = (pool_key in STATS_INJECTED)

    records = play_matchup(
        study_agent=study_agent,
        pool_agent=pool_agent,
        pool_agent_key=pool_key,
        is_stat_aware=is_stat_aware,
        pool_is_stats_injected=pool_is_stats_injected,
        pool_means=pool_means,
        num_rounds=num_rounds,
    )

    summary = _compute_summary(records)
    _save_matchup(agent_name, seed, pool_key, records, summary)

    elapsed = time.time() - t0
    if verbose:
        print(
            f" {summary['chips_per_round']:+.4f}  "
            f"[{summary['ci_95_low']:+.4f}, {summary['ci_95_high']:+.4f}]"
            f"  cold:{summary['cold_mean']:+.4f}  warm:{summary['warm_mean']:+.4f}"
            f"  ({elapsed:.0f}s)"
        )

    return summary


# ── aggregate report ──────────────────────────────────────────────────────────

def _build_aggregate_report(root: str) -> dict:
    """
    Read all saved summary JSONs and build an aggregate report.
    Returns dict with per-study-agent mean/std/robustness across pool agents.
    """
    study_list = _study_agent_list(root)
    report = {}

    for agent_name, seed, _, _ in study_list:
        d = _result_dir(agent_name, seed)
        scores = {}
        for pk in POOL_AGENT_KEYS:
            summary_path = os.path.join(d, f"vs_{pk}.json")
            if os.path.isfile(summary_path):
                with open(summary_path) as f:
                    s = json.load(f)
                scores[pk] = s["chips_per_round"]

        if not scores:
            continue

        vals = list(scores.values())
        avg  = float(np.mean(vals))
        std  = float(np.std(vals))
        rob  = avg - 1.5 * std
        worst = float(min(vals))

        key = f"{agent_name}_seed{seed}" if seed is not None else agent_name
        report[key] = {
            "agent": agent_name,
            "seed":  seed,
            "per_opponent": scores,
            "avg":   avg,
            "std":   std,
            "robustness": rob,
            "worst_case": worst,
            "n_opponents_evaluated": len(scores),
        }

    return report


# ── parallel worker (module-level so it can be pickled) ───────────────────────

def _parallel_worker(task_args):
    """Top-level picklable worker for multiprocessing.Pool."""
    a_name, seed, ckpt, stat_aware, pk, pmeans, num_r = task_args
    # Rebuild pool in each worker process (agent objects can't cross process boundaries)
    _pool = build_pool(ROOT)
    return run_single_matchup(
        agent_name=a_name, seed=seed, ckpt_path=ckpt,
        is_stat_aware=stat_aware, pool_key=pk,
        pool=_pool, pool_means=pmeans,
        num_rounds=num_r, verbose=False,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="paper.evaluation gauntlet runner"
    )
    parser.add_argument("--study",   type=str, default=None,
                        help="Study agent name (e.g. full_modulation)")
    parser.add_argument("--seed",    type=int, default=None,
                        help="Seed index (omit for value_based / gated_modulation)")
    parser.add_argument("--pool",    type=str, default=None,
                        help="Pool agent key (e.g. cfr)")
    parser.add_argument("--all",     action="store_true",
                        help="Run all 143 matchups sequentially")
    parser.add_argument("--rounds",  type=int, default=DEFAULT_ROUNDS,
                        help=f"Rounds per matchup (default {DEFAULT_ROUNDS})")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default 1)")
    parser.add_argument("--report",  action="store_true",
                        help="Print aggregate report from saved results and exit")
    args = parser.parse_args()

    if args.report:
        rep = _build_aggregate_report(ROOT)
        if not rep:
            print("No results found yet.")
            return
        # Print ranked table
        ranked = sorted(rep.values(), key=lambda x: -x["robustness"])
        print(f"\n{'Agent':30s}  {'avg':>8s}  {'std':>6s}  {'rob':>8s}  {'worst':>8s}  {'N':>3s}")
        print("-" * 75)
        for r in ranked:
            key = f"{r['agent']}_seed{r['seed']}" if r['seed'] is not None else r['agent']
            print(f"{key:30s}  {r['avg']:>+8.4f}  {r['std']:>6.4f}  "
                  f"{r['robustness']:>+8.4f}  {r['worst_case']:>+8.4f}  {r['n_opponents_evaluated']:>3d}")
        report_path = os.path.join(RESULTS_DIR, "aggregate_report.json")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\nSaved to {report_path}")
        return

    pool_means = _load_pool_priors(ROOT)
    pool = build_pool(ROOT)
    study_list = _study_agent_list(ROOT)

    if args.all:
        # ── run all matchups ──────────────────────────────────────────────────
        if args.workers > 1:
            import multiprocessing as mp

            tasks = []
            for a_name, seed, ckpt, stat_aware in study_list:
                if not os.path.isfile(ckpt):
                    print(f"  [warn] checkpoint missing: {ckpt}")
                    continue
                for pk in POOL_AGENT_KEYS:
                    if not _already_done(a_name, seed, pk):
                        tasks.append((a_name, seed, ckpt, stat_aware, pk, pool_means, args.rounds))

            print(f"Launching {len(tasks)} matchups with {args.workers} workers...")
            t_all = time.time()
            with mp.Pool(args.workers) as pool_proc:
                pool_proc.map(_parallel_worker, tasks)
            print(f"Done in {time.time() - t_all:.0f}s")

        else:
            # Sequential
            total = sum(1 for (a_name, seed, ckpt, _) in study_list
                        for pk in POOL_AGENT_KEYS
                        if os.path.isfile(ckpt))
            done = 0
            t_all = time.time()
            print(f"Running all matchups × {args.rounds} rounds each...\n")
            for a_name, seed, ckpt, stat_aware in study_list:
                if not os.path.isfile(ckpt):
                    seed_str = f"seed{seed}" if seed is not None else "noseed"
                    print(f"  [warn] {a_name}_{seed_str}: checkpoint missing, skipping")
                    continue
                for pk in POOL_AGENT_KEYS:
                    run_single_matchup(
                        agent_name=a_name, seed=seed, ckpt_path=ckpt,
                        is_stat_aware=stat_aware, pool_key=pk,
                        pool=pool, pool_means=pool_means,
                        num_rounds=args.rounds,
                    )
                    done += 1
                    print(f"  Progress: {done}/{total}")

            print(f"\nAll done in {time.time() - t_all:.0f}s")
            # Auto-build aggregate report
            rep = _build_aggregate_report(ROOT)
            report_path = os.path.join(RESULTS_DIR, "aggregate_report.json")
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(rep, f, indent=2)
            print(f"Aggregate report saved to {report_path}")

    elif args.study and args.pool:
        # ── single matchup ────────────────────────────────────────────────────
        # Find matching entry
        match = None
        for a_name, seed, ckpt, stat_aware in study_list:
            if a_name == args.study and seed == args.seed:
                match = (a_name, seed, ckpt, stat_aware)
                break
        if match is None:
            print(f"ERROR: No study agent '{args.study}' with seed={args.seed}")
            print(f"Available: {[(a, s) for a, s, _, _ in study_list]}")
            sys.exit(1)

        a_name, seed, ckpt, stat_aware = match
        if not os.path.isfile(ckpt):
            print(f"ERROR: checkpoint not found: {ckpt}")
            sys.exit(1)

        if args.pool not in POOL_AGENT_KEYS:
            print(f"ERROR: unknown pool agent '{args.pool}'. Choose from: {POOL_AGENT_KEYS}")
            sys.exit(1)

        run_single_matchup(
            agent_name=a_name, seed=seed, ckpt_path=ckpt,
            is_stat_aware=stat_aware, pool_key=args.pool,
            pool=pool, pool_means=pool_means,
            num_rounds=args.rounds,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
