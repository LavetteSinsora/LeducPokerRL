"""
paper — evaluation runner
=====================================
Loads a checkpoint and runs a full stat-aware pool evaluation.

Usage:
  python -m paper.evaluation.run_eval --agent full_modulation --seed 0
  python -m paper.evaluation.run_eval --agent full_modulation --seed 0 --checkpoint ep050000
  python -m paper.evaluation.run_eval --agent state_only --seed 1
  python -m paper.evaluation.run_eval --agent finetuned_base --seed 0 --rounds 2000

Arguments:
  --agent       : one of full_modulation, gated_modulation, state_only, finetuned_base
  --seed        : seed index (default 0)
  --checkpoint  : checkpoint name without extension (default: checkpoint_final)
                  e.g. ep050000 loads checkpoints/checkpoint_ep050000.pt
  --rounds      : rounds per opponent (default 5000)
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from paper.evaluation.comparison_protocol import (
    build_standard_opponents,
    evaluate_stat_aware_pool,
    compute_pool_summary,
    format_pool_summary,
    STANDARD_OPPONENT_KEYS,
)
from paper.evaluation.shared.training_recipe import play_hand_v2

SESSION_LENGTH = 100
PRIOR_STRENGTH = 20.0

AGENT_DIRS = {
    "full_modulation":  os.path.join(HERE, "..", "agents", "full_modulation"),
    "gated_modulation": os.path.join(HERE, "..", "agents", "gated_modulation"),
    "state_only":       os.path.join(HERE, "..", "agents", "ablations", "state_only"),
    "finetuned_base":   os.path.join(HERE, "..", "agents", "ablations", "finetuned_base"),
}


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}


def load_agent(agent_name: str, checkpoint_path: str):
    """Load an agent from a checkpoint file."""
    if agent_name == "full_modulation":
        from paper.agents.full_modulation.agent import FullModulationAgent
        agent = FullModulationAgent()
        agent.load_model(checkpoint_path)
    elif agent_name == "gated_modulation":
        from preliminary_experiments.dali_variants.gated_modulation.agent import GatedModulationAgent
        agent = GatedModulationAgent()
        agent.load_model(checkpoint_path)
    elif agent_name == "state_only":
        from paper.agents.ablations.state_only.agent import StateOnlyAgent
        agent = StateOnlyAgent()
        agent.load_model(checkpoint_path)
    elif agent_name == "finetuned_base":
        from paper.agents.ablations.finetuned_base.agent import FinetunedBaseAgent
        agent = FinetunedBaseAgent()
        agent.load_model(checkpoint_path)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    return agent


def play_hand_state_only_eval(agent, opponent, tracker, learner_id=0):
    """Wrapper for state_only eval — same as training version."""
    from paper.agents.ablations.state_only.train import play_hand_state_only
    return play_hand_state_only(agent, opponent, tracker, learner_id=learner_id)


def main():
    parser = argparse.ArgumentParser(
        description="paper.evaluation runner")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["full_modulation", "gated_modulation", "state_only", "finetuned_base"],
        help="Agent variant to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed index (default 0)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_final",
        help="Checkpoint name without extension (default: checkpoint_final). "
             "e.g. ep050000 loads checkpoints/checkpoint_ep050000.pt",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5000,
        help="Rounds per opponent (default 5000)",
    )
    args = parser.parse_args()

    agent_name      = args.agent
    seed            = args.seed
    checkpoint_name = args.checkpoint
    num_rounds      = args.rounds

    # ── resolve output dir ────────────────────────────────────────────────────
    agent_dir = os.path.abspath(AGENT_DIRS[agent_name])
    out_dir   = os.path.join(agent_dir, "outputs", f"seed_{seed}")

    if not os.path.isdir(out_dir):
        raise FileNotFoundError(
            f"Output dir not found: {out_dir}\n"
            f"Run training first: python -m paper.agents.{agent_name.replace('state_only', 'ablations.state_only').replace('finetuned_base', 'ablations.finetuned_base')}.train --seed {seed}"
        )

    # ── resolve checkpoint path ───────────────────────────────────────────────
    if checkpoint_name == "checkpoint_final":
        ckpt_path = os.path.join(out_dir, "checkpoint_final.pt")
    else:
        # e.g. ep050000 → checkpoints/checkpoint_ep050000.pt
        ckpt_path = os.path.join(out_dir, "checkpoints", f"checkpoint_{checkpoint_name}.pt")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Agent:      {agent_name}")
    print(f"Seed:       {seed}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Rounds/opp: {num_rounds}")
    print()

    # ── load pool priors ──────────────────────────────────────────────────────
    priors_path   = os.path.join(out_dir, "pool_priors.json")
    shared_priors = os.path.join(
        ROOT, "preliminary_experiments", "opp_stats_input_aug",
        "outputs", "pool_random", "pool_priors.json")

    if os.path.exists(priors_path):
        pool_means = _load_json(priors_path)
        print(f"Loaded pool priors from {priors_path}")
    elif os.path.exists(shared_priors):
        pool_means = _load_json(shared_priors)
        print("Using shared pool priors from opp_stats_input_augmentation_v1")
    else:
        raise FileNotFoundError(
            f"Pool priors not found. Run training first to generate them.")

    # ── load agent ────────────────────────────────────────────────────────────
    agent = load_agent(agent_name, ckpt_path)
    agent.set_train_mode(False)

    # ── build opponents ───────────────────────────────────────────────────────
    opponents = build_standard_opponents(ROOT)

    # ── select play_hand function ─────────────────────────────────────────────
    if agent_name == "state_only":
        play_fn = play_hand_state_only_eval
    else:
        play_fn = play_hand_v2

    # ── run evaluation ────────────────────────────────────────────────────────
    print(f"Running evaluation ({num_rounds} rounds/opponent)...")
    result = evaluate_stat_aware_pool(
        agent=agent,
        opponents=opponents,
        play_hand_fn=play_fn,
        pool_means=pool_means,
        num_rounds=num_rounds,
        session_length=SESSION_LENGTH,
        prior_strength=PRIOR_STRENGTH,
        opponent_keys=list(STANDARD_OPPONENT_KEYS),
        alternate_positions=True,
    )

    scores  = result["scores"]
    summary = result["summary"]

    # ── print results table ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {agent_name} | seed={seed} | {checkpoint_name}")
    print(f"{'='*65}")
    print(f"  {'Opponent':<22} {'avg chips/round':>16}")
    print(f"  {'-'*22}  {'-'*16}")
    for key in STANDARD_OPPONENT_KEYS:
        print(f"  {key:<22} {scores[key]:>+16.4f}")
    print(f"  {'='*40}")
    print(f"  {'avg':<22} {summary['avg']:>+16.4f}")
    print(f"  {'worst_case':<22} {summary['worst_case']:>+16.4f}")
    print(f"  {'robustness':<22} {summary['robustness']:>+16.4f}")
    print(f"  {'std':<22} {summary['std']:>16.4f}")
    print(f"{'='*65}")
    print(f"\n  {format_pool_summary(summary)}")

    # ── save results ──────────────────────────────────────────────────────────
    eval_out_name = f"eval_{checkpoint_name}.json"
    eval_out_path = os.path.join(out_dir, eval_out_name)
    output = {
        "agent":       agent_name,
        "seed":        seed,
        "checkpoint":  checkpoint_name,
        "rounds":      num_rounds,
        "scores":      scores,
        "summary": {
            "avg":        round(summary["avg"], 4),
            "worst_case": round(summary["worst_case"], 4),
            "robustness": round(summary["robustness"], 4),
            "std":        round(summary["std"], 4),
        },
        "details": result.get("details", {}),
    }
    _write_json(eval_out_path, output)
    print(f"\nResults saved to {eval_out_path}")


if __name__ == "__main__":
    main()
