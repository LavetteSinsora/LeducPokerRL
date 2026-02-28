#!/usr/bin/env python3
"""
Multi-seed confirmation: Run 3 seeds for n=1 (TD0) vs n=3 vs MC at 5K episodes.
This verifies the variance effect is consistent, not a single-seed artifact.
"""

import sys, os, copy, random
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.value_based import ValueBasedAgent
from src.agents.nstep_value import NStepValueAgent
from src.agents.heuristic import HeuristicAgent
from src.training.nstep_value_trainer import NStepValueTrainer
from src.training.value_based_trainer import SelfPlayTrainer
from src.training.evaluation import quick_evaluate


def train_run(n_steps, num_episodes=5000, batch_size=32, eval_rounds=200,
              seed=42, init_state_dict=None, use_original_td0=False):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if use_original_td0:
        agent = ValueBasedAgent()
    else:
        agent = NStepValueAgent()

    if init_state_dict is not None:
        agent.model.load_state_dict(copy.deepcopy(init_state_dict))

    if use_original_td0:
        trainer = SelfPlayTrainer(agent, learning_rate=1e-4)
    else:
        trainer = NStepValueTrainer(agent, learning_rate=1e-4, n_steps=n_steps)

    loss_history = []
    agent.set_train_mode(True)
    batch_data = []

    for ep in range(num_episodes):
        trajectory = trainer.collect_episode()
        batch_data.append(trajectory)
        if len(batch_data) >= batch_size:
            loss = trainer.update_model(batch_data)
            loss_history.append(loss)
            batch_data = []

    agent.set_train_mode(False)
    final_score = quick_evaluate(agent, HeuristicAgent(), num_rounds=eval_rounds)

    return final_score, np.mean(loss_history), np.std(loss_history)


if __name__ == "__main__":
    ref_agent = ValueBasedAgent()
    init_weights = copy.deepcopy(ref_agent.model.state_dict())

    seeds = [42, 123, 7]
    configs = [
        ("TD(0) original", None, True),
        ("NStep n=1", 1, False),
        ("NStep n=2", 2, False),
        ("NStep n=3", 3, False),
        ("MC (n=100)", 100, False),
    ]

    print(f"{'Config':<18} | {'Seed':<6} | {'Score':>8} | {'Loss Mean':>10} | {'Loss Std':>10}")
    print("-" * 70)

    summary = {}
    for label, n, use_td0 in configs:
        scores = []
        for seed in seeds:
            score, lmean, lstd = train_run(
                n_steps=n if n else 1,
                seed=seed,
                init_state_dict=init_weights,
                use_original_td0=use_td0
            )
            scores.append(score)
            print(f"{label:<18} | {seed:<6} | {score:>+8.3f} | {lmean:>10.4f} | {lstd:>10.4f}")

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        summary[label] = (avg_score, std_score)
        print(f"  {'=> AVG':<16} |        | {avg_score:>+8.3f} +/- {std_score:.3f}")
        print()

    print("\n" + "=" * 50)
    print("MULTI-SEED SUMMARY")
    print("=" * 50)
    for label, (avg, std) in summary.items():
        print(f"  {label:<18}: {avg:>+.3f} +/- {std:.3f}")
