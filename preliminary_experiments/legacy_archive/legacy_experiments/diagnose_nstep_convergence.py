#!/usr/bin/env python3
"""
Key insight from prior experiments:
- Loss is 2-3x higher for n>=2 vs n=1, but value estimates evolve similarly
- The loss difference is because terminal rewards (range [-13, +13]) produce
  much larger MSE than bootstrap targets (range [-0.13, +0.10])
- With n=1 (TD(0)): 35% of targets are bootstrap (near 0), 65% terminal (large)
- With n=3: 99% terminal. The loss is dominated by large terminal targets.

This means the effective learning rate is MUCH higher for n>=2 because:
  MSE_loss = (pred - target)^2
  When targets are terminal rewards (+/-13), gradients are ~100x larger
  than when targets are bootstrap values (~0.05)

So n=3 is effectively training with a much larger learning rate.

Experiment: Try n=3 with REDUCED learning rate to compensate for the
larger gradient magnitudes. If the gradient magnitude hypothesis is correct,
reducing lr by the ratio of gradient norms should recover TD(0) performance.

Gradient norm ratio from exp: n=3 is ~1.87x that of n=1.
So try lr = 1e-4 / 1.87 ≈ 5.3e-5 for n=3.
Also try lr scaling proportional to bootstrap fraction.
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


def train_run(n_steps, lr, num_episodes=5000, batch_size=32, eval_rounds=200,
              seed=42, init_state_dict=None):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    agent = NStepValueAgent()
    if init_state_dict is not None:
        agent.model.load_state_dict(copy.deepcopy(init_state_dict))

    trainer = NStepValueTrainer(agent, learning_rate=lr, n_steps=n_steps)
    loss_history = []
    eval_history = []

    agent.set_train_mode(True)
    batch_data = []

    for ep in range(num_episodes):
        trajectory = trainer.collect_episode()
        batch_data.append(trajectory)
        if len(batch_data) >= batch_size:
            loss = trainer.update_model(batch_data)
            loss_history.append(loss)
            batch_data = []

        if (ep + 1) % 1000 == 0:
            agent.set_train_mode(False)
            avg_chips = quick_evaluate(agent, HeuristicAgent(), num_rounds=eval_rounds)
            eval_history.append({"episode": ep + 1, "avg_chips": avg_chips})
            agent.set_train_mode(True)

    agent.set_train_mode(False)
    final = quick_evaluate(agent, HeuristicAgent(), num_rounds=eval_rounds)
    return final, eval_history, np.mean(loss_history), np.std(loss_history)


def train_td0(lr=1e-4, num_episodes=5000, batch_size=32, eval_rounds=200,
              seed=42, init_state_dict=None):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    agent = ValueBasedAgent()
    if init_state_dict is not None:
        agent.model.load_state_dict(copy.deepcopy(init_state_dict))

    trainer = SelfPlayTrainer(agent, learning_rate=lr)
    loss_history = []
    eval_history = []

    agent.set_train_mode(True)
    batch_data = []

    for ep in range(num_episodes):
        trajectory = trainer.collect_episode()
        batch_data.append(trajectory)
        if len(batch_data) >= batch_size:
            loss = trainer.update_model(batch_data)
            loss_history.append(loss)
            batch_data = []

        if (ep + 1) % 1000 == 0:
            agent.set_train_mode(False)
            avg_chips = quick_evaluate(agent, HeuristicAgent(), num_rounds=eval_rounds)
            eval_history.append({"episode": ep + 1, "avg_chips": avg_chips})
            agent.set_train_mode(True)

    agent.set_train_mode(False)
    final = quick_evaluate(agent, HeuristicAgent(), num_rounds=eval_rounds)
    return final, eval_history, np.mean(loss_history), np.std(loss_history)


if __name__ == "__main__":
    ref_agent = ValueBasedAgent()
    init_weights = copy.deepcopy(ref_agent.model.state_dict())

    seeds = [42, 123, 7]

    configs = [
        ("TD(0) lr=1e-4", "td0", 1, 1e-4),
        ("n=3 lr=1e-4",   "nstep", 3, 1e-4),
        ("n=3 lr=5e-5",   "nstep", 3, 5e-5),
        ("n=3 lr=3e-5",   "nstep", 3, 3e-5),
        ("MC lr=1e-4",    "nstep", 100, 1e-4),
        ("MC lr=5e-5",    "nstep", 100, 5e-5),
        ("MC lr=3e-5",    "nstep", 100, 3e-5),
    ]

    print(f"{'Config':<20} | {'Seed':<6} | {'Score':>8} | {'Loss Mean':>10} | {'Loss Std':>10}")
    print("-" * 70)

    summary = {}
    for label, trainer_type, n, lr in configs:
        scores = []
        for seed in seeds:
            if trainer_type == "td0":
                score, evals, lmean, lstd = train_td0(
                    lr=lr, seed=seed, init_state_dict=init_weights
                )
            else:
                score, evals, lmean, lstd = train_run(
                    n_steps=n, lr=lr, seed=seed, init_state_dict=init_weights
                )
            scores.append(score)
            print(f"{label:<20} | {seed:<6} | {score:>+8.3f} | {lmean:>10.4f} | {lstd:>10.4f}")

        avg = np.mean(scores)
        std = np.std(scores)
        summary[label] = (avg, std)
        print(f"  {'=> AVG':<18} |        | {avg:>+8.3f} +/- {std:.3f}")
        print()

    print("\n" + "=" * 50)
    print("LEARNING RATE COMPENSATION SUMMARY")
    print("=" * 50)
    for label, (avg, std) in summary.items():
        print(f"  {label:<20}: {avg:>+.3f} +/- {std:.3f}")

    print("""
INTERPRETATION:
  If reducing lr for n=3 recovers TD(0) performance, the root cause is
  that MC targets create ~2x larger gradients, effectively doubling the
  learning rate and causing overshoot/instability.

  If reducing lr does NOT help, the issue is fundamental to the variance
  of MC targets in non-stationary self-play, not just gradient magnitude.
""")
