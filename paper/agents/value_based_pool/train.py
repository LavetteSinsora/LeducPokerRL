"""
paper — value_based_pool training
=============================================
Architecture:  15 → 64 → 64 → 1  (identical to canonical value_based)
Training:      Weighted fixed-opponent pool (CFR/heuristic 3×, others 1×)

This is the clean control for the depth ablation.

  value_based       → shallow arch, unknown original training recipe
  value_based_pool  → shallow arch, weighted pool recipe      ← this agent
  value_based_deep  → deep arch (64×3), weighted pool recipe

Comparing value_based_pool vs value_based_deep isolates depth.
Comparing value_based_pool vs full_modulation isolates modulation head
(both use the same shallow architecture and the same training recipe).

3 seeds, 200K episodes, lr=1e-4, batch=32, Adam.

Usage:
  python -m paper.agents.value_based_pool.train --seed 0
  python -m paper.agents.value_based_pool.train --seed 0 --smoke
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

# Reuse ValueDeepAgent's encoding + play loop (identical to value_based encoding)
from preliminary_experiments.dali_variants.value_based_deep.agent import ValueDeepAgent
from preliminary_experiments.dali_variants.value_based_deep.train import play_hand_deep, td_update
from paper.evaluation.shared.training_recipe import (
    OPPONENT_WEIGHTS,
    WeightedOpponentSampler,
)
from paper.evaluation.comparison_protocol import build_standard_opponents

# ── constants ─────────────────────────────────────────────────────────────────

LR             = 1e-4
BATCH_SIZE     = 32
NUM_EPISODES   = 200_000
CKPT_INTERVAL  = 10_000


class ValueShallowNet(nn.Module):
    """15 → 64 → 64 → 1 (2 hidden layers). Same as canonical value_based."""

    def __init__(self, input_size: int = 15, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


class ValuePoolAgent(ValueDeepAgent):
    """
    Shallow value agent (15→64→64→1) trained with weighted pool recipe.
    Subclasses ValueDeepAgent to reuse encoding + action selection;
    simply replaces the network with a 2-layer version.
    """

    def __init__(self, model_path: str = None, temperature: float = 1.0):
        super().__init__(model_path=None, temperature=temperature)
        # Replace the 3-layer net with a 2-layer net
        self.net = ValueShallowNet(15, hidden_size=64)
        if model_path:
            self.load_model(model_path)
        self.net.eval()


# ── training ──────────────────────────────────────────────────────────────────

def train(out_dir: str, num_episodes: int, smoke: bool, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    opponents = build_standard_opponents(ROOT)
    sampler   = WeightedOpponentSampler(opponents, OPPONENT_WEIGHTS)

    agent     = ValuePoolAgent()
    optimizer = optim.Adam(agent.net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    agent.set_train_mode(True)

    config = {
        "agent":            "ValuePoolAgent",
        "architecture":     "15→64→64→1 (shallow, 2 hidden layers)",
        "num_episodes":     num_episodes,
        "learning_rate":    LR,
        "batch_size":       BATCH_SIZE,
        "ckpt_interval":    CKPT_INTERVAL,
        "opponent_weights": OPPONENT_WEIGHTS,
        "seed":             seed,
        "smoke":            smoke,
    }
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    freqs = sampler.expected_frequencies()
    print(f"\n{'='*60}")
    print(f"  DALI — value_based_pool  seed={seed}")
    print(f"  {'SMOKE' if smoke else 'FULL'} | episodes={num_episodes:,} "
          f"batch={BATCH_SIZE} lr={LR}")
    print(f"{'='*60}")
    print("Opponent sampling weights:")
    for k, p in freqs.items():
        print(f"  {k:<20s} {p:.1%}")
    print()

    batch_data = []
    log_path   = os.path.join(out_dir, "train_log.jsonl")
    last_loss  = last_gnorm = None

    t0 = time.time()
    with open(log_path, "w") as log_file:
        for ep in range(1, num_episodes + 1):
            learner_id = ep % 2
            opp_name, opp = sampler.sample()

            chain, reward = play_hand_deep(agent, opp, learner_id)
            batch_data.append((chain, reward))

            ep_loss = ep_gnorm = None
            if len(batch_data) >= BATCH_SIZE:
                loss_val, gnorm_val = td_update(agent, optimizer, criterion, batch_data)
                batch_data.clear()
                last_loss  = loss_val
                last_gnorm = gnorm_val
                ep_loss    = round(loss_val, 6)
                ep_gnorm   = round(gnorm_val, 6)

            log_file.write(json.dumps({
                "ep": ep, "opp": opp_name,
                "reward": float(reward),
                "loss": ep_loss, "gnorm": ep_gnorm,
            }) + "\n")

            if ep % CKPT_INTERVAL == 0:
                agent.save_model(os.path.join(ckpt_dir, f"checkpoint_ep{ep:06d}.pt"))
                l_str  = f"{last_loss:.4f}"  if last_loss  is not None else "n/a"
                gn_str = f"{last_gnorm:.4f}" if last_gnorm is not None else "n/a"
                elapsed = time.time() - t0
                print(f"[ep={ep:,}] loss={l_str} gnorm={gn_str} "
                      f"opp={opp_name} ({elapsed:.0f}s)")

    agent.save_model(os.path.join(out_dir, "checkpoint_final.pt"))
    print(f"\nDone in {time.time() - t0:.0f}s  →  {out_dir}/")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",     type=int,  default=0)
    parser.add_argument("--episodes", type=int,  default=None)
    parser.add_argument("--smoke",    action="store_true")
    args = parser.parse_args()

    smoke    = args.smoke
    episodes = args.episodes or (500 if smoke else NUM_EPISODES)
    out_dir  = os.path.join(HERE, "outputs", f"seed_{args.seed}")

    train(out_dir=out_dir, num_episodes=episodes, smoke=smoke, seed=args.seed)
