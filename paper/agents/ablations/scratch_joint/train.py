"""
paper — scratch_joint ablation training
===================================================
Ablation C: ScratchJointAgent (random-init base, jointly trained with mod head).

Architecture:  V(s, opp) = V_base(s) [RANDOM INIT, UNFROZEN] + Δ(s, opp_stats)

Key difference from finetuned_base: the base starts from random weights,
NOT from the pretrained value_based checkpoint.

Training recipe: identical to finetuned_base (weighted pool, sessions,
stats tracker, TD(0) total-value targets).

3 seeds, 300K episodes, lr=1e-4, batch=32, Adam.

Usage:
  python -m paper.agents.ablations.scratch_joint.train --seed 0
  python -m paper.agents.ablations.scratch_joint.train --seed 0 --smoke
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
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from paper.agents.ablations.scratch_joint.agent import ScratchJointAgent
from paper.evaluation.shared.training_recipe import (
    OPPONENT_WEIGHTS,
    WeightedOpponentSampler,
    SessionManager,
    play_hand_v2,
)
from paper.evaluation.comparison_protocol import build_standard_opponents
from paper.evaluation.shared.stats_tracker import compute_pool_means

# ── constants ──────────────────────────────────────────────────────────────────

SESSION_LENGTH      = 100
PRIOR_STRENGTH      = 20.0
CALIBRATION_HANDS   = 500
BATCH_SIZE          = 32
LR                  = 1e-4
NUM_EPISODES        = 300_000
CHECKPOINT_INTERVAL = 10_000


# ── TD(0) update ──────────────────────────────────────────────────────────────

def td_update(agent: ScratchJointAgent, optimizer, criterion, batch_data):
    """
    Total-value TD(0). Gradients flow through both base and mod.

    Terminal    : target = float(reward)
    Non-terminal: target = (V_base(s_{t+1}) + Δ(s_{t+1}, stats_{t+1})).detach()
    """
    optimizer.zero_grad()
    losses = []

    for chain, reward in batch_data:
        if not chain:
            continue
        for t, (game_enc, stats) in enumerate(chain):
            game_t  = game_enc.unsqueeze(0)
            stats_t = torch.tensor(stats, dtype=torch.float32).unsqueeze(0)

            v_total_t = agent.compute_value(game_t, stats_t).squeeze()

            if t == len(chain) - 1:
                target = torch.tensor(float(reward), dtype=torch.float32)
            else:
                game_t1  = chain[t + 1][0].unsqueeze(0)
                stats_t1 = torch.tensor(
                    chain[t + 1][1], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    target = agent.compute_value(game_t1, stats_t1).squeeze()

            losses.append(criterion(v_total_t, target.detach()))

    if not losses:
        return 0.0, 0.0

    mean_loss = torch.stack(losses).mean()
    mean_loss.backward()

    params = list(agent.base.parameters()) + list(agent.mod.parameters())
    grad_norm = sum(
        p.grad.norm() ** 2 for p in params if p.grad is not None
    ) ** 0.5
    grad_norm = float(grad_norm)

    optimizer.step()
    return mean_loss.item(), grad_norm


# ── main training loop ────────────────────────────────────────────────────────

def train(out_dir: str, num_episodes: int, smoke: bool, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── pool priors ────────────────────────────────────────────────────────────
    shared_priors = os.path.join(
        ROOT, "preliminary_experiments", "opp_stats_input_aug",
        "outputs", "pool_random", "pool_priors.json")
    priors_path = os.path.join(out_dir, "pool_priors.json")

    opponents = build_standard_opponents(ROOT)

    if os.path.exists(shared_priors):
        with open(shared_priors) as f:
            pool_means = json.load(f)
        with open(priors_path, "w") as f:
            json.dump(pool_means, f, indent=2)
        print("Reused pool priors from opp_stats_input_augmentation_v1")
    else:
        print("Computing pool priors...")
        pool_means = compute_pool_means(opponents, 50 if smoke else CALIBRATION_HANDS)
        with open(priors_path, "w") as f:
            json.dump(pool_means, f, indent=2)

    # ── agent + optimiser ──────────────────────────────────────────────────────
    agent     = ScratchJointAgent()
    trainable = list(agent.base.parameters()) + list(agent.mod.parameters())
    optimizer = optim.Adam(trainable, lr=LR)
    criterion = nn.MSELoss()
    agent.set_train_mode(True)

    # ── session + sampler ──────────────────────────────────────────────────────
    sampler = WeightedOpponentSampler(opponents, OPPONENT_WEIGHTS)
    session = SessionManager(SESSION_LENGTH, pool_means, PRIOR_STRENGTH, sampler)

    # ── config ─────────────────────────────────────────────────────────────────
    config = {
        "agent":               "ScratchJointAgent",
        "architecture":        "V_base(RANDOM_INIT, 15→64→64→1) + ModHead(22→32→32→1)",
        "ablation":            "C — no pretrained base",
        "td_target":           "total_value (V_base + delta)",
        "training_recipe":     "weighted_pool (cfr×3, heuristic×3, others×1)",
        "opponent_weights":    OPPONENT_WEIGHTS,
        "learning_rate":       LR,
        "batch_size":          BATCH_SIZE,
        "session_length":      SESSION_LENGTH,
        "prior_strength":      PRIOR_STRENGTH,
        "num_episodes":        num_episodes,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "seed":                seed,
        "smoke":               smoke,
    }
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    freqs = sampler.expected_frequencies()
    print(f"\n{'='*65}")
    print(f"  paper — scratch_joint (Ablation C)  seed={seed}")
    print(f"  {'SMOKE' if smoke else 'FULL'} | episodes={num_episodes:,} "
          f"batch={BATCH_SIZE} session={SESSION_LENGTH}")
    print(f"{'='*65}")
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
            opp_name, opp = session.current_opponent()

            chain, reward = play_hand_v2(
                agent, opp, session.tracker(learner_id), learner_id=learner_id)
            session.record_hand(learner_id)

            confidence = float(session.tracker(learner_id).get_features()[6])
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
                "ep": ep, "opp": opp_name, "reward": float(reward),
                "confidence": round(confidence, 6),
                "loss": ep_loss, "gnorm": ep_gnorm,
            }) + "\n")

            if ep % CHECKPOINT_INTERVAL == 0:
                agent.save_model(os.path.join(ckpt_dir, f"checkpoint_ep{ep:06d}.pt"))
                l_str  = f"{last_loss:.4f}"  if last_loss  is not None else "n/a"
                gn_str = f"{last_gnorm:.4f}" if last_gnorm is not None else "n/a"
                print(f"[ep={ep:,}] loss={l_str} gnorm={gn_str} "
                      f"opp={opp_name} conf={confidence:.4f} "
                      f"({time.time()-t0:.0f}s)")

    agent.save_model(os.path.join(out_dir, "checkpoint_final.pt"))
    print(f"\nDone in {time.time()-t0:.0f}s  →  {out_dir}/")


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
