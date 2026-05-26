"""
DALI_modulation — full_modulation_deep training
================================================
Architecture:  V(s,opp) = V_base_deep(s) [frozen] + Δ(s, opp_stats)
               Deep base: 15→64→64→64→1 (from value_based_deep)
               Mod head:  22→32→32→1

Same training recipe as full_modulation (cfr/heuristic 3×, 300K episodes).
Each seed trains on top of the corresponding value_based_deep seed checkpoint.

Usage:
  python -m preliminary_experiments.dali_variants.full_modulation_deep.train --seed 0
  python -m preliminary_experiments.dali_variants.full_modulation_deep.train --seed 0 --smoke
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

from preliminary_experiments.dali_variants.full_modulation_deep.agent import FullModulationDeepAgent
from paper.evaluation.shared.training_recipe import (
    OPPONENT_WEIGHTS,
    WeightedOpponentSampler,
    SessionManager,
    play_hand_v2,
)
from paper.evaluation.comparison_protocol import build_standard_opponents
from paper.evaluation.shared.stats_tracker import compute_pool_means

# ── constants ─────────────────────────────────────────────────────────────────

SESSION_LENGTH  = 100
PRIOR_STRENGTH  = 20.0
CAL_HANDS       = 500
BATCH_SIZE      = 32
LR              = 1e-4
NUM_EPISODES    = 300_000
CKPT_INTERVAL   = 10_000

DEEP_BASE_DIR = os.path.join(ROOT, "preliminary_experiments", "dali_variants", "value_based_deep")


# ── TD(0) update — residual only ──────────────────────────────────────────────

def td_update(agent: FullModulationDeepAgent, optimizer, criterion, batch_data):
    """
    Residual TD(0): only ModulationHead receives gradients.
    Terminal    : target_residual = reward − V_base_deep(s_T)
    Non-terminal: target_residual = (V_base_deep(s_{t+1}) + Δ(s_{t+1})) − V_base_deep(s_t)
    """
    optimizer.zero_grad()
    losses = []
    deltas = []

    for chain, reward in batch_data:
        if not chain:
            continue
        for t, (game_enc, stats) in enumerate(chain):
            game_t  = game_enc.unsqueeze(0)
            stats_t = torch.tensor(stats, dtype=torch.float32).unsqueeze(0)
            mod_inp = torch.cat([game_t, stats_t], dim=1)

            with torch.no_grad():
                v_base_t = agent.base(game_t).squeeze()

            delta_t = agent.mod(mod_inp).squeeze()
            deltas.append(delta_t.detach().abs().item())

            if t == len(chain) - 1:
                target_residual = torch.tensor(float(reward), dtype=torch.float32) - v_base_t
            else:
                game_t1  = chain[t + 1][0].unsqueeze(0)
                stats_t1 = torch.tensor(chain[t + 1][1], dtype=torch.float32).unsqueeze(0)
                mod_inp1 = torch.cat([game_t1, stats_t1], dim=1)
                with torch.no_grad():
                    v_base_t1 = agent.base(game_t1).squeeze()
                    delta_t1  = agent.mod(mod_inp1).squeeze()
                    target_residual = (v_base_t1 + delta_t1) - v_base_t

            losses.append(criterion(delta_t, target_residual.detach()))

    if not losses:
        return 0.0, 0.0, 0.0

    mean_loss = torch.stack(losses).mean()
    mean_loss.backward()

    params = list(agent.mod.parameters())
    grad_norm = float(
        sum(p.grad.norm() ** 2 for p in params if p.grad is not None) ** 0.5
    )
    optimizer.step()

    return mean_loss.item(), grad_norm, float(np.mean(deltas)) if deltas else 0.0


# ── training ──────────────────────────────────────────────────────────────────

def train(out_dir: str, num_episodes: int, smoke: bool, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resolve deep base checkpoint (seed-matched)
    base_ckpt = os.path.join(
        DEEP_BASE_DIR, "outputs", f"seed_{seed}", "checkpoint_final.pt"
    )
    if not os.path.isfile(base_ckpt):
        raise FileNotFoundError(
            f"Deep base checkpoint not found: {base_ckpt}\n"
            f"Run: python -m preliminary_experiments.dali_variants.value_based_deep.train --seed {seed}"
        )

    # Pool priors
    shared_priors = os.path.join(
        ROOT, "preliminary_experiments", "opp_stats_input_aug",
        "outputs", "pool_random", "pool_priors.json")
    priors_path = os.path.join(out_dir, "pool_priors.json")
    opponents   = build_standard_opponents(ROOT)

    if os.path.exists(priors_path):
        with open(priors_path) as f:
            pool_means = json.load(f)
    elif os.path.exists(shared_priors):
        with open(shared_priors) as f:
            pool_means = json.load(f)
        with open(priors_path, "w") as f:
            json.dump(pool_means, f)
    else:
        pool_means = compute_pool_means(opponents, 50 if smoke else CAL_HANDS)
        with open(priors_path, "w") as f:
            json.dump(pool_means, f)

    # Agent + optimiser
    agent     = FullModulationDeepAgent(base_ckpt=base_ckpt)
    optimizer = optim.Adam(agent.mod.parameters(), lr=LR)
    criterion = nn.MSELoss()
    agent.set_train_mode(True)

    sampler = WeightedOpponentSampler(opponents, OPPONENT_WEIGHTS)
    session = SessionManager(SESSION_LENGTH, pool_means, PRIOR_STRENGTH, sampler)

    config = {
        "agent":         "FullModulationDeepAgent",
        "architecture":  "V_base_deep(frozen,15→64→64→64→1) + ModHead(22→32→32→1)",
        "base_ckpt":     base_ckpt,
        "num_episodes":  num_episodes,
        "learning_rate": LR,
        "batch_size":    BATCH_SIZE,
        "session_length": SESSION_LENGTH,
        "opponent_weights": OPPONENT_WEIGHTS,
        "seed":          seed,
        "smoke":         smoke,
    }
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    freqs = sampler.expected_frequencies()
    print(f"\n{'='*65}")
    print(f"  DALI — full_modulation_deep  seed={seed}")
    print(f"  base: {base_ckpt}")
    print(f"  {'SMOKE' if smoke else 'FULL'} | episodes={num_episodes:,} "
          f"batch={BATCH_SIZE} lr={LR}")
    print(f"{'='*65}")
    print("Opponent sampling weights:")
    for k, p in freqs.items():
        print(f"  {k:<20s} {p:.1%}")
    print()

    batch_data   = []
    log_path     = os.path.join(out_dir, "train_log.jsonl")
    last_loss    = last_gnorm = last_delta = None

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

            ep_loss = ep_gnorm = ep_delta = None
            if len(batch_data) >= BATCH_SIZE:
                lv, gv, dv = td_update(agent, optimizer, criterion, batch_data)
                batch_data.clear()
                last_loss = lv; last_gnorm = gv; last_delta = dv
                ep_loss = round(lv, 6); ep_gnorm = round(gv, 6); ep_delta = round(dv, 6)

            log_file.write(json.dumps({
                "ep": ep, "opp": opp_name,
                "session": session._counts[learner_id],
                "reward": float(reward),
                "confidence": round(confidence, 6),
                "loss": ep_loss, "grad_norm": ep_gnorm, "delta_mean": ep_delta,
            }) + "\n")

            if ep % CKPT_INTERVAL == 0:
                agent.save_model(os.path.join(ckpt_dir, f"checkpoint_ep{ep:06d}.pt"))
                l_str = f"{last_loss:.4f}" if last_loss is not None else "n/a"
                g_str = f"{last_gnorm:.4f}" if last_gnorm is not None else "n/a"
                d_str = f"{last_delta:.4f}" if last_delta is not None else "n/a"
                print(f"[ep={ep:,}] loss={l_str} gnorm={g_str} "
                      f"delta={d_str} opp={opp_name} conf={confidence:.4f}")

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
