"""
Actor-Critic v2 — pool training.

Replaces self-play with weighted opponent pool (same recipe as value_based_pool):
  CFR ×3, heuristic ×3, 6 rule-based ×1 each.

Algorithm unchanged: online TD(0) advantage at each learner step.
Episodes increased to 500K.

Usage:
    python -m paper.baselines.actor_critic.train_v2 [--smoke] [--seed N]
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.leduc_game import Action, LeducGame
from paper.baselines.actor_critic.agent import ActorCriticAgent
from paper.evaluation.comparison_protocol import build_standard_opponents
from paper.evaluation.shared.training_recipe import (
    OPPONENT_WEIGHTS, WeightedOpponentSampler,
)

# ── hyperparameters ────────────────────────────────────────────────────────────
LR              = 1e-4
NUM_EPISODES    = 500_000
GAMMA           = 0.99
ENTROPY_COEF    = 0.01
VALUE_LOSS_COEF = 0.5
PRINT_EVERY     = 20_000
CKPT_EVERY      = 50_000
SMOKE_EPS       = 500


def train_step(agent, optimizer, obs_enc, action_idx, reward, next_obs_enc,
               done, legal_actions):
    logits, value = agent.forward(obs_enc)
    logits  = logits.squeeze(0)
    value   = value.squeeze()

    with torch.no_grad():
        next_value = torch.tensor(0.0) if done else agent.forward(next_obs_enc)[1].squeeze()

    advantage = reward + GAMMA * next_value - value

    mask = torch.full((3,), -1e9)
    for a in legal_actions:
        mask[a.value] = 0.0
    masked_logits = logits + mask
    log_prob = torch.log_softmax(masked_logits, dim=-1)[action_idx]

    policy_loss = -log_prob * advantage.detach()
    value_loss  = advantage.pow(2)

    probs   = torch.softmax(masked_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum()

    loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def play_episode(agent: ActorCriticAgent, optimizer, opponent, learner_id: int):
    game = LeducGame()
    game.reset()
    total_loss = 0.0
    num_steps  = 0
    pending    = None  # (enc, action_idx, legal_actions)

    while not game.is_finished:
        cp  = game.current_player
        obs = game.get_observation(viewer_id=cp)

        if cp == learner_id:
            enc    = agent.encode_observation(obs, viewer_id=cp)
            logits, _ = agent.forward(enc)
            logits = logits.squeeze(0)
            mask   = torch.full((3,), -1e9)
            for a in obs.legal_actions:
                mask[a.value] = 0.0
            probs  = torch.softmax(logits + mask, dim=-1)
            action_idx = torch.distributions.Categorical(probs).sample().item()

            if pending is not None:
                prev_enc, prev_act, prev_legal = pending
                total_loss += train_step(agent, optimizer, prev_enc, prev_act,
                                         0.0, enc, False, prev_legal)
                num_steps += 1

            pending = (enc, action_idx, obs.legal_actions)
            action  = Action(action_idx)
        else:
            action = opponent.select_action(obs)

        game.step(action)

    reward = game.get_reward()[learner_id]

    if pending is not None:
        prev_enc, prev_act, prev_legal = pending
        zeros = torch.zeros_like(prev_enc)
        total_loss += train_step(agent, optimizer, prev_enc, prev_act,
                                  float(reward), zeros, True, prev_legal)
        num_steps += 1

    return total_loss / max(num_steps, 1), reward


def train(num_episodes: int, output_dir: str, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    opponents = build_standard_opponents(ROOT)
    sampler   = WeightedOpponentSampler(opponents, OPPONENT_WEIGHTS)

    agent     = ActorCriticAgent()
    agent.set_train_mode(True)
    optimizer = optim.Adam(agent.parameters(), lr=LR)

    log_file = open(os.path.join(output_dir, 'train_log.jsonl'), 'w')

    running_reward = 0.0
    running_loss   = 0.0
    t0 = time.time()

    for ep in range(1, num_episodes + 1):
        learner_id         = ep % 2
        opp_name, opponent = sampler.sample()

        avg_loss, reward = play_episode(agent, optimizer, opponent, learner_id)

        running_reward += reward
        running_loss   += avg_loss

        log_file.write(json.dumps({
            'ep': ep, 'opp': opp_name, 'reward': reward, 'loss': avg_loss,
        }) + '\n')

        if ep % PRINT_EVERY == 0:
            elapsed = time.time() - t0
            print(f"[A2C-v2] ep={ep:,}/{num_episodes:,}  "
                  f"avg_r={running_reward/PRINT_EVERY:+.4f}  "
                  f"avg_loss={running_loss/PRINT_EVERY:.4f}  "
                  f"({elapsed:.0f}s)")
            running_reward = 0.0
            running_loss   = 0.0

        if ep % CKPT_EVERY == 0:
            agent.save_model(os.path.join(output_dir, 'checkpoints', f'ep{ep:06d}.pt'))

    agent.save_model(os.path.join(output_dir, 'checkpoint_final.pt'))
    log_file.close()
    print(f"[A2C-v2] Done → {output_dir}  ({time.time()-t0:.0f}s total)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seed',  type=int, default=0)
    args = parser.parse_args()

    n_ep    = SMOKE_EPS if args.smoke else NUM_EPISODES
    out_dir = os.path.join(HERE, 'outputs_v2', f'seed_{args.seed}')
    print(f"[A2C-v2] seed={args.seed}  episodes={n_ep:,}  → {out_dir}")
    train(n_ep, out_dir, args.seed)
