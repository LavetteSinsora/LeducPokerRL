"""Value-based network representation analysis.

Measures how well the TD(0)-trained value network's internal representation
aligns with reward structure, compared to explicitly contrastive-trained encoders.

Metrics:
  1. Scalar Spearman rho: V(s) output vs true terminal reward R
  2. Hidden Spearman rho: pairwise L2 distance in 64-dim penultimate layer vs pairwise |delta_R|
  3. Raw 15-dim Spearman rho: baseline using raw input features (pairwise L2 vs pairwise |delta_R|)

Run:
    python -m experiments.representation_learning.value_based_repr_analysis.analyze \
        --checkpoint agents/value_based/checkpoint.pt \
        --num-states 1000 \
        --num-pairs 2000 \
        --output-dir outputs/value_based_repr_analysis
"""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from agents.value_based.agent import ValueBasedAgent
from engine.leduc_game import LeducGame


# ── Penultimate layer hook ────────────────────────────────────────────────────

_penultimate_activations: list = []


def _hook_fn(module: nn.Module, input, output):
    """Forward hook: captures 64-dim output of second ReLU."""
    _penultimate_activations.append(output.detach().squeeze(0))  # [64]


def register_penultimate_hook(agent: ValueBasedAgent):
    """Attach a forward hook to the second ReLU in the value network.

    ValueNetwork.net layout:
        net[0] = Linear(15, 64)
        net[1] = ReLU           <- first ReLU
        net[2] = Linear(64, 64)
        net[3] = ReLU           <- second ReLU (penultimate, 64-dim)
        net[4] = Linear(64, 1)
    """
    second_relu = agent.model.net[3]
    handle = second_relu.register_forward_hook(_hook_fn)
    return handle


# ── Data collection ───────────────────────────────────────────────────────────

def collect_states(agent: ValueBasedAgent, num_states: int, seed: int = 42):
    """Generate game states via self-play.

    For each terminal game, records post-action states encountered along the
    way, paired with the terminal reward for the acting player.

    Returns:
        raw_features:      np.ndarray [N, 15]
        value_outputs:     np.ndarray [N]    -- V(s) from value network
        hidden_activations: np.ndarray [N, 64] -- penultimate layer
        terminal_rewards:  np.ndarray [N]    -- true outcome for the player
    """
    random.seed(seed)
    torch.manual_seed(seed)

    game = LeducGame()
    agent.set_train_mode(False)

    raw_list = []
    value_list = []
    hidden_list = []
    reward_list = []

    episodes_played = 0

    while len(raw_list) < num_states:
        game.reset()
        episode_states = []  # (raw_tensor, player_id)

        while not game.is_finished:
            current_player = game.current_player
            obs = game.get_observation(viewer_id=current_player)
            action = agent.select_action(obs)
            if isinstance(action, tuple):
                action = action[0]

            # Post-action state (same logic as trainer)
            from engine.leduc_game import LeducGame as LG, Action
            post_obs, _ = LG.simulate_action(obs, action)
            encoded = agent.encode_observation(post_obs, viewer_id=current_player)  # [1, 15]

            # Capture hidden activations via hook
            _penultimate_activations.clear()
            with torch.no_grad():
                v_out = agent.model(encoded).item()
            hidden = _penultimate_activations[0].numpy()  # [64]

            episode_states.append({
                'raw': encoded.squeeze(0).numpy(),  # [15]
                'v_out': v_out,
                'hidden': hidden,
                'player_id': current_player,
            })

            game.step(action)

        rewards = game.get_reward()
        episodes_played += 1

        for s in episode_states:
            r = float(rewards[s['player_id']])
            raw_list.append(s['raw'])
            value_list.append(s['v_out'])
            hidden_list.append(s['hidden'])
            reward_list.append(r)

            if len(raw_list) >= num_states:
                break

    raw_arr = np.array(raw_list[:num_states], dtype=np.float32)
    value_arr = np.array(value_list[:num_states], dtype=np.float32)
    hidden_arr = np.array(hidden_list[:num_states], dtype=np.float32)
    reward_arr = np.array(reward_list[:num_states], dtype=np.float32)

    print(f"Collected {len(raw_arr)} states from {episodes_played} episodes.")
    return raw_arr, value_arr, hidden_arr, reward_arr


# ── Spearman rho helpers ───────────────────────────────────────────────────────

def scalar_spearman(v_values: np.ndarray, rewards: np.ndarray):
    """Spearman rho between V(s) scalar and terminal reward."""
    rho, p = spearmanr(v_values, rewards)
    return float(rho), float(p)


def pairwise_spearman(embeddings: np.ndarray, rewards: np.ndarray,
                      max_pairs: int = 2000, seed: int = 0):
    """Spearman rho between pairwise L2 distances and pairwise |delta_R|.

    Subsample `max_pairs` pairs from N*(N-1)/2 if there are too many.
    """
    rng = np.random.default_rng(seed)
    N = len(embeddings)
    total_pairs = N * (N - 1) // 2

    if total_pairs <= max_pairs:
        # All pairs
        i_idx, j_idx = np.triu_indices(N, k=1)
    else:
        # Random subsample
        all_i, all_j = np.triu_indices(N, k=1)
        chosen = rng.choice(len(all_i), size=max_pairs, replace=False)
        i_idx = all_i[chosen]
        j_idx = all_j[chosen]

    emb_dists = np.linalg.norm(embeddings[i_idx] - embeddings[j_idx], axis=1)
    reward_diffs = np.abs(rewards[i_idx] - rewards[j_idx])

    rho, p = spearmanr(emb_dists, reward_diffs)
    return float(rho), float(p), len(i_idx)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze value_based network representation")
    parser.add_argument("--checkpoint", type=str,
                        default="agents/value_based/checkpoint.pt",
                        help="Path to value_based checkpoint.pt")
    parser.add_argument("--num-states", type=int, default=1000,
                        help="Number of game states to collect")
    parser.add_argument("--num-pairs", type=int, default=2000,
                        help="Max pairs for pairwise Spearman computation")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / 'outputs'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load agent
    print(f"Loading checkpoint: {args.checkpoint}")
    agent = ValueBasedAgent(model_path=args.checkpoint)
    agent.model.eval()

    # Register penultimate layer hook
    hook_handle = register_penultimate_hook(agent)

    # Collect states
    print(f"Collecting {args.num_states} game states via self-play...")
    raw_arr, value_arr, hidden_arr, reward_arr = collect_states(
        agent, num_states=args.num_states
    )

    # Deregister hook
    hook_handle.remove()

    # --- Metric 1: Scalar Spearman rho ---
    scalar_rho, scalar_p = scalar_spearman(value_arr, reward_arr)

    # --- Metric 2: Hidden Spearman rho (64-dim) ---
    hidden_rho, hidden_p, n_pairs_hidden = pairwise_spearman(
        hidden_arr, reward_arr, max_pairs=args.num_pairs
    )

    # --- Metric 3: Raw 15-dim Spearman rho (baseline) ---
    raw_rho, raw_p, n_pairs_raw = pairwise_spearman(
        raw_arr, reward_arr, max_pairs=args.num_pairs
    )

    # --- Print results ---
    print()
    print("=== Value-Based Network Representation Analysis ===")
    print(f"Scalar: Spearman rho (V(s) vs true R):          {scalar_rho:+.3f}  (p={scalar_p:.3e})")
    print(f"Hidden: Spearman rho (64-dim layer vs |dR|):    {hidden_rho:+.3f}  (p={hidden_p:.3e})")
    print(f"Raw:    Spearman rho (15-dim features vs |dR|): {raw_rho:+.3f}  (p={raw_p:.3e})")
    print()
    print("Comparison to contrastive encoders:")
    print("  contrastive_repr_v1 (L1, 8-dim):  0.163")
    print("  dual_axis_repr_v4   (EMA, 8-dim): 0.672")
    print("===================================================")

    # --- Save results ---
    results = {
        "checkpoint": args.checkpoint,
        "num_states": int(len(raw_arr)),
        "scalar_spearman": {
            "rho": scalar_rho,
            "p": scalar_p,
            "description": "V(s) scalar output vs true terminal reward R",
        },
        "hidden_spearman": {
            "rho": hidden_rho,
            "p": hidden_p,
            "n_pairs": n_pairs_hidden,
            "description": "Pairwise L2 distance in 64-dim penultimate layer vs pairwise |delta_R|",
        },
        "raw_spearman": {
            "rho": raw_rho,
            "p": raw_p,
            "n_pairs": n_pairs_raw,
            "description": "Pairwise L2 distance in 15-dim raw features vs pairwise |delta_R|",
        },
        "contrastive_baselines": {
            "contrastive_repr_v1_L1": 0.163,
            "dual_axis_repr_v4_EMA": 0.672,
        },
    }

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
