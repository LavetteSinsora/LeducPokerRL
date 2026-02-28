#!/usr/bin/env python3
"""
Investigate the non-monotonic pattern: n=3 is worse than both TD(0) and MC.

Hypothesis: The problem is the MIXTURE of target types. With n=3:
  - 99% of transitions use terminal reward (high variance, unbiased)
  - 1% of transitions use bootstrapped V(s_{t+n}) (low variance, biased)

This creates an inconsistent training signal. The network tries to fit
two very different distributions simultaneously. Pure TD(0) or pure MC
would be more consistent.

With n=2: 93% terminal, 7% bootstrap -> still mostly inconsistent
With n=1: 65% terminal, 35% bootstrap -> more balanced mixture
MC: 100% terminal -> consistent (but high variance)
TD(0): ~65% terminal, ~35% bootstrap -> same as n=1, but TD(0)'s bootstrap
targets are from the NEXT state (strongly correlated), creating smoother targets.

Let's measure target bimodality for different n values.
"""

import sys, os, random, copy
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.leduc_game import LeducGame, Action
from src.agents.value_based import ValueBasedAgent
from src.agents.nstep_value import NStepValueAgent


def analyze_target_distribution():
    """For a partially-trained agent, look at the actual target values."""
    print("=" * 70)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Train a value agent briefly so the bootstrap targets are non-trivial
    from src.training.value_based_trainer import SelfPlayTrainer
    agent = ValueBasedAgent()
    trainer = SelfPlayTrainer(agent, learning_rate=1e-4)

    # Train for 2K episodes to get a rough value function
    agent.set_train_mode(True)
    batch_data = []
    for ep in range(2000):
        traj = trainer.collect_episode()
        batch_data.append(traj)
        if len(batch_data) >= 32:
            trainer.update_model(batch_data)
            batch_data = []

    # Now collect episodes and analyze targets
    agent.set_train_mode(True)  # keep in train mode for exploration
    game = LeducGame()

    episodes_data = []
    for _ in range(1000):
        game.reset()
        chains = [[], []]
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)
            action = random.choice(obs.legal_actions)
            post_obs, _ = LeducGame.simulate_action(obs, action)
            encoded = agent.encode_observation(post_obs, viewer_id=cp)
            chains[cp].append(encoded)
            game.step(action)
        rewards = game.get_reward()
        episodes_data.append((chains, rewards))

    for n in [1, 2, 3, 100]:
        label = f"n={n}" if n < 100 else "MC"
        bootstrap_targets = []
        terminal_targets = []

        for chains, rewards in episodes_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                L = len(chain)
                for t in range(L):
                    if t + n >= L:
                        terminal_targets.append(rewards[p_idx])
                    else:
                        with torch.no_grad():
                            val = agent.model(chain[t + n]).squeeze(0).item()
                        bootstrap_targets.append(val)

        all_targets = bootstrap_targets + terminal_targets
        bt = np.array(bootstrap_targets) if bootstrap_targets else np.array([])
        tt = np.array(terminal_targets)
        at = np.array(all_targets)

        print(f"\n  {label}:")
        print(f"    Bootstrap targets: N={len(bt)}")
        if len(bt) > 0:
            print(f"      mean={bt.mean():.4f}, std={bt.std():.4f}, range=[{bt.min():.2f}, {bt.max():.2f}]")
        print(f"    Terminal targets:  N={len(tt)}")
        print(f"      mean={tt.mean():.4f}, std={tt.std():.4f}, range=[{tt.min():.2f}, {tt.max():.2f}]")
        print(f"    Combined targets: N={len(at)}")
        print(f"      mean={at.mean():.4f}, std={at.std():.4f}")

        # Measure bimodality: how different are bootstrap vs terminal distributions?
        if len(bt) > 0:
            mean_gap = abs(bt.mean() - tt.mean())
            std_ratio = bt.std() / tt.std() if tt.std() > 0 else float('inf')
            print(f"    BIMODALITY: mean_gap={mean_gap:.4f}, std_ratio(boot/term)={std_ratio:.4f}")


def analyze_gradient_magnitude():
    """Measure gradient magnitudes under different n to see if MC causes larger/noisier gradients."""
    print("\n" + "=" * 70)
    print("GRADIENT MAGNITUDE ANALYSIS")
    print("=" * 70)

    import torch.nn as nn

    ref_agent = ValueBasedAgent()
    init_weights = copy.deepcopy(ref_agent.model.state_dict())

    game = LeducGame()

    # Collect a fixed batch of episodes
    random.seed(42)
    episodes_data = []
    for _ in range(32):
        game.reset()
        chains = [[], []]
        while not game.is_finished:
            cp = game.current_player
            obs = game.get_observation(viewer_id=cp)
            action = random.choice(obs.legal_actions)
            post_obs, _ = LeducGame.simulate_action(obs, action)
            encoded = ref_agent.encode_observation(post_obs, viewer_id=cp)
            chains[cp].append(encoded)
            game.step(action)
        rewards = game.get_reward()
        episodes_data.append((chains, rewards))

    criterion = nn.MSELoss()

    for n in [1, 3, 100]:
        label = f"n={n}" if n < 100 else "MC"

        # Fresh agent with same init weights
        agent = ValueBasedAgent()
        agent.model.load_state_dict(copy.deepcopy(init_weights))
        agent.model.train()

        # Compute loss and gradients for this batch
        total_losses = []
        for chains, rewards in episodes_data:
            for p_idx in [0, 1]:
                chain = chains[p_idx]
                L = len(chain)
                for t in range(L):
                    prediction = agent.model(chain[t]).squeeze(0)
                    if t + n >= L:
                        target = torch.FloatTensor([rewards[p_idx]])
                    else:
                        with torch.no_grad():
                            target = agent.model(chain[t + n]).squeeze(0)
                    loss = criterion(prediction, target)
                    total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            agent.model.zero_grad()
            mean_loss.backward()

            # Measure gradient norms
            total_norm = 0
            param_norms = {}
            for name, param in agent.model.named_parameters():
                if param.grad is not None:
                    pn = param.grad.data.norm(2).item()
                    param_norms[name] = pn
                    total_norm += pn ** 2
            total_norm = total_norm ** 0.5

            print(f"\n  {label}:")
            print(f"    Loss: {mean_loss.item():.4f}")
            print(f"    Total gradient norm: {total_norm:.4f}")
            for name, norm in param_norms.items():
                print(f"      {name}: {norm:.4f}")


def analyze_training_oscillation():
    """Track value predictions on a fixed probe set during training for different n."""
    print("\n" + "=" * 70)
    print("VALUE OSCILLATION ANALYSIS")
    print("=" * 70)
    print("  Tracking V(probe_states) every 500 episodes during training")

    from src.training.nstep_value_trainer import NStepValueTrainer
    from src.training.value_based_trainer import SelfPlayTrainer

    ref_agent = ValueBasedAgent()
    init_weights = copy.deepcopy(ref_agent.model.state_dict())

    # Create fixed probe states
    game = LeducGame()
    random.seed(99)
    probe_states = []
    probe_labels = []
    for hand in ['J', 'Q', 'K']:
        for board in [None, 'J', 'Q', 'K']:
            obs = type('Obs', (), {
                'player_hand': hand,
                'board': board,
                'pot': [1, 1],
                'current_player': 0,
                'current_round': 0 if board is None else 1,
                'legal_actions': [],
                'is_finished': False,
                'raises_this_round': 0,
                'opponent_stats': None,
            })()
            enc = ref_agent.encode_observation(obs, viewer_id=0)
            probe_states.append(enc)
            probe_labels.append(f"{hand}/{board or '-'}")

    for n_config in [(1, False, "TD(0)"), (3, False, "n=3"), (100, False, "MC")]:
        n, _, label = n_config
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        agent = NStepValueAgent()
        agent.model.load_state_dict(copy.deepcopy(init_weights))
        trainer = NStepValueTrainer(agent, learning_rate=1e-4, n_steps=n)

        agent.set_train_mode(True)
        batch_data = []
        value_tracks = {l: [] for l in probe_labels}

        for ep in range(5000):
            traj = trainer.collect_episode()
            batch_data.append(traj)
            if len(batch_data) >= 32:
                trainer.update_model(batch_data)
                batch_data = []

            if (ep + 1) % 500 == 0:
                agent.model.eval()
                with torch.no_grad():
                    for enc, lbl in zip(probe_states, probe_labels):
                        val = agent.model(enc).item()
                        value_tracks[lbl].append(val)
                agent.model.train()

        print(f"\n  {label}:")
        print(f"    {'State':<10}", end="")
        for ep_idx in range(10):
            print(f"  ep{(ep_idx+1)*500}", end="")
        print(f"  | range")

        for lbl in probe_labels:
            vals = value_tracks[lbl]
            print(f"    {lbl:<10}", end="")
            for v in vals:
                print(f"  {v:>+6.2f}", end="")
            vrange = max(vals) - min(vals)
            print(f"  | {vrange:.2f}")

        # Summary: average oscillation (range across epochs)
        avg_range = np.mean([max(value_tracks[l]) - min(value_tracks[l]) for l in probe_labels])
        print(f"    Average value range across training: {avg_range:.3f}")


if __name__ == "__main__":
    analyze_target_distribution()
    analyze_gradient_magnitude()
    analyze_training_oscillation()
