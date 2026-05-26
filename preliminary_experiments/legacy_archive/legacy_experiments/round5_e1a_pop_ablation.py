"""
Round 5 -- Ablation: Pure Nash CFR + Population Training + 40K Sessions

This experiment resolves a critical confound from E1b (belief_modulated).

E1b changed THREE variables from E1a simultaneously:
  1. Likelihood source: pure Nash -> Nash + learned modulation
  2. Training methodology: self-play -> population-based
  3. Training duration: 40K episodes (~60K hands) -> 40K sessions (~1.2M hands)

This ablation uses:
  - Pure Nash CFR likelihoods (NO modulation) -- same as E1a
  - Population-based training (heuristic, value_based, adaptive_value) -- same as E1b
  - 40K sessions (~1.2M hands) -- same as E1b

If this ablation achieves similar performance to E1b (+0.315), then modulation
is a no-op and population+duration explain everything.
If this is much worse than E1b, modulation is genuinely contributing.

Protocol:
  1. Train BeliefCfrAgent for 40K sessions against rotating opponents
  2. Evaluate against all major opponents (500 rounds each, both positions)
  3. Run diagnostics: belief correctness, shift, action distribution
  4. Save results + model
"""

import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.leduc_game import LeducGame, Action
from src.engine.observation import Observation
from src.engine.poker_session import PokerSession
from src.agents.belief_cfr import BeliefCfrAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.value_based import ValueBasedAgent
from src.training.evaluation import evaluate_agents, compute_robustness_metrics


# -----------------------------------------------
# Configuration
# -----------------------------------------------

TRAIN_SESSIONS = 40000  # sessions (each session = 30 hands = ~1.2M total hands)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
HANDS_PER_SESSION = 30
ROTATE_EVERY = 100  # rotate opponent every N sessions
EVAL_ROUNDS = 500
CFR_MODEL_PATH = "models/cfr_agent.pt"
MODEL_PATH = "models/belief_cfr_pop_ablation_agent.pt"
RESULTS_PATH = "experiments/round5_e1a_pop_ablation_results.json"

OPPONENTS = {
    "heuristic": {"class": "HeuristicAgent", "model_path": None},
    "value_based": {"class": "ValueBasedAgent", "model_path": "models/value_based_agent.pt"},
    "adaptive_value": {"class": "AdaptiveValueAgent", "model_path": "models/adaptive_value_agent.pt"},
    "modulated_value": {"class": "ModulatedValueAgent", "model_path": "models/modulated_value_agent.pt"},
    "entropy_ac": {"class": "EntropyACAgent", "model_path": "models/entropy_ac_agent.pt"},
    "cfr": {"class": "CFRAgent", "model_path": "models/cfr_agent.pt"},
}


# -----------------------------------------------
# Population-based trainer for BeliefCfrAgent
# -----------------------------------------------

class PopulationBeliefCfrTrainer:
    """
    Population-based trainer for BeliefCfrAgent.

    Same training methodology as BeliefModulatedTrainer but with NO modulation:
      - PokerSession for multi-hand sessions
      - Rotating opponent pool (heuristic, value_based, adaptive_value)
      - TD(0) on post-action state chains (value network only)
      - No modulation loss (pure Nash likelihoods, frozen)
    """

    def __init__(self, agent: BeliefCfrAgent, learning_rate: float = 1e-4,
                 hands_per_session: int = 30, rotate_every: int = 100):
        self.agent = agent
        self.hands_per_session = hands_per_session
        self.rotate_every = rotate_every
        self.session = PokerSession()

        # Value network optimizer (only thing we train)
        self.value_optimizer = optim.Adam(
            self.agent.model.parameters(), lr=learning_rate
        )
        self.value_criterion = nn.MSELoss()

        # Build opponent pool (same as E1b)
        self.opponent_pool = self._build_opponent_pool()
        self.current_opponent_idx = 0
        self.session_count = 0
        self.stop_requested = False

    def _build_opponent_pool(self):
        """Create the opponent pool from pre-trained agents."""
        pool = []

        # 1. Heuristic
        pool.append(("heuristic", HeuristicAgent()))

        # 2. Value-based
        vb = ValueBasedAgent()
        vb_path = "models/value_based_agent.pt"
        if os.path.exists(vb_path):
            vb.load_model(vb_path)
        vb.set_train_mode(False)
        pool.append(("value_based", vb))

        # 3. Adaptive value
        from src.agents.adaptive_value import AdaptiveValueAgent
        av = AdaptiveValueAgent()
        av_path = "models/adaptive_value_agent.pt"
        if os.path.exists(av_path):
            av.load_model(av_path)
        av.set_train_mode(False)
        pool.append(("adaptive_value", av))

        return pool

    def _get_current_opponent(self):
        return self.opponent_pool[self.current_opponent_idx][1]

    def _maybe_rotate_opponent(self):
        if self.session_count > 0 and self.session_count % self.rotate_every == 0:
            self.current_opponent_idx = (
                (self.current_opponent_idx + 1) % len(self.opponent_pool)
            )
            name = self.opponent_pool[self.current_opponent_idx][0]
            print(f"  Rotating opponent to: {name}")

    def collect_session(self):
        """
        Play one session of hands against a pool opponent.

        Returns a list of (chain, reward) tuples for player 0.
        """
        self.session.reset()
        session_data = []
        opponent = self._get_current_opponent()

        for _ in range(self.hands_per_session):
            self.session.new_hand()
            chain = []  # post-action states for player 0

            while not self.session.is_finished:
                current_player = self.session.current_player
                obs = self.session.get_observation(viewer_id=current_player)

                if current_player == 0:
                    # Training agent (player 0)
                    action = self.agent.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]

                    # Record post-action state for TD(0)
                    belief = self.agent.compute_belief_from_history(obs)
                    post_obs, _ = LeducGame.simulate_action(obs, action)
                    if obs.opponent_stats is not None:
                        post_obs = replace(post_obs, opponent_stats=obs.opponent_stats)
                    encoded = self.agent.encode_observation(
                        post_obs, viewer_id=current_player, belief=belief
                    )
                    chain.append(encoded)
                else:
                    # Opponent from pool
                    action = opponent.select_action(obs)
                    if isinstance(action, tuple):
                        action = action[0]

                self.session.step(action)

            rewards = self.session.game.get_reward()
            session_data.append((chain, rewards[0]))

        self.session_count += 1
        self._maybe_rotate_opponent()
        return session_data

    def update_model(self, batch_data):
        """TD(0) update on post-action state chains."""
        self.value_optimizer.zero_grad()
        total_losses = []

        for session_data in batch_data:
            for chain, reward in session_data:
                if not chain:
                    continue

                for t in range(len(chain)):
                    prediction = self.agent.model(chain[t]).squeeze(0)

                    if t == len(chain) - 1:
                        target = torch.FloatTensor([reward])
                    else:
                        with torch.no_grad():
                            target = self.agent.model(chain[t + 1]).squeeze(0)

                    loss = self.value_criterion(prediction, target)
                    total_losses.append(loss)

        if total_losses:
            mean_loss = torch.stack(total_losses).mean()
            mean_loss.backward()
            self.value_optimizer.step()
            return mean_loss.item()
        return 0.0

    def evaluate(self, num_games: int = 100):
        """Evaluate against heuristic."""
        self.agent.set_train_mode(False)
        result = evaluate_agents(self.agent, HeuristicAgent(), num_rounds=num_games)
        self.agent.set_train_mode(True)
        return result.agent_0_avg_chips

    def train(self, num_sessions: int, batch_size: int = 32, save_path: str = None,
              callback=None):
        """Session-based training loop with population rotation."""
        self.agent.set_train_mode(True)
        self.stop_requested = False

        batch_data = []
        episode_counter = 0
        eval_every = max(1, 50 // self.hands_per_session)

        for session_idx in range(num_sessions):
            if self.stop_requested:
                print("Training stop requested.")
                break

            session_data = self.collect_session()
            batch_data.append(session_data)
            episode_counter += len(session_data)

            if len(batch_data) >= batch_size // self.hands_per_session or \
               len(batch_data) * self.hands_per_session >= batch_size:
                loss = self.update_model(batch_data)
                batch_data = []

                if callback:
                    callback({
                        "episode": episode_counter,
                        "loss": loss,
                        "type": "batch_update",
                    })

                if session_idx < 2 or episode_counter % 100 == 0:
                    print(f"Session {session_idx + 1}, Episode {episode_counter}, "
                          f"Batch Loss: {loss:.4f}")

            # Evaluate periodically
            if (session_idx + 1) % eval_every == 0:
                avg_chips = self.evaluate(num_games=100)
                if callback:
                    callback({
                        "episode": episode_counter,
                        "avg_chips_per_round": avg_chips,
                        "type": "evaluation",
                    })
                print(f"Session {session_idx + 1}, Episode {episode_counter}, "
                      f"Avg Chips/Round: {avg_chips:+.2f}")

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            self.agent.save_model(save_path)
            print(f"Model saved to {save_path}")


# -----------------------------------------------
# Training
# -----------------------------------------------

def train_agent():
    """Train the BeliefCfr Agent with population-based training."""
    print("=" * 60)
    print("  ABLATION: Pure Nash CFR + Population Training + 40K Sessions")
    print(f"  Sessions: {TRAIN_SESSIONS}, Hands/session: {HANDS_PER_SESSION}")
    print(f"  Total hands: ~{TRAIN_SESSIONS * HANDS_PER_SESSION:,}")
    print(f"  Value LR: {LEARNING_RATE}")
    print(f"  Opponent rotation: every {ROTATE_EVERY} sessions")
    print(f"  NO modulation -- pure Nash likelihoods only")
    print("=" * 60)

    agent = BeliefCfrAgent(
        cfr_path=CFR_MODEL_PATH if os.path.exists(CFR_MODEL_PATH) else None,
        temperature=1.0,
    )
    trainer = PopulationBeliefCfrTrainer(
        agent,
        learning_rate=LEARNING_RATE,
        hands_per_session=HANDS_PER_SESSION,
        rotate_every=ROTATE_EVERY,
    )

    losses = []
    eval_scores = []

    def callback(data):
        if data["type"] == "batch_update":
            losses.append({"episode": data["episode"], "loss": data["loss"]})
            if len(losses) % 100 == 0:
                print(f"    Update {len(losses)}: loss={data['loss']:.6f}")
        elif data["type"] == "evaluation":
            eval_scores.append({
                "episode": data["episode"],
                "avg_chips": data["avg_chips_per_round"]
            })

    start_time = time.time()
    trainer.train(
        num_sessions=TRAIN_SESSIONS,
        batch_size=BATCH_SIZE,
        save_path=MODEL_PATH,
        callback=callback,
    )
    elapsed = time.time() - start_time

    print(f"\n  Training complete in {elapsed:.1f}s")
    if losses:
        print(f"  Final loss: {losses[-1]['loss']:.6f}")
    if eval_scores:
        print(f"  Final eval vs heuristic: {eval_scores[-1]['avg_chips']:+.3f}")

    return {
        "training_time_s": round(elapsed, 1),
        "num_updates": len(losses),
        "final_loss": losses[-1]["loss"] if losses else None,
        "final_eval_vs_heuristic": eval_scores[-1]["avg_chips"] if eval_scores else None,
        "eval_history": eval_scores[-10:] if eval_scores else [],
    }


# -----------------------------------------------
# Evaluation
# -----------------------------------------------

def load_opponent(name, config):
    """Dynamically load an opponent agent."""
    if name == "heuristic":
        return HeuristicAgent()
    elif name == "value_based":
        agent = ValueBasedAgent()
        if config["model_path"] and os.path.exists(config["model_path"]):
            agent.load_model(config["model_path"])
        agent.set_train_mode(False)
        return agent
    elif name == "adaptive_value":
        from src.agents.adaptive_value import AdaptiveValueAgent
        agent = AdaptiveValueAgent()
        if config["model_path"] and os.path.exists(config["model_path"]):
            agent.load_model(config["model_path"])
        agent.set_train_mode(False)
        return agent
    elif name == "modulated_value":
        from src.preliminary_experiments.promoted_registry.modulated_value import ModulatedValueAgent
        agent = ModulatedValueAgent()
        if config["model_path"] and os.path.exists(config["model_path"]):
            agent.load_model(config["model_path"])
        agent.set_train_mode(False)
        return agent
    elif name == "entropy_ac":
        from src.agents.entropy_ac import EntropyACAgent
        agent = EntropyACAgent()
        if config["model_path"] and os.path.exists(config["model_path"]):
            agent.load_model(config["model_path"])
        agent.set_train_mode(False)
        return agent
    elif name == "cfr":
        from src.agents.cfr_agent import CFRAgent
        agent = CFRAgent()
        if config["model_path"] and os.path.exists(config["model_path"]):
            agent.load_model(config["model_path"])
        return agent
    else:
        raise ValueError(f"Unknown opponent: {name}")


def evaluate_against_opponents(agent):
    """Evaluate against all opponents."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION ({EVAL_ROUNDS} rounds per matchup, both positions)")
    print("=" * 60)

    agent.set_train_mode(False)
    results = {}

    for name, config in OPPONENTS.items():
        try:
            opponent = load_opponent(name, config)
        except Exception as e:
            print(f"  WARNING: Could not load {name}: {e}")
            continue

        result = evaluate_agents(agent, opponent, num_rounds=EVAL_ROUNDS)
        avg_chips = result.agent_0_avg_chips
        results[name] = round(avg_chips, 4)
        print(f"  vs {name:20s}: {avg_chips:+.4f} chips/round")

    robustness = compute_robustness_metrics(results)
    print(f"\n  Robustness metrics:")
    print(f"    Avg:       {robustness['avg']:+.4f}")
    print(f"    Worst:     {robustness['worst_case']:+.4f}")
    print(f"    Best:      {robustness['best_case']:+.4f}")
    print(f"    Std:       {robustness['std']:.4f}")
    print(f"    Robustness:{robustness['robustness']:+.4f}")

    return results, robustness


# -----------------------------------------------
# Diagnostics
# -----------------------------------------------

def diagnose_belief_quality(agent, num_games=500):
    """Measure belief correctness and shift."""
    print("\n" + "=" * 60)
    print("  DIAGNOSIS: Belief Quality")
    print("=" * 60)

    session = PokerSession()
    heuristic = HeuristicAgent()
    agent.set_train_mode(False)

    belief_correct_mass = []  # P(true_hand) at each decision point
    shift_magnitudes = []
    per_round_shifts = {0: [], 1: []}

    for game_idx in range(num_games):
        if game_idx % 30 == 0:
            session.reset()

        session.new_hand()
        prev_belief = None

        while not session.is_finished:
            cp = session.current_player
            obs = session.get_observation(viewer_id=cp)

            if cp == 0:
                belief = agent.compute_belief_from_history(obs)

                # Record belief correctness (P assigned to true hand)
                opp_hand = session.game.player_hands[1]
                opp_idx = agent.CARD_MAP.get(opp_hand)
                if opp_idx is not None:
                    belief_correct_mass.append(belief[opp_idx])

                # Record belief shift
                if prev_belief is not None:
                    shift = np.sum(np.abs(belief - prev_belief))
                    shift_magnitudes.append(shift)
                    rnd = obs.current_round
                    if rnd in per_round_shifts:
                        per_round_shifts[rnd].append(shift)
                prev_belief = belief.copy()

                action = agent.select_action(obs)
            else:
                action = heuristic.select_action(obs)

            if isinstance(action, tuple):
                action = action[0]
            session.step(action)

    avg_correctness = np.mean(belief_correct_mass) if belief_correct_mass else 0
    avg_shift = np.mean(shift_magnitudes) if shift_magnitudes else 0
    std_shift = np.std(shift_magnitudes) if shift_magnitudes else 0

    print(f"  Belief correctness (avg P(true_hand)): {avg_correctness:.4f}")
    print(f"    Random baseline: 0.50")
    print(f"    Status: {'ABOVE random' if avg_correctness > 0.50 else 'BELOW random'}")
    print(f"  Avg belief shift (L1): {avg_shift:.4f} +/- {std_shift:.4f}")

    for rnd in [0, 1]:
        if per_round_shifts[rnd]:
            avg = np.mean(per_round_shifts[rnd])
            print(f"  Round {rnd} avg shift: {avg:.4f} (n={len(per_round_shifts[rnd])})")

    return {
        "belief_correctness": round(float(avg_correctness), 4),
        "avg_belief_shift_l1": round(float(avg_shift), 4),
        "std_belief_shift_l1": round(float(std_shift), 4),
        "total_decision_points": len(belief_correct_mass),
        "per_round_shift": {
            f"round_{r}": round(float(np.mean(s)), 4) if s else 0.0
            for r, s in per_round_shifts.items()
        },
    }


def diagnose_action_distribution(agent, num_games=1000):
    """Analyze action distribution per hand card."""
    print("\n" + "=" * 60)
    print("  DIAGNOSIS: Action Distribution Per Hand")
    print("=" * 60)

    session = PokerSession()
    heuristic = HeuristicAgent()
    agent.set_train_mode(False)

    action_counts = {}
    for card in ['J', 'Q', 'K']:
        action_counts[card] = {
            0: {0: 0, 1: 0, 2: 0},
            1: {0: 0, 1: 0, 2: 0},
        }

    for game_idx in range(num_games):
        if game_idx % 30 == 0:
            session.reset()

        session.new_hand()
        while not session.is_finished:
            cp = session.current_player
            obs = session.get_observation(viewer_id=cp)

            if cp == 0:
                action = agent.select_action(obs)
                hand = obs.player_hand
                rnd = obs.current_round
                if hand in action_counts and rnd in action_counts[hand]:
                    action_counts[hand][rnd][int(action)] += 1
            else:
                action = heuristic.select_action(obs)

            if isinstance(action, tuple):
                action = action[0]
            session.step(action)

    action_names = {0: "FOLD", 1: "CALL", 2: "RAISE"}
    round_names = {0: "Preflop", 1: "Flop"}

    results = {}
    for card in ['J', 'Q', 'K']:
        results[card] = {}
        print(f"\n  Hand: {card}")
        for rnd in [0, 1]:
            counts = action_counts[card][rnd]
            total = sum(counts.values())
            if total == 0:
                continue
            dist = {action_names[a]: round(counts[a] / total, 3) for a in range(3)}
            results[card][round_names[rnd]] = dist
            dist_str = "  ".join(f"{n}:{p:.3f}" for n, p in dist.items())
            print(f"    {round_names[rnd]:>7s}: {dist_str}  (n={total})")

    return results


# -----------------------------------------------
# Main
# -----------------------------------------------

def main():
    print("=" * 60)
    print("  ROUND 5 -- ABLATION: Pure Nash + Population + 40K Sessions")
    print("  (Resolves modulation confound from E1b)")
    print("=" * 60)

    all_results = {
        "agent": "belief_cfr_pop_ablation",
        "experiment": "E1a_pop_ablation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Isolate modulation contribution: pure Nash + population + 40K sessions",
        "config": {
            "train_sessions": TRAIN_SESSIONS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "hands_per_session": HANDS_PER_SESSION,
            "rotate_every": ROTATE_EVERY,
            "eval_rounds": EVAL_ROUNDS,
            "cfr_model_path": CFR_MODEL_PATH,
            "modulation": False,  # KEY: no modulation
        },
        "comparison": {
            "E1a_self_play": {"avg": -0.724, "training": "40K episodes, self-play"},
            "E1b_modulated": {"avg": 0.315, "training": "40K sessions, population, modulation"},
        },
    }

    # Phase 1: Training
    print(f"\n{'#' * 60}")
    print(f"  PHASE 1: TRAINING")
    print(f"{'#' * 60}")
    training_results = train_agent()
    all_results["training"] = training_results

    # Phase 2: Evaluation
    print(f"\n{'#' * 60}")
    print(f"  PHASE 2: EVALUATION")
    print(f"{'#' * 60}")
    agent = BeliefCfrAgent(
        model_path=MODEL_PATH,
        cfr_path=CFR_MODEL_PATH if os.path.exists(CFR_MODEL_PATH) else None,
    )
    agent.set_train_mode(False)
    eval_results, robustness = evaluate_against_opponents(agent)
    all_results["evaluation"] = eval_results
    all_results["robustness"] = robustness

    # Phase 3: Diagnostics
    print(f"\n{'#' * 60}")
    print(f"  PHASE 3: DIAGNOSTICS")
    print(f"{'#' * 60}")
    belief_diag = diagnose_belief_quality(agent)
    all_results["diagnostics"] = {"belief_quality": belief_diag}

    action_diag = diagnose_action_distribution(agent)
    all_results["diagnostics"]["action_distribution"] = action_diag

    # Phase 4: Save results
    os.makedirs(os.path.dirname(RESULTS_PATH) if os.path.dirname(RESULTS_PATH) else ".", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Summary & interpretation
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY & INTERPRETATION")
    print(f"{'=' * 60}")
    print(f"  Training time: {training_results['training_time_s']}s")
    print(f"\n  Evaluation (avg chips/round):")
    for name, score in eval_results.items():
        print(f"    vs {name:20s}: {score:+.4f}")
    print(f"\n  Avg: {robustness['avg']:+.4f}")
    print(f"  Robustness: {robustness['robustness']:+.4f}")
    print(f"  Belief correctness: {belief_diag['belief_correctness']:.4f}")

    # Interpretation
    ablation_avg = robustness['avg']
    e1b_avg = 0.315
    e1a_avg = -0.724

    print(f"\n  --- Confound Resolution ---")
    print(f"  E1a (Nash + self-play + 40K ep):      {e1a_avg:+.3f}")
    print(f"  THIS (Nash + population + 40K sess):   {ablation_avg:+.3f}")
    print(f"  E1b (Nash+mod + population + 40K sess): {e1b_avg:+.3f}")

    diff_from_e1b = ablation_avg - e1b_avg
    diff_from_e1a = ablation_avg - e1a_avg

    if abs(diff_from_e1b) < 0.15:
        print(f"\n  CONCLUSION: Ablation ≈ E1b (diff={diff_from_e1b:+.3f})")
        print(f"  => Modulation is a NO-OP. Population + duration explain E1b's success.")
    elif diff_from_e1b < -0.15:
        print(f"\n  CONCLUSION: Ablation < E1b by {abs(diff_from_e1b):.3f}")
        print(f"  => Modulation IS contributing {abs(diff_from_e1b):.3f} chips/round.")
    else:
        print(f"\n  CONCLUSION: Ablation > E1b (diff={diff_from_e1b:+.3f})")
        print(f"  => Modulation is HARMFUL. Pure Nash is better than modulated Nash.")

    return all_results


if __name__ == "__main__":
    main()
