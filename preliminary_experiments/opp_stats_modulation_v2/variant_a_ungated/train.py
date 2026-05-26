"""
opp_stats_modulation_v2 — Variant A: Ungated Residual Training
==============================================================
Architecture:  V(s, opp) = V_base(s) [frozen] + Δ(s, opp_stats)
               22 → 32 → 32 → 1 modulation head, near-zero output init.

Training recipe (new vs v1):
  - CFR + heuristic oversampled 3× (vs uniform sampling in v1)
  - 200K episodes (vs 300K in v1)
  - Eval every 5K episodes (checkpoint_best_robust.pt and checkpoint_best_avg.pt)
  - Session length: 100 hands (matches evaluation)

Usage:
  python train.py           # full run (200K episodes)
  python train.py --smoke   # quick pipeline check (500 episodes)
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from paper.evaluation.comparison_protocol import (
    STANDARD_OPPONENT_KEYS,
    build_standard_opponents,
    evaluate_stat_aware_pool,
    format_pool_summary,
    compute_pool_summary,
)
from paper.evaluation.shared.stats_tracker import (
    compute_pool_means,
)
from paper.evaluation.shared.training_recipe import (
    OPPONENT_WEIGHTS,
    WeightedOpponentSampler,
    SessionManager,
    play_hand_v2,
)
from preliminary_experiments.opp_stats_modulation_v2.variant_a_ungated.agent import (
    UngatedModAgent,
)

# ── constants ──────────────────────────────────────────────────────────────────

EXPERIMENT_ID    = "opp_stats_modulation_v2_variant_a_ungated"
SESSION_LENGTH   = 100
PRIOR_STRENGTH   = 20.0
CALIBRATION_HANDS = 500
BATCH_SIZE       = 32
LR               = 1e-4
NUM_EPISODES     = 200_000
EVAL_INTERVAL    = 5_000    # episodes between in-training evals
EVAL_ROUNDS      = 500      # rounds per opponent for in-training eval
FINAL_EVAL_ROUNDS = 5_000   # rounds per opponent for final eval
FLUSH_EVERY      = 10       # eval records before flushing to disk
OPPONENT_KEYS    = list(STANDARD_OPPONENT_KEYS)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}


def _convergence_check(losses):
    n = len(losses)
    if n < 10:
        return False, float("inf")
    seg = max(1, n // 5)
    prev = sum(losses[-2 * seg:-seg]) / seg
    last = sum(losses[-seg:]) / seg
    if prev == 0:
        return True, 0.0
    pct = abs(prev - last) / abs(prev) * 100
    return pct < 5.0, pct


# ── TD(0) update ──────────────────────────────────────────────────────────────

def td_update(agent: UngatedModAgent, optimizer, criterion, batch_data):
    """
    Residual TD(0) update. Only the modulation head receives gradients.

    Terminal    : target_residual = r − V_base(s_T)
    Non-terminal: target_residual = (V_base(s_{t+1}) + Δ(s_{t+1})) − V_base(s_t)
    Loss        = MSE(Δ(s_t), target_residual)
    """
    optimizer.zero_grad()
    losses = []
    for chain, reward in batch_data:
        if not chain:
            continue
        for t, (game_enc, stats) in enumerate(chain):
            game_t  = game_enc.unsqueeze(0)
            stats_t = torch.tensor(stats, dtype=torch.float32).unsqueeze(0)
            mod_inp = torch.cat([game_t, stats_t], dim=1)          # (1, 22)

            with torch.no_grad():
                v_base_t = agent.base(game_t).squeeze()             # scalar

            delta_t = agent.mod(mod_inp).squeeze()                  # scalar (grad)

            if t == len(chain) - 1:
                # Terminal: target residual = reward − V_base(s_T)
                target_residual = torch.tensor(
                    float(reward), dtype=torch.float32) - v_base_t
            else:
                # Non-terminal: bootstrap from next state total value
                game_t1  = chain[t + 1][0].unsqueeze(0)
                stats_t1 = torch.tensor(
                    chain[t + 1][1], dtype=torch.float32).unsqueeze(0)
                mod_inp1 = torch.cat([game_t1, stats_t1], dim=1)
                with torch.no_grad():
                    v_base_t1 = agent.base(game_t1).squeeze()
                    delta_t1  = agent.mod(mod_inp1).squeeze()
                    v_total_t1 = v_base_t1 + delta_t1
                target_residual = v_total_t1 - v_base_t

            losses.append(criterion(delta_t, target_residual.detach()))

    if not losses:
        return 0.0
    mean_loss = torch.stack(losses).mean()
    mean_loss.backward()
    optimizer.step()
    return mean_loss.item()


# ── evaluation ────────────────────────────────────────────────────────────────

def quick_eval(agent, opponents, num_rounds, pool_means):
    return evaluate_stat_aware_pool(
        agent=agent,
        opponents=opponents,
        play_hand_fn=play_hand_v2,
        pool_means=pool_means,
        num_rounds=num_rounds,
        session_length=SESSION_LENGTH,
        prior_strength=PRIOR_STRENGTH,
        opponent_keys=OPPONENT_KEYS,
        alternate_positions=True,
    )


# ── main training loop ────────────────────────────────────────────────────────

def train(out_dir: str, num_episodes: int, smoke: bool):
    os.makedirs(out_dir, exist_ok=True)

    # ── pool priors ────────────────────────────────────────────────────────────
    priors_path  = os.path.join(out_dir, "pool_priors.json")
    shared_priors = os.path.join(
        ROOT, "preliminary_experiments", "opp_stats_input_aug",
        "outputs", "pool_random", "pool_priors.json")

    opponents = build_standard_opponents(ROOT)

    if os.path.exists(priors_path):
        pool_means = _load_json(priors_path)
        print(f"Loaded pool priors from {priors_path}")
    elif os.path.exists(shared_priors):
        pool_means = _load_json(shared_priors)
        _write_json(priors_path, pool_means)
        print("Reused pool priors from opp_stats_input_augmentation_v1")
    else:
        print("Computing pool priors (calibrating stats tracker)...")
        pool_means = compute_pool_means(opponents, 50 if smoke else CALIBRATION_HANDS)
        _write_json(priors_path, pool_means)

    # ── agent + optimiser ──────────────────────────────────────────────────────
    agent     = UngatedModAgent()
    optimizer = optim.Adam(agent.mod.parameters(), lr=LR)
    criterion = nn.MSELoss()
    agent.set_train_mode(True)

    # ── session + sampler ──────────────────────────────────────────────────────
    sampler = WeightedOpponentSampler(opponents, OPPONENT_WEIGHTS)
    session = SessionManager(SESSION_LENGTH, pool_means, PRIOR_STRENGTH, sampler)

    # ── training config ────────────────────────────────────────────────────────
    config = {
        "experiment_id":    EXPERIMENT_ID,
        "variant":          "variant_a_ungated",
        "architecture":     "UngatedModAgent(base=frozen, mod=22→32→32→1, zero-init-output)",
        "training_recipe":  "weighted_pool (cfr×3, heuristic×3, others×1)",
        "opponent_weights": OPPONENT_WEIGHTS,
        "learning_rate":    LR,
        "batch_size":       BATCH_SIZE,
        "session_length":   SESSION_LENGTH,
        "num_episodes":     num_episodes,
        "eval_interval":    EVAL_INTERVAL,
        "checkpoint_metric": "robustness",
    }
    _write_json(os.path.join(out_dir, "train_config.json"), config)

    # log expected opponent frequencies
    freqs = sampler.expected_frequencies()
    print("Opponent sampling weights:")
    for k, p in freqs.items():
        print(f"  {k:<20s} {p:.1%}")

    batch_data = []
    train_history = []
    eval_history  = []
    new_train_buf = []
    new_eval_buf  = []
    best_robust   = float("-inf")
    best_avg      = float("-inf")
    last_eval_ep  = -1

    def flush():
        if new_train_buf:
            train_history.extend(new_train_buf)
            new_train_buf.clear()
            _write_json(os.path.join(out_dir, "train_history.json"), train_history)
        if new_eval_buf:
            eval_history.extend(new_eval_buf)
            new_eval_buf.clear()
            _write_json(os.path.join(out_dir, "eval_history.json"), eval_history)

    print(f"\n{'='*65}")
    print(f"  {EXPERIMENT_ID}")
    print(f"  {'SMOKE' if smoke else 'FULL'} | episodes={num_episodes:,} "
          f"batch={BATCH_SIZE} session={SESSION_LENGTH}")
    print(f"{'='*65}\n")

    t0 = time.time()
    for ep in range(1, num_episodes + 1):
        learner_id  = ep % 2
        _, opp      = session.current_opponent()

        chain, reward = play_hand_v2(
            agent, opp, session.tracker(learner_id), learner_id=learner_id)
        session.record_hand(learner_id)
        batch_data.append((chain, reward))

        if len(batch_data) >= BATCH_SIZE:
            loss = td_update(agent, optimizer, criterion, batch_data)
            batch_data.clear()
            new_train_buf.append({"episode": ep, "loss": round(loss, 6)})

        # ── periodic eval ──────────────────────────────────────────────────────
        if ep % EVAL_INTERVAL == 0 and ep != last_eval_ep:
            last_eval_ep = ep
            agent.set_train_mode(False)
            result  = quick_eval(agent, opponents, EVAL_ROUNDS, pool_means)
            scores  = result["scores"]
            summary = result["summary"]
            rob     = summary["robustness"]
            avg     = summary["avg"]

            new_eval_buf.append({"episode": ep, **scores,
                                 "_rob": round(rob, 4), "_avg": round(avg, 4)})

            if rob > best_robust:
                best_robust = rob
                agent.save_model(os.path.join(out_dir, "checkpoint_best_robust.pt"))
            if avg > best_avg:
                best_avg = avg
                agent.save_model(os.path.join(out_dir, "checkpoint_best_avg.pt"))

            print(f"  [ep={ep:>7,}]  "
                  f"h:{scores['heuristic']:+.3f}  cfr:{scores['cfr']:+.3f}  "
                  f"tp:{scores['tight_passive']:+.2f}  "
                  f"ta:{scores['tight_aggressive']:+.2f}  "
                  f"lp:{scores['loose_passive']:+.2f}  "
                  f"la:{scores['loose_aggressive']:+.2f}  "
                  f"mn:{scores['maniac']:+.2f}  rnd:{scores['random']:+.2f}  "
                  f"[rob:{rob:+.3f} best:{best_robust:+.3f}]  "
                  f"{format_pool_summary(summary)}")
            agent.set_train_mode(True)

            if len(new_eval_buf) >= FLUSH_EVERY:
                flush()

    flush()
    elapsed = time.time() - t0

    # ── final checkpoint + full eval ───────────────────────────────────────────
    agent.save_model(os.path.join(out_dir, "checkpoint.pt"))
    agent.set_train_mode(False)

    print(f"\nFinal evaluation ({FINAL_EVAL_ROUNDS} rounds/opponent)...")
    final = quick_eval(agent, opponents, FINAL_EVAL_ROUNDS, pool_means)
    final_scores  = final["scores"]
    final_summary = final["summary"]

    all_losses  = [r["loss"] for r in train_history if "loss" in r]
    converged, pct = _convergence_check(all_losses)

    results = {
        "experiment_id":    EXPERIMENT_ID,
        "variant":          "variant_a_ungated",
        "training_episodes": num_episodes,
        "training_time_s":  round(elapsed, 1),
        "converged":        converged,
        "loss_plateau_pct": round(pct, 2),
        "best_robust_ep":   best_robust,
        "final_scores":     final_scores,
        "final_summary": {
            "avg":        round(final_summary["avg"], 4),
            "worst_case": round(final_summary["worst_case"], 4),
            "robustness": round(final_summary["robustness"], 4),
            "std":        round(final_summary["std"], 4),
        },
    }
    _write_json(os.path.join(out_dir, "results.json"), results)

    print(f"\nTraining complete in {elapsed:.0f}s | "
          f"converged={converged} (Δloss={pct:.1f}%)")
    print(f"Final: {format_pool_summary(final_summary)}")
    print(f"Checkpoints saved to {out_dir}/")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Variant A — Ungated Residual")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick pipeline check (500 episodes)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of training episodes")
    args = parser.parse_args()

    smoke    = args.smoke
    episodes = args.episodes or (500 if smoke else NUM_EPISODES)
    out_dir  = os.path.join(HERE, "..", "outputs", "variant_a_ungated")

    train(out_dir=out_dir, num_episodes=episodes, smoke=smoke)
