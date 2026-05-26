"""
Training entry point for the AlphaZero-style agent.

Usage:
    # Fresh run
    python -m preliminary_experiments.alphazero.train

    # Resume from checkpoint
    python -m preliminary_experiments.alphazero.train --resume outputs/alphazero_v1/checkpoint.pt

    # Smoke test (10 episodes)
    python -m preliminary_experiments.alphazero.train --smoke

    # Custom settings
    python -m preliminary_experiments.alphazero.train --episodes 200000 --k 10 --output-dir outputs/az_run2

Checkpoints are saved every --checkpoint-every episodes and on Ctrl+C.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preliminary_experiments.alphazero.config import AZConfig
from preliminary_experiments.alphazero.trainer import build_agent_and_trainer
from preliminary_experiments.alphazero.eval import evaluate, analyze_raise_rates, RandomAgent

EXPERIMENT_ID = "alphazero_v1"


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero-style Leduc agent.")
    parser.add_argument("--episodes",          type=int,   default=200_000, help="Total training episodes (target, not additional — resumes are handled automatically).")
    parser.add_argument("--k",                 type=int,   default=10,      help="PIMC rollouts per action per hand.")
    parser.add_argument("--temperature",       type=float, default=1.0,     help="Softmax temperature.")
    parser.add_argument("--lr",                type=float, default=1e-3,    help="Learning rate.")
    parser.add_argument("--lambda-belief",     type=float, default=0.1,     help="Belief CE loss weight.")
    parser.add_argument("--log-every",         type=int,   default=1000,    help="Print progress every N episodes.")
    parser.add_argument("--checkpoint-every",  type=int,   default=5000,    help="Save checkpoint every N episodes.")
    parser.add_argument("--resume",            type=str,   default=None,    help="Path to checkpoint to resume from.")
    parser.add_argument("--eval-every",        type=int,   default=10_000,  help="Run lightweight eval every N episodes (0 = off).")
    parser.add_argument("--eval-games",        type=int,   default=200,     help="Games per opponent during periodic eval.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / EXPERIMENT_ID,
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny budget (10 episodes) to verify the pipeline.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 10
        args.k = 2
        args.log_every = 5
        args.checkpoint_every = 10
        args.eval_every = 5
        args.eval_games = 20

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(args.output_dir / "checkpoint.pt")
    history_path    = args.output_dir / "train_history.json"
    config_path     = args.output_dir / "train_config.json"

    # ── Config ──────────────────────────────────────────────────────────────
    config = AZConfig(
        k_rollouts=args.k,
        temperature=args.temperature,
        lr=args.lr,
        lambda_belief=args.lambda_belief,
        n_episodes=args.episodes,
    )

    config_dict = {
        "experiment_id":   EXPERIMENT_ID,
        "episodes":        args.episodes,
        "k_rollouts":      args.k,
        "temperature":     args.temperature,
        "lr":              args.lr,
        "lambda_belief":   args.lambda_belief,
        "d_model":         config.d_model,
        "q_hidden":        list(config.q_hidden),
        "belief_hidden":   list(config.belief_hidden),
        "state_hidden":    list(config.state_hidden),
        "resumed_from":    args.resume,
    }
    config_path.write_text(json.dumps(config_dict, indent=2))
    print(f"Config saved → {config_path}")

    # ── Build trainer ────────────────────────────────────────────────────────
    trainer, agent0, agent1 = build_agent_and_trainer(config)

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume}  (episode {trainer.episode_count})")

    # ── Eval setup ───────────────────────────────────────────────────────────
    eval_log_path = args.output_dir / "eval_history.json"
    eval_history  = []

    if args.eval_every > 0:
        from agents.heuristic.agent import HeuristicAgent
        _heuristic = HeuristicAgent()
        _random    = RandomAgent()
        print(f"Periodic eval enabled: every {args.eval_every} episodes, "
              f"{args.eval_games} games/opponent (greedy Q, no search).")

    # ── History logging ──────────────────────────────────────────────────────
    history = []

    def _run_eval(ep_count):
        agent0.set_train_mode(False)
        r_rng = evaluate(agent0, _random,    "random",    args.eval_games, use_search=False)
        r_heu = evaluate(agent0, _heuristic, "heuristic", args.eval_games, use_search=False)
        rates  = analyze_raise_rates(agent0, n_games=args.eval_games, use_search=False)
        agent0.set_train_mode(True)

        spread = max(rates.values()) - min(rates.values())
        entry  = {
            "episode":        ep_count,
            "vs_random":      r_rng["avg_chips"],
            "vs_heuristic":   r_heu["avg_chips"],
            "raise_J":        rates["J"],
            "raise_Q":        rates["Q"],
            "raise_K":        rates["K"],
            "raise_spread":   round(spread, 3),
        }
        eval_history.append(entry)
        eval_log_path.write_text(json.dumps(eval_history, indent=2))
        print(
            f"  [eval] ep {ep_count:>7d} | "
            f"vs random {r_rng['avg_chips']:+.3f} | "
            f"vs heuristic {r_heu['avg_chips']:+.3f} | "
            f"raise J/Q/K {rates['J']:.0%}/{rates['Q']:.0%}/{rates['K']:.0%} "
            f"(spread {spread:.0%})"
        )

    def _history_callback(event):
        history.append(event)
        # Write every 100 episodes to keep the file reasonably fresh
        if len(history) % 100 == 0:
            history_path.write_text(json.dumps(history, indent=2))
        # Lightweight periodic eval (greedy Q, no search)
        if args.eval_every > 0 and event["episode"] % args.eval_every == 0:
            _run_eval(event["episode"])

    # ── Tournament checkpointer (standard protocol) ───────────────────────────
    from preliminary_experiments.alphazero.tournament import az_tournament_checkpointer
    checkpointer = az_tournament_checkpointer(
        agent0=agent0,
        agent1=agent1,
        output_dir=args.output_dir,
        pass_through_callback=_history_callback,
    )
    callback = checkpointer.callback

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\nStarting training: {args.episodes} episodes, k={args.k}, lr={args.lr}")
    print(f"Output dir: {args.output_dir}\n")

    t_start = time.time()
    trainer.train(
        log_every=args.log_every,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        callback=callback,
    )
    elapsed = time.time() - t_start

    # ── Final flush ──────────────────────────────────────────────────────────
    history_path.write_text(json.dumps(history, indent=2))
    print(f"\nDone. {trainer.episode_count} episodes in {elapsed/60:.1f} min.")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"History    : {history_path}")


if __name__ == "__main__":
    main()
