"""
Tournament Evaluation Module — Standard Evaluation Protocol for PokerRL

This module implements the canonical evaluation protocol for comparing agents.
All architecture experiments MUST use TournamentCheckpointer during training to
ensure results are directly comparable across experiments.

Protocol summary (see experiments/EVAL_CONFIG.json for frozen parameters):
  - Round-robin tournament vs all promoted agents every 10,000 training episodes
  - 2,000 rounds per matchup, session_length=100 (mirrors training distribution)
  - Position-swapped: subject agent plays both P0 and P1 in every matchup
  - Two best checkpoints maintained: checkpoint_best_avg.pt + checkpoint_best_robust.pt
  - Snapshot checkpoint saved at each tournament point
  - RNG state isolated: tournament execution does not shift the training random stream

Usage in experiment train.py:
    from agents.tournament_eval import TournamentCheckpointer

    checkpointer = TournamentCheckpointer(
        agent=agent,
        output_dir=output_dir,
        pass_through_callback=my_existing_callback,
    )
    trainer.train(num_episodes=200_000, callback=checkpointer.callback)

For one-shot post-hoc evaluation (eval.py scripts):
    from agents.tournament_eval import run_tournament_eval

    results = run_tournament_eval(
        agent=agent,
        checkpoint_path="outputs/my_exp/checkpoint_best_robust.pt",
        output_path="outputs/my_exp/final_tournament.json",
    )
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch

from agents.base import BaseAgent
from agents.evaluation import compute_robustness_metrics, evaluate_agents


# ---------------------------------------------------------------------------
# Load frozen evaluation config
# ---------------------------------------------------------------------------

def _load_eval_config() -> dict:
    config_path = Path(__file__).parent.parent / "experiments" / "EVAL_CONFIG.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    # Fallback defaults if file missing
    return {
        "rounds_per_matchup": 2000,
        "session_length": 100,
        "tournament_interval": 10000,
        "opponent_pool": ["heuristic", "value_based", "adaptive_value",
                          "modulated_value", "cfr"],
    }


_EVAL_CONFIG = _load_eval_config()

DEFAULT_ROUNDS = _EVAL_CONFIG["rounds_per_matchup"]
DEFAULT_SESSION_LENGTH = _EVAL_CONFIG["session_length"]
DEFAULT_INTERVAL = _EVAL_CONFIG["tournament_interval"]
DEFAULT_OPPONENTS = _EVAL_CONFIG["opponent_pool"]


# ---------------------------------------------------------------------------
# TournamentCheckpointer
# ---------------------------------------------------------------------------

class TournamentCheckpointer:
    """
    Callback-compatible tournament runner for BaseTrainer.train().

    Fires a round-robin tournament against the promoted agent pool every
    `tournament_interval` training episodes. Maintains two best-checkpoint
    pointers and saves a timestamped snapshot at each tournament point.

    The training random stream is fully isolated: torch and Python RNG states
    are saved before the tournament and restored after, so training is
    deterministic regardless of tournament frequency.
    """

    def __init__(
        self,
        agent: BaseAgent,
        output_dir: Union[str, Path],
        tournament_interval: int = DEFAULT_INTERVAL,
        rounds_per_matchup: int = DEFAULT_ROUNDS,
        session_length: int = DEFAULT_SESSION_LENGTH,
        opponent_ids: Optional[List[str]] = None,
        pass_through_callback: Optional[Callable] = None,
    ):
        """
        Args:
            agent: The agent being trained. Must implement save_model() and
                set_train_mode().
            output_dir: Experiment output directory. All tournament artifacts
                (history JSON, snapshot checkpoints, best-pointer files) go here.
            tournament_interval: Run tournament every N training episodes.
                Default read from EVAL_CONFIG.json.
            rounds_per_matchup: Rounds per head-to-head matchup.
                Default read from EVAL_CONFIG.json.
            session_length: Reset PokerSession every N hands to mirror training
                distribution for stateful agents. Default read from EVAL_CONFIG.json.
            opponent_ids: Opponent agent IDs from the promoted pool. None = use
                all agents listed in EVAL_CONFIG.json. Loaded fresh from registry
                at each tournament (not cached at construction time).
            pass_through_callback: Existing callback to chain. Called first on
                every event, then tournament logic runs. This lets you keep your
                existing history-writing callback unchanged.
        """
        self._agent = agent
        self._output_dir = Path(output_dir)
        self._tournament_interval = tournament_interval
        self._rounds_per_matchup = rounds_per_matchup
        self._session_length = session_length
        self._opponent_ids = opponent_ids if opponent_ids is not None else list(DEFAULT_OPPONENTS)
        self._pass_through_callback = pass_through_callback

        self._history: List[Dict] = []
        self._best_avg_score: float = float("-inf")
        self._best_robust_score: float = float("-inf")
        self._last_tournament_episode: int = -1

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Resume: reload history if a previous tournament run exists
        history_path = self._output_dir / "tournament_history.json"
        if history_path.exists():
            try:
                with open(history_path) as f:
                    self._history = json.load(f)
                for record in self._history:
                    if record.get("is_best_avg"):
                        self._best_avg_score = record["tournament_avg_chips"]
                    if record.get("is_best_robust"):
                        self._best_robust_score = record["tournament_robustness"]
                print(f"[TournamentCheckpointer] Resumed: {len(self._history)} prior "
                      f"tournaments loaded. Best avg={self._best_avg_score:.4f}, "
                      f"best robust={self._best_robust_score:.4f}")
            except Exception as e:
                print(f"[TournamentCheckpointer] Warning: could not load history: {e}")

    # ------------------------------------------------------------------
    # Public callback — plug into trainer.train(callback=checkpointer.callback)
    # ------------------------------------------------------------------

    def callback(self, data: Dict) -> None:
        """
        Callback compatible with BaseTrainer.train(callback=...).

        Passes every event to pass_through_callback first, then checks whether
        a tournament should fire.
        """
        if self._pass_through_callback is not None:
            self._pass_through_callback(data)

        episode = data.get("episode", 0)
        if (data.get("type") == "batch_update"
                and episode > 0
                and episode % self._tournament_interval == 0
                and episode != self._last_tournament_episode):
            self._last_tournament_episode = episode
            self._run_tournament(episode)

    # ------------------------------------------------------------------
    # Internal: run full tournament
    # ------------------------------------------------------------------

    def _run_tournament(self, episode: int) -> None:
        print(f"\n[Tournament] Episode {episode} — starting round-robin "
              f"({len(self._opponent_ids)} opponents × {self._rounds_per_matchup} rounds)")

        # 1. Snapshot training RNG state
        rng_state_random = random.getstate()
        rng_state_torch = torch.get_rng_state()

        # 2. Eval mode
        self._agent.set_train_mode(False)

        # 3. Load fresh opponent pool
        opponents = self._load_opponent_pool()
        if not opponents:
            print("[Tournament] Warning: no opponents loaded, skipping tournament.")
            self._agent.set_train_mode(True)
            random.setstate(rng_state_random)
            torch.set_rng_state(rng_state_torch)
            return

        # 4. Run matchups
        scores: Dict[str, float] = {}
        matchup_details: Dict[str, dict] = {}
        ts_start = time.time()

        for opp_id, opp_agent in opponents.items():
            with torch.no_grad():
                result = evaluate_agents(
                    self._agent, opp_agent,
                    num_rounds=self._rounds_per_matchup,
                    session_length=self._session_length,
                )
            # Per-position breakdown (first half = agent as P0, second half = P1)
            half = self._rounds_per_matchup // 2
            p0_rewards = [r[0] for r in result.round_results[:half]]
            p1_rewards = [r[0] for r in result.round_results[half:]]
            avg_as_p0 = sum(p0_rewards) / len(p0_rewards) if p0_rewards else 0.0
            avg_as_p1 = sum(p1_rewards) / len(p1_rewards) if p1_rewards else 0.0

            scores[opp_id] = result.agent_0_avg_chips
            matchup_details[opp_id] = {
                "avg_chips": round(result.agent_0_avg_chips, 4),
                "avg_chips_as_p0": round(avg_as_p0, 4),
                "avg_chips_as_p1": round(avg_as_p1, 4),
                "total_chips": round(result.agent_0_total_chips, 2),
                "rounds_counted": result.num_rounds,
            }

        elapsed = time.time() - ts_start

        # 5. Aggregate metrics
        metrics = compute_robustness_metrics(scores)

        # 6. Restore training RNG state
        random.setstate(rng_state_random)
        torch.set_rng_state(rng_state_torch)

        # 7. Back to train mode
        self._agent.set_train_mode(True)

        # 8. Save timestamped snapshot
        snapshot_path = self._output_dir / f"checkpoint_ep{episode:07d}.pt"
        self._agent.save_model(str(snapshot_path))

        # 9. Update best-pointer checkpoints
        avg_score = metrics["avg"]
        robust_score = metrics["robustness"]
        new_best_avg = avg_score > self._best_avg_score
        new_best_robust = robust_score > self._best_robust_score

        if new_best_avg:
            self._best_avg_score = avg_score
            self._agent.save_model(str(self._output_dir / "checkpoint_best_avg.pt"))

        if new_best_robust:
            self._best_robust_score = robust_score
            self._agent.save_model(str(self._output_dir / "checkpoint_best_robust.pt"))

        # 10. Record and persist
        record = {
            "episode": episode,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "elapsed_seconds": round(elapsed, 1),
            "tournament_avg_chips": metrics["avg"],
            "tournament_robustness": metrics["robustness"],
            "tournament_std": metrics["std"],
            "tournament_worst_case": metrics["worst_case"],
            "tournament_best_case": metrics["best_case"],
            "n_opponents": metrics["n_opponents"],
            "is_best_avg": new_best_avg,
            "is_best_robust": new_best_robust,
            "snapshot_path": str(snapshot_path),
            "matchups": matchup_details,
        }
        self._history.append(record)
        self._save_history()

        # 11. Print summary
        self._print_summary(episode, metrics, matchup_details, elapsed)

    def _load_opponent_pool(self) -> Dict[str, BaseAgent]:
        """Load fresh opponent snapshot from registry at each tournament."""
        from agents.registry import registry

        opponents = {}
        for opp_id in self._opponent_ids:
            try:
                agent = registry.create(opp_id)
                checkpoint_path = registry.get_checkpoint_path(opp_id)
                if checkpoint_path and os.path.exists(checkpoint_path):
                    agent.load_model(checkpoint_path)
                agent.set_train_mode(False)
                opponents[opp_id] = agent
            except Exception as e:
                print(f"[TournamentCheckpointer] Warning: could not load {opp_id}: {e}")
        return opponents

    def _save_history(self) -> None:
        path = self._output_dir / "tournament_history.json"
        with open(path, "w") as f:
            json.dump(self._history, f, indent=2)

    def _print_summary(
        self,
        episode: int,
        metrics: dict,
        matchup_details: dict,
        elapsed: float,
    ) -> None:
        print(f"\n{'='*60}")
        print(f"[Tournament] Episode {episode} — {elapsed:.0f}s")
        print(f"{'─'*60}")
        print(f"  {'Opponent':<20} {'Avg chips':>10} {'P0':>8} {'P1':>8}")
        print(f"  {'─'*46}")
        for opp_id, detail in matchup_details.items():
            print(f"  {opp_id:<20} {detail['avg_chips']:>+10.4f} "
                  f"{detail['avg_chips_as_p0']:>+8.4f} {detail['avg_chips_as_p1']:>+8.4f}")
        print(f"  {'─'*46}")
        print(f"  {'TOURNAMENT AVG':<20} {metrics['avg']:>+10.4f}")
        print(f"  {'ROBUSTNESS':<20} {metrics['robustness']:>+10.4f}  "
              f"(avg - 1.5×std, std={metrics['std']:.4f})")
        print(f"  {'WORST CASE':<20} {metrics['worst_case']:>+10.4f}")
        best_avg_marker = " ← new best avg" if metrics["avg"] == self._best_avg_score else ""
        best_rob_marker = " ← new best robust" if metrics["robustness"] == self._best_robust_score else ""
        if best_avg_marker:
            print(f"  {best_avg_marker.strip()}")
        if best_rob_marker:
            print(f"  {best_rob_marker.strip()}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Standalone one-shot tournament (for eval.py scripts)
# ---------------------------------------------------------------------------

def run_tournament_eval(
    agent: BaseAgent,
    checkpoint_path: str,
    output_path: str,
    rounds_per_matchup: int = DEFAULT_ROUNDS,
    session_length: int = DEFAULT_SESSION_LENGTH,
    opponent_ids: Optional[List[str]] = None,
) -> Dict:
    """
    One-shot tournament evaluation at a single checkpoint.

    Loads the agent from checkpoint_path, runs a round-robin tournament
    against the promoted pool, and saves a results dict to output_path.

    Use in experiment eval.py scripts instead of rolling your own evaluation
    loop — this ensures consistency with training-time tournament parameters.

    Args:
        agent: Agent instance (weights loaded from checkpoint_path).
        checkpoint_path: Path to .pt checkpoint to load.
        output_path: Where to save the results JSON.
        rounds_per_matchup: Rounds per matchup (default from EVAL_CONFIG.json).
        session_length: Session reset interval (default from EVAL_CONFIG.json).
        opponent_ids: Opponents to evaluate against (default: all promoted agents).

    Returns:
        Dict with keys: tournament_avg_chips, tournament_robustness, matchups, etc.
    """
    from agents.registry import registry

    if opponent_ids is None:
        opponent_ids = list(DEFAULT_OPPONENTS)

    agent.load_model(checkpoint_path)
    agent.set_train_mode(False)

    opponents = {}
    for opp_id in opponent_ids:
        try:
            opp = registry.create(opp_id)
            cp = registry.get_checkpoint_path(opp_id)
            if cp and os.path.exists(cp):
                opp.load_model(cp)
            opp.set_train_mode(False)
            opponents[opp_id] = opp
        except Exception as e:
            print(f"[run_tournament_eval] Warning: could not load {opp_id}: {e}")

    scores: Dict[str, float] = {}
    matchup_details: Dict[str, dict] = {}

    for opp_id, opp_agent in opponents.items():
        with torch.no_grad():
            result = evaluate_agents(
                agent, opp_agent,
                num_rounds=rounds_per_matchup,
                session_length=session_length,
            )
        half = rounds_per_matchup // 2
        p0_rewards = [r[0] for r in result.round_results[:half]]
        p1_rewards = [r[0] for r in result.round_results[half:]]
        scores[opp_id] = result.agent_0_avg_chips
        matchup_details[opp_id] = {
            "avg_chips": round(result.agent_0_avg_chips, 4),
            "avg_chips_as_p0": round(sum(p0_rewards) / len(p0_rewards) if p0_rewards else 0.0, 4),
            "avg_chips_as_p1": round(sum(p1_rewards) / len(p1_rewards) if p1_rewards else 0.0, 4),
            "total_chips": round(result.agent_0_total_chips, 2),
            "rounds_counted": result.num_rounds,
        }

    metrics = compute_robustness_metrics(scores)
    results = {
        "checkpoint_path": checkpoint_path,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "rounds_per_matchup": rounds_per_matchup,
        "session_length": session_length,
        "tournament_avg_chips": metrics["avg"],
        "tournament_robustness": metrics["robustness"],
        "tournament_std": metrics["std"],
        "tournament_worst_case": metrics["worst_case"],
        "tournament_best_case": metrics["best_case"],
        "n_opponents": metrics["n_opponents"],
        "matchups": matchup_details,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[run_tournament_eval] avg={metrics['avg']:+.4f}  "
          f"robust={metrics['robustness']:+.4f}  → {output_path}")
    return results
