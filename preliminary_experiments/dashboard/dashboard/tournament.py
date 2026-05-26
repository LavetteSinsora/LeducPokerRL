"""
Round-robin tournament runner.

Runs every registered agent (that has a trained model) against every other
agent and computes head-to-head and aggregate statistics, including
per-epoch chip tracking for the "chip race" visualization.
"""

import os
import datetime
import threading
from typing import Dict, List, Optional, Sequence

from agents import registry
from agents.evaluation import compute_robustness_metrics, evaluate_agents


DEFAULT_SHOWCASE_AGENT_IDS = [
    "heuristic",
    "value_based",
    "adaptive_value",
    "modulated_value",
    "cfr",
    "entropy_ac",
    "belief_value",
    "belief_cfr",
    "belief_modulated",
    "belief_oracle",
    "belief_confident",
    "belief_stable",
    "distributional_value",
    "opponent_model",
    "opp_encoder_modulation_v1",
]


class TournamentRunner:
    """Runs round-robin tournaments between a selected set of registered agents."""

    def __init__(self):
        self.last_result: Optional[dict] = None
        self.is_running = False
        self.progress = ""
        self._thread: Optional[threading.Thread] = None
        self.stop_requested = False
        self.current_agent_ids: List[str] = []
        self.current_num_rounds: int = 0

    def run_async(
        self,
        num_rounds: int = 500,
        agent_ids: Optional[Sequence[str]] = None,
    ):
        """Run the tournament in a background thread. Returns True if started."""
        if self.is_running:
            return False
        self.is_running = True
        self.stop_requested = False
        self.current_num_rounds = num_rounds
        self.current_agent_ids = list(agent_ids) if agent_ids else list(DEFAULT_SHOWCASE_AGENT_IDS)
        self.progress = "Starting..."
        self._thread = threading.Thread(
            target=self._run,
            args=(num_rounds, self.current_agent_ids),
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self, num_rounds: int, agent_ids: Sequence[str]):
        try:
            self.last_result = self._execute(num_rounds, agent_ids)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[tournament] Error: {e}")
            self.last_result = {"error": str(e)}
        finally:
            self.is_running = False

    def _resolve_agent_ids(self, requested_ids: Sequence[str]) -> List[str]:
        eligible_ids = []
        for agent_id in requested_ids:
            meta = registry.get_metadata(agent_id)
            if meta is None:
                print(f"[tournament] Skipping unknown agent: {agent_id}")
                continue
            model_path = registry.get_checkpoint_path(agent_id)
            if model_path is not None and not os.path.exists(model_path):
                print(f"[tournament] Skipping {agent_id}: missing checkpoint at {model_path}")
                continue
            eligible_ids.append(agent_id)
        return eligible_ids

    def _execute(self, num_rounds: int, requested_ids: Sequence[str]) -> dict:
        eligible_ids = self._resolve_agent_ids(requested_ids)
        if len(eligible_ids) < 2:
            return {"error": "Need at least 2 eligible agents for a tournament"}

        EPOCHS = 10
        rounds_per_epoch = max(1, num_rounds // EPOCHS)
        total_rounds_per_matchup = rounds_per_epoch * EPOCHS

        num_matchups = len(eligible_ids) * (len(eligible_ids) - 1) // 2

        # Head-to-head accumulators
        h2h: Dict[str, Dict[str, float]] = {aid: {} for aid in eligible_ids}
        for aid in eligible_ids:
            for opp in eligible_ids:
                if aid != opp:
                    h2h[aid][opp] = 0.0

        # Chip race tracking
        chip_totals = {aid: 1000.0 for aid in eligible_ids}
        tournament_history = [{"epoch": 0, **chip_totals}]

        for epoch_idx in range(EPOCHS):
            self.progress = f"Epoch {epoch_idx + 1}/{EPOCHS} ({num_matchups} matchups)"
            print(f"[tournament] {self.progress}")

            for i, a_id in enumerate(eligible_ids):
                for j, b_id in enumerate(eligible_ids):
                    if i >= j:
                        continue
                    if self.stop_requested:
                        self.progress = "Stopped"
                        return {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "stopped": True,
                            "num_rounds": rounds_per_epoch * epoch_idx,
                            "agents": eligible_ids,
                            "message": "Tournament stopped before completion",
                        }

                    # Create fresh agents for each matchup
                    agent_a = self._load_agent(a_id)
                    agent_b = self._load_agent(b_id)

                    result = evaluate_agents(agent_a, agent_b, num_rounds=rounds_per_epoch)

                    for r0, r1 in result.round_results:
                        chip_totals[a_id] += r0
                        chip_totals[b_id] += r1
                        h2h[a_id][b_id] += r0
                        h2h[b_id][a_id] += r1

            # Snapshot chip totals at end of epoch
            snapshot = {"epoch": epoch_idx + 1}
            for aid in eligible_ids:
                snapshot[aid] = round(chip_totals[aid], 1)
            tournament_history.append(snapshot)

        # Average h2h stats
        for aid in eligible_ids:
            for opp in eligible_ids:
                if aid != opp:
                    h2h[aid][opp] = round(h2h[aid][opp] / total_rounds_per_matchup, 4)

        # Compute rankings
        rankings = []
        for aid in eligible_ids:
            scores = {opp: h2h[aid][opp] for opp in eligible_ids if opp != aid}
            metrics = compute_robustness_metrics(scores)
            meta = registry.get_metadata(aid)

            wins = sum(1 for v in scores.values() if v > 0)
            total = len(scores)
            win_rate = round(wins / total * 100, 1) if total > 0 else 0

            rankings.append({
                "agent_id": aid,
                "display_name": meta.display_name,
                "win_rate": win_rate,
                **metrics,
            })

        rankings.sort(key=lambda x: x["robustness"], reverse=True)

        self.progress = "Complete"
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "num_rounds": total_rounds_per_matchup,
            "agents": eligible_ids,
            "head_to_head": h2h,
            "rankings": rankings,
            "chip_race": {
                "agents": eligible_ids,
                "history": tournament_history,
                "epochs": EPOCHS,
            },
        }

    def _load_agent(self, agent_id: str):
        """Create and load a single agent."""
        agent = registry.create(agent_id)
        model_path = registry.get_checkpoint_path(agent_id)
        if model_path and os.path.exists(model_path):
            try:
                agent.load_model(model_path)
            except Exception as e:
                print(f"[tournament] Warning: could not load {model_path}: {e}")
        agent.set_train_mode(False)
        return agent

    def get_status(self) -> dict:
        return {
            "is_running": self.is_running,
            "progress": self.progress,
            "has_results": self.last_result is not None,
            "num_rounds": self.current_num_rounds,
            "agent_ids": self.current_agent_ids,
            "num_agents": len(self.current_agent_ids),
        }

    def get_results(self) -> Optional[dict]:
        return self.last_result

    def request_stop(self) -> bool:
        if not self.is_running:
            return False
        self.stop_requested = True
        self.progress = "Stopping..."
        return True
