"""
Round-robin tournament runner.

Runs every registered agent (that has a trained model) against every other
agent and computes head-to-head and aggregate statistics, including
per-epoch chip tracking for the "chip race" visualization.
"""

import os
import datetime
import threading
from typing import Dict, Optional

from src.agents import registry
from src.training.evaluation import evaluate_agents, compute_robustness_metrics

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class TournamentRunner:
    """Runs round-robin tournaments between all registered agents."""

    def __init__(self):
        self.last_result: Optional[dict] = None
        self.is_running = False
        self.progress = ""
        self._thread: Optional[threading.Thread] = None

    def run_async(self, num_rounds: int = 500):
        """Run the tournament in a background thread. Returns True if started."""
        if self.is_running:
            return False
        self.is_running = True
        self.progress = "Starting..."
        self._thread = threading.Thread(target=self._run, args=(num_rounds,), daemon=True)
        self._thread.start()
        return True

    def _run(self, num_rounds: int):
        try:
            self.last_result = self._execute(num_rounds)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[tournament] Error: {e}")
            self.last_result = {"error": str(e)}
        finally:
            self.is_running = False

    def _execute(self, num_rounds: int) -> dict:
        # Discover agents that are ready to compete
        agent_metas = registry.list_agents()
        eligible_ids = []
        for meta in agent_metas:
            model_path = os.path.join(ROOT_DIR, 'models', f'{meta.id}_agent.pt')
            if not meta.requires_model_path or os.path.exists(model_path):
                eligible_ids.append(meta.id)

        eligible_ids.sort()
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
        model_path = os.path.join(ROOT_DIR, 'models', f'{agent_id}_agent.pt')
        if os.path.exists(model_path):
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
        }

    def get_results(self) -> Optional[dict]:
        return self.last_result
