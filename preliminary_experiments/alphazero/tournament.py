"""
Tournament evaluation adapter for the AlphaZero-style agent.

AZAgent has a stateful lifecycle (new_game / observe_event) that the standard
evaluate_agents() / TournamentCheckpointer infrastructure does not call.

AZEvalAdapter wraps AZAgent and manages the lifecycle automatically by
inspecting obs.action_history and obs.board on each select_action() call.
This makes AZAgent drop-in compatible with evaluate_agents() and therefore
with TournamentCheckpointer unchanged.

Usage in train.py:
    from preliminary_experiments.alphazero.tournament import az_tournament_checkpointer

    checkpointer = az_tournament_checkpointer(
        agent0=agent0,        # player_id=0
        agent1=agent1,        # player_id=1 (shared weights)
        output_dir=output_dir,
        pass_through_callback=history_callback,
    )
    trainer.train(..., callback=checkpointer.callback)
"""

from agents.tournament_eval import TournamentCheckpointer
from agents.base import BaseAgent
from engine.observation import Observation
from engine.leduc_game import Action

from .state_encoder import action_event_id, deal_event_id
from .agent import AZAgent


# ── Adapter ───────────────────────────────────────────────────────────────────

class AZEvalAdapter(BaseAgent):
    """
    Makes AZAgent compatible with evaluate_agents().

    evaluate_agents() only calls select_action(obs). This adapter:
      1. Detects new hands by comparing action_history length.
      2. Calls agent.new_game(hand) and sets agent.player_id when a new hand begins.
      3. Replays any new actions from obs.action_history via agent.observe_event().
      4. Injects deal events when obs.current_round transitions to 1.
    """

    def __init__(self, agent: AZAgent):
        self._agent = agent
        self._n_processed: int = 0       # actions already applied to belief state
        self._in_game:     bool = False   # whether new_game() has been called
        self._deal_done:   bool = False   # whether deal event has been applied

    def select_action(self, obs: Observation) -> Action:
        history = obs.action_history or ()
        my_id   = obs.current_player
        hand    = obs.player_hand

        # ── New hand detection ────────────────────────────────────────────────
        # Trigger: belief not initialized, OR history shrank (PokerSession reset)
        if not self._in_game or len(history) < self._n_processed:
            self._agent.player_id = my_id
            self._agent.new_game(hand)
            self._n_processed = 0
            self._deal_done   = False
            self._in_game     = True

        # ── Inject deal event if we've entered round 1 ────────────────────────
        # The deal happens between rounds; the first round-1 call is the first time
        # the AZAgent can observe the board card.
        if obs.current_round == 1 and not self._deal_done and obs.board is not None:
            # Apply all pending round-0 actions FIRST, then the deal.
            # (Round-0 actions may still be unprocessed if the agent is P1
            # and acts first in round 1.)
            self._apply_history(history, stop_at=len(history))
            from engine.leduc_game import Action as _A  # noqa: F401 — circular guard
            self._agent.observe_event(None, deal_event_id(obs.board))
            self._deal_done = True

        # ── Apply any remaining unprocessed actions ───────────────────────────
        self._apply_history(history, stop_at=len(history))

        return self._agent.select_action(obs)

    def _apply_history(self, history: tuple, stop_at: int) -> None:
        """Apply actions from history[self._n_processed:stop_at] to belief state."""
        _NAME_TO_VAL = {"FOLD": 0, "CALL": 1, "RAISE": 2}
        for i in range(self._n_processed, stop_at):
            player, action_name = history[i]
            act_val = _NAME_TO_VAL[str(action_name).upper()]
            act_eid = action_event_id(int(player), act_val)
            self._agent.observe_event(int(player), act_eid)
            self._n_processed += 1

    # ── BaseAgent forwarding ──────────────────────────────────────────────────

    def set_train_mode(self, mode: bool) -> None:
        self._agent.set_train_mode(mode)

    def save_model(self, path: str) -> None:
        self._agent.save_model(path)

    def load_model(self, path: str) -> None:
        self._agent.load_model(path)

    def get_action_evaluations(self, obs: Observation) -> list:
        return self._agent.get_action_evaluations(obs)


# ── Factory ───────────────────────────────────────────────────────────────────

def az_tournament_checkpointer(
    agent0: AZAgent,
    agent1: AZAgent,
    output_dir,
    pass_through_callback=None,
    tournament_interval: int = 10_000,
    rounds_per_matchup:  int = 2_000,
    session_length:      int = 100,
    opponent_ids=None,
) -> TournamentCheckpointer:
    """
    Build a standard TournamentCheckpointer that evaluates an AZAgent correctly.

    agent0 / agent1 must share the same underlying networks (same state_enc,
    belief_net, q_net) and differ only in player_id.  AZEvalAdapter wraps agent0
    for the standard evaluate_agents() calls; AZEvalAdapter also updates agent0's
    player_id dynamically per hand so position-swapping works correctly.

    The returned checkpointer emits {episode, loss} events and is wired to fire
    every `tournament_interval` episodes.  It uses the standard output schema
    (tournament_history.json, checkpoint_best_avg.pt, checkpoint_best_robust.pt).

    NOTE: TournamentCheckpointer.callback checks for data["type"] == "batch_update".
    The AZ trainer emits {"episode": N, "loss": X}.  We handle this below by
    overriding the callback trigger.
    """
    adapter = AZEvalAdapter(agent0)

    checkpointer = TournamentCheckpointer(
        agent=adapter,
        output_dir=output_dir,
        tournament_interval=tournament_interval,
        rounds_per_matchup=rounds_per_matchup,
        session_length=session_length,
        opponent_ids=opponent_ids,
        pass_through_callback=pass_through_callback,
    )

    # TournamentCheckpointer.callback checks for data["type"] == "batch_update".
    # AZ trainer emits {"episode": N, "loss": X} — no "type" key.
    # Patch the callback to accept AZ-style events.
    _orig_callback = checkpointer.callback

    def _az_callback(data: dict) -> None:
        if pass_through_callback is not None:
            pass_through_callback(data)
        episode = data.get("episode", 0)
        if (episode > 0
                and episode % checkpointer._tournament_interval == 0
                and episode != checkpointer._last_tournament_episode):
            checkpointer._last_tournament_episode = episode
            checkpointer._run_tournament(episode)

    checkpointer.callback = _az_callback
    return checkpointer
