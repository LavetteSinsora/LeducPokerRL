"""
Standalone evaluation for the AlphaZero-style agent.

Usage:
    # Evaluate current checkpoint vs heuristic + random (200 games each)
    python -m preliminary_experiments.alphazero.eval

    # Custom checkpoint / game count
    python -m preliminary_experiments.alphazero.eval --checkpoint outputs/alphazero_v1/checkpoint.pt --games 500

    # Faster (skip PIMC search — uses Q-values directly)
    python -m preliminary_experiments.alphazero.eval --no-search

Can be run while training is still in progress (reads the latest checkpoint).
"""

import argparse
import random
import sys
from pathlib import Path
from collections import defaultdict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.leduc_game import LeducGame, Action
from engine.observation import Observation

from preliminary_experiments.alphazero.config import AZConfig
from preliminary_experiments.alphazero.trainer import build_agent_and_trainer
from preliminary_experiments.alphazero.state_encoder import (
    CARD_TO_IDX, IDX_TO_CARD, action_event_id, deal_event_id,
)
from preliminary_experiments.alphazero.belief import make_belief_state, update_belief_state
from preliminary_experiments.alphazero.agent import AZAgent, QNet, hand_onehot, masked_softmax


# ── Simple opponents ──────────────────────────────────────────────────────────

class RandomAgent:
    def select_action(self, obs: Observation) -> Action:
        return random.choice(obs.legal_actions)


# ── AZ agent action selection (with or without PIMC search) ──────────────────

def _az_select_action_greedy(agent: AZAgent, obs: Observation) -> Action:
    """
    Pick action using Q_θ directly — no PIMC search.
    ~100× faster than search; useful for eval during training.
    """
    bs     = agent._bs
    q_vals = agent.q_net(bs.P_current, hand_onehot(agent._hand), bs.b_mine)
    probs  = masked_softmax(q_vals, obs.legal_actions, agent.config.temperature)
    return Action(torch.multinomial(probs, 1).item())


# ── Single-game evaluation ────────────────────────────────────────────────────

def _play_eval_game(
    az_agent: AZAgent,
    opponent,
    az_player_id: int,
    use_search: bool,
) -> float:
    """
    Play one complete game. Returns the chip reward for the AZ agent.
    Handles AZAgent lifecycle (new_game / observe_event) correctly.
    """
    game = LeducGame()
    game.reset()

    hand_az = game.player_hands[az_player_id]
    az_agent.new_game(hand_az)

    with torch.no_grad():
        while not game.is_finished:
            p   = game.current_player
            obs = game.get_observation(viewer_id=p)

            if p == az_player_id:
                if use_search:
                    action = az_agent.select_action(obs)
                else:
                    action = _az_select_action_greedy(az_agent, obs)
            else:
                action = opponent.select_action(obs)

            actor     = game.current_player
            pre_board = game.board
            _, rewards, done, _ = game.step(action)

            # Notify AZ agent of action event
            act_eid = action_event_id(actor, int(action))
            az_agent.observe_event(actor, act_eid)

            # Notify of deal event if round transitioned
            if game.board is not None and game.board != pre_board:
                az_agent.observe_event(None, deal_event_id(game.board))

            if done:
                return rewards[az_player_id]

    return game.get_reward()[az_player_id]


# ── Multi-game evaluation ─────────────────────────────────────────────────────

def evaluate(
    az_agent: AZAgent,
    opponent,
    opponent_name: str,
    n_games: int = 200,
    use_search: bool = False,
) -> dict:
    """
    Evaluate AZ agent vs opponent over n_games, alternating positions.

    Returns dict with avg_chips, win_rate, draw_rate, loss_rate, std.
    """
    half    = n_games // 2
    rewards = []

    for pos in [0, 1]:
        for _ in range(half):
            r = _play_eval_game(az_agent, opponent, az_player_id=pos, use_search=use_search)
            rewards.append(r)

    n     = len(rewards)
    avg   = sum(rewards) / n
    wins  = sum(1 for r in rewards if r > 0)
    draws = sum(1 for r in rewards if r == 0)
    loss  = sum(1 for r in rewards if r < 0)

    variance = sum((r - avg) ** 2 for r in rewards) / max(n - 1, 1)
    std      = variance ** 0.5

    return {
        "opponent":  opponent_name,
        "n_games":   n,
        "avg_chips": round(avg,  4),
        "win_rate":  round(wins  / n, 3),
        "draw_rate": round(draws / n, 3),
        "loss_rate": round(loss  / n, 3),
        "std":       round(std,   4),
    }


# ── Hand-stratified breakdown ─────────────────────────────────────────────────

def evaluate_by_hand(
    az_agent: AZAgent,
    opponent,
    opponent_name: str,
    n_games_per_hand: int = 100,
    use_search: bool = False,
) -> dict:
    """
    Evaluate separately for each starting hand (J, Q, K).
    Returns dict mapping hand → avg_chips.
    """
    results = {}
    for hand in IDX_TO_CARD:
        rewards = []
        for _ in range(n_games_per_hand):
            # Force the AZ agent to hold this hand by monkey-patching the game
            game = LeducGame()
            game.reset()
            # Swap hand if needed
            if game.player_hands[0] != hand:
                # Re-deal until AZ gets the desired hand (position 0)
                for _ in range(50):
                    game = LeducGame()
                    game.reset()
                    if game.player_hands[0] == hand:
                        break

            az_agent.new_game(game.player_hands[0])

            with torch.no_grad():
                while not game.is_finished:
                    p   = game.current_player
                    obs = game.get_observation(viewer_id=p)
                    actor     = game.current_player
                    pre_board = game.board

                    if p == 0:
                        if use_search:
                            action = az_agent.select_action(obs)
                        else:
                            action = _az_select_action_greedy(az_agent, obs)
                    else:
                        action = opponent.select_action(obs)

                    _, rews, done, _ = game.step(action)
                    az_agent.observe_event(actor, action_event_id(actor, int(action)))
                    if game.board is not None and game.board != pre_board:
                        az_agent.observe_event(None, deal_event_id(game.board))
                    if done:
                        rewards.append(rews[0])
                        break

        results[hand] = round(sum(rewards) / max(len(rewards), 1), 4)
    return results


# ── Raise-rate analysis ───────────────────────────────────────────────────────

def analyze_raise_rates(
    az_agent: AZAgent,
    n_games: int = 500,
    use_search: bool = False,
) -> dict:
    """
    Sample raise rates by hand for early detection of strategy collapse
    (if all hands raise at the same rate, the agent hasn't differentiated).
    """
    raise_counts  = defaultdict(int)
    action_counts = defaultdict(int)

    for _ in range(n_games):
        game = LeducGame()
        game.reset()
        hand_az = game.player_hands[0]
        az_agent.new_game(hand_az)

        with torch.no_grad():
            while not game.is_finished:
                p   = game.current_player
                obs = game.get_observation(viewer_id=p)
                actor     = game.current_player
                pre_board = game.board

                if p == 0:
                    if use_search:
                        action = az_agent.select_action(obs)
                    else:
                        action = _az_select_action_greedy(az_agent, obs)
                    action_counts[hand_az] += 1
                    if action == Action.RAISE:
                        raise_counts[hand_az] += 1
                else:
                    action = Action(random.randint(0, 1))  # random fold/call for speed

                _, _, done, _ = game.step(action)
                az_agent.observe_event(actor, action_event_id(actor, int(action)))
                if game.board is not None and game.board != pre_board:
                    az_agent.observe_event(None, deal_event_id(game.board))
                if done:
                    break

    rates = {}
    for hand in IDX_TO_CARD:
        n = action_counts[hand]
        rates[hand] = round(raise_counts[hand] / n, 3) if n > 0 else 0.0
    return rates


# ── Pretty print ──────────────────────────────────────────────────────────────

def _print_result(r: dict):
    print(
        f"  vs {r['opponent']:12s} | "
        f"avg {r['avg_chips']:+.3f} chips/hand | "
        f"W/D/L {r['win_rate']:.0%}/{r['draw_rate']:.0%}/{r['loss_rate']:.0%} | "
        f"n={r['n_games']}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate the AlphaZero-style Leduc agent.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "outputs" / "alphazero_v1" / "checkpoint.pt"),
    )
    parser.add_argument("--games",     type=int,  default=200, help="Games per opponent.")
    parser.add_argument("--no-search", action="store_true",
                        help="Use Q-values directly (no PIMC search). Much faster.")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load
    config = AZConfig()
    trainer, agent0, _ = build_agent_and_trainer(config)
    trainer.load(str(ckpt_path))
    ep = trainer.episode_count
    agent0.set_train_mode(False)

    use_search = not args.no_search
    mode_str   = "PIMC search" if use_search else "greedy Q (no search)"
    print(f"\n=== AlphaZero Eval — ep {ep:,} | mode: {mode_str} ===\n")

    from agents.heuristic.agent import HeuristicAgent
    heuristic = HeuristicAgent()
    rng_opp   = RandomAgent()

    # ── Head-to-head ──────────────────────────────────────────────────────────
    print("Head-to-head results:")
    r_rng = evaluate(agent0, rng_opp,   "random",    args.games, use_search)
    r_heu = evaluate(agent0, heuristic, "heuristic", args.games, use_search)
    _print_result(r_rng)
    _print_result(r_heu)

    # ── Per-hand performance vs heuristic ─────────────────────────────────────
    per_hand = evaluate_by_hand(agent0, heuristic, "heuristic",
                                n_games_per_hand=args.games // 2, use_search=use_search)
    print("\nAvg chips/hand vs heuristic, by starting hand:")
    for hand, v in per_hand.items():
        print(f"  {hand}: {v:+.3f}")

    # ── Raise rate stratification ─────────────────────────────────────────────
    raise_rates = analyze_raise_rates(agent0, n_games=args.games * 2, use_search=use_search)
    print("\nRaise rates by hand (non-degenerate strategy = J<Q<K roughly):")
    for hand, rate in raise_rates.items():
        bar = "█" * int(rate * 20)
        print(f"  {hand}: {rate:.1%}  {bar}")

    spread = max(raise_rates.values()) - min(raise_rates.values())
    if spread < 0.05:
        print("  ⚠  WARNING: raise rates nearly identical — possible strategy collapse.")
    else:
        print(f"  ✓  Spread {spread:.1%} — agent is differentiating hands.")

    print()


if __name__ == "__main__":
    main()
