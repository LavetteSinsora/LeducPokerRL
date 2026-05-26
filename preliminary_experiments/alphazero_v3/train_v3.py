"""
AlphaZero v3 — Reduced Q-Net + Higher k Rollouts.

Key differences from v2:
  - q_hidden: (64,64) → (32,32)  — ~3× faster Q-net forward pass
  - k_rollouts: 10 → 30          — same compute budget, 1.7× lower Q* variance
  - state_hidden: (8,16) → (8,)  — single layer sufficient for 9 event types
  - belief_hidden: (16,16) → (8,8)

Usage:
    python experiments/alphazero_v3/train_v3.py
    python experiments/alphazero_v3/train_v3.py --smoke
    python experiments/alphazero_v3/train_v3.py --resume experiments/alphazero_v3/outputs/checkpoint.pt
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preliminary_experiments.alphazero.config import AZConfig
from preliminary_experiments.alphazero.state_encoder import StateEncoder, CARD_TO_IDX, IDX_TO_CARD, action_event_id, deal_event_id
from preliminary_experiments.alphazero.belief import BeliefNet, BeliefState, make_belief_state, update_belief_state, informed_prior
from preliminary_experiments.alphazero.agent import QNet, AZAgent, hand_onehot, masked_softmax
from preliminary_experiments.alphazero.rollout import _step_game
from preliminary_experiments.alphazero.fast_rollout import fast_pimc_search as pimc_search
from preliminary_experiments.alphazero.trainer import DecisionRecord, Episode, _replay_episode, AZTrainer
from preliminary_experiments.alphazero.tournament import az_tournament_checkpointer
from preliminary_experiments.alphazero.eval import evaluate, analyze_raise_rates

from agents.base import BaseAgent
from engine.leduc_game import LeducGame, Action


# ── Reduced architecture config ───────────────────────────────────────────────

V3_CONFIG = AZConfig(
    d_model=4,
    state_hidden=(8,),
    belief_hidden=(8, 8),
    q_hidden=(32, 32),
    k_rollouts=30,
    temperature=1.0,
    n_episodes=200_000,
    lr=1e-3,
    lambda_belief=0.1,
)


# ── Fixed-opponent episode ────────────────────────────────────────────────────

def _play_episode_fixed_opp(
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    q_net: QNet,
    config: AZConfig,
    opponent: BaseAgent,
) -> Episode:
    """
    Play one game where:
      - Player 0 uses PIMC search (generates training signal)
      - Player 1 uses the provided frozen opponent agent

    Both players' belief states are updated after every event so that P_t
    and b_mine remain meaningful during the training replay phase.
    """
    game = LeducGame()
    game.reset()

    hands = list(game.player_hands)

    with torch.no_grad():
        bs = [make_belief_state(hands[p], config.d_model) for p in range(2)]

    events: list = []
    decisions: List[DecisionRecord] = []

    with torch.no_grad():
        while not game.is_finished:
            p = game.current_player
            obs = game.get_observation(viewer_id=p)
            legal = game.get_legal_actions()

            if p == 0:
                # PIMC search — records Q* for training
                q_star = pimc_search(
                    obs=obs,
                    player_i=p,
                    h_i=hands[p],
                    bs_i=bs[p],
                    legal_actions=legal,
                    state_enc=state_enc,
                    belief_net=belief_net,
                    q_net=q_net,
                    k=config.k_rollouts,
                    T=config.temperature,
                )
                decisions.append(DecisionRecord(
                    step=len(events),
                    player=p,
                    legal_actions=legal,
                    q_star=q_star.clone(),
                ))
                probs = masked_softmax(q_star, legal, config.temperature)
                action_idx = torch.multinomial(probs, 1).item()
                action = Action(action_idx)
            else:
                # Fixed frozen opponent — no training signal recorded
                action = opponent.select_action(obs)

            _, done, act_eid, deal_eid, actor = _step_game(game, action)
            events.append((act_eid, actor))
            if deal_eid is not None:
                events.append((deal_eid, None))

            # Update both players' beliefs (needed so P_t and b_mine are live
            # during _replay_episode gradient computation)
            for player_idx in range(2):
                e_prime, P_new = state_enc.encode_event(
                    act_eid, bs[player_idx].P_current, bs[player_idx].P_history
                )
                update_belief_state(bs[player_idx], actor, player_idx,
                                    e_prime, P_new, belief_net)
                if deal_eid is not None:
                    e_prime_d, P_new_d = state_enc.encode_event(
                        deal_eid, bs[player_idx].P_current, bs[player_idx].P_history
                    )
                    update_belief_state(bs[player_idx], None, player_idx,
                                        e_prime_d, P_new_d, belief_net)

    return Episode(hands=hands, events=events, decisions=decisions)


# ── Trainer subclass ──────────────────────────────────────────────────────────

class AZTrainerFixedOpp(AZTrainer):
    """AZTrainer that plays P1 with a rotating frozen opponent pool."""

    def __init__(
        self,
        config: AZConfig,
        state_enc: StateEncoder,
        belief_net: BeliefNet,
        q_net: QNet,
        opponents: List[BaseAgent],
    ):
        super().__init__(config, state_enc, belief_net, q_net)
        self.opponents = opponents

    def train_one_episode(self) -> float:
        self.state_enc.eval()
        self.belief_net.eval()
        self.q_net.eval()

        opponent = random.choice(self.opponents)
        opponent.set_train_mode(False)

        episode = _play_episode_fixed_opp(
            self.state_enc, self.belief_net, self.q_net, self.config, opponent
        )

        self.state_enc.train()
        self.belief_net.train()
        self.q_net.train()

        self.optimizer.zero_grad()
        loss = _replay_episode(episode, self.state_enc, self.belief_net, self.q_net, self.config)
        loss.backward()
        self.optimizer.step()

        self.episode_count += 1
        return loss.item()


# ── Opponent pool loader ──────────────────────────────────────────────────────

def load_opponents() -> List[BaseAgent]:
    from agents.heuristic.agent import HeuristicAgent
    from agents.value_based.agent import ValueBasedAgent
    from agents.cfr.agent import CFRAgent

    opponents = [
        HeuristicAgent(),
        ValueBasedAgent(model_path=str(ROOT / "agents" / "value_based" / "checkpoint.pt")),
        CFRAgent(model_path=str(ROOT / "agents" / "cfr" / "checkpoint.pt")),
    ]
    for opp in opponents:
        opp.set_train_mode(False)
    print(f"Loaded {len(opponents)} opponents: [heuristic, value_based, cfr]")
    return opponents


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero v2 with fixed opponent pool.")
    parser.add_argument("--episodes",         type=int,   default=200_000)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--lambda-belief",    type=float, default=0.1)
    parser.add_argument("--log-every",        type=int,   default=1000)
    parser.add_argument("--checkpoint-every", type=int,   default=5000)
    parser.add_argument("--eval-every",       type=int,   default=10_000)
    parser.add_argument("--eval-games",       type=int,   default=200)
    parser.add_argument("--resume",           type=str,   default=None)
    parser.add_argument("--smoke",            action="store_true",
                        help="Tiny budget (10 episodes) to verify the pipeline.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 10
        args.log_every = 5
        args.checkpoint_every = 10
        args.eval_every = 5
        args.eval_games = 20

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(args.output_dir / "checkpoint.pt")
    history_path    = args.output_dir / "train_history.json"
    config_path     = args.output_dir / "train_config.json"

    # ── Config ────────────────────────────────────────────────────────────────
    config = AZConfig(
        d_model=V3_CONFIG.d_model,
        state_hidden=V3_CONFIG.state_hidden,
        belief_hidden=V3_CONFIG.belief_hidden,
        q_hidden=V3_CONFIG.q_hidden,
        k_rollouts=V3_CONFIG.k_rollouts,
        temperature=V3_CONFIG.temperature,
        n_episodes=args.episodes,
        lr=args.lr,
        lambda_belief=args.lambda_belief,
    )

    config_path.write_text(json.dumps({
        "experiment":     "alphazero_v3",
        "d_model":        config.d_model,
        "state_hidden":   list(config.state_hidden),
        "belief_hidden":  list(config.belief_hidden),
        "q_hidden":       list(config.q_hidden),
        "k_rollouts":     config.k_rollouts,
        "temperature":    config.temperature,
        "n_episodes":     config.n_episodes,
        "lr":             config.lr,
        "lambda_belief":  config.lambda_belief,
        "opponent_pool":  ["heuristic", "value_based", "cfr"],
        "resumed_from":   args.resume,
    }, indent=2))
    print(f"Config saved → {config_path}")

    # ── Build networks ────────────────────────────────────────────────────────
    state_enc  = StateEncoder(config)
    belief_net = BeliefNet(config)
    q_net      = QNet(config)

    opponents = load_opponents()

    trainer = AZTrainerFixedOpp(config, state_enc, belief_net, q_net, opponents)

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume}  (episode {trainer.episode_count})")

    # Agents for tournament eval (player_id set dynamically by AZEvalAdapter)
    agent0 = AZAgent(config, state_enc, belief_net, q_net, player_id=0)
    agent1 = AZAgent(config, state_enc, belief_net, q_net, player_id=1)

    # ── Eval setup ────────────────────────────────────────────────────────────
    eval_history: list = []
    eval_log_path = args.output_dir / "eval_history.json"

    if args.eval_every > 0:
        from agents.heuristic.agent import HeuristicAgent
        _heuristic_eval = HeuristicAgent()
        print(f"Periodic eval: every {args.eval_every} episodes, {args.eval_games} games/opponent.")

    def _run_eval(ep_count):
        agent0.set_train_mode(False)
        r_heu = evaluate(agent0, _heuristic_eval, "heuristic", args.eval_games, use_search=False)
        rates  = analyze_raise_rates(agent0, n_games=args.eval_games, use_search=False)
        agent0.set_train_mode(True)
        spread = max(rates.values()) - min(rates.values())
        entry = {
            "episode":       ep_count,
            "vs_heuristic":  r_heu["avg_chips"],
            "raise_J":       rates["J"],
            "raise_Q":       rates["Q"],
            "raise_K":       rates["K"],
            "raise_spread":  round(spread, 3),
        }
        eval_history.append(entry)
        eval_log_path.write_text(json.dumps(eval_history, indent=2))
        print(f"  [eval] ep {ep_count:>7d} | vs heuristic {r_heu['avg_chips']:+.3f} | "
              f"raise J/Q/K {rates['J']:.0%}/{rates['Q']:.0%}/{rates['K']:.0%} "
              f"(spread {spread:.0%})")

    # ── History callback ──────────────────────────────────────────────────────
    history: list = []

    def _history_callback(event):
        history.append(event)
        if len(history) % 100 == 0:
            history_path.write_text(json.dumps(history, indent=2))
        if args.eval_every > 0 and event["episode"] % args.eval_every == 0:
            _run_eval(event["episode"])

    # ── Tournament checkpointer ───────────────────────────────────────────────
    checkpointer = az_tournament_checkpointer(
        agent0=agent0,
        agent1=agent1,
        output_dir=args.output_dir,
        pass_through_callback=_history_callback,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting alphazero_v3 training: {args.episodes} episodes")
    print(f"Architecture: d={config.d_model}, state{config.state_hidden}, "
          f"belief{config.belief_hidden}, Q{config.q_hidden}")
    print(f"Opponent pool: [heuristic, value_based, cfr] (random per episode)")
    print(f"Output dir: {args.output_dir}\n")

    t_start = time.time()
    trainer.train(
        log_every=args.log_every,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        callback=checkpointer.callback,
    )
    elapsed = time.time() - t_start

    history_path.write_text(json.dumps(history, indent=2))
    print(f"\nDone. {trainer.episode_count} episodes in {elapsed/60:.1f} min.")
    print(f"Checkpoint : {checkpoint_path}")


if __name__ == "__main__":
    main()
