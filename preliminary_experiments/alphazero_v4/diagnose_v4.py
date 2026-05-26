"""
AlphaZero v4 — Degradation Diagnosis

Investigates four hypotheses for why tournament performance peaks at ep 10K
and oscillates/degrades afterward despite stable loss:

  D1: Player-0 training bias  — agent only trains as P0; P1 strategies
      are never practiced. Measure: P0 vs P1 Q* spread and raise rates.

  D2: Q* signal collapse      — Q*(legal actions) may be converging to
      near-equal values, providing weak gradient signal to Q-net.
      Measure: mean max-min spread of Q* at decision points.

  D3: Belief accuracy         — b_mine may not improve over the game, or
      may be different quality for P0 vs P1.
      Measure: b_mine[h_opp_true] at each game step.

  D4: K raise rate collapse   — K raise rate falls from 73% (ep10K) to
      45% (ep90K). Is this P1-side only (positional), or both sides?

Run:
    python experiments/alphazero_v4/diagnose_v4.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preliminary_experiments.alphazero.config import AZConfig
from preliminary_experiments.alphazero.state_encoder import StateEncoder, IDX_TO_CARD, CARD_TO_IDX
from preliminary_experiments.alphazero.belief import BeliefNet, BeliefState, make_belief_state, update_belief_state, informed_prior
from preliminary_experiments.alphazero.agent import QNet, hand_onehot
from preliminary_experiments.alphazero.fast_rollout import fast_pimc_search
from preliminary_experiments.alphazero.rollout import _step_game
from engine.leduc_game import LeducGame, Action

# ── Config (must match train_v4) ──────────────────────────────────────────────

V4_CONFIG = AZConfig(
    d_model=4, state_hidden=(8,), belief_hidden=(8, 8), q_hidden=(32, 32),
    k_rollouts=10,          # reduced for speed (was 30 in training)
    temperature=1.0, T_self=0.5, T_opp=2.0,
    lambda_entropy=0.01, target_sync_freq=500,
    replay_buffer_size=50_000, replay_batch_size=256,
    n_episodes=200_000, lr=1e-3, lambda_belief=0.1,
)

CHECKPOINTS = {
    10_000:  "outputs/checkpoint_ep0010000.pt",
    40_000:  "outputs/checkpoint_ep0040000.pt",
    80_000:  "outputs/checkpoint_ep0080000.pt",
}

BASE_DIR = Path(__file__).resolve().parent

N_GAMES = 500   # games per checkpoint per position


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_checkpoint(rel_path: str):
    path = BASE_DIR / rel_path
    state_enc  = StateEncoder(V4_CONFIG)
    belief_net = BeliefNet(V4_CONFIG)
    q_net      = QNet(V4_CONFIG)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_enc.load_state_dict(ckpt["state_enc"])
    belief_net.load_state_dict(ckpt["belief_net"])
    q_net.load_state_dict(ckpt["q_net"])
    state_enc.eval(); belief_net.eval(); q_net.eval()
    return state_enc, belief_net, q_net


# ── Heuristic opponent for evaluation ────────────────────────────────────────

def _load_heuristic():
    from agents.heuristic.agent import HeuristicAgent
    opp = HeuristicAgent()
    opp.set_train_mode(False)
    return opp


# ── Run diagnostic games ──────────────────────────────────────────────────────

def run_diagnostic_games(state_enc, belief_net, q_net, az_player: int, n_games: int, opponent):
    """
    Run n_games with the AZ agent as `az_player` (0 or 1).

    Returns per-decision records:
        q_spread:       max(Q*[legal]) - min(Q*[legal])
        belief_acc:     b_mine[h_opp_true]  (probability assigned to correct hand)
        action_taken:   0=fold, 1=call, 2=raise
        hand:           'J','Q','K'
        game_step:      0=preflop, 1=postflop
        reward:         final chip reward for az_player
    """
    records = []

    with torch.no_grad():
        for _ in range(n_games):
            game = LeducGame()
            game.reset()
            hands = list(game.player_hands)
            bs = [make_belief_state(hands[p], V4_CONFIG.d_model) for p in range(2)]
            game_step = 0   # increments after deal event
            episode_records = []
            final_reward = None

            while not game.is_finished:
                p = game.current_player
                obs = game.get_observation(viewer_id=p)
                legal = game.get_legal_actions()

                if p == az_player:
                    q_star = fast_pimc_search(
                        obs=obs, player_i=p, h_i=hands[p], bs_i=bs[p],
                        legal_actions=legal, state_enc=state_enc,
                        belief_net=belief_net, q_net=q_net,
                        k=V4_CONFIG.k_rollouts, T=V4_CONFIG.temperature,
                        T_self=V4_CONFIG.T_self, T_opp=V4_CONFIG.T_opp,
                    )
                    opp_true = hands[1 - p]
                    opp_idx  = CARD_TO_IDX[opp_true]
                    b_acc    = bs[p].b_mine[opp_idx].item()

                    legal_q = q_star[[int(a) for a in legal]]
                    q_spread = (legal_q.max() - legal_q.min()).item()

                    probs = F.softmax(q_star[[int(a) for a in legal]] / V4_CONFIG.temperature, dim=-1)
                    action_idx_local = torch.multinomial(probs, 1).item()
                    action = legal[action_idx_local]
                else:
                    action = opponent.select_action(obs)

                _, done, act_eid, deal_eid, actor = _step_game(game, action)

                if p == az_player:
                    episode_records.append({
                        "q_spread":    q_spread,
                        "belief_acc":  b_acc,
                        "action":      int(action),
                        "hand":        hands[p],
                        "game_step":   game_step,
                    })

                for pi in range(2):
                    e_prime, P_new = state_enc.encode_event(
                        act_eid, bs[pi].P_current, bs[pi].P_history
                    )
                    update_belief_state(bs[pi], actor, pi, e_prime, P_new, belief_net)
                    if deal_eid is not None:
                        e_d, P_new_d = state_enc.encode_event(
                            deal_eid, bs[pi].P_current, bs[pi].P_history
                        )
                        update_belief_state(bs[pi], None, pi, e_d, P_new_d, belief_net)

                if deal_eid is not None:
                    game_step = 1

            if done:
                rew = game.get_reward()
                final_reward = rew[az_player]

            for rec in episode_records:
                rec["reward"] = final_reward
            records.extend(episode_records)

    return records


# ── Aggregate results ─────────────────────────────────────────────────────────

def summarise(records, label):
    if not records:
        print(f"  {label}: no records")
        return {}

    q_spreads   = [r["q_spread"]   for r in records]
    belief_accs = [r["belief_acc"] for r in records]

    # Raise rates by hand
    for hand in ("J", "Q", "K"):
        hand_recs = [r for r in records if r["hand"] == hand]
        if not hand_recs:
            continue
        raise_rate = sum(1 for r in hand_recs if r["action"] == 2) / len(hand_recs)
        print(f"  {label} | hand={hand} n={len(hand_recs):4d} | raise={raise_rate:.0%}")

    mean_q_spread  = sum(q_spreads)  / len(q_spreads)
    mean_b_acc     = sum(belief_accs) / len(belief_accs)
    baseline_b_acc = 1 / 3
    avg_reward     = sum(r["reward"] for r in records if r["reward"] is not None) / \
                     max(1, sum(1 for r in records if r["reward"] is not None))

    # Fraction of decisions with Q* spread < 0.1 (near-degenerate signal)
    degenerate_frac = sum(1 for s in q_spreads if s < 0.10) / len(q_spreads)

    print(f"  {label} | Q*-spread avg={mean_q_spread:.3f} | degenerate(<0.1)={degenerate_frac:.0%}")
    print(f"  {label} | belief_acc avg={mean_b_acc:.3f} (baseline=33%) | avg_reward={avg_reward:+.3f}")

    return {
        "q_spread_mean": mean_q_spread,
        "q_degenerate_frac": degenerate_frac,
        "belief_acc": mean_b_acc,
        "avg_reward": avg_reward,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    opp = _load_heuristic()
    results = {}

    for ep, rel_path in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"Checkpoint ep {ep:,}")
        print(f"{'='*60}")
        state_enc, belief_net, q_net = load_checkpoint(rel_path)

        for pos in (0, 1):
            label = f"ep{ep//1000:03d}K as_P{pos}"
            recs = run_diagnostic_games(state_enc, belief_net, q_net, pos, N_GAMES, opp)
            stats = summarise(recs, label)
            results[f"ep{ep}_P{pos}"] = stats

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY — Q* spread and belief accuracy")
    print(f"{'='*60}")
    print(f"{'Key':<20} {'Q*spread':>10} {'Degen%':>8} {'Belief':>8} {'Reward':>8}")
    for key, s in results.items():
        if s:
            print(f"{key:<20} {s['q_spread_mean']:>10.3f} {s['q_degenerate_frac']:>7.0%} "
                  f"{s['belief_acc']:>8.3f} {s['avg_reward']:>8.3f}")

    out_path = BASE_DIR / "outputs" / "diag_v4.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
