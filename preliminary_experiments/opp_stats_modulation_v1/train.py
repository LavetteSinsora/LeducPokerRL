"""
opp_stats_modulation_v1 — Training Script
==========================================
Trains the modulation head of StatModValueAgent.

Architecture:  V(s, opp) = V_base(s) [frozen] + Δ(s, opp_stats)
  V_base : frozen 15-dim value network (agents/value_based/checkpoint.pt)
  Δ      : trainable 22→32→32→1 MLP

Two variants:

  variant_a_td  — Variant A: TD(0) online training
    - Agent plays 100-hand sessions vs a randomly sampled opponent each session.
    - Modulation head trained via TD(0): at terminal state, target = r − V_base(s);
      at non-terminal, bootstrapped from next state total value.
    - Training schedule: pool_random (uniformly sample opponent each session).
    - 300K episodes total.

  variant_b_supervised  — Variant B: Supervised residual regression
    - Uses oracle EV data from EV_variation_analysis/data.json.
    - Per-opponent prototype stats collected by playing 500 hands vs each opponent.
    - Info-set EV targets computed by marginalizing over opponent's unknown hand
      using card-removal probability weighting.
    - Residual = EV_infoset(s, opp) − V_base(s) → train Δ to predict this.
    - Train set: tight_passive, tight_aggressive, loose_passive, loose_aggressive,
                 maniac, random  (6 opponents)
    - Validation set: heuristic, cfr  (held out)
    - Train for up to 1000 epochs with early stopping on validation MSE.

Usage:
  python train.py --variant variant_a_td
  python train.py --variant variant_b_supervised
  python train.py --variant variant_a_td --smoke        # quick pipeline check
  python train.py --variant variant_b_supervised --smoke
"""

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.value_based.agent import ValueBasedAgent
from paper.evaluation.comparison_protocol import (
    STANDARD_OPPONENT_KEYS,
    build_standard_opponents,
    evaluate_stat_aware_pool,
    format_pool_summary,
)

from paper.evaluation.shared.stats_tracker import (
    OpponentStatsTracker, STAT_KEYS, N_STATS,
    compute_pool_means, play_hand as _aug_play_hand,
)

from preliminary_experiments.opp_stats_modulation_v1.agent import (
    StatModValueAgent, GAME_DIMS, STAT_DIMS,
)
from engine.leduc_game import LeducGame, Action

# ── constants ──────────────────────────────────────────────────────────────────

SESSION_LENGTH   = 100      # hands per session (TD variant)
PRIOR_STRENGTH   = 20.0
CALIBRATION_HANDS = 500     # per opponent for prototype stats
BATCH_SIZE       = 32       # TD variant
LR_TD            = 1e-4
LR_SUP           = 1e-3
WEIGHT_DECAY_SUP = 1e-4     # L2 regularization for supervised (small dataset)
EVAL_INTERVAL    = 100      # episodes between evals (TD variant)
EVAL_ROUNDS      = 200
FLUSH_EVERY      = 500
CHECKPOINT_METRIC = "robustness"

OPPONENT_KEYS = list(STANDARD_OPPONENT_KEYS)
# EV_variation_analysis has these 8 opponents (note: "value_based" not "heuristic")
EV_OPPONENT_KEYS = [
    "cfr", "value_based",
    "tight_passive", "tight_aggressive",
    "loose_passive", "loose_aggressive",
    "maniac", "random",
]
# Train on 6 rule-based + value_based; validate on cfr (near-optimal, hardest test)
TRAIN_KEYS = [
    "tight_passive", "tight_aggressive",
    "loose_passive", "loose_aggressive",
    "maniac", "random", "value_based",
]
VAL_KEYS = ["cfr"]

CARD_MAP  = {'J': 0, 'Q': 1, 'K': 2}
MAX_CHIPS = 13


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _ema(values, alpha=0.95):
    if not values:
        return []
    s = values[0]
    out = []
    for v in values:
        s = alpha * s + (1 - alpha) * v
        out.append(s)
    return out


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


# ── game-loop helpers (adapted for StatModValueAgent) ─────────────────────────

def _encode_game_state(obs, viewer_id: int) -> torch.Tensor:
    """15-dim game encoding (shared logic with agent, avoids agent dependency)."""
    hand_idx = CARD_MAP.get(obs.player_hand)
    hand_vec = torch.zeros(3)
    if hand_idx is not None:
        hand_vec[hand_idx] = 1.0
    board_idx = CARD_MAP.get(obs.board, 3)
    board_vec = torch.zeros(4)
    board_vec[board_idx] = 1.0
    p0, p1 = obs.pot
    pot_rel = [p0, p1] if viewer_id == 0 else [p1, p0]
    pot_vec = torch.tensor(pot_rel, dtype=torch.float32) / MAX_CHIPS
    feats = torch.tensor([
        1.0 if viewer_id == obs.current_player else 0.0,
        float(viewer_id),
        float(obs.current_round),
        1.0 if obs.is_finished else 0.0,
        1.0 if (obs.board is not None and obs.player_hand == obs.board) else 0.0,
        obs.raises_this_round / 2.0,
    ])
    return torch.cat([hand_vec, board_vec, pot_vec, feats])  # (15,)


def play_hand_mod(agent: StatModValueAgent, opponent, tracker: OpponentStatsTracker,
                  learner_id: int = 0):
    """
    Play one hand for TD training. Returns:
        chain  : list of (game_enc_15, stats_7) tuples for the learner's post-action states
        reward : terminal reward for the learner
    """
    game = LeducGame()
    game.reset()
    chain = []
    prev_raise = False
    prev_round = -1
    opp_id = 1 - learner_id

    while not game.is_finished:
        cp = game.current_player
        obs = game.get_observation(viewer_id=cp)
        if obs.current_round != prev_round:
            prev_raise = False
            prev_round = obs.current_round

        if cp == learner_id:
            stats = tracker.get_features()
            action = agent.select_action(obs, opp_stats=stats)
            post_obs, _ = LeducGame.simulate_action(obs, action)
            game_enc = _encode_game_state(post_obs, viewer_id=learner_id)
            chain.append((game_enc, stats.copy()))
        else:
            action = opponent.select_action(obs)
            tracker.update_action(action, obs.current_round,
                                  prev_raise, obs.legal_actions)
        prev_raise = (action == Action.RAISE)
        game.step(action)

    tracker.update_hand_end()
    rewards = game.get_reward()
    return chain, rewards[learner_id]


# ── Variant A: TD(0) training ─────────────────────────────────────────────────

def td_update(agent: StatModValueAgent, optimizer, criterion, batch_data):
    """
    TD(0) update — only modulation head parameters receive gradients.

    At terminal state T: target = r − V_base(s_T)  →  effectively direct residual regression
    At non-terminal t : target = (V_base(s_{t+1}) + Δ(s_{t+1})) − V_base(s_t)
                                  bootstrapped residual
    Loss = MSE(Δ(s_t) − target_residual)
    """
    optimizer.zero_grad()
    losses = []
    for chain, reward in batch_data:
        if not chain:
            continue
        for t, (game_enc, stats) in enumerate(chain):
            game_t = game_enc.unsqueeze(0)              # (1, 15)
            stats_t = torch.tensor(stats, dtype=torch.float32).unsqueeze(0)  # (1, 7)
            mod_inp = torch.cat([game_t, stats_t], dim=1)  # (1, 22)

            # frozen base
            with torch.no_grad():
                v_base_t = agent.base(game_t).squeeze()      # scalar

            delta_t = agent.mod(mod_inp).squeeze()           # scalar (grad flows)

            if t == len(chain) - 1:
                # terminal: target residual = r − V_base(s_T)
                target_residual = torch.tensor(reward, dtype=torch.float32) - v_base_t
            else:
                # non-terminal: bootstrap from next state total value
                game_t1 = chain[t + 1][0].unsqueeze(0)
                stats_t1 = torch.tensor(chain[t + 1][1], dtype=torch.float32).unsqueeze(0)
                mod_inp1 = torch.cat([game_t1, stats_t1], dim=1)
                with torch.no_grad():
                    v_base_t1 = agent.base(game_t1).squeeze()
                    delta_t1  = agent.mod(mod_inp1).squeeze()
                    v_total_t1 = v_base_t1 + delta_t1
                # target residual = next total value − base(current)
                target_residual = v_total_t1 - v_base_t

            losses.append(criterion(delta_t, target_residual.detach()))

    if not losses:
        return 0.0
    mean_loss = torch.stack(losses).mean()
    mean_loss.backward()
    optimizer.step()
    return mean_loss.item()


def quick_eval_mod(agent, opponents, eval_rounds, pool_means):
    return evaluate_stat_aware_pool(
        agent=agent,
        opponents=opponents,
        play_hand_fn=play_hand_mod,
        pool_means=pool_means,
        num_rounds=eval_rounds,
        session_length=SESSION_LENGTH,
        prior_strength=PRIOR_STRENGTH,
        opponent_keys=OPPONENT_KEYS,
        alternate_positions=True,
    )


def run_variant_a(out_dir, num_episodes, smoke, opponents):
    """Variant A: TD(0) pool_random training."""
    os.makedirs(out_dir, exist_ok=True)

    # pool priors (shared from opp_stats_input_augmentation_v1 if available)
    priors_path = os.path.join(out_dir, "pool_priors.json")
    aug_priors  = os.path.join(HERE, "..", "opp_stats_input_augmentation_v1",
                               "outputs", "pool_random", "pool_priors.json")
    if os.path.exists(priors_path):
        pool_means = json.load(open(priors_path))
        print(f"Loaded pool priors from {priors_path}")
    elif os.path.exists(aug_priors):
        pool_means = json.load(open(aug_priors))
        _write_json(priors_path, pool_means)
        print(f"Reused pool priors from input_augmentation_v1")
    else:
        print("Running pool calibration...")
        cal_hands = 100 if smoke else CALIBRATION_HANDS
        pool_means = compute_pool_means(opponents, cal_hands)
        _write_json(priors_path, pool_means)

    agent     = StatModValueAgent()
    optimizer = optim.Adam(agent.mod.parameters(), lr=LR_TD)
    criterion = nn.MSELoss()
    agent.set_train_mode(True)

    config = {
        "experiment_id": "opp_stats_modulation_v1_variant_a_td",
        "variant": "variant_a_td",
        "architecture": "StatModValueAgent(base=frozen, mod=22→32→32→1)",
        "training_schedule": "pool_random",
        "learning_rate": LR_TD,
        "batch_size": BATCH_SIZE,
        "session_length": SESSION_LENGTH,
        "num_episodes": num_episodes,
        "checkpoint_metric": CHECKPOINT_METRIC,
        "seat_protocol": "balanced_alternating_training",
        "eval_protocol": "both_seats_with_100_hand_resets",
    }
    _write_json(os.path.join(out_dir, "train_config.json"), config)

    ordered_opps = [opponents[k] for k in OPPONENT_KEYS]
    current_opponent = random.choice(ordered_opps)
    trackers = {
        0: OpponentStatsTracker(pool_means, PRIOR_STRENGTH, SESSION_LENGTH),
        1: OpponentStatsTracker(pool_means, PRIOR_STRENGTH, SESSION_LENGTH),
    }
    hands_in_session = {0: 0, 1: 0}
    batch_data = []
    train_history = []; eval_history = []
    new_train = [];     new_eval    = []
    best_metric = float("-inf")
    last_eval_bucket = -1

    def flush():
        if new_train:
            train_history.extend(new_train); new_train.clear()
            _write_json(os.path.join(out_dir, "train_history.json"), train_history)
        if new_eval:
            eval_history.extend(new_eval); new_eval.clear()
            _write_json(os.path.join(out_dir, "eval_history.json"), eval_history)

    print(f"\n{'='*65}")
    print(f"  opp_stats_modulation_v1 | VARIANT A TD | {'SMOKE' if smoke else 'FULL'}")
    print(f"  episodes: {num_episodes:,}  batch={BATCH_SIZE}  session={SESSION_LENGTH}")
    print(f"{'='*65}\n")

    t0 = time.time()
    for ep in range(1, num_episodes + 1):
        learner_id = ep % 2

        if hands_in_session[0] >= SESSION_LENGTH and hands_in_session[1] >= SESSION_LENGTH:
            for tracker in trackers.values():
                tracker.reset()
            hands_in_session = {0: 0, 1: 0}
            current_opponent = random.choice(ordered_opps)

        chain, reward = play_hand_mod(agent, current_opponent, trackers[learner_id], learner_id=learner_id)
        hands_in_session[learner_id] += 1
        batch_data.append((chain, reward))

        if len(batch_data) >= BATCH_SIZE:
            loss = td_update(agent, optimizer, criterion, batch_data)
            batch_data.clear()
            new_train.append({"episode": ep, "loss": round(loss, 6)})
            if len(new_train) >= FLUSH_EVERY:
                flush()
            if ep <= BATCH_SIZE or ep % 1000 == 0:
                print(f"Episode {ep:,}, Loss: {loss:.4f}")

        bucket = ep // EVAL_INTERVAL
        if bucket > last_eval_bucket:
            last_eval_bucket = bucket
            pool_eval = quick_eval_mod(agent, opponents, EVAL_ROUNDS, pool_means)
            scores = pool_eval["scores"]
            summary = pool_eval["summary"]
            new_eval.append({"episode": ep, **scores})
            if len(new_eval) >= FLUSH_EVERY:
                flush()
            metric_value = summary["metric_values"][CHECKPOINT_METRIC]
            if metric_value > best_metric:
                best_metric = metric_value
                agent.save_model(os.path.join(out_dir, "checkpoint_best.pt"))
            print(f"  [ep={ep:>7,}]  heuristic:{scores['heuristic']:+.3f}  cfr:{scores['cfr']:+.3f}"
                  f"  tp:{scores['tight_passive']:+.2f}  ta:{scores['tight_aggressive']:+.2f}"
                  f"  lp:{scores['loose_passive']:+.2f}  la:{scores['loose_aggressive']:+.2f}"
                  f"  mn:{scores['maniac']:+.2f}  rnd:{scores['random']:+.2f}"
                  f"  [{CHECKPOINT_METRIC}:{metric_value:+.3f}  best:{best_metric:+.3f}]"
                  f"  {format_pool_summary(summary)}")

    flush()
    elapsed = time.time() - t0
    agent.save_model(os.path.join(out_dir, "checkpoint.pt"))

    all_losses  = [r["loss"] for r in train_history]
    converged, pct = _convergence_check(all_losses)
    agent.set_train_mode(False)
    final_eval = quick_eval_mod(agent, opponents, EVAL_ROUNDS, pool_means)
    final_scores = final_eval["scores"]
    final_summary = final_eval["summary"]

    results = {
        "experiment_id":    "opp_stats_modulation_v1_variant_a_td",
        "variant":          "variant_a_td",
        "training_episodes": num_episodes,
        "converged":        converged,
        "loss_plateau_pct": round(pct, 2),
        "checkpoint_metric": CHECKPOINT_METRIC,
        "best_checkpoint_metric_value": round(best_metric, 4),
        "peak_heuristic":   round(best_metric, 4),
        "final_heuristic":  round(final_scores["heuristic"], 4),
        "final_cfr":        round(final_scores["cfr"], 4),
        "overall_avg":      final_summary["avg"],
        "worst_case":       final_summary["worst_case"],
        "best_case":        final_summary["best_case"],
        "robustness":       final_summary["robustness"],
        "elapsed_seconds":  round(elapsed, 1),
    }
    _write_json(os.path.join(out_dir, "results.json"), results)
    print(f"\nVariant A complete ({elapsed:.1f}s). checkpoint.pt + results.json saved.")
    print(f"  best {CHECKPOINT_METRIC:>10}: {best_metric:+.3f}")
    print(f"  {format_pool_summary(final_summary)}")
    print(f"  converged      : {converged} ({pct:.1f}%)")


# ── Variant B: Supervised residual regression ─────────────────────────────────

def _compute_prototype_stats(opponents, smoke):
    """
    Play CALIBRATION_HANDS against each opponent individually and record
    per-opponent 7-dim stat vectors (prototype embeddings).
    Covers all opponents present in the EV_variation_analysis data, including
    'value_based' which is not in the standard OPPONENT_KEYS pool.
    Returns: dict[opp_name -> np.ndarray (7,)]
    """
    from agents.rule_based.random_agent import RandomAgent as Rnd

    dummy_pool_means = {k: 0.5 for k in STAT_KEYS}
    learner = Rnd()
    cal = 50 if smoke else CALIBRATION_HANDS
    prototypes = {}

    # build extended opponent dict that includes value_based
    vb_ckpt  = os.path.join(ROOT, "agents", "value_based", "checkpoint.pt")
    from agents.value_based.agent import ValueBasedAgent as VBA
    extended_opps = dict(opponents)  # copy
    extended_opps["value_based"] = VBA(model_path=vb_ckpt)
    extended_opps["value_based"].set_train_mode(False)

    for name in EV_OPPONENT_KEYS:
        opp = extended_opps[name]
        tracker = OpponentStatsTracker(dummy_pool_means, PRIOR_STRENGTH,
                                       session_length=cal + 1)
        game = LeducGame()
        prev_raise = False
        prev_round = -1
        for _ in range(cal):
            game.reset()
            prev_raise = False
            prev_round = -1
            while not game.is_finished:
                cp = game.current_player
                obs = game.get_observation(viewer_id=cp)
                if obs.current_round != prev_round:
                    prev_raise = False
                    prev_round = obs.current_round
                if cp == 0:
                    action = learner.select_action(obs)
                else:
                    action = opp.select_action(obs)
                    tracker.update_action(action, obs.current_round,
                                          prev_raise, obs.legal_actions)
                prev_raise = (action == Action.RAISE)
                game.step(action)
            tracker.update_hand_end()

        stats = tracker.get_features()
        prototypes[name] = stats
        print(f"  {name:<22}: {np.round(stats, 3)}")

    return prototypes


def _card_removal_probs(my_hand: str, board=None) -> dict:
    """
    Return P(opp_hand = h | my_hand, board) for h in {J, Q, K}.
    Uses card-removal from a 6-card deck (2×J, 2×Q, 2×K).
    """
    counts = {'J': 2, 'Q': 2, 'K': 2}
    counts[my_hand] -= 1
    if board is not None:
        counts[board] -= 1
    total = sum(counts.values())
    if total <= 0:
        return {'J': 1/3, 'Q': 1/3, 'K': 1/3}
    return {h: c / total for h, c in counts.items()}


def _build_supervised_dataset(ev_data, base_agent: StatModValueAgent, prototypes,
                               train_keys, val_keys):
    """
    Build (inputs, targets) tensors for supervised regression.

    For each record in ev_data:
      - Determine my_hand = hand of current_player
      - Compute card-removal probability over opponent's possible hands
      - Weighted-average EV across opponent hands for the same (state config, opponent)
      - Encode info-set state (15-dim)
      - Append prototype stats (7-dim)
      - Target = weighted_ev − V_base(state_enc)

    Returns: (X_train, y_train, X_val, y_val) as torch.Tensor
    """
    # Group records by (state_config, opponent):
    # state_config = (round, pot, current_player, raises, board)
    # Within each group, average over opp-hand dimension with card-removal weights.

    # Step 1: group by (my_hand, board, pot tuple, current_player, raises, opponent)
    groups = {}  # key -> {opp_hand: (ev, count)}
    for rec in ev_data:
        cp = rec["current_player"]
        my_hand  = rec["hand0"] if cp == 0 else rec["hand1"]
        opp_hand = rec["hand1"] if cp == 0 else rec["hand0"]
        board    = rec.get("board")
        key = (my_hand, board, tuple(rec["pot"]), cp, rec["raises"], rec["opponent"])
        if key not in groups:
            groups[key] = {}
        # Store (ev, n) per opp_hand; average if duplicates exist
        if opp_hand not in groups[key]:
            groups[key][opp_hand] = []
        groups[key][opp_hand].append(rec["ev"])

    # Step 2: build samples
    train_inputs, train_targets = [], []
    val_inputs,   val_targets   = [], []

    for (my_hand, board, pot, cp, raises, opp_name), hand_evs in groups.items():
        probs = _card_removal_probs(my_hand, board)
        # weighted average EV over opponent hands
        ev_infoset = 0.0
        weight_total = 0.0
        for opp_hand, ev_list in hand_evs.items():
            p = probs.get(opp_hand, 0.0)
            ev_infoset += p * (sum(ev_list) / len(ev_list))
            weight_total += p
        if weight_total > 0:
            ev_infoset /= weight_total

        if opp_name not in prototypes:
            continue

        # Build a minimal obs-like structure for encoding
        class _Obs:
            pass
        obs = _Obs()
        obs.player_hand = my_hand
        obs.board = board
        obs.pot = list(pot)
        obs.current_player = cp
        obs.current_round = 0 if board is None else 1
        obs.is_finished = False
        obs.raises_this_round = raises

        game_enc = _encode_game_state(obs, viewer_id=cp).unsqueeze(0)  # (1, 15)
        with torch.no_grad():
            v_base = base_agent.base(game_enc).item()

        residual = ev_infoset - v_base
        opp_stats = torch.tensor(prototypes[opp_name], dtype=torch.float32)
        inp = torch.cat([game_enc.squeeze(0), opp_stats])  # (22,)
        tgt = torch.tensor([residual], dtype=torch.float32)

        if opp_name in train_keys:
            train_inputs.append(inp)
            train_targets.append(tgt)
        elif opp_name in val_keys:
            val_inputs.append(inp)
            val_targets.append(tgt)

    X_train = torch.stack(train_inputs)
    y_train = torch.stack(train_targets).squeeze(1)
    X_val   = torch.stack(val_inputs)
    y_val   = torch.stack(val_targets).squeeze(1)
    return X_train, y_train, X_val, y_val


def run_variant_b(out_dir, num_epochs, smoke, opponents):
    """Variant B: Supervised residual regression."""
    os.makedirs(out_dir, exist_ok=True)

    # load EV oracle data
    ev_data_path = os.path.join(HERE, "..", "EV_variation_analysis", "data.json")
    if not os.path.exists(ev_data_path):
        print(f"EV data not found: {ev_data_path}")
        sys.exit(1)
    with open(ev_data_path) as f:
        raw = json.load(f)
    ev_data = raw["records"] if isinstance(raw, dict) and "records" in raw else raw
    print(f"Loaded {len(ev_data)} EV records from {ev_data_path}")

    # compute per-opponent prototype stats
    proto_path = os.path.join(out_dir, "prototype_stats.json")
    if os.path.exists(proto_path):
        raw = json.load(open(proto_path))
        prototypes = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
        print(f"Loaded prototype stats from {proto_path}")
    else:
        print("Computing per-opponent prototype stats...")
        prototypes = _compute_prototype_stats(opponents, smoke)
        _write_json(proto_path, {k: v.tolist() for k, v in prototypes.items()})
        print(f"Prototype stats saved → {proto_path}")

    # build supervised dataset
    base_agent = StatModValueAgent()  # loads frozen base
    print("Building supervised dataset...")
    X_train, y_train, X_val, y_val = _build_supervised_dataset(
        ev_data, base_agent, prototypes, TRAIN_KEYS, VAL_KEYS
    )
    print(f"  Train: {X_train.shape[0]} samples,  Val: {X_val.shape[0]} samples")

    # train modulation head
    optimizer = optim.Adam(base_agent.mod.parameters(),
                           lr=LR_SUP, weight_decay=WEIGHT_DECAY_SUP)
    criterion = nn.MSELoss()
    base_agent.set_train_mode(True)

    config = {
        "experiment_id": "opp_stats_modulation_v1_variant_b_supervised",
        "variant": "variant_b_supervised",
        "architecture": "StatModValueAgent(base=frozen, mod=22→32→32→1)",
        "training_method": "supervised_residual_regression",
        "train_opponents": TRAIN_KEYS,
        "val_opponents": VAL_KEYS,
        "n_train_samples": X_train.shape[0],
        "n_val_samples": X_val.shape[0],
        "learning_rate": LR_SUP,
        "weight_decay": WEIGHT_DECAY_SUP,
        "num_epochs": num_epochs,
    }
    _write_json(os.path.join(out_dir, "train_config.json"), config)

    history = []
    best_val_loss  = float("inf")
    patience       = 50 if not smoke else 5
    patience_count = 0

    print(f"\n{'='*65}")
    print(f"  opp_stats_modulation_v1 | VARIANT B SUPERVISED | {'SMOKE' if smoke else 'FULL'}")
    print(f"  epochs: {num_epochs}  train={X_train.shape[0]}  val={X_val.shape[0]}")
    print(f"{'='*65}\n")

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        # shuffle train set
        perm = torch.randperm(X_train.shape[0])
        X_s  = X_train[perm]
        y_s  = y_train[perm]

        # mini-batch SGD
        base_agent.mod.train()
        train_losses = []
        for start in range(0, X_s.shape[0], BATCH_SIZE):
            xb = X_s[start:start + BATCH_SIZE]
            yb = y_s[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            pred = base_agent.mod(xb).squeeze(1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)

        # validation
        base_agent.mod.eval()
        with torch.no_grad():
            val_pred = base_agent.mod(X_val).squeeze(1)
            val_loss = criterion(val_pred, y_val).item()

        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "val_loss": round(val_loss, 6)})

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            base_agent.save_model(os.path.join(out_dir, "checkpoint_best.pt"))
        else:
            patience_count += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  train_mse={train_loss:.5f}  "
                  f"val_mse={val_loss:.5f}  [best={best_val_loss:.5f}  patience={patience_count}/{patience}]")

        if patience_count >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience} exhausted)")
            break

    elapsed = time.time() - t0
    _write_json(os.path.join(out_dir, "train_history.json"), history)
    base_agent.save_model(os.path.join(out_dir, "checkpoint.pt"))

    results = {
        "experiment_id":   "opp_stats_modulation_v1_variant_b_supervised",
        "variant":         "variant_b_supervised",
        "final_train_mse": history[-1]["train_loss"] if history else None,
        "best_val_mse":    round(best_val_loss, 6),
        "epochs_trained":  len(history),
        "elapsed_seconds": round(elapsed, 1),
    }
    _write_json(os.path.join(out_dir, "results.json"), results)
    print(f"\nVariant B complete ({elapsed:.1f}s). Best val MSE={best_val_loss:.5f}")
    print("Run eval.py --variant variant_b_supervised for final evaluation.")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["variant_a_td", "variant_b_supervised"])
    parser.add_argument("--smoke",   action="store_true",
                        help="Short pipeline smoke-test")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episode count (variant_a_td)")
    parser.add_argument("--epochs",  type=int, default=1000,
                        help="Max epochs (variant_b_supervised)")
    args = parser.parse_args()

    opponents = build_standard_opponents(ROOT)
    out_dir   = os.path.join(HERE, "outputs", args.variant)

    if args.variant == "variant_a_td":
        episodes = args.episodes or (500 if args.smoke else 300_000)
        run_variant_a(out_dir, episodes, args.smoke, opponents)
    else:
        epochs = 10 if args.smoke else args.epochs
        run_variant_b(out_dir, epochs, args.smoke, opponents)


if __name__ == "__main__":
    main()
