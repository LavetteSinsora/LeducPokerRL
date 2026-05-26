# opp_stats_input_augmentation_v1 — Opponent Statistics as Input Augmentation

**Experiment thread:** OpponentModeling
**Builds on:** `baseline_value_v1`
**Date:** 2026-03-15
**Status:** Complete

---

## Motivation

The baseline value network (`baseline_value_v1`) uses a 15-dimensional game-state observation: hand,
board card, pot sizes, round, position, and raise count. It has **no representation of the
opponent** — it plays the same strategy regardless of whether it is facing a maniac or a tight
passive player.

This experiment asks: *does augmenting the value network's input with on-the-fly opponent
statistics improve performance?*

The hypothesis is that round-stratified behavioral statistics (e.g., pre-flop raise rate vs.
flop raise rate) provide an informative signal that lets the network adapt its value estimates
to the opponent's playing style.

---

## What Changed from Baseline

| Dimension | Baseline | opp_stats_input_augmentation_v1 |
|---|---|---|
| Input size | 15 | **22** (15 game + 7 opponent stats) |
| Hidden size | 64 | 64 (unchanged) |
| Optimizer | Adam lr=1e-3 | Adam lr=1e-3 (unchanged) |
| Batch size | 32 | 32 (unchanged) |
| TD(0) discount | 0.99 | 0.99 (unchanged) |
| Training scheme | Self-play | Three variants (see below) |

---

## The 7 Opponent Statistic Features

All features use **Beta-Bernoulli Bayesian smoothing** with pool-average priors and prior
strength S = 20.

```
p̂ = (k + α) / (n + α + β)
    where α = pool_mean × S,  β = (1 − pool_mean) × S
```

- At n = 0 (session start): p̂ = pool_mean — starts at a realistic average, not a neutral 0.5
- At n = S = 20 hands: prior and observed data contribute equally
- At n >> S: estimate converges to the observed frequency

| # | Feature | Numerator | Denominator | Key signal |
|---|---|---|---|---|
| 1 | `preflop_fold_rate` | pre-flop folds | pre-flop decisions | Tight vs loose entry |
| 2 | `preflop_raise_rate` | pre-flop raises | pre-flop decisions | Pre-flop aggression |
| 3 | `flop_raise_rate` | flop raises | flop decisions | Post-board aggression; spike over feature 2 → likely paired with board |
| 4 | `preflop_fold_to_raise` | pre-flop folds when facing raise | pre-flop raise-facing decisions | Pressure tolerance pre-flop |
| 5 | `flop_fold_to_raise` | flop folds when facing raise | flop raise-facing decisions | Pressure tolerance post-board |
| 6 | `raise_after_raise_rate` | re-raises when RAISE is legal | raise-facing decisions with RAISE legal | Distinguishes maniac-level aggression |
| 7 | `confidence` | `hands_seen / (hands_seen + S)` | — | How much to trust features 1–6 |

Stats are **reset at the start of each session** (100 rounds). The network therefore starts each
new encounter with the prior and gradually adapts as data accumulates.

**Pool calibration:** Before training, 500 hands are played against each of the 8 opponents
using a random learner agent. The empirical average across all opponents forms the pool-mean
prior centers. Results are saved to `outputs/<variant>/pool_priors.json`.

---

## Three Training Variants

### Variant A: `self_play`
- Agent plays against itself (same as baseline regime)
- Stats track the agent's own behavior in the opposite seat
- Session: 100 hands, then stats reset
- Total episodes: 300,000

**Expected behavior:** Stats are stable (agent models itself), network may not learn to exploit
the stat features, but this tests whether the augmented input dimension interferes with
baseline performance.

### Variant B: `pool_random`
- Each session: one opponent sampled uniformly at random from the pool of 8
  (tight_passive, tight_aggressive, loose_passive, loose_aggressive, maniac, random, heuristic, cfr)
- Stats track the randomly sampled opponent for that session
- Session: 100 hands, then stats reset and new opponent sampled
- Total episodes: 300,000

**Expected behavior:** Network sees diverse stat profiles during training, learning to use stat
features as genuine signals about opponent style. Gradient updates are consistent within each
session (same opponent, same stats) but varied across sessions.

### Variant C: `pool_seq`
- Train K = 500 sessions (50,000 episodes) against each opponent sequentially
- Opponent order: tight_passive → tight_aggressive → loose_passive → loose_aggressive →
  maniac → random → heuristic → cfr
- Stats reset at the start of each opponent block
- Total episodes: 500 × 100 × 8 = 400,000

**Expected behavior:** Deep specialization per opponent, but risk of catastrophic forgetting
when the opponent switches. Later opponents receive better-initialized weights but also face
the most forgetting pressure.

---

## Hyperparameters

```python
# Model
INPUT_SIZE       = 22          # 15 game + 7 opponent stats
HIDDEN_SIZE      = 64
OUTPUT_SIZE      = 1

# Training
LEARNING_RATE    = 1e-3
BATCH_SIZE       = 32
GAMMA            = 0.99        # TD(0) discount
TEMPERATURE      = 1.0         # Boltzmann exploration temperature

# Stats tracker
PRIOR_STRENGTH   = 20.0        # S: effective prior sample count
SESSION_LENGTH   = 100         # hands per session before stats reset
CALIBRATION_HANDS = 500        # hands per opponent for pool calibration

# Training budget
EPISODES_SELF_PLAY   = 300_000
EPISODES_POOL_RANDOM = 300_000
SESSIONS_PER_OPP_SEQ = 500     # → 400K total episodes

# Evaluation (during training)
EVAL_INTERVAL    = 96          # episodes between eval checkpoints
EVAL_ROUNDS      = 200         # rounds per eval vs heuristic + cfr

# Final evaluation
FINAL_EVAL_ROUNDS = 5_000      # rounds vs each of 8 opponents
```

---

## Results

### Final Evaluation (5,000 rounds vs each opponent, best checkpoint)

| Opponent | baseline_v1 | self_play | pool_random | pool_seq |
|---|---|---|---|---|
| heuristic | **+0.346** | -0.215 | -0.009 | +0.187 |
| cfr | -0.098 | -0.383 | -0.375 | **-0.141** |
| tight_passive | **+0.316** | +0.179 | +0.306 | +0.157 |
| tight_aggressive | **+0.300** | -0.470 | -0.142 | +0.154 |
| loose_passive | **+0.510** | -0.254 | +0.244 | +0.438 |
| loose_aggressive | **+0.798** | +0.812 | +0.316 | +0.736 |
| maniac | +1.165 | +1.292 | **+1.675** | +0.543 |
| random | +1.304 | +1.515 | **+1.689** | +0.974 |
| **Overall avg** | **+0.580** | +0.310 | +0.463 | +0.381 |
| **Worst case** | -0.098 | -0.470 | -0.375 | **-0.141** |

### Training Summary

| Variant | Episodes | Converged | Loss (final) | Peak vs heuristic | Time |
|---|---|---|---|---|---|
| baseline_v1 | 300K | ❌ 9.3% | 12.33 | +1.435 | 27 min |
| self_play | 300K | ✅ 1.5% | 4.41 | +0.880 | 34 min |
| pool_random | 300K | ✅ 4.0% | 5.06 | +0.970 | 30 min |
| pool_seq | 400K | ❌ 7.0% | 8.03 | +1.165 | 41 min |

---

## Analysis

### Finding 1: Baseline outperforms all opp_stats_input_augmentation_v1 variants overall

Despite 22-dimensional input, no variant beats `baseline_value_v1` on overall average (+0.580).
This is a **meaningful negative result**. Adding opponent statistics in this form does not
improve — and in most cases hurts — overall performance.

### Finding 2: Cold-start noise is the primary suspect

Each session begins with `confidence = 0`, meaning all 7 stat features sit at the pool prior.
For the first ~20 hands of every 100-hand session, the opponent-stat features carry almost no
signal and introduce noise into the network's inputs. Since 20% of every session is
near-zero-signal, a substantial fraction of training gradients are computed on noisy inputs.
The baseline has no such noise — its 15-dim input is always clean.

### Finding 3: pool_random learns to exploit weak opponents

pool_random achieves **+1.675 vs maniac** and **+1.689 vs random** — both better than baseline
(+1.165 and +1.304 respectively). Exposure to diverse opponents during training appears to
teach the network to identify and exploit low-skill opponents more aggressively. However, this
comes at the cost of structured opponents: -0.375 vs cfr and -0.142 vs tight_aggressive.

### Finding 4: pool_seq gradient interference confirmed

pool_seq did not converge (7.0% loss plateau vs ≤5% threshold). The sequential opponent
switching causes training instability consistent with catastrophic forgetting. Interestingly,
pool_seq has the best worst-case of the three variants (-0.141 vs cfr) — suggesting that seeing
each opponent in depth creates a more conservative, balanced strategy even if overall exploitability
is poor.

### Finding 5: self_play is the worst variant

Without meaningful variation in the stat features (the agent always models itself), the network
cannot learn to use the 7-dimensional opponent representation. The extra input dimensions add
pure noise, degrading performance below the 15-dim baseline.

---

## Hypotheses for Follow-up

1. **Cold-start mitigation:** Longer sessions (e.g., 500 rounds) would reduce the fraction of
   time spent at low confidence. Alternatively, a "warm-start" that pre-loads stats from the
   pool prior before the first hand might help.

2. **Separate opponent encoder:** Rather than concatenating raw stats to the game features,
   a separate encoder could project the 7 stats into a latent opponent embedding, potentially
   learning richer opponent representations.

3. **Soft session boundary:** Instead of hard resets at 100-hand boundaries, use exponential
   decay (`k ← 0.95k, n ← 0.95n`) for gradual forgetting while preserving recent history.

---

## Replication

```bash
cd /path/to/PokerRL_Vanilla

# Train
python OpponentModeling/opp_stats_input_augmentation_v1/train.py --variant self_play   --episodes 300000
python OpponentModeling/opp_stats_input_augmentation_v1/train.py --variant pool_random --episodes 300000
python OpponentModeling/opp_stats_input_augmentation_v1/train.py --variant pool_seq    --k 500

# Evaluate
python OpponentModeling/opp_stats_input_augmentation_v1/eval.py --variant self_play
python OpponentModeling/opp_stats_input_augmentation_v1/eval.py --variant pool_random
python OpponentModeling/opp_stats_input_augmentation_v1/eval.py --variant pool_seq
```

Outputs are saved under `outputs/<variant>/`:
- `train_history.json` — per-batch loss and per-episode avg-chips at every 50-episode step
- `eval_history.json` — periodic evaluation vs heuristic + cfr during training
- `checkpoint_best.pt` — best checkpoint by heuristic eval score
- `checkpoint.pt` — final checkpoint
- `results.json` — training summary
- `evaluation.json` — final 5K-round evaluation results
- `pool_priors.json` — calibrated pool-average priors used for Beta smoothing
- `training_curve.png` — loss and chips/round over training
- `eval_curve.png` — periodic eval curve vs 8 opponents

---

## Files

| File | Description |
|---|---|
| `stats_tracker.py` | `OpponentStatsTracker`: incremental Beta-Bernoulli stats collection + `play_hand()` game loop |
| `agent.py` | `StatAugValueAgent`: 22-dim value network, encodes game state + opponent stats |
| `train.py` | Training script for all 3 variants (`--variant` flag) |
| `eval.py` | Final evaluation script (5K rounds vs 8 opponents with live stats accumulation) |
