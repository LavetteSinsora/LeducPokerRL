# alphazero_v5 — Position-Symmetric Training

## Diagnosis from v4

v4 best checkpoint (ep 10K): avg=−0.789, robust=−0.956.
Despite asymmetric T and replay buffer, the agent degrades monotonically after ep 10K.

Diagnosis (diagnose_v4.py on checkpoints ep 10K/40K/80K) identified **one root cause**:

### Belief Network P0/P1 Asymmetry

| Checkpoint | P0 belief accuracy | P1 belief accuracy |
|---|---|---|
| ep 10K | 0.448 (+11% over random) | 0.332 (≈ random) |
| ep 40K | 0.465 (+13%) | 0.303 (−3%, below random) |
| ep 80K | 0.477 (+14%) | 0.317 (−2%) |

**The BeliefNet learns to track opponents only from P0's perspective.** The cause:
belief CE loss is computed only at P0 decision steps. P1's belief gets zero gradient.

Consequence: every Q* computed for P1 decisions is built on random belief → PIMC
averages over uniform opponent hands → Q* is noisy for P1 → Q-net learns broken P1
strategies. This explains the growing P0/P1 performance gap (−0.157 at ep 10K →
−0.452 at ep 60K).

Q* value spread (max−min legal Q*) was 3.5–4.6 at all checkpoints — **not degenerate**.
Bootstrap instability is not the remaining bottleneck.

---

## Hypothesis

Three targeted fixes cure the positional asymmetry:

### Fix 1 — Belief loss for both players

In `_belief_loss_only`, extend CE loss to P1 decision steps.
Both belief states are already tracked during episode replay; this is a 1-line change
in the loss computation gate.

**Expected effect:** P1 belief accuracy rises from ~33% to ~44%+ (matching P0).
BeliefNet becomes useful for PIMC when the agent plays as P1.

### Fix 2 — Alternating player roles

50% of training episodes: AZ agent = P0, opponent = P1 (same as v4).
50% of training episodes: AZ agent = P1, opponent = P0.

When az_player=1, P1 makes PIMC decisions, P1 records go into the replay buffer,
and the Q-net trains on correct P1 Q* targets.

**Expected effect:** Closes the P0/P1 performance gap entirely.
Replay buffer becomes position-balanced (50% P0 records, 50% P1 records).

### Fix 3 — Position feature in Q-net

Add a 1-bit `player_id` (0 or 1) to Q-net input: `d+6 → d+7` input dimension.
Allows the Q-net to learn explicitly position-dependent strategies without relying
on `P_t` to carry implicit positional information.

Input: `[P_t (d=4) | h_onehot (3) | b_mine (3) | pos_bit (1)]` = 11-dim.

**Expected effect:** Q-net value estimates for P0 and P1 scenarios become independent,
preventing cross-contamination between position-specific strategies.

### Speed: k=20 rollouts (was k=30)

The replay buffer compensates for higher Q* variance from fewer rollouts.
Reduces PIMC runtime by ~33% per episode.

---

## Changed Axes vs v4

| Parameter | v4 | v5 |
|-----------|----|----|
| Belief CE loss | P0 decisions only | **both players** |
| Training position | always P0 | **alternates P0/P1 50/50** |
| Q-net input dim | d+6 = 10 | **d+7 = 11 (+pos_bit)** |
| k_rollouts | 30 | **20** |

Architecture otherwise unchanged: d=4, state(8,), belief(8,8), Q(32,32).
Opponent pool unchanged: [heuristic, value_based, cfr].
All v4 stability fixes retained: target Q-net, replay buffer, asymmetric T,
entropy bonus.

---

## Success Criteria

| Metric | v4 best (ep10K) | v5 target |
|--------|-----------------|-----------|
| Best robust | −0.956 | **> −0.85** |
| Best avg | −0.789 | **> −0.70** |
| P0/P1 gap | −0.157 to −0.462 | **< −0.10** |
| P1 belief accuracy | ~33% (random) | **> 40%** |
| Loss convergence | stable ~3.0 | **stable < 2.5 by ep 50K** |

---

## Risk

- **Alternating roles doubles variance of per-episode signal** — mitigated by replay buffer
  which holds decisions from both positions.
- **Position bit may be redundant** — `P_t` already encodes action history implicitly.
  If v5 shows the same results without it, it's safe to drop in v6.
- **k=20 Q* variance** — with 50K replay buffer and batch 256, stale Q* from earlier
  target versions is the bigger concern. Monitor whether best checkpoint shifts earlier.
