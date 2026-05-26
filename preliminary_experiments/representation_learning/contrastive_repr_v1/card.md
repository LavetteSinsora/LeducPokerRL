# Contrastive State Representation Learning v1

## Origin and motivation

The value-based agent learns V(s): a scalar mapping from 15-dim state to expected terminal reward via TD(0). This works well but compresses all state information into a single number. Two states with V(s) = 0 are treated identically even if one is "safe neutral" (always +/-1) and the other is "risky neutral" (could be +5 or -5).

This experiment asks: can contrastive/metric learning produce a multi-dimensional state embedding where geometric distance reflects reward similarity -- and does this representation capture meaningful structure beyond what a scalar V(s) provides?

This is Phase 1: representation learning only. No downstream action selection. We evaluate purely by diagnosing the learned embedding structure.

## Parent and changed axis

- Parent agent: `value_based` (same 15-dim input encoding)
- Changed axis: training objective (contrastive/metric learning instead of TD(0))

## Mechanism story

1. Take the same 15-dim observation encoding as `value_based`.
2. Feed it into an encoder (15->64->64->8) to produce an 8-dim embedding z.
3. Train the encoder using contrastive loss so that states with similar terminal rewards are close in embedding space and states with different rewards are far apart.
4. Data is collected via self-play with the trained value-based agent (not random), so terminal rewards are meaningful estimates of V^pi(s).

The training signal (terminal reward proximity) mostly supervises organizing by expected reward. The hoped-for benefit -- capturing distributional/strategic structure beyond expectation -- is not strongly forced by the loss. Whether extra dimensions capture genuinely useful structure or just noise is the empirical question this experiment tests.

## Three formulations tested

### L0: Control (encoder + linear value head, TD(0))
Same architecture (15->8->1) but trained with standard TD(0). Isolates the architectural bottleneck from the loss change.

### L1: Soft Distance Correlation
loss = mean( (||z_i - z_j|| - beta * |R_i - R_j|)^2 ) over pairs.
Embedding distance should be proportional to reward distance. Acts like a spring system. Beta auto-calibrated from random initialization.

### L2: Rank-N-Contrast (based on NeurIPS 2023 paper)
For each (anchor i, candidate j), negatives are all samples k with |R_i - R_k| > |R_i - R_j|.
Uses InfoNCE-style loss with dynamic positive/negative based on reward ranking.
Only requires ordering to be correct, not exact distances. More robust to noisy labels.

## Main risks

1. The whole thing may just relearn V(s) in disguise -- the training signal is reward-based, so extra dimensions may encode noise rather than meaningful structure.
2. Reward stochasticity (same state, different terminal rewards due to hidden opponent hand) creates a noisy supervision signal.
3. Same-trajectory states share terminal rewards, creating shortcut learning opportunities.
4. L1 may face inconsistent constraint systems under noisy labels.

## Training setup

- Data collection: self-play with promoted `value_based` agent
- Replay buffer (capacity 5000) for batch diversity
- Contrastive batch size: 256 samples from buffer
- Episodes per step: 8
- Budget: 20K episodes minimum, extend to 40K if loss still decreasing
- VICReg variance regularization for L1 (L2 has built-in anti-collapse)

## What success looks like

A method is genuinely interesting if it satisfies ALL THREE:
1. Preserves reward order (Spearman rho, k-NN reward error)
2. Retains strategically meaningful info not reducible to scalar value (same-mean/different-variance states separate; same-reward states organize by strategic features)
3. Robust to shortcut removal (performance doesn't collapse when same-trajectory pairs excluded)

## What failure would still teach us

- All formulations collapse to 1D: contrastive learning adds nothing beyond V(s) for Leduc
- Good reward ordering but no variance/strategic structure: training objective supervises reward proximity, so that's all it learns
- L0 surprisingly good: the 8-dim bottleneck itself induces useful structure, not the loss function
- L2 >> L1: ranking is more appropriate than exact metric matching for noisy continuous labels

## Files in this experiment folder

- `agent.py`: encoder network and observation encoding
- `losses.py`: L0, L1, L2 loss implementations + VICReg
- `trainer.py`: replay-buffer-based contrastive trainer
- `train.py`: entrypoint for training runs
- `diagnose.py`: D1-D10 diagnostic probes
- `report.md`: results (filled after training)
- `summary.json`: machine-readable experiment record
