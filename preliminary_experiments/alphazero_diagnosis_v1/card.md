# Experiment Card: alphazero_diagnosis_v1

## Hypothesis

The AlphaZeroStyle agent degraded from ep 70K (robustness -1.02) to ep 120K (robustness -1.28)
with raise rates collapsing from ~{21%, 19%, 25%} to ~{10%, 12%, 17%} for hands J/Q/K.

We hypothesize that at least one of the three jointly-trained modules (StateEncoder, BeliefNet, QNet)
lost representational quality during training, causing the observed strategy collapse.

**Three candidate failure modes:**
1. BeliefNet stopped improving its estimate of the opponent's hand
2. P_t vectors (public state) became indistinguishable across game steps
3. Q-network lost its ability to differentiate Q-values across hands

## Success Criteria

For each of D1–D6, produce plots comparing checkpoints at ep 70K–120K that either:
- **Confirm** a specific failure mode (e.g., belief accuracy stays flat after ep 90K), or
- **Rule it out** (e.g., P_t vectors remain discriminative throughout)

At minimum, D4 and D5 must produce interpretable plots that tell us whether the
collapse is architectural (P_t / embedding) or behavioral (strategy-level).

## Single Changed Axis

This is a diagnostic experiment — no training axis is changed. We read checkpoints
at fixed intervals and probe internal representations and strategy statistics.

## Checkpoints Analyzed

| Episode | Tournament Avg | Robustness | Raise Spread |
|---------|---------------|-----------|--------------|
| 70K     | -0.919        | -1.019    | ~10%         |
| 80K     | -0.910        | -1.071    | ~10%         |
| 90K     | -0.894        | -1.131    | ~11%         |
| 100K    | -0.899        | -1.102    | ~6%          |
| 110K    | -0.901        | -1.278    | ~14%         |
| 120K    | -0.983        | -1.146    | ~8%          |

## Diagnosis Scripts

| Script         | What it probes                              | Runtime  |
|---------------|---------------------------------------------|----------|
| d4_embeddings | Event embedding structure (9×8 weight matrix) | ~1 min |
| d5_strategy   | Hand-stratified action distributions        | ~5 min   |
| d1_belief     | Belief accuracy vs game step                | ~15 min  |
| d3_pubstate   | P_t cosine distances across game steps      | ~15 min  |
| d6_portfolio  | b_opp diversity (portfolio coherence)       | ~15 min  |
| d2_q_agreement| Q vs Q* agreement (requires PIMC search)   | ~30 min  |
