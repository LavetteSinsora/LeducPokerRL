# alphazero_v3 — Reduced Q-Net + Higher k Rollouts

## Hypothesis

v2 diagnosis showed the primary loss driver is high Q* target variance from k=10 rollouts
against an increasingly aggressive policy (Q* std grew 3.28 → 4.41 between ep 10K–30K).
The Q-net (64,64) is massively overparameterized for a 10-dim input; cutting it to (32,32)
gives ~3× faster Q-net forward passes, which buys k=30 rollouts at the same compute budget.
More rollouts reduce Q* std by √3 ≈ 1.7×, directly targeting the noise problem.

Additionally: state_hidden reduced to a single (8,) layer (Leduc has only 9 event types;
one MLP layer is sufficient), and belief_hidden reduced to (8,8).

## Changed Axes vs v2

| Param | v2 | v3 |
|-------|----|----|
| state_hidden | (8, 16) | **(8,)** |
| belief_hidden | (16, 16) | **(8, 8)** |
| q_hidden | (64, 64) | **(32, 32)** |
| k_rollouts | 10 | **30** |

d_model=4, fixed opponent pool [heuristic, value_based, cfr], lr=1e-3 unchanged.

## Success Criteria

- Loss stabilizes below 5.0 avg by ep 30K (v2 was 8.05 in 30-40K window)
- Tournament avg > −0.9 at best checkpoint
- Robustness > −1.1 at best checkpoint (v2 best was −1.169 at ep 30K)
- Raise spread converges rather than oscillates
