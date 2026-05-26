# PokerRL — Claude Code Instructions

**This file is the single authoritative guide for AI coding agents.**
It is auto-loaded at the start of every session. When there is a conflict
between this file and any other documentation, this file wins.

---

## Project Overview

Leduc Hold'em reinforcement learning research. Two-player, 6-card game (J/Q/K × 2),
2 betting rounds. The headline result is in the paper *Opponent-Modulated Value
Networks for Exploiting Suboptimal Play in Leduc Hold'em*; the repo is structured
around reproducing that paper.

## Repo Layout

```
engine/                  Leduc Hold'em game engine (shared by all code)
agents/                  Framework + canonical opponents
  ├ base.py, evaluation.py, registry.py, tournament_eval.py
  ├ value_based/         Base value-network architecture
  ├ heuristic/, cfr/, adaptive_value/
  └ rule_based/          Tight/loose × passive/aggressive, maniac, random
paper/                   Paper-canonical code (see paper/README.md)
  ├ agents/{value_based_pool, full_modulation, ablations/*}
  ├ baselines/{reinforce, actor_critic, dqn, opp_encoder_v1}
  ├ evaluation/          Round-robin tournament harness + shared utilities
  │   └ meta/            EVAL_CONFIG.json, STANDARDS.md, METRICS_GLOSSARY.md
  ├ ev_analysis/         Monte-Carlo ground-truth EV (for Figure 3)
  ├ figures/             Figure-2 and Figure-3 generation
  └ checkpoints/         Final-epoch weights, committed (~500 KB total)
preliminary_experiments/ Non-paper research threads (AlphaZero, opp encoders,
                         representation learning, hand-conditioned belief,
                         capacity studies, the old dashboard, etc.)
writeup/                 LaTeX source for the paper
tests/                   Engine unit tests
```

## Paper-Canonical Agents

| Paper label                       | Code path                                     |
|-----------------------------------|-----------------------------------------------|
| Modulated Value Net (Ours)        | `paper/agents/full_modulation/`               |
| Value Network (base)              | `paper/agents/value_based_pool/`              |
| + State Mod. (no opp. stats)      | `paper/agents/ablations/state_only/`          |
| + Opp. Stats Mod., unfrozen base  | `paper/agents/ablations/finetuned_base/`      |
| Joint Training from scratch       | `paper/agents/ablations/scratch_joint/`       |
| REINFORCE / Actor-Critic / DQN    | `paper/baselines/{reinforce,actor_critic,dqn}/` |
| CFR (Nash)                        | `agents/cfr/`                                 |
| Heuristic                         | `agents/heuristic/`                           |

Rule-based opponents (Section 4) live in `agents/rule_based/`.

---

## Experiment Lifecycle

1. **Write `card.md` first** (hypothesis, success criteria, single changed axis) before
   writing any code.
2. **Experiment code lives in** `preliminary_experiments/<exp_id>/` — unless the work is a
   direct extension of the paper (then `paper/`).
3. **Raw outputs go to** `<exp_id>/outputs/` — gitignored via `**/outputs/`.
4. **Use TournamentCheckpointer** from `agents/tournament_eval.py` for any architecture
   experiment (see Evaluation Protocol below).
5. **Write `report.md` + `summary.json`** after evaluation.
6. **Promote to `agents/`** only if the experiment produces a reusable framework piece
   or a stable canonical opponent. Otherwise keep the folder and report in
   `preliminary_experiments/`.
7. **Validate before reporting:**
   `python paper/evaluation/meta/validate_experiment.py <exp_id>/outputs/`

## Standard Training Configuration

These values MUST be used for architecture comparison experiments. Deviations require
explicit justification in `card.md`.

| Parameter      | Value |
|----------------|-------|
| Episodes       | 200,000 |
| Learning rate  | 1e-4  |
| Batch size     | 32    |
| Optimizer      | Adam  |

Source of truth for all parameters: `paper/evaluation/meta/EVAL_CONFIG.json`

---

## Evaluation Protocol (Non-Negotiable)

**All architecture experiments MUST use `TournamentCheckpointer`** from
`agents/tournament_eval.py`. Using it correctly is a requirement, not optional.

```python
from agents.tournament_eval import TournamentCheckpointer

checkpointer = TournamentCheckpointer(
    agent=agent,
    output_dir=output_dir,
    pass_through_callback=my_existing_callback,  # chains your history callback
)
trainer.train(num_episodes=200_000, callback=checkpointer.callback)
```

### Checkpoint selection
- **`checkpoint_best_robust.pt`** is the primary candidate for promotion decisions.
- Robustness = `avg - 1.5 × std` across per-opponent scores. Penalizes fragility.
- Report both alongside `checkpoint.pt` (final) in `report.md`.

### Committed artifacts
- Only **curated final-epoch checkpoints** live in `paper/checkpoints/` (committed).
- Per-epoch intermediate checkpoints in `outputs/checkpoints/` are gitignored.
- Round-robin **summary JSONs** under `paper/evaluation/results/**/vs_*.json` are committed; per-hand `.jsonl` replay logs are gitignored.

---

## Reference Documents (Read On Demand)

Do not read these at session start unless the task specifically requires them:

| Document                                         | When to read |
|--------------------------------------------------|--------------|
| `paper/README.md`                                | Reproducing figures or retraining paper agents |
| `paper/evaluation/meta/STANDARDS.md`             | Running an experiment; checking required artifacts |
| `paper/evaluation/meta/METRICS_GLOSSARY.md`      | Working with specific metrics (Spearman ρ, hand probe, etc.) |
| `paper/evaluation/meta/EVAL_CONFIG.json`         | Checking or changing any evaluation/training parameter |
| `AGENTS.md`                                      | Full agent genealogy, tournament history, and research insights |
| `preliminary_experiments/README.md`              | Locating prior exploratory work |

---

## Documentation Maintenance Rule

When you change any standard, protocol, or parameter:
1. **Update `CLAUDE.md` immediately** — this file is the authority.
2. **If a number changes, update `paper/evaluation/meta/EVAL_CONFIG.json`** — not just prose.
3. **Do not create new documentation files** except experiment-specific ones
   (`card.md`, `report.md`) — unless the user explicitly requests new documentation.
4. If you find a contradiction between this file and any other document, **this file wins**
   and the other document should be updated to match.
