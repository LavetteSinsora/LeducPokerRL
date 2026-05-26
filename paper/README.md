# `paper/` — Reproducing the Results

Everything in this directory exists to reproduce the numbers and figures in *Opponent-Modulated Value Networks for Exploiting Suboptimal Play in Leduc Hold'em*.

If you only want the figures, you can regenerate them from the **committed checkpoints + summary JSONs** without retraining anything. If you want to train from scratch, read [`Training`](#training) below.

---

## Layout

```
paper/
├── agents/
│   ├── value_based_pool/             # Base value network (Section 3.1)
│   ├── full_modulation/              # "Modulated Value Net (Ours)" — Section 3.2
│   └── ablations/                    # Table 2 ablations
│       ├── state_only/               # + State Mod. (no opp. stats)
│       ├── finetuned_base/           # + Opp. Stats Mod., unfrozen base
│       └── scratch_joint/            # Joint Training from scratch
├── baselines/
│   ├── reinforce/                    # Policy gradient
│   ├── actor_critic/
│   ├── dqn/                          # Double Q-Learning
│   └── opp_encoder_v1/               # Trained OOD opponent (related-work baseline)
├── evaluation/
│   ├── pool.py                       # Tournament opponent pool builder
│   ├── gauntlet.py                   # Per-agent gauntlet evaluation
│   ├── run_eval.py                   # Round-robin entry point
│   ├── comparison_protocol.py        # Standard opponent set
│   ├── results/                      # Per-matchup summary JSONs (committed)
│   ├── shared/                       # Stats tracker, training recipe, prototype data
│   └── meta/                         # EVAL_CONFIG.json, STANDARDS.md, METRICS_GLOSSARY.md
├── ev_analysis/
│   └── data.json                     # Monte-Carlo ground-truth EV (Figure 3)
├── figures/
│   ├── plot_performance_profile.py   # Figure 2
│   ├── plot_value_modulation.py      # Figure 3
│   ├── collect_modulation_deltas.py  # Figure 3 data collection
│   └── *.pdf, *.png                  # Published figures
└── checkpoints/                      # Curated final-epoch weights (committed)
    ├── value_based_pool/seed_{0,1,2}.pt
    ├── full_modulation/seed_{0,1,2}.pt
    ├── ablations/{state_only,finetuned_base,scratch_joint}/seed_{0,1,2}.pt
    └── baselines/{reinforce_v3,actor_critic_v3,dqn_v3}/seed_0.pt
```

---

## Regenerate Figures (no training)

```bash
pip install -r ../requirements.txt
python -m paper.figures.plot_performance_profile   # Figure 2
python -m paper.figures.plot_value_modulation      # Figure 3
```

Figure 2 reads summary JSONs from [`paper/evaluation/results/`](evaluation/results/).
Figure 3 reads `paper/ev_analysis/data.json` plus the `full_modulation` checkpoints in `paper/checkpoints/`.

---

## Training

Each agent has its own `train.py`. Standard config: 200,000 episodes, lr=1e-4, batch=32, Adam, 3 seeds. See [`evaluation/meta/EVAL_CONFIG.json`](evaluation/meta/EVAL_CONFIG.json).

```bash
# Base value network (Section 3.1 — required before training the modulated model)
python -m paper.agents.value_based_pool.train --seed 0
python -m paper.agents.value_based_pool.train --seed 1
python -m paper.agents.value_based_pool.train --seed 2

# Modulated value net (Section 3.2 — uses value_based_pool as frozen base)
python -m paper.agents.full_modulation.train --seed 0
python -m paper.agents.full_modulation.train --seed 1
python -m paper.agents.full_modulation.train --seed 2

# Ablations
python -m paper.agents.ablations.state_only.train       --seed {0,1,2}
python -m paper.agents.ablations.finetuned_base.train   --seed {0,1,2}
python -m paper.agents.ablations.scratch_joint.train    --seed {0,1,2}

# Baselines (one seed each — paper Figure 2)
python -m paper.baselines.train_v3_all                  # trains all three in parallel
```

Outputs go to `paper/agents/<name>/outputs/seed_<n>/` (gitignored). After training, copy
final-epoch checkpoints into `paper/checkpoints/<name>/seed_<n>.pt` for use by the figure
scripts.

---

## Evaluation

The round-robin tournament:

```bash
python -m paper.evaluation.run_eval                    # all agents × all opponents
python -m paper.evaluation.gauntlet --agent full_modulation --seed 0
```

Results are written to `paper/evaluation/results/<agent>_<seed>/vs_<opponent>.json` with the structure `{"chips_per_round": <float>, ...}`. The committed JSONs in this directory are the ones used to plot Figure 2.

---

## Paper ↔ Code Crosswalk

| Paper reference                          | Code path                                                 |
|------------------------------------------|-----------------------------------------------------------|
| Section 3.1, Eq. (1) – one-step lookahead | [`agents/base.py`](../agents/base.py), [`agents/value_based/agent.py`](../agents/value_based/agent.py) |
| Section 3.1, Eq. (2) – TD(0) loss          | [`paper/agents/value_based_pool/train.py`](agents/value_based_pool/train.py) |
| Section 3.2, Eq. (3) – modulation         | [`paper/agents/full_modulation/agent.py`](agents/full_modulation/agent.py) |
| Section 3.2 – opponent stats              | [`paper/evaluation/shared/stats_tracker.py`](evaluation/shared/stats_tracker.py) |
| Figure 1 (architecture)                   | [`writeup/Value network architecture.png`](../writeup/Value%20network%20architecture.png) |
| Figure 2 (performance profile)            | [`paper/figures/plot_performance_profile.py`](figures/plot_performance_profile.py) |
| Figure 3 (modulation per info-set)        | [`paper/figures/plot_value_modulation.py`](figures/plot_value_modulation.py) |
| Table 1 – overall                         | [`paper/evaluation/results/`](evaluation/results/) (load + aggregate) |
| Table 2 – ablations                       | [`paper/agents/ablations/`](agents/ablations/) + same eval pipeline |
| Appendix A.4 – opponent statistics        | [`paper/evaluation/shared/stats_tracker.py`](evaluation/shared/stats_tracker.py) |
| Appendix A.5 – agent pool                 | [`paper/evaluation/pool.py`](evaluation/pool.py) |
