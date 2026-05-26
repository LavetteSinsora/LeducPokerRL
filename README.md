# PokerRL — Opponent-Modulated Value Networks for Leduc Hold'em

Code accompanying the paper *Opponent-Modulated Value Networks for Exploiting Suboptimal Play in Leduc Hold'em* (He, Pant, Kharat, Dang).

We train a value network with a **value-modulation head** that conditions on online-collected opponent statistics. Across a round-robin tournament our method earns **+21.1 chips/session more than CFR Nash** and is the only baseline that maintains positive net gain against out-of-distribution opponents.

---

## Repository Map

| Directory | Purpose |
|---|---|
| [`paper/`](paper/) | **Everything needed to reproduce the paper.** Models, baselines, evaluation harness, figure-generation scripts, curated checkpoints. Start here. |
| [`engine/`](engine/) | Leduc Hold'em game engine — shared by all code in the repo. |
| [`agents/`](agents/) | Agent framework + canonical opponents: `BaseAgent`, value-network architecture, CFR Nash baseline, heuristic, rule-based opponents (tight/loose × passive/aggressive, maniac, random). |
| [`preliminary_experiments/`](preliminary_experiments/) | Earlier research threads — AlphaZero-style search, opponent encoders, representation learning, hand-conditioned belief, capacity studies, the legacy dashboard, etc. **Not required to reproduce the paper.** |
| [`writeup/`](writeup/) | LaTeX source for the paper, figure images, references. |
| [`tests/`](tests/) | Engine unit tests. |

---

## Quick Start: Reproduce the Paper

```bash
pip install -r requirements.txt

# Regenerate Figure 2 (performance profile) and Figure 3 (value modulation)
# using the committed checkpoints + summary JSONs. No retraining needed.
python -m paper.figures.plot_performance_profile
python -m paper.figures.plot_value_modulation

# Re-train a model from scratch (~hours on CPU):
python -m paper.agents.full_modulation.train --seed 0

# Re-run the round-robin tournament (overwrites paper/evaluation/results/):
python -m paper.evaluation.run_eval
```

See [`paper/README.md`](paper/README.md) for the full reproduction guide.

---

## Paper Agents

| Paper label                          | Code                                                |
|--------------------------------------|-----------------------------------------------------|
| Modulated Value Net. (**Ours**)      | [`paper/agents/full_modulation/`](paper/agents/full_modulation/) |
| Value Network (base, self-play)      | [`paper/agents/value_based_pool/`](paper/agents/value_based_pool/) |
| + State Mod. (no opp. stats)         | [`paper/agents/ablations/state_only/`](paper/agents/ablations/state_only/) |
| + Opp. Stats Mod., unfrozen base     | [`paper/agents/ablations/finetuned_base/`](paper/agents/ablations/finetuned_base/) |
| Joint Training from scratch          | [`paper/agents/ablations/scratch_joint/`](paper/agents/ablations/scratch_joint/) |
| REINFORCE                            | [`paper/baselines/reinforce/`](paper/baselines/reinforce/) |
| Actor-Critic                         | [`paper/baselines/actor_critic/`](paper/baselines/actor_critic/) |
| DQN                                  | [`paper/baselines/dqn/`](paper/baselines/dqn/) |
| CFR (Nash)                           | [`agents/cfr/`](agents/cfr/) |
| Heuristic                            | [`agents/heuristic/`](agents/heuristic/) |

## Opponent Pool

In-distribution opponents (rule-based, used during training):

- [`agents/rule_based/tight_passive.py`](agents/rule_based/tight_passive.py)
- [`agents/rule_based/tight_aggressive.py`](agents/rule_based/tight_aggressive.py)
- [`agents/rule_based/loose_passive.py`](agents/rule_based/loose_passive.py)
- [`agents/rule_based/loose_aggressive.py`](agents/rule_based/loose_aggressive.py)
- [`agents/rule_based/maniac.py`](agents/rule_based/maniac.py)
- [`agents/rule_based/random_agent.py`](agents/rule_based/random_agent.py)

Out-of-distribution opponents (held out for the OOD tournament): the trained PG baselines above plus an opponent-encoder model in [`paper/baselines/opp_encoder_v1/`](paper/baselines/opp_encoder_v1/).

---

## Looking at Preliminary Work

The paper presents the final, working method. Many things were tried first and didn't make it in — AlphaZero-style search, contrastive representations, hand-conditioned belief models, capacity sweeps, opponent encoders. If you're reading the paper and wondering "did they try X?", check [`preliminary_experiments/`](preliminary_experiments/) — each subdirectory has a `card.md` or `report.md` documenting the hypothesis, design, and outcome.

---

## Citation

```bibtex
@misc{he2026opponentmodulated,
  title  = {Opponent-Modulated Value Networks for Exploiting Suboptimal Play in Leduc Hold'em},
  author = {He, Chris and Pant, Devang and Kharat, Rutvij and Dang, Mark},
  year   = {2026}
}
```
