# When Opponent Awareness Helps, and When It Breaks

This document is a blog-style narrative of the current opponent-aware value-learning line in this repo.

It is not the machine-readable ledger. The ledger lives in each experiment folder.

This file is the human story:

- what we tried
- why we tried it
- what surprised us
- what we now believe is actually hard about the problem

## The starting point

The strongest simple recipe in this repo was never the fanciest one.

The basic `value_based` agent worked because it was stable:

- small observation space
- TD(0) value learning
- simple decision rule

Then `adaptive_value` showed the first real opponent-aware gain. A few macro opponent statistics were enough to improve decisions. That was the first strong sign that opponent-specific information mattered.

The real breakthrough came with `modulated_value`.

Its idea was modest:

- keep a strong pretrained base value network
- freeze it
- learn only a small gated correction on top

That turned out to be much more important than it looked. The agent did not win by learning an elaborate exploit model. It won by preserving a strong base and refusing to move too far away from it.

That became the central design lesson:

**in this domain, "do no harm" is an architectural advantage.**

## Why the encoder direction was attractive

Once the frozen-base modulated architecture worked, the next obvious question was:

what if the opponent signal itself were learned, instead of hand-designed?

The macro stats in `adaptive_value` and `modulated_value` were useful, but crude. They were summaries chosen by us. A learned opponent encoder offered a more ambitious possibility:

- compress opponent behavior into a learned embedding
- train that embedding with an auxiliary next-action prediction task
- use the embedding to modulate value predictions

This became `opp_encoder_modulation_v1`.

Conceptually, it was appealing because it separated the problem into two parts:

1. learn a representation of how the opponent behaves
2. use that representation to correct a frozen equilibrium-like value estimate

This is exactly the kind of structure that feels right in imperfect-information games.

## What v1 taught us

`opp_encoder_modulation_v1` was a mixed success, but an important one.

It proved three things.

First, the encoder was not fake. The action-prediction head did learn something real, especially against learned value-style opponents.

Second, the modulation path mattered. A base-only ablation was clearly worse, which meant the encoder-conditioned correction was doing real work.

Third, and most importantly, the model broke the one property that had made the parent strong:

the gate saturated near `1.0`.

That meant the correction was no longer a small correction. It became the dominant signal.

So v1 gave us a very sharp lesson:

**representation learning can help, but it can also destroy the protective geometry of a good architecture.**

This was a deeper lesson than "v1 underperformed."

The real point was that the failure was mechanistic, not mysterious.

The model did not fail because the encoder was useless.
It failed because the architecture stopped being conservative.

## What v2 taught us

`opp_encoder_modulation_v2` was designed as a repair experiment.

It changed one thing:

- add explicit regularization to push the gate back toward the parent regime
- penalize large modulation

At first glance, it looked promising.

The gate no longer saturated near `1.0`.
It moved down to roughly `0.55`.

But after full training and matched-budget evaluation, the result was still not good enough.

Why?

Because the residual network simply adapted to the new constraint.

The gate got smaller, but the raw residual got larger.
So the effective modulation stayed large anyway.

That was the crucial v2 insight:

**regularizing the gate is not the same as controlling the residual path.**

This is the kind of result that matters a lot in research, even though it does not look like a "win" on the leaderboard.

v2 told us that we were fixing a symptom, not the mechanism.

It also exposed another tradeoff:

- action prediction got better
- value learning got worse

So the auxiliary opponent-modeling task was helping the encoder become predictive, but not necessarily helping the value objective.

That leads to a more precise understanding of the bottleneck.

## What we now believe is hard

At this point, the problem no longer looks like "how do we add more opponent modeling?"

It looks like a tighter, more structural problem:

### 1. Opponent awareness is useful

That part is real. We have enough evidence for it.

The repo already shows that opponent statistics help, and the encoder experiments show that learned opponent signal is not empty.

### 2. The useful part of opponent awareness is small

This is the bigger lesson.

The best architecture so far is not one that aggressively rewrites the base value estimate.
It is one that makes small, carefully limited corrections.

So the challenge is not just "learn a better opponent embedding."
The challenge is:

**how do we let opponent information matter without letting it dominate?**

### 3. Self-play is still a poor teacher for opponent-aware components

This has shown up repeatedly across the repo:

- belief models
- opponent models
- adversarial information-hiding ideas
- encoder-based opponent prediction

Self-play tends to teach "everyone looks like me."

Population training helps, but it does not magically solve the problem. It mostly prevents total collapse into one style. The representation still needs enough structure and the value path still needs protection.

### 4. Auxiliary tasks are dangerous when they become optimization shortcuts

Predicting the opponent's next action is intuitively aligned with opponent modeling.

But in practice, the auxiliary loss can become a shortcut objective:

- it is easier to optimize than value quality
- it can dominate gradient budget
- it may encourage the encoder to represent what is predictive, not what is useful for value correction

That means "better opponent prediction" is not automatically the same thing as "better exploitative value estimation."

## The current follow-up queue

Three new single-axis experiments are now running or queued from this diagnosis.

### 1. `opp_encoder_modulation_v3_relative_cap`

This directly addresses the main v2 lesson.

Instead of only regularizing the gate, it hard-caps the **effective residual** relative to base value size.

This is the cleanest test of the current main hypothesis:

if we directly constrain the correction path, can the encoder help without destabilizing the architecture?

### 2. `opp_encoder_modulation_v3_aux_schedule`

This isolates the loss-budget question.

The architecture stays the same as v2, but the auxiliary action-prediction loss is delayed and ramped in gradually.

This tests whether the encoder can still learn useful opponent structure without overwhelming value learning early.

### 3. `opp_encoder_modulation_v3_state_gate`

This addresses the gate-flatness problem.

In v2, the gate barely varied. That suggests the gate may not be seeing the right information.

This variant lets the gate depend on both:

- the state
- the opponent embedding

The hope is that modulation should depend not only on *who* the opponent is, but also on *whether this particular situation is one where opponent-specific deviation matters*.

## What we should not do yet

The temptation now is to jump to bigger models:

- MoE
- FiLM everywhere
- more auxiliary objectives
- richer multitask representation learning

That would be premature.

The current results do not say "the model is too weak."
They say:

- the control mechanism is still not right
- the optimization coupling is still not right

So the next good experiments are not about making the model bigger.
They are about making the correction path safer and more selective.

## The best current understanding

The current best mental model for this project is:

1. Learn a strong generic value prior.
2. Learn opponent-aware signal on top of it.
3. Treat opponent-aware modulation as a **small, conditional correction**, not a replacement value function.

That sounds conservative, but that is exactly what the evidence in this repo points toward.

The biggest mistake would be to interpret the v1 and v2 results as "opponent encoders do not work."

That is not what happened.

What happened is more interesting:

**the opponent encoder learned something real, but we still do not know how to let that signal influence value estimates safely.**

That is now the core research question.
