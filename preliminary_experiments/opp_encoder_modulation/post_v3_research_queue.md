# Post-v3 Research Queue

This file records what to do **after** the currently running v3 experiments finish.

Do not launch any new training jobs until these complete:

- `opp_encoder_modulation_v3_relative_cap`
- `opp_encoder_modulation_v3_aux_schedule`
- `opp_encoder_modulation_v3_state_gate`

The purpose of this file is to keep the next move deliberate.

## Selection rules

Any new direction should satisfy most of these:

1. It changes one main thing.
2. It keeps the strong frozen-base safety story unless there is a very good reason not to.
3. It uses opponent information in a more meaningful way than "just predict the next action."
4. It is simple enough that failure teaches something specific.
5. It has a plausible path to good performance without needing a huge architecture jump.

## Five candidate directions

### Direction 1: Confidence-weighted modulation cap

**Idea**

Keep the current opponent-encoder modulation family, but make the allowed correction size depend on explicit confidence in the opponent estimate.

Examples of confidence features:

- number of observed opponent decisions
- action diversity
- recency-weighted evidence mass
- disagreement/entropy of the action head

Then use confidence to scale the residual cap:

- low confidence -> tiny correction
- high confidence -> slightly larger correction

**Why it is attractive**

- very simple
- directly aligned with the "do no harm" story
- likely robust because it defaults toward the base when evidence is weak

**Self-critique**

- confidence may become a glorified hand-coded schedule
- if the confidence signal is poor, this may not add much beyond `v3_relative_cap`
- not very novel unless the confidence mechanism is genuinely informative

**What it would test**

Whether the missing ingredient is not just "small residual," but "small residual unless evidence is strong."

### Direction 2: Hand-conditioned opponent action model

**Idea**

Train an opponent action model of the form:

`P(opponent_action | public_state, macro_stats, candidate_opponent_hand)`

This is different from the current auxiliary head. The current head predicts the next action directly from our view of the opponent. This new model explicitly conditions on a candidate private hand for the opponent.

Once trained, use it to update a belief over opponent hands with Bayes' rule after each observed action.

**Why it is attractive**

- simple and conceptually clean
- directly tied to imperfect-information poker structure
- gives a real belief object instead of only a style embedding
- can use the same population data already being generated

**Self-critique**

- requires careful supervised targets because the opponent's private hand is only visible during simulation/training
- if the action model is weak, the Bayes update will be noisy or misleading
- self-play bias can still make the action model too uniform

**What it would test**

Whether opponent embeddings are too indirect, and whether hand-conditioned likelihood modeling is the more natural object.

### Direction 3: Belief-augmented one-step search

**Idea**

Use the belief from Direction 2 and run a conservative one-step lookahead:

1. enumerate legal actions
2. simulate each action
3. weight possible opponent hands by the current belief
4. evaluate continuation with the frozen value model or a belief-aware leaf evaluator

This is not full CFR or deep search. It is just a bounded one-step exploitative lookahead.

**Why it is attractive**

- strategically meaningful
- still simple enough to debug
- likely safer than replacing the whole value model
- uses beliefs for decision-time improvement instead of only as training supervision

**Self-critique**

- depends on Direction 2 being good enough
- can become slow if implemented carelessly
- belief quality may matter more than the search itself

**What it would test**

Whether opponent-aware information is more useful at decision time than as a direct value-network conditioning signal.

### Direction 4: Prototype opponent memory

**Idea**

Instead of one continuous opponent embedding, map each opponent to a mixture over a small number of archetypes:

- bluff-heavy
- passive caller
- equilibrium-like
- over-folder
- aggressive raiser

Then let the residual be a weighted combination of prototype corrections.

**Why it is attractive**

- interpretable
- may stabilize learning by avoiding an unconstrained continuous correction
- easier to diagnose than a generic latent embedding

**Self-critique**

- this is getting closer to MoE, which may be too early
- choosing the right archetypes is itself a modeling problem
- higher implementation complexity than the other directions

**What it would test**

Whether the value shift induced by opponent style is more discrete/regime-like than continuous.

### Direction 5: Action-value delta head

**Idea**

Instead of learning one scalar residual on the post-action value, learn a bounded correction per legal action:

- frozen base evaluates post-action states
- new head predicts small action-specific adjustments

This moves the learned correction closer to the actual decision boundary.

**Why it is attractive**

- simple and directly tied to action selection
- could make modulation more interpretable
- avoids forcing one scalar correction to explain everything

**Self-critique**

- may duplicate information already present in post-action base values
- still needs a strong safety constraint
- less clearly principled than a belief-based approach

**What it would test**

Whether the opponent-aware signal is more naturally an action-ranking correction than a state-value correction.

## Ranking after self-critique

### Most promising

1. **Direction 2: Hand-conditioned opponent action model**
2. **Direction 3: Belief-augmented one-step search**
3. **Direction 1: Confidence-weighted modulation cap**

### Why these three

**Direction 2** is the strongest new idea because it turns opponent modeling into something poker-native: likelihood over actions given candidate hands. That gives a real Bayesian object to work with.

**Direction 3** is strong because it uses opponent information at decision time rather than asking a value network to absorb everything into one latent correction.

**Direction 1** is the safest engineering bet. It may not be the most novel, but it has the highest chance of preserving performance if the current v3 family still looks close to competitive.

## Recommended decision rule after v3 finishes

### If a v3 variant clearly wins

Stay inside the modulation family for one more step.

Best follow-up:

- run **Direction 1** next
- keep the winning v3 mechanism fixed
- add explicit confidence-based control of the allowed correction size

### If all v3 variants are mixed or negative

Shift to the belief/search branch.

Best follow-up order:

1. **Direction 2**: hand-conditioned opponent action model
2. **Direction 3**: belief-augmented one-step search

This is the right pivot if the current evidence keeps saying:

- opponent signal is real
- but direct value modulation remains too hard to control

### If one v3 variant improves safety but not exploitative strength

Run a split program:

- one safe modulation follow-up from Direction 1
- one more novel branch from Direction 2

That gives one near-term performance line and one higher-upside research line.

## Suggested next concrete experiment names

- `opp_encoder_modulation_v4_confidence_cap`
- `hand_conditioned_action_model_v1`
- `belief_lookahead_v1`

## Notes for future design

- Do not jump to large MoE or FiLM stacks yet.
- Keep the frozen-base story unless there is clear evidence it is the wrong bottleneck.
- If using beliefs, measure both:
  - belief quality
  - decision quality after belief use
- If using search, keep it shallow and diagnose where it actually changes actions.
