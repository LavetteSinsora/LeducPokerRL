# hand_posterior_belief_v1

**Question**: Can a GRU-based belief network learn P(opp_rank | action_sequence), and can a value agent conditioned on that posterior outperform the base value agent?

**Parent agent**: `hand_conditioned_action_model_v1` (for the belief model) + `value_based` (for the policy)

**Single change**: Replace the static hand-conditioned likelihood model with a GRU that tracks the posterior over opponent hand rank across the full action sequence. The value network receives (15-dim state + 3-dim posterior) = 18-dim input.

**Architecture**:
- Phase 1: `BeliefNetwork` — GRU (hidden=32) conditioned on context (hand+board, 7-dim) and action tokens (4-dim), outputs P(opp_rank ∈ {J,Q,K})
- Phase 2: `BeliefPosteriorValueAgent` — freeze `BeliefNetwork`, train value net on 18-dim input (15 + 3 posterior)

**Success criteria**:
- Belief posterior top-1 accuracy > 50% (above card-removal prior ~33%)
- Value agent with frozen belief net beats base value_based agent on 5-agent eval suite

**Status**: In progress — agent.py complete, training not yet run.
