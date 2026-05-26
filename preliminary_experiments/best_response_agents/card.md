# Experiment Card: opponent_value_tables_v1

## Research Question

Can we train a separate value network per rule-based opponent to obtain approximate
ground-truth EV estimates V(s | opponent)?

## Hypothesis

With enough training against a fixed, stationary opponent, a value network should
converge to a good approximation of the true expected value of each game state under
that opponent's policy. Since the opponent policy does not change, the TD target
distribution is stationary and convergence should be reliable.

## Setup

- **Agent**: ValueBasedAgent (15-feature input, 64-hidden MLP, TD(0))
- **Trainer**: FixedOpponentTrainer — learning agent vs fixed rule-based opponent
  (alternates P0/P1 each episode for position balance)
- **Training budget**: 200,000 episodes per opponent × 6 opponents
- **Optimizer**: Adam, lr=1e-4, batch_size=32, temperature=1.0
- **Opponents**: tight_passive, tight_aggressive, loose_passive, loose_aggressive, maniac, random

## Success Criteria

1. Each agent achieves positive avg chips/round vs its training opponent (exploits the fixed policy)
2. Loss converges (< 5% change over last 20% of training) for all 6 agents
3. Cross-eval matrix shows meaningful variation between agents — values are
   opponent-specific, not generic

## Intended Use

These per-opponent value networks serve as approximate ground truth for:
- Measuring regret of adaptive agents vs known opponents
- Validating modulation experiments (does opp_encoder_modulation produce values
  close to the per-opponent ground truth?)
- Feature importance analysis: which state features matter most against each
  opponent archetype?
