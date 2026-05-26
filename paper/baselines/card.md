# Pool Agents for DALI Modulation Ablation Study

## Hypothesis
Three training-paradigm baselines using the same 15→64→64 backbone as value_based,
differing only in learning algorithm, provide diverse behavioral coverage for the
DALI modulation pool. Diversity in learning paradigm should produce qualitatively
different strategies that stress-test the modulation mechanism.

## Agents
| ID | Algorithm | Description |
|----|-----------|-------------|
| `reinforce` | REINFORCE (Monte Carlo PG) | Full-episode returns, stochastic policy |
| `actor_critic` | A2C | TD(0) advantage, shared trunk, online updates |
| `dqn` | DQN | Off-policy Q-learning, target network, replay buffer |

## Success Criteria
- All three smoke tests pass (500 episodes)
- All three full training runs complete (200K episodes)
- Each agent achieves positive reward against random opponent

## Changed Axis
Learning algorithm only. Architecture (15→64→64), optimizer (Adam, lr=1e-4),
and training budget (200K episodes) are held constant across all three.

## Notes
- Self-play training: agent plays both seats, learner_id = ep % 2
- No periodic pool evaluation during training
- One seed each (seed not fixed)
