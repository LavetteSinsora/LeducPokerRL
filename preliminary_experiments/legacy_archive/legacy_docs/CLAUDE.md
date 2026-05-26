# PokerRL Research Principles

These principles encode the experience and knowledge built up in this project.
Apply them as a lens, not a checklist.

- **Be curious.** The goal is understanding the model — why it behaves as it does,
  why it succeeds or fails. Performance numbers are just evidence toward that
  understanding. Follow unexpected findings and investigate the depth of what the model
  has learned, even when that wasn't the plan.

- **Train to convergence.** Dismissing a change because of insufficient training or
  evaluation is a costly mistake. When in doubt, train longer before drawing conclusions.

- **Think about what the model is actually learning.** Before and after any change,
  ask: given this data and structure, what is the model actually learning — and does
  that match the intended behavior? Let intuition guide the research; use sanity checks
  to test it.

- **Build understanding from the ground up.** General RL wisdom doesn't always transfer
  to Leduc Hold'em's specific setting. Be rigorous. Test ideas in context and form your
  own understanding of why techniques work or don't — borrowed intuitions can mislead.

- **Elegance is a signal.** The clearest ideas tend to be the best ones. If a change
  is hard to explain clearly, that's worth noticing.
