# PokerRL

Leduc Hold'em RL research project. Iteratively develops poker agents through
rounds of experiments, each changing exactly one aspect from a parent agent.

## Structure
- `src/agents/` — agent implementations (inherit BaseAgent, register in registry.py)
- `src/engine/` — game engine, observation, poker session
- `experiments/` — experiment scripts + result JSON files
- `experiment_reports/` — markdown reports summarizing each round's findings
- `web/` — wiki dashboard with agent pages and evolution tree
- `models/` — saved model weights (.pt files)
- `AGENTS.md` — canonical agent family tree with all results

## Conventions
- New agents inherit from an existing parent, changing exactly one aspect
- Agents register in `src/agents/registry.py` with a string ID
- Evaluation: round-robin tournament, robustness = avg - 1.5×std
- Experiment reports go to both `experiment_reports/` and Obsidian vault
- Wiki pages: one per agent at `web/wiki/{agent_id}.md`

## Experiment Workflow
Direction → Implement → Train → Evaluate → Diagnose → Report
