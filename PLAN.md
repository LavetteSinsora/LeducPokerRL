# Repo Cleanup & Collaboration Plan

## Summary of Current State

The repo is a clean, working RL project (Leduc Hold'em) with three agent types, a training
framework, and a web dashboard. The core code is well-structured. The main gaps are:

- **No README.md** — the first file a teammate looks for doesn't exist
- **COOKBOOK.md is misleading** — it reads like a tutorial for building features that already exist,
  making it hard to distinguish "examples" from "current implementation"
- **experiments/ is unorganized** — research artifacts and scripts live at the top level with no
  context for what they are or whether they're still relevant
- **tests/debug/ contains loose scripts** — `reproduce_freeze.py`, `verify_bug.py` appear to be
  one-off debugging artifacts with no indication of whether the bugs are resolved

---

## Outdated / Misleading Docs

### COOKBOOK.md

| Section | Issue |
|---|---|
| Part 2: Implementing Your Agent | Shows `PolicyGradientAgent` as an example to build — but it **already exists** in `src/agents/policy_gradient.py`. A new teammate reading this will be confused about whether they are supposed to create this or if it's already done. |
| Part 3: Implementing Your Trainer | Same problem — `PolicyGradientTrainer` already exists in `src/training/policy_gradient_trainer.py`. |
| Part 4: Connecting to Dashboard | The `registry.py` example in the cookbook may not match the actual current registry (which only registers `heuristic` and `value_based` agents by default — `policy_gradient` exists as a file but is not in the default registry). |
| Quick Reference | Appears accurate and useful as-is. |

### experiments/training_target_report.md

This is an internal research analysis, not onboarding documentation. Its recommendations
(switch to TD(0), normalize rewards, etc.) appear **not yet implemented** in the current
training code. A teammate might interpret this as authoritative current behavior when it's
actually a list of open improvement items.

---

## Proposed Changes

### 1. Add README.md (highest priority)

Create a top-level `README.md` as the single entry point for any teammate. It should cover:

- **What this project is** (2-3 sentences)
- **Quick start** — how to install deps and run the server
- **Project structure** — a short annotated directory tree
- **How to train an agent** — minimal steps
- **How to add a new agent** — pointer to COOKBOOK
- **How to run tests**

This alone will dramatically reduce onboarding friction.

### 2. Revise COOKBOOK.md

Two options:

**Option A (Recommended): Reframe as a "How to Extend" guide**
Clearly label the PolicyGradient sections as "Here is an example of what has already been done —
follow this pattern to add your own agent." Add a note at the top: "The agents described here
already exist in the codebase. Use them as a reference template."

**Option B: Replace with a leaner CONTRIBUTING.md**
Keep the Quick Reference section, move the step-by-step agent tutorial to a `docs/` folder,
and use CONTRIBUTING.md for collaboration norms (branching, testing, etc.).

### 3. Reorganize experiments/

Rename or restructure to make the research-vs-documentation distinction clear:

```
experiments/
├── README.md              ← new: explains what this folder is
├── reports/
│   └── training_target_report.md   ← moved here
└── scripts/
    ├── training_target_analysis.py
    └── verify_td_agent.py
```

### 4. Clean up tests/debug/

The `debug/` subfolder contains what appear to be one-off bug investigation scripts.
Options:
- **Delete** `reproduce_freeze.py` and `verify_bug.py` if the bugs are resolved (confirm first)
- **Move to experiments/scripts/** if they have ongoing diagnostic value
- At minimum, add a `tests/debug/README.md` explaining what each script was for

### 5. (Optional) Add docs/ folder

For a team context, a lightweight `docs/` folder with an architecture diagram or overview doc
could help. Suggested minimal content:

```
docs/
└── architecture.md    ← describes BaseAgent → Trainer → Registry → Server flow
```

---

## Recommended Priority Order

1. **README.md** — write from scratch, immediate impact
2. **COOKBOOK.md** — add a clarifying header/framing, update registry example
3. **experiments/README.md** — one-paragraph note explaining the folder's purpose
4. **tests/debug/** — confirm bug status, then delete or annotate the loose scripts
5. **docs/architecture.md** — optional but useful for larger teams

---

## What Will NOT Be Changed

- Source code in `src/` — all agent, engine, training, and server code is unchanged
- `models/` — model weights untouched
- `requirements.txt`, `.gitignore` — these are fine as-is
- `web/` — frontend code is out of scope
