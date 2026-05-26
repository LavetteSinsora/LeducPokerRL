"""
Train REINFORCE, Actor-Critic, DQN at 200K episodes with pool training recipe.
Outputs go to outputs_v3/seed_0 for each agent.
All three run in parallel via subprocess.

Usage:
    python -m paper.baselines.train_v3_all
"""

import os
import sys
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

NUM_EPISODES = 200_000
SEED         = 0

AGENTS = ["reinforce", "actor_critic", "dqn"]

RUNNER = """
import sys, os
sys.path.insert(0, {root!r})
from paper.baselines.{agent}.train_v2 import train
out = os.path.join({here!r}, {agent!r}, "outputs_v3", f"seed_{seed}")
train({n_ep}, out, {seed})
"""


if __name__ == "__main__":
    procs = []
    for agent in AGENTS:
        code = RUNNER.format(
            root=ROOT, here=HERE, agent=agent,
            n_ep=NUM_EPISODES, seed=SEED,
        )
        p = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        procs.append((agent, p))
        print(f"[v3] launched {agent} (pid={p.pid})", flush=True)

    # Stream output and wait for all
    t0 = time.time()
    for agent, p in procs:
        out, _ = p.communicate()
        elapsed = time.time() - t0
        status  = "OK" if p.returncode == 0 else f"ERROR ({p.returncode})"
        print(f"\n[v3/{agent}] {status} ({elapsed:.0f}s)")
        # Print last 5 lines of output
        for line in out.strip().splitlines()[-5:]:
            print(f"  {line}")

    print(f"\nAll v3 training complete ({time.time()-t0:.0f}s total).")
