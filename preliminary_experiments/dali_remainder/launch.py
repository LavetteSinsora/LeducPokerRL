"""
Launch all DALI_modulation training runs.

Usage:
  python -m DALI_modulation.launch              # all agents, all seeds, parallel
  python -m DALI_modulation.launch --sequential # one at a time
  python -m DALI_modulation.launch --agent full_modulation  # one agent only
  python -m DALI_modulation.launch --smoke      # quick test (500 eps each)
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")

# Mapping from agent name to dotted module path (for -m argument)
AGENT_MODULE_PATHS = {
    "full_modulation":  "full_modulation",
    "gated_modulation": "gated_modulation",
    "state_only":       "ablations.state_only",
    "finetuned_base":   "ablations.finetuned_base",
}


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def build_cmd(agent_name: str, seed: int, smoke: bool) -> list:
    dotted = AGENT_MODULE_PATHS[agent_name]
    cmd = [
        sys.executable, "-m",
        f"DALI_modulation.agents.{dotted}.train",
        "--seed", str(seed),
    ]
    if smoke:
        cmd.append("--smoke")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch DALI_modulation training runs")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run one at a time instead of in parallel",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        choices=list(AGENT_MODULE_PATHS.keys()),
        help="Run only this agent (default: all agents)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick test (500 episodes each)",
    )
    args = parser.parse_args()

    config  = load_config()
    seeds   = config["seeds"]
    sequential = args.sequential
    smoke   = args.smoke

    # Build list of (agent, seed) runs
    if args.agent:
        agents_to_run = [args.agent]
    else:
        agents_to_run = list(AGENT_MODULE_PATHS.keys())

    runs = []
    for agent_name in agents_to_run:
        agent_seeds = seeds.get(agent_name, [0])
        for seed in agent_seeds:
            runs.append((agent_name, seed))

    print(f"DALI_modulation launcher")
    print(f"  mode:      {'sequential' if sequential else 'parallel'}")
    print(f"  smoke:     {smoke}")
    print(f"  runs ({len(runs)}):")
    for agent_name, seed in runs:
        print(f"    {agent_name}  seed={seed}")
    print()

    if sequential:
        # Run one at a time
        for agent_name, seed in runs:
            cmd = build_cmd(agent_name, seed, smoke)
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print()
    else:
        # Launch all in parallel
        processes = []
        for agent_name, seed in runs:
            cmd = build_cmd(agent_name, seed, smoke)
            print(f"Launching: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd)
            processes.append((agent_name, seed, proc))

        print(f"\nAll {len(processes)} processes launched. Waiting for completion...\n")

        try:
            for agent_name, seed, proc in processes:
                proc.wait()
                rc = proc.returncode
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                print(f"  {agent_name} seed={seed}: {status}")
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt — terminating all child processes...")
            for agent_name, seed, proc in processes:
                proc.terminate()
                print(f"  Terminated {agent_name} seed={seed}")
            sys.exit(1)

    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
