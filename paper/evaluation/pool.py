"""
paper — Pool Agent Loader
=====================================
Loads all pool agents used for the gauntlet evaluation.

Pool composition (16 agents):
  In-distribution (8 — in training pool of study agents):
    cfr, heuristic, tight_passive, tight_aggressive,
    loose_passive, loose_aggressive, maniac, random

  Out-of-distribution v1 (5 — NOT in training pool):
    adaptive_value, opp_encoder_v1,
    reinforce (self-play), actor_critic (self-play), dqn (self-play)

  Out-of-distribution v2 (3 — pool-trained, stronger OOD opponents):
    reinforce_v2, actor_critic_v2, dqn_v2

  Stats-injected (need obs.opponent_stats):
    adaptive_value, opp_encoder_v1

Usage:
  from paper.evaluation.pool import build_pool, STATS_INJECTED
  pool = build_pool(root)
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from agents.cfr.agent import CFRAgent
from agents.heuristic.agent import HeuristicAgent
from agents.adaptive_value.agent import AdaptiveValueAgent
from agents.rule_based.tight_passive import TightPassiveAgent
from agents.rule_based.tight_aggressive import TightAggressiveAgent
from agents.rule_based.loose_passive import LoosePassiveAgent
from agents.rule_based.loose_aggressive import LooseAggressiveAgent
from agents.rule_based.maniac import ManiacAgent
from agents.rule_based.random_agent import RandomAgent
from paper.baselines.reinforce.agent import REINFORCEAgent
from paper.baselines.actor_critic.agent import ActorCriticAgent
from paper.baselines.dqn.agent import DQNAgent
import importlib.util as _ilu
_opp_enc_spec = _ilu.spec_from_file_location(
    "opp_encoder_modulation_v1_agent",
    os.path.join(
        ROOT,
        "experiments", "opp_encoder_modulation",
        "opp_encoder_modulation_v1", "agent.py",
    ),
)
_opp_enc_mod = _ilu.module_from_spec(_opp_enc_spec)
_opp_enc_spec.loader.exec_module(_opp_enc_mod)
OpponentEncoderModulationAgent = _opp_enc_mod.OpponentEncoderModulationAgent

# Pool agents that need obs.opponent_stats injected (4-dim feature vector).
# All others use plain select_action(obs).
STATS_INJECTED = frozenset({"adaptive_value", "opp_encoder_v1"})

POOL_AGENT_KEYS = [
    # In-distribution (in training pool of study agents)
    "cfr",
    "heuristic",
    "tight_passive",
    "tight_aggressive",
    "loose_passive",
    "loose_aggressive",
    "maniac",
    "random",
    # OOD v1 — self-play trained (weak)
    "adaptive_value",
    "opp_encoder_v1",
    "reinforce",
    "actor_critic",
    "dqn",
    # OOD v2 — pool-trained (stronger, better held-out test)
    "reinforce_v2",
    "actor_critic_v2",
    "dqn_v2",
    # OOD v3 — pool-trained at 200K episodes (matched training budget)
    "reinforce_v3",
    "actor_critic_v3",
    "dqn_v3",
]

# Subset used for in-distribution evaluation only
IN_DIST_KEYS = ["cfr","heuristic","tight_passive","tight_aggressive",
                "loose_passive","loose_aggressive","maniac","random"]

# Subset used for OOD evaluation
OOD_KEYS = ["adaptive_value","opp_encoder_v1",
            "reinforce","actor_critic","dqn",
            "reinforce_v2","actor_critic_v2","dqn_v2"]


def build_pool(root: str = ROOT) -> dict:
    """
    Load and return all 13 pool agents in eval mode.

    Parameters
    ----------
    root : str
        Repo root directory.

    Returns
    -------
    dict[str, agent]
        Keys match POOL_AGENT_KEYS.
    """
    pool = {}

    # ── CFR ──────────────────────────────────────────────────────────────────
    cfr_path = os.path.join(root, "agents", "cfr", "checkpoint.pt")
    pool["cfr"] = CFRAgent(model_path=cfr_path)

    # ── Heuristic ─────────────────────────────────────────────────────────────
    pool["heuristic"] = HeuristicAgent()

    # ── Rule-based ────────────────────────────────────────────────────────────
    pool["tight_passive"]    = TightPassiveAgent()
    pool["tight_aggressive"] = TightAggressiveAgent()
    pool["loose_passive"]    = LoosePassiveAgent()
    pool["loose_aggressive"] = LooseAggressiveAgent()
    pool["maniac"]           = ManiacAgent()
    pool["random"]           = RandomAgent()

    # ── Adaptive value (stats-injected) ───────────────────────────────────────
    av_path = os.path.join(root, "agents", "adaptive_value", "checkpoint.pt")
    pool["adaptive_value"] = AdaptiveValueAgent(model_path=av_path)

    # ── Opponent encoder v1 (stats-injected) ──────────────────────────────────
    opp_path = os.path.join(
        root,
        "experiments", "opp_encoder_modulation",
        "opp_encoder_modulation_v1", "outputs", "checkpoint.pt",
    )
    pool["opp_encoder_v1"] = OpponentEncoderModulationAgent(model_path=opp_path)

    # ── REINFORCE ─────────────────────────────────────────────────────────────
    rein_path = os.path.join(
        root, "paper", "baselines", "reinforce",
        "outputs", "checkpoint_final.pt",
    )
    pool["reinforce"] = REINFORCEAgent(model_path=rein_path)

    # ── Actor-Critic ──────────────────────────────────────────────────────────
    ac_path = os.path.join(
        root, "paper", "baselines", "actor_critic",
        "outputs", "checkpoint_final.pt",
    )
    pool["actor_critic"] = ActorCriticAgent(model_path=ac_path)

    # ── DQN ───────────────────────────────────────────────────────────────────
    dqn_path = os.path.join(
        root, "paper", "baselines", "dqn",
        "outputs", "checkpoint_final.pt",
    )
    pool["dqn"] = DQNAgent(model_path=dqn_path)

    # ── OOD v2: pool-trained, stronger held-out opponents ─────────────────────
    # Uses seed_0 checkpoint. Seeds 1/2 train in parallel; update path when done.
    # Strongest seed selected by eval vs value_based (2K hands):
    #   reinforce_v2 seed_0: -0.18  actor_critic_v2 seed_1: -0.22  dqn_v2 seed_2: -0.07
    rein_v2_path = os.path.join(
        root, "paper", "baselines", "reinforce",
        "outputs_v2", "seed_0", "checkpoint_final.pt",
    )
    if os.path.isfile(rein_v2_path):
        pool["reinforce_v2"] = REINFORCEAgent(model_path=rein_v2_path)

    ac_v2_path = os.path.join(
        root, "paper", "baselines", "actor_critic",
        "outputs_v2", "seed_1", "checkpoint_final.pt",
    )
    if os.path.isfile(ac_v2_path):
        pool["actor_critic_v2"] = ActorCriticAgent(model_path=ac_v2_path)

    dqn_v2_path = os.path.join(
        root, "paper", "baselines", "dqn",
        "outputs_v2", "seed_2", "checkpoint_final.pt",
    )
    if os.path.isfile(dqn_v2_path):
        pool["dqn_v2"] = DQNAgent(model_path=dqn_v2_path)

    # ── OOD v3: pool-trained at 200K episodes — matched training budget ────────
    rein_v3_path = os.path.join(
        root, "paper", "baselines", "reinforce",
        "outputs_v3", "seed_0", "checkpoint_final.pt",
    )
    if os.path.isfile(rein_v3_path):
        pool["reinforce_v3"] = REINFORCEAgent(model_path=rein_v3_path)

    ac_v3_path = os.path.join(
        root, "paper", "baselines", "actor_critic",
        "outputs_v3", "seed_0", "checkpoint_final.pt",
    )
    if os.path.isfile(ac_v3_path):
        pool["actor_critic_v3"] = ActorCriticAgent(model_path=ac_v3_path)

    dqn_v3_path = os.path.join(
        root, "paper", "baselines", "dqn",
        "outputs_v3", "seed_0", "checkpoint_final.pt",
    )
    if os.path.isfile(dqn_v3_path):
        pool["dqn_v3"] = DQNAgent(model_path=dqn_v3_path)

    # ── Set all to eval mode ──────────────────────────────────────────────────
    for agent in pool.values():
        agent.set_train_mode(False)

    return pool
