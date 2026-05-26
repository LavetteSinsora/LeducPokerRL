"""Auto-discovered registry for promoted agents."""

import importlib
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from agents.base import BaseAgent


@dataclass
class AgentMetadata:
    """Metadata describing a promoted agent package."""

    id: str
    display_name: str
    description: str
    agent_class: Type[BaseAgent]
    is_trainable: bool = False
    category: str = "general"
    trainer_class: Optional[Type] = None
    agent_dir: str = ""
    checkpoint_name: Optional[str] = "checkpoint.pt"
    history_name: str = "training_history.json"
    checkpoint_path: Optional[str] = None
    history_path: Optional[str] = None


class AgentRegistry:
    """Central registry backed by agent package discovery."""

    def __init__(self):
        self._agents: Dict[str, AgentMetadata] = {}

    def register(self, metadata: AgentMetadata):
        self._agents[metadata.id] = metadata

    def list_agents(self, category: Optional[str] = None) -> List[AgentMetadata]:
        agents = list(self._agents.values())
        if category:
            agents = [a for a in agents if a.category == category]
        return agents

    def get_metadata(self, agent_id: str) -> Optional[AgentMetadata]:
        return self._agents.get(agent_id)

    def create(self, agent_id: str, **kwargs) -> BaseAgent:
        meta = self._agents.get(agent_id)
        if meta is None:
            raise ValueError(
                f"Unknown agent: {agent_id}. Available: {sorted(self._agents.keys())}"
            )
        return meta.agent_class(**kwargs)

    def is_registered(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def get_trainable_agents(self) -> List[AgentMetadata]:
        return [a for a in self._agents.values() if a.is_trainable]

    def get_checkpoint_path(self, agent_id: str) -> Optional[str]:
        meta = self.get_metadata(agent_id)
        if meta is None:
            return None
        if meta.checkpoint_path:
            return meta.checkpoint_path
        if not meta.checkpoint_name:
            return None
        return os.path.join(meta.agent_dir, meta.checkpoint_name)

    def get_history_path(self, agent_id: str) -> Optional[str]:
        meta = self.get_metadata(agent_id)
        if meta is None:
            return None
        if meta.history_path:
            return meta.history_path
        if not meta.history_name:
            return None
        return os.path.join(meta.agent_dir, meta.history_name)

    def get_readme_path(self, agent_id: str) -> Optional[str]:
        meta = self.get_metadata(agent_id)
        if meta is None:
            return None
        path = os.path.join(meta.agent_dir, "README.md")
        return path if os.path.exists(path) else None


registry = AgentRegistry()


def _discover_agents():
    agents_dir = os.path.dirname(os.path.abspath(__file__))
    print("[registry] Scanning agents/...")

    for entry in sorted(os.listdir(agents_dir)):
        entry_path = os.path.join(agents_dir, entry)
        if not os.path.isdir(entry_path) or entry.startswith("_"):
            continue

        init_path = os.path.join(entry_path, "__init__.py")
        if not os.path.exists(init_path):
            continue

        module_name = f"agents.{entry}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            print(f"[registry] x {entry}: import failed - {exc}")
            continue

        meta_dict = getattr(module, "AGENT_META", None)
        if meta_dict is None:
            continue

        required = {"id", "display_name", "description", "agent_class"}
        missing = required - set(meta_dict.keys())
        if missing:
            print(f"[registry] x {entry}: missing keys {sorted(missing)}")
            continue

        if meta_dict.get("is_trainable", False) and not meta_dict.get("trainer_class"):
            print(f"[registry] x {entry}: trainable agent missing trainer_class")
            continue

        metadata = AgentMetadata(
            id=meta_dict["id"],
            display_name=meta_dict["display_name"],
            description=meta_dict["description"],
            agent_class=meta_dict["agent_class"],
            is_trainable=meta_dict.get("is_trainable", False),
            category=meta_dict.get("category", "general"),
            trainer_class=meta_dict.get("trainer_class"),
            agent_dir=entry_path,
            checkpoint_name=meta_dict.get("checkpoint_name", "checkpoint.pt"),
            history_name=meta_dict.get("history_name", "training_history.json"),
        )
        registry.register(metadata)

        status = metadata.display_name
        if metadata.is_trainable:
            status += f" - {metadata.category}, trainable"
        else:
            status += f" - {metadata.category}"
        print(f"[registry] ok {metadata.id} ({status})")

    print(f"[registry] Discovered {len(registry.list_agents())} agents")


def _register_experiment_agents():
    """Temporarily expose experiment agents for tournament/demo use."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    experiment_entries = [
        (
            "opp_encoder_modulation_v1",
            "Opponent Encoder Modulation v1",
            "Experiment agent: learned opponent encoder plus auxiliary action prediction",
            "experiments.opp_encoder_modulation_v1.agent",
            "OpponentEncoderModulationAgent",
            os.path.join(repo_root, "outputs", "opp_encoder_modulation_v1", "checkpoint.pt"),
        ),
        (
            "opp_encoder_modulation_v2",
            "Opponent Encoder Modulation v2",
            "Experiment agent: encoder modulation with gate and residual regularization",
            "experiments.opp_encoder_modulation_v2.agent",
            "OpponentEncoderModulationAgent",
            os.path.join(repo_root, "outputs", "opp_encoder_modulation_v2", "checkpoint.pt"),
        ),
    ]

    for agent_id, display_name, description, module_name, class_name, checkpoint_path in experiment_entries:
        if registry.is_registered(agent_id):
            continue
        try:
            module = importlib.import_module(module_name)
            agent_class = getattr(module, class_name)
        except Exception as exc:
            print(f"[registry] x {agent_id}: experiment import failed - {exc}")
            continue

        registry.register(
            AgentMetadata(
                id=agent_id,
                display_name=display_name,
                description=description,
                agent_class=agent_class,
                is_trainable=False,
                category="experiment",
                agent_dir=os.path.dirname(os.path.abspath(module.__file__)),
                checkpoint_name=None,
                history_name="",
                checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
            )
        )
        print(f"[registry] ok {agent_id} ({display_name} - experiment)")


def _legacy_checkpoint_path(models_dir: str, agent_id: str) -> Optional[str]:
    direct = os.path.join(models_dir, f"{agent_id}_agent.pt")
    if os.path.exists(direct):
        return direct

    special = {
        "distributional_value": "distributional_agent.pt",
    }
    filename = special.get(agent_id)
    if filename:
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            return path
    return None


def _legacy_history_path(models_dir: str, agent_id: str) -> Optional[str]:
    direct = os.path.join(models_dir, f"{agent_id}_agent_history.json")
    return direct if os.path.exists(direct) else None


def _register_legacy_agents():
    """Temporarily merge the archived agent suite for round-robin demos."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    legacy_root = os.path.join(repo_root, "experiments", "archive", "legacy_source")
    models_dir = os.path.join(repo_root, "outputs", "legacy", "models")
    if not os.path.isdir(legacy_root):
        return

    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)

    try:
        from src.agents.registry import registry as legacy_registry
    except Exception as exc:
        print(f"[registry] x legacy registry import failed - {exc}")
        return

    registered = 0
    for legacy_meta in legacy_registry.list_agents():
        if registry.is_registered(legacy_meta.id):
            continue

        agent_class = legacy_registry._agents.get(legacy_meta.id)
        if agent_class is None:
            continue

        checkpoint_path = _legacy_checkpoint_path(models_dir, legacy_meta.id)
        history_path = _legacy_history_path(models_dir, legacy_meta.id)
        registry.register(
            AgentMetadata(
                id=legacy_meta.id,
                display_name=legacy_meta.display_name,
                description=legacy_meta.description,
                agent_class=agent_class,
                is_trainable=legacy_meta.is_trainable,
                category=f"legacy_{legacy_meta.category}",
                trainer_class=legacy_meta.trainer_class,
                agent_dir=os.path.join(legacy_root, "src", "agents"),
                checkpoint_name=None,
                history_name="",
                checkpoint_path=checkpoint_path,
                history_path=history_path,
            )
        )
        registered += 1
        state = "with checkpoint" if checkpoint_path else "without checkpoint"
        print(f"[registry] ok {legacy_meta.id} ({legacy_meta.display_name} - {state})")

    print(f"[registry] Added {registered} legacy agents")


_discover_agents()
_register_experiment_agents()
_register_legacy_agents()
