"""
Strategy storage for CFR.

InfoSetData stores per-infoset regret and strategy accumulators.
TabularStrategyStore is the dict-based implementation for vanilla CFR.
StrategyStore is the abstract interface — the extensibility point for Deep CFR.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

NUM_ACTIONS = 3  # FOLD=0, CALL=1, RAISE=2


@dataclass
class InfoSetData:
    """Regret and strategy accumulators for one information set."""

    regret_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
    )
    strategy_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
    )

    def get_current_strategy(self) -> np.ndarray:
        """Regret matching: normalize positive regrets into a probability distribution."""
        positive = np.maximum(self.regret_sum, 0.0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS

    def get_average_strategy(self) -> np.ndarray:
        """Average strategy across all iterations — converges to Nash equilibrium."""
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS


class StrategyStore(ABC):
    """Abstract strategy lookup — tabular now, neural later for Deep CFR."""

    @abstractmethod
    def get_info_set(self, key: str) -> InfoSetData:
        """Get or create InfoSetData for the given key."""
        ...

    @abstractmethod
    def get_strategy(self, key: str, legal_actions: list) -> np.ndarray:
        """Current iteration's strategy (masked to legal actions)."""
        ...

    @abstractmethod
    def get_average_strategy(self, key: str, legal_actions: list) -> np.ndarray:
        """Converged Nash strategy (masked to legal actions)."""
        ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    @abstractmethod
    def num_info_sets(self) -> int: ...


def _mask_and_normalize(strategy: np.ndarray, legal_actions: list) -> np.ndarray:
    """Zero out illegal actions and renormalize."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for a in legal_actions:
        idx = a.value if hasattr(a, "value") else int(a)
        mask[idx] = 1.0
    masked = strategy * mask
    total = masked.sum()
    if total > 0:
        return masked / total
    # Fallback: uniform over legal actions
    return mask / mask.sum()


class TabularStrategyStore(StrategyStore):
    """Dict-based strategy store. ~300 infosets for Leduc Hold'em."""

    def __init__(self):
        self.data: Dict[str, InfoSetData] = {}

    def get_info_set(self, key: str) -> InfoSetData:
        if key not in self.data:
            self.data[key] = InfoSetData()
        return self.data[key]

    def get_strategy(self, key: str, legal_actions: list) -> np.ndarray:
        return _mask_and_normalize(
            self.get_info_set(key).get_current_strategy(), legal_actions
        )

    def get_average_strategy(self, key: str, legal_actions: list) -> np.ndarray:
        return _mask_and_normalize(
            self.get_info_set(key).get_average_strategy(), legal_actions
        )

    def num_info_sets(self) -> int:
        return len(self.data)

    def save(self, path: str) -> None:
        serialized = {
            key: {
                "regret_sum": info.regret_sum.tolist(),
                "strategy_sum": info.strategy_sum.tolist(),
            }
            for key, info in self.data.items()
        }
        with open(path, "w") as f:
            json.dump(serialized, f)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            serialized = json.load(f)
        self.data = {}
        for key, vals in serialized.items():
            info = InfoSetData()
            info.regret_sum = np.array(vals["regret_sum"], dtype=np.float64)
            info.strategy_sum = np.array(vals["strategy_sum"], dtype=np.float64)
            self.data[key] = info
