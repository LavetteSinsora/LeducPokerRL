from .tight_passive import TightPassiveAgent
from .tight_aggressive import TightAggressiveAgent
from .loose_passive import LoosePassiveAgent
from .loose_aggressive import LooseAggressiveAgent
from .maniac import ManiacAgent
from .random_agent import RandomAgent

ALL_AGENTS = {
    "tight_passive":    TightPassiveAgent,
    "tight_aggressive": TightAggressiveAgent,
    "loose_passive":    LoosePassiveAgent,
    "loose_aggressive": LooseAggressiveAgent,
    "maniac":           ManiacAgent,
    "random":           RandomAgent,
}
