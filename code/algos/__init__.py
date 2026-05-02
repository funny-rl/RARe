
from .maxminq import MAXMINQ
from .ddpg import DDPG

from .ute import UTE
from .temporl import TempoRL
from .rare import RARe
from .taac import TAAC

BASE_REGISTRY = {
    "MAXMINQ": MAXMINQ,
    "DDPG": DDPG
}

ALGO_REGISTRY = {
    "TempoRL": TempoRL,
    "RARe": RARe,
    "UTE": UTE,
    "TAAC": TAAC
}
