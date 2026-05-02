
from .maxminq import MAXMINQ
# from .DDPG import DDPG
# from .TD3 import TD3
# from .ExpectedSARSA import ExpectedSARSA

from .ute import UTE
from .temporl import TempoRL
from .rare import RARe
# from .TAAC import TAAC

BASE_REGISTRY = {
    "MAXMINQ": MAXMINQ,
    # "DDPG": DDPG,
    # "TD3": TD3,
}

ALGO_REGISTRY = {
    "TempoRL": TempoRL,
    "RARe": RARe,
    "UTE": UTE,
    # "TAAC": TAAC
}
