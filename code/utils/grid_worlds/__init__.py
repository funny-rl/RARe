from utils.grid_worlds.envs.chainmdp import ChainMDP
from utils.grid_worlds.envs.field import Field
from utils.grid_worlds.envs.cliffwalking import CliffWalking
from utils.grid_worlds.envs.bridge import Bridge
from utils.grid_worlds.envs.zigzag import ZigZag
from utils.grid_worlds.envs.zigzag import RZigZag
from utils.grid_worlds.envs.dnr import DiscreteNoisyRewards
ENVS_REGISTRY = {
    "ChainMDP": ChainMDP,
    "CliffWalking": CliffWalking,
    "Bridge": Bridge,
    "ZigZag": ZigZag,
    "RZigZag": RZigZag,
    "Field": Field,
    "DiscreteNoisyRewards": DiscreteNoisyRewards
}
