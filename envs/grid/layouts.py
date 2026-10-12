"""Original RARe layouts with optional randomized start and goal regions."""

from dataclasses import replace

from .core import GridEnv, GridSpec


CLIFF_WALKING = GridSpec(
    name="CliffWalking",
    shape=(8, 12),
    starts=((7, 1),),
    goals=((7, 10),),
    pits=tuple((row, col) for row in (6, 7) for col in range(2, 10)),
)

BRIDGE = GridSpec(
    name="Bridge",
    shape=(8, 12),
    starts=((0, 0),),
    goals=((7, 10),),
    pits=tuple((row, col) for row in (0, 1, 2, 5, 6, 7) for col in range(2, 10)),
)

ZIGZAG = GridSpec(
    name="ZigZag",
    shape=(8, 12),
    starts=((0, 0),),
    goals=((7, 10),),
    pits=(
        tuple((row, col) for row in range(4) for col in (2, 3, 4))
        + tuple((row, col) for row in range(4, 8) for col in (7, 8, 9))
    ),
)

# Both layouts have safe side regions covering the two outermost columns.
_START_REGION = tuple((row, col) for row in range(8) for col in (0, 1))
_GOAL_REGION = tuple((row, col) for row in range(8) for col in (10, 11))

_LAYOUTS = {spec.name: spec for spec in (CLIFF_WALKING, BRIDGE, ZIGZAG)}
_RANDOM_LAYOUTS = {
    "Bridge": replace(BRIDGE, starts=_START_REGION, goals=_GOAL_REGION),
    "ZigZag": replace(ZIGZAG, starts=_START_REGION, goals=_GOAL_REGION),
    "CliffWalking": replace(
        CLIFF_WALKING,
        starts=tuple((row, col) for row in (6, 7) for col in (0, 1)),
        goals=tuple((row, col) for row in (6, 7) for col in (10, 11)),
    ),
}


def make_grid_env(name: str, *, random: bool = False) -> GridEnv:
    """Choose fixed endpoints or resample from agreed regions at every reset.

    Randomness comes only from the reset key, never from construction or a
    hidden RNG. Custom layouts can use GridEnv(GridSpec(...)) directly.
    """
    if name not in _LAYOUTS:
        raise ValueError(f"Unknown grid environment {name!r}; choose from {tuple(_LAYOUTS)}")
    if type(random) is not bool:
        raise TypeError("random must be a bool")
    return GridEnv((_RANDOM_LAYOUTS if random else _LAYOUTS)[name])
