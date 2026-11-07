"""MJX adapter for the original Gymnasium-Robotics PointMaze experiment."""

from dataclasses import dataclass
from importlib.metadata import version
import math
from pathlib import Path
from typing import NamedTuple
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp


ENV_ID = "PointMaze_Medium_Diverse_GDense-v3"
_MAX_GEOM_PAIRS = 8
_MAX_CONTACT_POINTS = 8


def _optimized_model(original, mujoco):
    """Keep the original maze while reducing MJX's collision work."""
    xml_path = Path(original.tmp_xml_file_path)
    root = ET.parse(xml_path).getroot()
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "numeric", name="max_geom_pairs", data=str(_MAX_GEOM_PAIRS))
    ET.SubElement(custom, "numeric", name="max_contact_points", data=str(_MAX_CONTACT_POINTS))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    reference = original.model
    if (model.ngeom, model.nbody, model.nq, model.nv) != (
        reference.ngeom, reference.nbody, reference.nq, reference.nv
    ):
        raise RuntimeError("The PointMaze MJCF layout changed")

    particle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "particle_geom")
    if particle < 0 or model.geom_type[particle] != mujoco.mjtGeom.mjGEOM_SPHERE:
        raise RuntimeError("The PointMaze particle geometry changed")
    other = [i for i in range(model.ngeom) if i != particle]
    walls = [i for i in other if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX]
    if (model.ngeom != 40 or len(walls) != 38
            or any(model.geom_bodyid[i] != 0 for i in other)):
        raise RuntimeError("The PointMaze collision layout changed")
    margins = [float(model.geom_margin[i] + model.geom_margin[particle]) for i in other]
    if any(not math.isclose(margin, margins[0], abs_tol=1e-9) for margin in margins):
        raise RuntimeError("PointMaze contact margins are no longer uniform")

    # MJX-JAX's sphere-box collision omits positive-distance margin contacts.
    # Absorbing the same pair margin into the sphere restores their constraint distance.
    # The compiled body's mass stays at the original value.
    model.geom_size[particle, 0] += margins[0]
    model.geom_margin[:] = 0
    # Four Newton iterations matched the original 20 in the contact stress check.
    model.opt.iterations = 4
    return model, margins[0]


@dataclass(frozen=True)
class PointMazeSpec:
    max_steps: int = 600
    position_noise_range: float = 0.25

    def __post_init__(self) -> None:
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if (isinstance(self.position_noise_range, bool)
                or not math.isfinite(self.position_noise_range)
                or not 0 <= self.position_noise_range < 0.5):
            raise ValueError("position_noise_range must be finite and in [0, 0.5)")


class PointMazeState(NamedTuple):
    data: object
    goal: jax.Array
    steps: jax.Array


def observe(data, goal: jax.Array) -> jax.Array:
    position = data.qpos[:2]
    return jnp.concatenate((position, data.qvel[:2], goal, position))


class PointMazeEnv:
    """Use the original MJCF and goal rules with a functional MJX state."""

    backend = "mjx_jax"
    supports_exact_risk = False
    requires_global_step = False
    observation_storage_dtype = jnp.float32
    observation_codec = "raw"

    def __init__(self, spec: PointMazeSpec = PointMazeSpec()):
        try:
            import gymnasium as gym
            import gymnasium_robotics
            import mujoco
            from mujoco import mjx
        except ImportError as error:
            raise ImportError(
                "PointMaze requires gymnasium-robotics==1.4.1, "
                "gymnasium>=1.2.0, mujoco>=3.4.0 and matching mujoco-mjx"
            ) from error

        if not hasattr(gym, "register_envs"):
            raise RuntimeError("PointMaze requires gymnasium>=1.2.0")
        gym.register_envs(gymnasium_robotics)
        reference = gym.make(ENV_ID)
        try:
            original = reference.unwrapped
            if reference.spec.max_episode_steps != 600:
                raise RuntimeError("PointMaze registry horizon changed from 600")
            if not original.continuing_task or original.reset_target:
                raise RuntimeError("PointMaze goal-continuation defaults changed")
            try:
                optimized_model, absorbed_margin = _optimized_model(original, mujoco)
                self.model = mjx.put_model(optimized_model)
            except NotImplementedError as error:
                raise RuntimeError(
                    "This PointMaze model needs mujoco-mjx>=3.4.0; "
                    "the installed MJX cannot handle its contact dimension"
                ) from error
            self.initial_data = mjx.make_data(self.model)
            self.goal_centers = jnp.asarray(original.maze.unique_goal_locations,
                                            dtype=jnp.float32)
            self.reset_centers = jnp.asarray(original.maze.unique_reset_locations,
                                             dtype=jnp.float32)
            if self.goal_centers.shape != (3, 2) or self.reset_centers.shape != (1, 2):
                raise RuntimeError("PointMaze Medium Diverse G layout changed")
        finally:
            reference.close()
        self.spec = spec
        self._mjx = mjx
        self.physics_optimization = {
            "max_geom_pairs": _MAX_GEOM_PAIRS,
            "max_contact_points": _MAX_CONTACT_POINTS,
            "solver_iterations": 4,
            "margin_absorbed_into_particle_radius": absorbed_margin,
        }
        self.software_versions = {
            "gymnasium": gym.__version__,
            "gymnasium_robotics": gymnasium_robotics.__version__,
            "mujoco": mujoco.__version__,
            "mujoco_mjx": version("mujoco-mjx"),
        }

    @property
    def name(self) -> str:
        return ENV_ID

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (8,)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (2,)

    @property
    def action_low(self) -> jax.Array:
        return jnp.full((2,), -1., jnp.float32)

    @property
    def action_high(self) -> jax.Array:
        return jnp.full((2,), 1., jnp.float32)

    def _sample_goal(self, key: jax.Array) -> jax.Array:
        cell_key, noise_key = jax.random.split(key)
        index = jax.random.randint(cell_key, (), 0, self.goal_centers.shape[0])
        noise = jax.random.uniform(noise_key, (2,), minval=-self.spec.position_noise_range,
                                   maxval=self.spec.position_noise_range)
        return self.goal_centers[index] + noise

    def reset(self, key: jax.Array, *, global_step: int | jax.Array = 0
              ) -> tuple[jax.Array, PointMazeState]:
        del global_step
        goal_key, noise_key = jax.random.split(key)
        goal = self._sample_goal(goal_key)
        noise = jax.random.uniform(noise_key, (2,), minval=-self.spec.position_noise_range,
                                   maxval=self.spec.position_noise_range)
        position = self.reset_centers[0] + noise
        data = self.initial_data.replace(
            qpos=self.initial_data.qpos.at[:2].set(position),
            qvel=jnp.zeros_like(self.initial_data.qvel),
        )
        # mjx.step refreshes kinematics; observations here use only qpos and qvel.
        state = PointMazeState(data, goal, jnp.int32(0))
        return observe(data, goal), state

    def with_global_step(self, observation: jax.Array, state: PointMazeState,
                         global_step: int | jax.Array) -> tuple[jax.Array, PointMazeState]:
        del global_step
        return observation, state

    def step(self, key: jax.Array, state: PointMazeState, action: jax.Array):
        del key
        clipped_action = jnp.clip(jnp.asarray(action, jnp.float32).reshape((2,)), -1., 1.)
        # PointEnv clips velocity before simulation, so clip the incoming state.
        data = state.data.replace(qvel=jnp.clip(state.data.qvel, -5., 5.),
                                  ctrl=clipped_action)
        data = self._mjx.step(self.model, data)
        position = data.qpos[:2]
        distance = jnp.linalg.norm(position - state.goal)
        reward = jnp.exp(-distance)
        success = distance <= 0.45
        # The original v1.4.1 default keeps the same goal after success.
        next_state = PointMazeState(data, state.goal, state.steps + 1)
        observation = observe(data, state.goal)
        return observation, next_state, reward, jnp.asarray(False), \
            next_state.steps >= self.spec.max_steps, {"success": success}


def make_point_maze_env(**settings) -> PointMazeEnv:
    return PointMazeEnv(PointMazeSpec(**settings))
