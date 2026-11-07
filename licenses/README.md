# Third-party licenses

This project is released under the Apache License 2.0 (see `../LICENSE`). It installs the
packages below as dependencies; they remain under their own licenses.

| Asset | Use | License |
| --- | --- | --- |
| JAX, Flax, Optax | learners | Apache 2.0 |
| Brax | HalfCheetah simulator and PPO reference | Apache 2.0 |
| MuJoCo, MJX | PointMaze physics | Apache 2.0 (`assets/MUJOCO_license.txt`) |
| Gymnasium | PointMaze interface; Pendulum and MountainCar dynamics reimplemented in JAX | MIT (`assets/Gymnasium_license.txt`) |
| Gymnasium-Robotics | PointMaze model | MIT |
| Safety-Gymnasium | Goal and Button tasks | Apache 2.0 (`assets/SafeRL-gymnasium_license.txt`) |

The baselines are our own JAX implementations of the published methods. The licenses of the
original implementations are kept for reference: `assets/TempoRL_license.txt`,
`assets/UTE_license.txt` and `assets/TAAC_license.txt`.

Check the original repositories for the current license terms.
