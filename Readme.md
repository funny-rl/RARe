# Risk-aware Action Repetition (RARe)

This repository contains the official code for the paper **"Risk-aware Action Repetition Learning"**.

It includes the implementation of the RARe algorithm and several baselines (TempoRL, UTE, TAAC, etc.) evaluated across diverse domains (Classic Control, Grid Worlds, Robotics, and Safe RL).

## Repository Structure

- `code/algos/`: Implementations of RARe and other baselines.
- `code/scripts/`: Bash scripts to run experiments across multiple domains.
- `code/configs/`: Configuration files used for training scenarios.
- `analysis/`: Scripts for evaluating models and generating figures.

## Usage

To train the agents, you can run the provided execution scripts located inside the `code/scripts/` folder. The experiments are organized by domain categories. The general syntax for running a script is:

```bash
cd code/scripts/<domain>/
bash <script_name>.sh [ALGO] [USE_WANDB] [Grop Name]
```

### Parameters
- **`ALGO`**: The algorithm to run. Examples include `RARe`, `TempoRL`, `UTE`, `TAAC`, or `null` (for the base agent such as `MAXMINQ` or `DDPG`).
- **`USE_WANDB`**: Set to `true` to log training with Weights & Biases, otherwise `false`.
- **`GROUP_NAME`**: The Weights & Biases group name used to organize related runs, such as different seeds or variants of the same experiment.

### Available Domains

Below are the bash scripts grouped by domain:

- **Grid Worlds (`code/scripts/grid/`)**
  - `bridge.sh`, `cliff.sh`, `field.sh`, `rzigzag.sh`, `zigzag.sh`
- **Classic Control (`code/scripts/classic/`)**
  - `mountaincar.sh`, `pendulum.sh`, `sparse_pendulum.sh`
- **Robotics (`code/scripts/robotics/`)**
  - `pointmaze_medium.sh`
- **Safe RL (`code/scripts/safe/`)**
  - `button.sh`, `circle.sh`, `goal.sh`
- **Reward Estimation (`code/scripts/estimation/`)**
  - `cnr.sh`, `dnr.sh`

### Example Execution

To train **RARe** on the benchmark **Mountain Car** locally on GPU **0** and log to Weights & Biases:

```bash
cd code/scripts/classic/
bash mountaincar.sh RARe true "Group name"
```

## License

This anonymized supplementary code is released under the MIT License for review and reproducibility purposes. See the `LICENSE` file for details.

## Third-party assets

This repository uses or builds upon open-source packages, benchmark environments, and baseline implementations. We cite the corresponding papers in the manuscript and acknowledge the original creators and maintainers of these assets.

For transparency, we provide a `licenses/` directory containing license files or license notes for the third-party assets used in this work. These third-party assets remain subject to their original licenses, copyright notices, and terms of use.

We use all third-party assets solely for research and reproducibility purposes, and we respect their original licenses and terms of use. Please refer to the original repositories for the most up-to-date license information.
