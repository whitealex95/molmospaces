# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MolmoSpaces is a robotics simulation framework for generating manipulation/navigation data and running benchmarks across MuJoCo, Isaac, and ManiSkill. Data generation and benchmarking are MuJoCo-only; the `molmo_spaces_isaac/` and `molmo_spaces_maniskill/` directories are separate sub-packages with their own `pyproject.toml` providing asset interop for those simulators.

Python 3.11 only. Linux and macOS supported.

## Install

This developer uses **conda** on Ubuntu with an NVIDIA RTX 4090 (sm_89). The conda env is named `mlspaces`. Always use this profile unless the user says otherwise.

### Preferred install profile (Ubuntu + RTX 4090, filament + dev + housegen)

This developer uses the **Filament renderer** and does **not** install `curobo`. Do not add `curobo` to the install command or introduce its prerequisites (CUDA toolkit, pre-installed torch, `TORCH_CUDA_ARCH_LIST`/`CUDA_HOME`/`CPATH` exports) unless explicitly asked.

```bash
conda deactivate
conda env remove -n mlspaces -y                    # if rebuilding from scratch
conda create -n mlspaces python=3.11 -y
conda activate mlspaces

# `mujoco-filament` requires uv (see gotcha below). Install uv into the env.
pip install uv
export VIRTUAL_ENV=$CONDA_PREFIX
uv pip install -e ".[mujoco-filament,dev,housegen]"

pre-commit install
```

### Gotcha: `mujoco-filament` requires `uv`

`pyproject.toml`'s `mujoco-filament` extra contains `mujoco @ file://${PROJECT_ROOT}/bin/wheels/...`. `${PROJECT_ROOT}` is a uv-specific substitution; plain `pip` errors with `ValueError: non-local file URIs are not supported`. The fix is to install `uv` inside the conda env and run `uv pip install` — as shown in the profile above. Setting `VIRTUAL_ENV=$CONDA_PREFIX` tells `uv` to target the active conda env.

The classic `mujoco` extra does work with plain pip, but this developer wants Filament — do not silently swap to `mujoco`.

### Available extras

Exactly one of `mujoco` / `mujoco-filament` must be selected. This developer always uses `mujoco-filament` (Filament renderer; installs from local wheel at `bin/wheels/`).

- `dev` — code development (ruff, mypy, pre-commit, pybind11-stubgen, ty)
- `grasp` — grasp generation pipeline
- `housegen` — house generation pipeline from iTHOR, ProcTHOR, or Holodeck JSONs
- `curobo` — CuRobo GPU-accelerated planning (used for RB-Y1 tasks). **Not installed in this developer's profile.** If ever re-added: requires `conda install cuda-toolkit=12.8 ninja cuda-nvcc cuda-cudart-dev`, pre-installing torch with cu128, and exporting `CUDA_HOME=$CONDA_PREFIX`, `CPATH=...`, `TORCH_CUDA_ARCH_LIST="8.9"` before the project install (see README's cuRobo section).
- `docs` — MkDocs site build tooling (only needed to preview/build the documentation site).

## Common commands

```bash
# Format / lint (CI enforces format only)
ruff format .
ruff check .

# Tests — PYTHONPATH=. is required
PYTHONPATH=. pytest mlspaces_tests/component_tests        # what CI runs
PYTHONPATH=. pytest mlspaces_tests/data_generation        # full datagen tests
PYTHONPATH=. pytest mlspaces_tests/data_generation_curobo # requires curobo extra
PYTHONPATH=. pytest mlspaces_tests/data_generation/test_franka_pick.py::test_name --log-cli-level DEBUG

# Install benchmark assets (downloads to MLSPACES_ASSETS_DIR)
python -m molmo_spaces.molmo_spaces_constants

# Quick smoke test (Linux). On macOS replace `python` with `mjpython`.
python scripts/datagen/run_pipeline.py --viewer --seed 3

# Pre-commit hooks
pre-commit install
```

Data-generation tests under `mlspaces_tests/data_generation*` compare against versioned fixture archives pinned in `molmo_spaces/molmo_spaces_constants.py` under `test_data`. Regenerating fixtures involves running the `generate_test_data_*.py` scripts, uploading with `mjt_upload`, and bumping the version string — see `mlspaces_tests/README.md`.

## Three entry points

```
molmo_spaces/evaluation/eval_main.py    # benchmark evaluation (config-driven)
molmo_spaces/data_generation/main.py    # data generation (config-driven)
scripts/datagen/run_pipeline.py         # debug / one-off runs (hand-edited)
```

Both config-driven entry points accept a fully-qualified config reference, e.g.:

```bash
python molmo_spaces/data_generation/main.py FrankaPickOmniCamConfig
python molmo_spaces/evaluation/eval_main.py \
    molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig \
    --benchmark_dir <path/to/benchmark> --task_horizon_steps 500
```

Configs are auto-discovered via `molmo_spaces/data_generation/config_registry.py`; list them with:

```python
from molmo_spaces.data_generation.main import auto_import_configs
from molmo_spaces.data_generation.config_registry import list_available_configs
auto_import_configs(); print(list_available_configs())
```

## Architecture

Everything is driven by **experiment configs** that inherit `MlSpacesExpConfig` (`molmo_spaces/configs/abstract_exp_config.py`). A config bundles a task sampler config, robot config, policy config, camera config, and fixed task params; the runners take a config class and execute parallel rollouts.

### Robot stack (three layers — see `docs/concepts.md` for the canonical writeup)

```
Robot                            (molmo_spaces/robots/abstract.py)
├── robot_view: RobotView        (composes move groups; bulk qpos/ctrl/Jacobian access)
│   └── move_groups: dict[str, MoveGroup]   # keys like "arm", "gripper", "base"
├── controllers: dict[str, Controller]      # one per commanded move group
└── kinematics / parallel_kinematics
```

- **Move groups** are the atomic unit of robot control. They abstract over MuJoCo joints/actuators — a group's number of joints, qpos dims, and actuators do not have to match (mirrored grippers, mocap-driven "fake" actuators in `FloatingRUMBaseGroup`, etc.). The rest of the system only sees `joint_pos`, `ctrl`, `noop_ctrl`.
- **Action format everywhere is `dict[str, np.ndarray]`** keyed by move group ID, e.g. `{"arm": ..., "gripper": ...}`. Configs reference the same keys (`init_qpos`, `command_mode`).
- **Command modes** (`"joint_position"`, `"joint_rel_position"`) select which `Controller` subclass runs. Controllers run at `ctrl_dt` (~20ms); MuJoCo physics runs at `sim_dt` (~2ms); the policy is queried at `policy_dt` (~200ms). One `task.step()` = one policy step = several control steps = many sim sub-steps.

### Env / Task / TaskSampler

```
TaskSampler   owns env lifecycle, loads scenes, randomizes placement, returns a Task
   │
   └── Task   wraps (does not own) the env; Gymnasium-style reset/step; reward/success
          │
          └── Env  MuJoCo model + MjData + robots + CameraManager + ObjectManager
```

Key behaviors that bite if you miss them:

- `task.reset()` does **NOT** call `env.reset()` — the sampler is responsible for putting the env in the desired state. Task reset only clears bookkeeping (step counter, sensors, policy).
- The env API is nominally batched but `n_batch > 1` is not well tested — assume 1.
- Episode flow: `sampler.sample_task()` → `task.reset()` → loop `task.step(action)` until `task.is_done()` → next episode reuses or reloads scene via another `sample_task()`.

### Data generation pipeline

`molmo_spaces/data_generation/pipeline.py` (`ParallelRolloutRunner`) spawns worker processes; each creates its own sampler, samples tasks, runs rollouts, and saves trajectories (HDF5/video via `molmo_spaces/utils/save_utils.py`).

### Evaluation / benchmarks

A **benchmark** is a `benchmark.json` containing self-contained episode specs (scene, robot pose, object poses, cameras, language). An eval config is a normal datagen config with a policy attached; when pointed at a benchmark, per-episode fields override sampler randomization. To plug in a custom policy: subclass `InferencePolicy` (implement `prepare_model`, `reset`, `get_action`), make a `BasePolicyConfig`, and an eval config extending `JsonBenchmarkEvalConfig`. Full template in `molmo_spaces/evaluation/README.md`.

### Other key modules

- `molmo_spaces/molmo_spaces_constants.py` — global paths, asset version pins, resource manager entry point. Modify pinned versions here when rolling fixtures or assets.
- `molmo_spaces/planner/` — `astar_planner` for nav, `curobo_planner` (+ gRPC client/server) for arm motion planning.
- `molmo_spaces/policy/solvers/` — scripted planner-based policies (one of the primary action sources for datagen).
- `molmo_spaces/renderer/` — OpenGL and Filament backends; Filament requires the `mujoco-filament` extra.
- `molmo_spaces/env/arena/randomization/` — lighting/texture/dynamics domain randomization.

## Environment variables

| Variable | Purpose |
|---|---|
| `MLSPACES_ASSETS_DIR` | Where downloaded assets live (default `~/.cache/molmospaces/assets/<install-hash>`) |
| `MLSPACES_FORCE_INSTALL` | Override existing assets (default `True`) |
| `MLSPACES_PINNED_ASSETS_FILE` | JSON overriding asset versions in `molmo_spaces_constants.py` |
| `MUJOCO_EGL_DEVICE_ID` | Render device — indices do not match `CUDA_VISIBLE_DEVICES` (see issue #66) |
| `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` | Required for headless rendering on Linux |

## Conventions

- Robot base frame: +x forward, +y left, +z up.
- Parallel-jaw gripper: +z forward, fingers open along y.
- `setuptools` packages only `molmo_spaces*` — the `molmo_spaces_isaac` and `molmo_spaces_maniskill` sub-packages are explicitly excluded from this install and ship separately.
- `ruff` lint excludes `scripts/`, `tests/`, and `mlspaces_tests/scenes/` (see `pyproject.toml`). Formatting is enforced repo-wide by CI.
- Type stubs live in `typings/`; regenerate with `pybind11-stubgen mujoco -o ./typings/`.
