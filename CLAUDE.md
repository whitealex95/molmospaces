# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MolmoSpaces is a robotics simulation framework for generating manipulation/navigation data and running benchmarks across MuJoCo, Isaac, and ManiSkill. Data generation and benchmarking are MuJoCo-only; the `molmo_spaces_isaac/` and `molmo_spaces_maniskill/` directories are separate sub-packages with their own `pyproject.toml` providing asset interop for those simulators.

Python 3.11 only. Linux and macOS supported.

## Working with this codebase — read in-repo docs first

Before answering questions about install steps, asset/resource management, env vars, sub-package usage, or core abstractions, **search the in-repo docs and READMEs first** — most of what users ask is already documented in this repo, and re-deriving it from source code wastes effort (and is more error-prone, as repeated debugging sessions in this conversation have shown).

| Where to look | What you'll find |
|---|---|
| `README.md` | Install, optional extras, env-var summary table, asset overview, cuRobo install ordering |
| `docs/assets.md` | Resource manager, `MLSPACES_*` env-var defaults, bulk vs lazy download (`MLSPACES_DOWNLOAD_EXTRACT_ALL_SCENES_OBJECTS_GRASPS`), symlink layout, per-asset license helper |
| `docs/concepts.md` | Robot / RobotView / MoveGroup abstractions, Env / Task / TaskSampler lifecycle, timing model |
| `docs/code_structure.md` | Module-by-module layout |
| `docs/development.md` | Pre-commit hooks, ruff rules, IDE setup |
| `docs/evaluation_guide.md`, `molmo_spaces/evaluation/README.md` | Benchmark eval, custom-policy plug-in template, sample episode spec |
| `docs/data_format.md`, `docs/data_processing.md` | Trajectory file layout |
| `docs/tutorials/` | Task-shaped walkthroughs (e.g. `add_robot/`) |
| `molmo_spaces_isaac/README.md`, `molmo_spaces_maniskill/README.md` | Sub-package install + CLI scripts |
| `mlspaces_tests/README.md` | Test fixture regeneration + upload workflow |
| `docs/mansion_to_usd.md` | **MansionWorld floor JSON → USDA conversion.** Read first for anything related to the mansion dataset (`~/Projects/mansion`), AI2-THOR/Holodeck JSON → USD, or `scripts/mansion/`. Mansion uses its own asset caches (`~/.objathor-assets/`, mansion_patch) — the molmospaces housegen + `ms-convert-houses` pipeline cannot convert mansion scenes end-to-end. |
| `docs/mansion_to_mjcf.md` | **MansionWorld floor JSON → MJCF conversion** (sibling of the USDA doc). For loading mansion scenes in MuJoCo viewer / `MjModel`. Same source caches, same 100 % coverage, different output target. Note that molmospaces's objaverse MJCF cache is on-disk metadata-only and covers only ~48 % of mansion's objathor UIDs even after bulk download — direct `.pkl.gz` → `.obj` bake is required. |
| `docs/navigation_pipeline.md` | **2D A\* navigation + G1 egocentric-render pipeline** (`scripts/navigation/`). Read this first — and **keep it in sync** — for anything touching the navigation scripts: occupancy-grid building (`build_occupancy.py`), clearance-aware A\* planning (`plan.py`), the kinematic G1 ego/chase video runtime (`run_mujoco.py`), or the batch trajectory generator (`gen_trajectories.py`). Whenever you change a `scripts/navigation/` script, update this doc in the same change. |
| `docs/navigation_motion_matching.md` | **Full-body motion-matched G1 walk runtime** (`run_mujoco_mm.py`) — the Stage-3 alternative to the kinematic `run_mujoco.py`. Drives all 36 G1 qpos from `~/Projects/motionmatching-g1`'s `MotionMatcher` along the same A\* `path.npz`, output to `nav_runs_mm/`. Also documents the top-down control overlay (target + command) and the `motion_full.npz` / `motion_min.npz` saved motion + compression comparison. Read first — and **keep in sync** — for anything touching `run_mujoco_mm.py`. |
| `pyproject.toml` | Declared deps, extras, ruff config |

For runtime crashes or unfamiliar errors, also check open GitHub issues — the project's contributors often document workarounds there before they reach the docs:

```bash
gh issue list --repo allenai/molmospaces            # list all
gh search issues --repo allenai/molmospaces "<keyword>"
gh issue view <NUMBER> --repo allenai/molmospaces --comments
```

(The Filament THORMAP segfault, for example, was diagnosed via `gh issue view 79` in this repo's history.)

**Don't re-derive answers from source code when a doc covers it.** If a doc seems wrong, out of date, or contradicts observed behavior, flag the discrepancy in the response rather than silently working around it.

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

Exactly one of `mujoco` / `mujoco-filament` must be selected. For the main `mlspaces` env, this developer uses `mujoco-filament` (Filament renderer; installs from local wheel at `bin/wheels/`). A parallel `mlspaces-mujoco` env uses the classic `mujoco` extra — see the section below.

- `dev` — code development (ruff, mypy, pre-commit, pybind11-stubgen, ty)
- `grasp` — grasp generation pipeline
- `housegen` — house generation pipeline from iTHOR, ProcTHOR, or Holodeck JSONs
- `curobo` — CuRobo GPU-accelerated planning (used for RB-Y1 tasks). **Not installed in this developer's profile.** If ever re-added: requires `conda install cuda-toolkit=12.8 ninja cuda-nvcc cuda-cudart-dev`, pre-installing torch with cu128, and exporting `CUDA_HOME=$CONDA_PREFIX`, `CPATH=...`, `TORCH_CUDA_ARCH_LIST="8.9"` before the project install (see README's cuRobo section).
- `docs` — MkDocs site build tooling (only needed to preview/build the documentation site).

## Isaac sub-package (separate env)

`molmo_spaces_isaac/` is a separate Python package with its own `pyproject.toml` (`molmo-spaces-isaac`), explicitly excluded from the main install (see `pyproject.toml` `[tool.setuptools.packages.find]` exclude list). It provides MJCF→USD asset/house conversion and IsaacSim/IsaacLab integration.

**This developer keeps Isaac in a separate conda env named `mlspaces-isaac`.** Do not install it into the `mlspaces` env — there is an unresolvable torch conflict:

| Env | Torch | CUDA |
|---|---|---|
| `mlspaces` (main) | `~=2.7.0` (>=2.7,<2.8) | cu128 |
| `mlspaces-isaac` | `>=2.9.0` (uv override) | cu130 |

### Install profile (one-time)

```bash
conda deactivate
conda env remove -n mlspaces-isaac -y          # if rebuilding
conda create -n mlspaces-isaac python=3.11 -y
conda activate mlspaces-isaac

pip install uv
export VIRTUAL_ENV=$CONDA_PREFIX

cd /home/jkim3662/Projects/molmospaces/molmo_spaces_isaac
uv pip install -e ".[dev,sim]"
```

Notes:
- Must `cd` into `molmo_spaces_isaac/` first — the README is explicit about this.
- Must use `uv pip` — the sub-package's `pyproject.toml` uses uv-only features (`override-dependencies`, custom indices for `nvidia` and `torch`/cu130).
- The `sim` extra pulls in `isaaclab[all,isaacsim]>=2.3.1`, which downloads IsaacSim 5.1.0 + IsaacLab 2.3.1 — multi-GB, slow.

### Note: `flatdict` build constraint

The Isaac install transitively pulls `flatdict==4.0.1` (via `isaaclab`), which calls `pkg_resources` at build time. Setuptools ≥81 dropped `pkg_resources` from the implicit imports, so the sdist build fails with `ModuleNotFoundError: No module named 'pkg_resources'`. This is pinned via `[tool.uv] build-constraint-dependencies = ["setuptools<81"]` in `molmo_spaces_isaac/pyproject.toml` — no install-time flag is needed.

### Isaac-only CLI scripts (only available in `mlspaces-isaac` env)

- `ms-download --type usd --install-dir assets/usd --assets <dataset>` — fetch USD assets
- `ms-download --type usd --install-dir assets/usd --scenes <dataset>` — fetch USD scenes
- `ms-convert-assets` — MJCF → USD asset conversion
- `ms-convert-houses` — MJCF → USD house conversion

## Classic MuJoCo env (no Filament)

`mlspaces-mujoco` is a parallel env that installs the same `molmo-spaces` package as `mlspaces` but selects the classic `mujoco` extra instead of `mujoco-filament`. Use it for workflows that hit the Filament arena overflow (see Known issues) or that otherwise need the stock PyPI MuJoCo wheel rather than the custom Filament-bundled build.

| Env | mujoco wheel | Renderer |
|---|---|---|
| `mlspaces` (main) | 3.7.1 (custom Filament build, `bin/wheels/`) | Filament + OpenGL |
| `mlspaces-mujoco` | 3.5.0 (stock PyPI) | OpenGL only |

### Install profile (one-time)

```bash
conda deactivate
conda env remove -n mlspaces-mujoco -y          # if rebuilding
conda create -n mlspaces-mujoco python=3.11 -y
conda activate mlspaces-mujoco

# Classic `mujoco` extra has no ${PROJECT_ROOT} substitution — plain pip works.
# tyro is an undeclared dep of scripts/data/generate_maps.py; install it too.
pip install -e ".[mujoco,dev,housegen]" tyro

pre-commit install
```

Notes:
- No `uv` dance required (only `mujoco-filament` needs it).
- The renderer fallback note in Known issues — "fall back to the classic `mujoco` extra for that workflow" — means activating `mlspaces-mujoco`.

## Switching between envs

- `conda activate mlspaces` → MuJoCo + Filament work (datagen, evaluation, benchmarks; default)
- `conda activate mlspaces-mujoco` → same workflows but with stock MuJoCo + OpenGL renderer
- `conda activate mlspaces-isaac` → Isaac/USD conversion, IsaacSim/IsaacLab scripts

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

# Install benchmark assets (downloads to MLSPACES_ASSETS_DIR).
# Both env vars have defaults — set them only if you want a different location:
#   MLSPACES_CACHE_DIR   default: ~/.cache/molmo-spaces-resources
#   MLSPACES_ASSETS_DIR  default: ~/.cache/molmospaces/assets/<base64url(project_path)>
python -m molmo_spaces.molmo_spaces_constants

# Quick smoke test (no --viewer; mjviewer needs classic OpenGL which the
# Filament wheel lacks — see Known issues). Pre-generate maps once from
# `mlspaces-mujoco` if the THORMAP segfault hits.
python scripts/datagen/run_pipeline.py --seed 3

# Pre-commit hooks
pre-commit install
```

Data-generation tests under `mlspaces_tests/data_generation*` compare against versioned fixture archives pinned in `molmo_spaces/molmo_spaces_constants.py` under `test_data`. Regenerating fixtures involves running the `generate_test_data_*.py` scripts, uploading with `mjt_upload`, and bumping the version string — see `mlspaces_tests/README.md`.

## Visualizing a scene

**Always use `mlspaces-mujoco`** for any MuJoCo viewer workflow. The Filament-built wheel in `mlspaces` is missing the classic OpenGL UI symbols (`mjui_update`, etc.), so every viewer entry point — `python -m mujoco.viewer`, `mujoco.viewer.launch()`, `mujoco.viewer.launch_passive()`, and the `--viewer` flag of `run_pipeline.py` — fails with `ImportError: undefined symbol: mjui_update` in `mlspaces`. There is no env-var workaround.

```bash
conda activate mlspaces-mujoco
```

### Just look at a scene XML (no project code)

```bash
python -m mujoco.viewer --mjcf /path/to/scene.xml
```

Use this for raw MJCF inspection. Interactive: mouse to orbit, Ctrl+drag to manipulate joints, `w` for wireframe, spacebar to pause. Won't load lazily-fetched meshes/textures the project's resource manager would normally download — point it at scenes already on disk under `${MLSPACES_ASSETS_DIR}/scenes/`.

### Full project pipeline with viewer (robot + cameras + task)

```bash
python scripts/datagen/run_pipeline.py --viewer --seed 3
```

This is the `--viewer` flag that fails from `mlspaces` — works from `mlspaces-mujoco` because of the classic renderer. Adds the robot, cameras, and policy execution on top of the scene. Pre-generated `_map.png` files are not required here (the classic renderer can compute occupancy on the fly), but they speed up startup.

### Programmatic (REPL / scripts)

```python
import mujoco, mujoco.viewer
model = mujoco.MjModel.from_xml_path("/path/to/scene.xml")
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)              # blocking, modal
# or for non-blocking driven from Python:
viewer = mujoco.viewer.launch_passive(model, data)
```

`launch_passive` is what `run_pipeline.py --viewer` uses internally — your code keeps stepping `mj_step` while the viewer renders in another thread.

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

## Known issues

### Filament wheel + THORMAP scenes → segfault unless `_map.png` is precomputed

Tracked upstream: https://github.com/allenai/molmospaces/issues/79

When the Filament-wheel `mlspaces` env runs a pipeline that loads a THORMAP/iTHOR/procthor/holodeck scene, the worker segfaults during scene setup with a misleading message:

```
PanicLog in allocateHandleSlow:136
reason: HandleAllocator arena is full, using slower system heap.
  Please increase the appropriate constant (e.g. FILAMENT_OPENGL_HANDLE_ARENA_SIZE_IN_MB).
... Segmentation fault (core dumped)
```

**Don't be fooled** — bumping `FILAMENT_OPENGL_HANDLE_ARENA_SIZE_IN_MB` does NOT help. The arena message is from a second renderer that gets spun up to compute the scene's occupancy map (`get_thormap` in `molmo_spaces/env/env.py:690`). The custom Filament-built MuJoCo wheel **does not include the classic OpenGL renderer at all**, so any code path requesting a renderer goes through Filament, and the second Filament init crashes regardless of arena size. The `use_filament` flag in Python (e.g. in `ProcTHORMap.from_mj_model_path`) is effectively a no-op for this wheel.

`get_thormap` short-circuits the renderer if a precomputed `<scene_stem>_map.png` already exists next to the scene XML — so the fix is to pre-generate those maps in the **`mlspaces-mujoco`** env (classic OpenGL renderer), then run the actual workflow from `mlspaces` (Filament).

**Workflow:**

```bash
# 1. Generate the maps once, using the classic-mujoco env
conda activate mlspaces-mujoco
SCENES_DIR=$(python -c "from molmo_spaces.molmo_spaces_constants import ASSETS_DIR; print(ASSETS_DIR / 'scenes')")
PYTHONPATH=. python scripts/data/generate_maps.py --dataset ithor             --split train --scenes-dir "$SCENES_DIR"
PYTHONPATH=. python scripts/data/generate_maps.py --dataset procthor-10k      --split train --scenes-dir "$SCENES_DIR"
# (repeat --dataset / --split for other datasets you'll hit; only present XML files get processed)

# 2. Switch back to mlspaces (Filament) and run normally
conda activate mlspaces
python scripts/datagen/run_pipeline.py --seed 3
```

Notes:
- `scripts/data/generate_maps.py` uses `tyro`, which **is not declared** as a main-package dep — `pip install tyro` in the `mlspaces-mujoco` env if missing.
- Maps land in the shared asset cache (`~/.cache/molmospaces/.../scenes/<dataset>/`) so both envs see them.
- Drop the `--viewer` flag — `mjviewer` requires the classic OpenGL renderer that the Filament wheel lacks, so `--viewer` always crashes with the Filament wheel even after maps are precomputed.
- The "leaked semaphore objects" warning that follows the segfault is downstream (multiprocessing workers torn down after the crash), not a separate bug.

## Environment variables

All `MLSPACES_*` variables are **optional** — every one has a working default. Set them only to relocate caches or override pins.

| Variable | Purpose | Default |
|---|---|---|
| `MLSPACES_CACHE_DIR` | Where downloaded archives are extracted (the "store") | `~/.cache/molmo-spaces-resources` |
| `MLSPACES_ASSETS_DIR` | Where versioned symlinks are created (the "view" the project reads) | `~/.cache/molmospaces/assets/<base64url(project_path)>` |
| `MLSPACES_OBJAVERSE_ASSETS_DIR` | Override location for objaverse objects only | `${MLSPACES_ASSETS_DIR}/objects/objaverse` |
| `MLSPACES_FORCE_INSTALL` | Replace existing symlinks when version pin differs | `True` |
| `MLSPACES_PINNED_ASSETS_FILE` | JSON merged onto `DATA_TYPE_TO_SOURCE_TO_VERSION` (override versions) | _(unset)_ |
| `MLSPACES_DOWNLOAD_EXTRACT_ALL_SCENES_OBJECTS_GRASPS` | Bulk-download every scene/object/grasp, not just metadata for large datasets | `False` |
| `MUJOCO_EGL_DEVICE_ID` | Render device — indices do not match `CUDA_VISIBLE_DEVICES` (see issue #66) | `0` |
| `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` | Required for headless rendering on Linux | _(unset)_ |
| `FILAMENT_OPENGL_HANDLE_ARENA_SIZE_IN_MB` | Filament's GPU-handle arena size. **Does NOT fix the THORMAP segfault** (see Known issues) | `~8` (built-in) |

## Local data inventory (snapshot 2026-05-19)

Two parallel asset stores on disk. Re-verify with `du -sh ~/.molmospaces ~/.cache/molmospaces`.

### USD (Isaac) — `~/.molmospaces/usd/`, symlinked into `assets/usd/`

`du -sh assets/usd/` reports only the symlink entries (~4 KB). Use `du -shL` to follow symlinks, or measure the cache directly.

| Dataset | Real size |
|---|---|
| `objects/thor/20260128` | 3.0 GB |
| `objects/objaverse/20260128` | 24 GB |
| `scenes/ithor/20260121` | 323 MB |
| `scenes/procthor-10k-val/20260128` | 22 GB |
| `scenes/procthor-objaverse-val/20260128` | **176 GB** |
| **Total** | **224 GB** |

ProcTHOR scenes reference the standalone object library at runtime (`AddReference(rel_model_path)` in `molmo_spaces_isaac/src/molmo_spaces_isaac/assets/house_converter.py:740,777`) — they need `objects/thor` (and `objects/objaverse` for procthor-objaverse). iThor scenes bake geometry into their own `Payload/GeometryLibrary.usdc` and don't need the standalone library.

### MJCF (MuJoCo) — `~/.cache/molmospaces/assets/<base64url(project_path)>/`

In progress as of this snapshot — downloading via `python -m molmo_spaces.molmo_spaces_constants` (README's default install procedure). Last measured partial size: ~1.3 GB. The download covers scenes, objects, robots, benchmarks, grasps, datagen, test_data.

## Conventions

- Robot base frame: +x forward, +y left, +z up.
- Parallel-jaw gripper: +z forward, fingers open along y.
- `setuptools` packages only `molmo_spaces*` — the `molmo_spaces_isaac` and `molmo_spaces_maniskill` sub-packages are explicitly excluded from this install and ship separately.
- `ruff` lint excludes `scripts/`, `tests/`, and `mlspaces_tests/scenes/` (see `pyproject.toml`). Formatting is enforced repo-wide by CI.
- Type stubs live in `typings/`; regenerate with `pybind11-stubgen mujoco -o ./typings/`.
