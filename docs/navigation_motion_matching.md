# Navigation with motion matching

`scripts/navigation/run_mujoco_mm.py` is a full-body **motion-matching** runtime
for the navigation pipeline: it drives a Unitree G1 along an A\* path with a real
**walking gait** — swinging legs, arm sway, pelvis bob — instead of the
frozen-leg kinematic slide of `run_mujoco.py`.

It is a **drop-in alternative to Stage 3** (`run_mujoco.py`) of the navigation
pipeline. Stages 1–2 (`build_occupancy.py` → `plan.py`) are unchanged and
shared: `run_mujoco_mm.py` consumes the same `occupancy.npz` / `path.npz`. Read
[`docs/navigation_pipeline.md`](navigation_pipeline.md) first for those stages,
the occupancy/path data formats, and the kinematic runtime this replaces.
**Keep this doc in sync whenever you change `run_mujoco_mm.py`.**

```
occupancy.npz + path.npz  ─▶ run_mujoco_mm.py ─▶ combined.mp4   full-body G1 walk
   (Stages 1–2, shared)        (this doc)                       (nav_runs_mm/)
```

| | `run_mujoco.py` (kinematic) | `run_mujoco_mm.py` (motion matching) |
|---|---|---|
| Robot body | floating base only; legs/arms frozen | **all 36 qpos** articulated |
| G1 model | CAMDM `g1_29dof_rev_1_0.xml` | **Menagerie** `g1.xml` (matcher's model) |
| Drive | base slid along the smoothed path | matcher stepped by velocity command |
| Path tracking | exact (base placed on the path) | approximate (matcher drifts; realistic) |
| Chase camera | interior, ~−50° (`CHASE_Z` 2.3 m) | **near-overhead**, ~−70° (`CHASE_Z` 5 m) |
| Output tree | `nav_runs/` | `nav_runs_mm/` (separate, gitignored) |

## The motion-matching controller

The motion is produced by the standalone repo at **`~/Projects/motionmatching-g1`**
(`mm_g1.controller.MotionMatcher`) — a port of GenoView / Daniel Holden's "Simple
Motion Matching" over a **GMR-retargeted LAFAN1** walk/run/stumble library. The
controller is pure **numpy + scipy (`cKDTree`) + mujoco**, so it runs in either
the `mlspaces-mujoco` or `mlspaces` conda env.

API used by `run_mujoco_mm.py`:

```python
sys.path.insert(0, str(MM_ROOT))
from mm_g1.controller import MotionMatcher
from mm_g1.data import load_library

matcher = MotionMatcher(load_library())        # builds data/motion_lib.npz on first use (~5 MB), then caches
qpos = matcher.step(desiredVel, desiredFace)    # advance one 30 Hz frame; returns world-frame (36,) qpos
```

- **`matcher.step(desiredVel, desiredFace)`** — `desiredVel` is a desired velocity
  `[vx, vy, 0]` in m/s (world frame); `desiredFace` is an independent facing
  direction `[fx, fy, 0]` (zero ⇒ face the travel direction). It integrates the
  matcher's **own** world root from the motion database and returns the 36-D
  MuJoCo `qpos`: `qpos[0:3]` pelvis position, `qpos[3:7]` pelvis quaternion
  (wxyz), `qpos[7:36]` the 29 joint angles (radians, declaration order).
- **`matcher.rootPos` / `matcher.rootYaw`** — the controller's smoothed ground
  root (xy + heading) that the script reads each frame to steer and to map poses
  back into the scene.

Fixed at **30 Hz** (`STEP_HZ`); one `step()` = one rendered frame.

## How the A\* path drives a velocity-controlled character

The matcher is **velocity-controlled** — it cannot be teleported onto an
arbitrary path. `run_mujoco_mm.py` bridges the gap in two parts:

### 1. Rigid frame anchoring (matcher frame ↔ scene frame)

The matcher resets to some world pose taken from its motion DB, unrelated to the
scene. The script builds a **rigid 2D transform `T`** (z-rotation `dθ` +
xy-translation) that maps the matcher's reset pose onto the path's first
waypoint with the path's initial heading:

- `dθ = sy0 − my0`, where `my0 = matcher.rootYaw` at reset and `sy0` is the
  initial tangent of the planned path (`atan2` over the first ~8 densified pts).
- **matcher → scene** (applied to every output pose): `xy = R(dθ)·(p − m0) + s0`;
  the root quaternion is left-multiplied by `yaw_quat(dθ)`; joints are untouched;
  z is untouched.
- **scene → matcher** (`T⁻¹`, used to phrase steering targets in the matcher
  frame): `xy = R(−dθ)·(p − s0) + m0`.

Because `T` is rigid, headings and velocities transform consistently, so the
robot reproduces the path *shape* in the scene frame.

### 2. Pure-pursuit steering

Each frame the script:

1. Densifies the planned path to `PATH_STEP_M` (0.1 m) spacing.
2. Finds the closest densified point to the robot's current scene position, then
   targets a **lookahead point** `PURSUIT_LOOKAHEAD_M` (0.8 m) further along.
3. Sets `desiredVel` toward that target (in the matcher frame via `T⁻¹`), scaled
   to `--speed` (default `WALK_SPEED` = 1.3 m/s). `desiredFace` is left zero so
   the robot faces its travel direction.

**Drift is expected and realistic** — motion matching does not track the line
exactly. The top-down panel therefore draws **both** the planned path (orange
polyline) and the robot's *actual* position (red dot), so the gap is visible
rather than hidden.

**Arrival / termination** — when the robot is within `ARRIVE_TOL_M` (0.4 m) of
the final waypoint, `desiredVel` goes to zero and the gait settles for
`SETTLE_FRAMES` (45) before the run ends. A `--max-frames` cap (auto-sized from
path length × 2 + margin, or set explicitly) guards against a stuck pursuit and
is reported if hit.

## The merged G1 — Menagerie, not CAMDM

`run_mujoco_mm.py` attaches **`~/Projects/motionmatching-g1/assets/unitree_g1/g1.xml`**
(MuJoCo Menagerie's `unitree_g1`), **not** the CAMDM `g1_29dof_rev_1_0.xml` that
`run_mujoco.py` uses. This is mandatory: the matcher's 36-D `qpos` is authored
for the Menagerie joint order, so the legs would be scrambled if set into a
model with a different joint layout.

Conveniently, the Menagerie model shares the **`pelvis` / `floating_base_joint`
/ `torso_link`** body names, so after `frame.attach_body(g1, "g1_", "")` the
ego-camera mount (`g1_torso_link`, the D435 offset `[0.0576, 0.0175, 0.4299]`
looking +x forward) and the Filament fill light are **byte-identical** to
`run_mujoco.py`. Mesh paths are absolutised the same way (`meshdir="assets"`
relative to the g1.xml dir).

Per frame the script sets the full attached-body slice — `data.qpos[base_adr :
base_adr + 36] = qscene` (base_adr from `model.joint("g1_floating_base_joint")`)
— then `mj_forward` so the torso-mounted ego camera rides along. No physics; the
pose is placed directly (the matcher already provides a balanced gait + the real
pelvis z, so `PELVIS_Z` is *not* hard-coded as in the kinematic runtime).

## The more-overhead chase camera

The chase camera sits higher and steeper than `run_mujoco.py` so the full-body
gait **and** the route are both clearly visible:

| Constant | `run_mujoco_mm.py` | `run_mujoco.py` |
|---|---|---|
| `CHASE_BACK` | 1.5 m | 1.3 m |
| `CHASE_Z` | **5.0 m** | 2.3 m |
| `CHASE_LOOK_Z` | 0.9 m | 0.9 m |
| ⇒ elevation | ≈ **−70°** | ≈ −50° |

**Use the NON-ceiling scene variant** (`val_<N>.xml`, not `val_<N>_ceiling.xml`)
— at 5 m the chase camera is above the ~2.9 m ceiling, which would otherwise
occlude the whole view. Raise/lower `CHASE_Z` to taste (the robot reads small at
5 m; drop it for a tighter frame).

## Renderer and outputs

`--renderer` selects the backend, exactly as in `run_mujoco.py`:

- `opengl` — stock `mujoco.Renderer`; run in **`mlspaces-mujoco`**.
- `filament` — molmospaces `MjFilamentRenderer` (PBR lighting + shadows); run in
  **`mlspaces`**. A forward fill light is mounted on the torso (Filament ignores
  the MuJoCo headlight).

Outputs use the **same filenames** as `run_mujoco.py` so downstream tooling is
unchanged: `combined.mp4` (2×2: top-down map | overhead chase | ego RGB | ego
depth), plus `ego_rgb.mp4`, `ego_depth.mp4`, `follow.mp4`, and
`combined_montage.png`. Encoding is H.264 (libx264, yuv420p, `+faststart`) via
`ffmpeg`, falling back to OpenCV `mp4v` only if `ffmpeg` is absent.

By convention they land in a **parallel tree** `nav_runs_mm/` (gitignored) so the
motion-matched renders never overwrite the kinematic `nav_runs/` ones:

```
nav_runs_mm/<dataset>/<scene>/NN__<start>__to__<goal>/
  combined.mp4  ego_rgb.mp4  ego_depth.mp4  follow.mp4  combined_montage.png
```

The `occupancy.npz` / `path.npz` are read straight from the existing
`nav_runs/<dataset>/<scene>/...`, so no occupancy/plan rebuild is needed.

## Tuning constants

| Constant | Default | Effect |
|---|---|---|
| `--speed` / `WALK_SPEED` | 1.3 | desired travel speed fed to the matcher's velocity springs (m/s) |
| `PURSUIT_LOOKAHEAD_M` | 0.8 | pure-pursuit lookahead along the densified path |
| `PATH_STEP_M` | 0.1 | densification spacing for the pursuit target |
| `ARRIVE_TOL_M` | 0.4 | radius around the final waypoint counting as arrived |
| `SETTLE_FRAMES` | 45 | extra frames (`desiredVel`=0) after arrival so the gait settles |
| `--max-frames` | auto | hard frame cap (0 ⇒ path-length × 2 + 120) |
| `CHASE_BACK / CHASE_Z / CHASE_LOOK_Z` | 1.5, 5.0, 0.9 | near-overhead chase offset (needs the non-ceiling scene) |
| `STEP_HZ` | 30 | matcher data + render rate (fixed; the matcher is 30 fps) |

## Gotchas

- **Menagerie G1 only** — the matcher's qpos joint order is the Menagerie one;
  do not point `--g1` at the CAMDM model or the limbs will be scrambled.
- **Non-ceiling scene** — the 5 m chase camera is above the ceiling; pass
  `val_<N>.xml`, not the `_ceiling` variant.
- **`~/Projects/motionmatching-g1` must be on disk** — the script inserts it on
  `sys.path` and imports `mm_g1`; the first run builds `data/motion_lib.npz`.
- **Drift, not a bug** — the robot will not hug the planned line; the top-down
  panel intentionally shows planned (orange) vs actual (red).
- **Shared frame with the kinematic runtime** — occupancy + path are
  sim/runtime-agnostic, so the same `path.npz` drives `run_mujoco.py`,
  `run_mujoco_mm.py`, and `run_isaac.py` alike.

## Running it

```bash
# opengl (mlspaces-mujoco) — reuses the existing occupancy/path; NON-ceiling scene
conda run -n mlspaces-mujoco python scripts/navigation/run_mujoco_mm.py \
    --scene  <…/scenes/<dataset>/val_<N>.xml>            `# NON-ceiling variant` \
    --path   nav_runs/<dataset>/val_<N>/NN__.../path.npz \
    --occupancy nav_runs/<dataset>/val_<N>/occupancy.npz \
    --out-dir   nav_runs_mm/<dataset>/val_<N>/NN__... \
    [--speed 1.3]

# filament (mlspaces) — better RGB
conda run -n mlspaces python scripts/navigation/run_mujoco_mm.py \
    --scene <…/val_<N>.xml> --path <…/path.npz> \
    --occupancy <…/occupancy.npz> --out-dir <nav_runs_mm/…> --renderer filament
```

Example generated so far: procthor-10k-val `val_2`, room-2 → room-11 (27 m) →
`nav_runs_mm/procthor-10k-val/val_2/01__room-2__to__room-11/` (772 frames,
1280×960 H.264).
