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

matcher = MotionMatcher(load_library())          # builds data/motion_lib.npz on first use (~5 MB), then caches
qpos = matcher.step(desiredVel, desiredFace)      # velocity-driven: advance one 30 Hz frame
qpos = matcher.step_targets(Tpos, Tdir)           # path-driven: future targets handed in directly
```

Internally `MotionMatcher.step` is **decoupled into two halves** (a non-breaking
refactor; `step`'s behaviour is byte-identical to before):

- `_predict_trajectory(desiredVel, desiredFace)` — spring-predict the future
  trajectory query (`Tpos` / `Tdir` at the `HORIZONS` taps) from a desired velocity.
- `_query_from_trajectory()` — match (per-clip KD-tree search) → advance the
  playhead → integrate the root → reconstruct the pose, from whatever query is set.

`step` runs *predict → jump-trigger → query*; `step_targets` runs *set-targets →
query*, reusing the same query half (no copy-paste). The two entry points map onto
the script's two `--query-mode`s:

- **`matcher.step(desiredVel, desiredFace)`** (`--query-mode velocity`) — `desiredVel`
  is a desired velocity `[vx, vy, 0]` in m/s (matcher frame); `desiredFace` is an
  independent facing direction `[fx, fy, 0]` (zero ⇒ face the travel direction). The
  matcher's springs predict the future-target query from it.
- **`matcher.step_targets(Tpos, Tdir)`** (`--query-mode targets`, **default**) — the
  future trajectory is supplied **directly**: `Tpos` `(len(HORIZONS), 3)` world-frame
  target positions and `Tdir` the matching facing directions, one per horizon tap.
  No velocity middle-man — you hand in *where the character should be along the path*.

Both return the 36-D MuJoCo `qpos`: `qpos[0:3]` pelvis position, `qpos[3:7]` pelvis
quaternion (wxyz), `qpos[7:36]` the 29 joint angles (radians, declaration order), and
both update `matcher.Tpos` (the query trajectory, read back for the overlay).

- **`matcher.rootPos` / `matcher.rootYaw`** — the controller's smoothed ground
  root (xy + heading) that the script reads each frame to steer and to map poses
  back into the scene.

Fixed at **30 Hz** (`STEP_HZ`); one `step()` = one rendered frame.

## How the A\* path drives the matcher

The matcher runs its **own** world root from the motion database — it cannot be
teleported onto an arbitrary path. `run_mujoco_mm.py` bridges the gap with a
rigid anchor plus one of two steering modes (`--query-mode`).

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
robot reproduces the path *shape* in the scene frame. The anchor `s0` (the path's
first waypoint) is closed over by `m2s_xy` / `s2m_xy` — do **not** shadow that
name with a local (an earlier targets-mode bug reassigned `s0` to a scalar
arc-length, silently corrupting the whole scene↔matcher map; the projection
arc-length is now named `s_proj`).

### 2a. Target steering (`--query-mode targets`, default)

The cleaner path-follower: instead of a velocity, hand the matcher the future
**trajectory** it would otherwise spring-predict. Each frame the script:

1. Densifies the planned path to `PATH_STEP_M` (0.1 m) spacing and finds the
   robot's closest path point (its **projection**, arc-length `s_proj`).
2. Samples three **centerline** points ahead of the projection at
   `s_proj + --speed × [1/3, 2/3, 1] s` — the matcher's three `HORIZONS` taps.
3. **Off-path merge blend:** adds the robot's residual cross-track offset
   `err = robot − projection` to each centerline point, weighted `w: 1 → 0`
   linearly over `CONVERGE_TIME_S` (`w = max(0, 1 − t / CONVERGE_TIME_S)`). The
   near tap keeps most of the offset (sits beside the robot, a little ahead); the
   far tap drops to zero (on the line). Each facing is taken from the resulting
   approach chain `robot → t₁ → t₂ → t₃`, so the heading curves diagonally in and
   straightens onto the path.
4. Maps the targets + facings into the matcher frame (`T⁻¹`) and feeds them
   straight to `matcher.step_targets(Tpos, Tdir)`.

Why the blend: sampling **pure centerline** points (no blend) silently assumes
zero lateral error — the whole `--speed × t` budget goes to along-path progress,
so a robot 0.5 m off the line is told to be *on* the line 1/3 s ahead, i.e. an
≈1.8 m/s instantaneous snap (and the nearest tap may not even be reachable). The
decaying-offset blend turns that into a smooth diagonal **merge** spread over
`CONVERGE_TIME_S` (the same case drops to ≈1.1 m/s, uniform across taps). When the
robot is on the line (`err ≈ 0`) the blend is a no-op and step 3 reduces exactly
to the centerline samples. In practice it **tracks the path tightly** (≈0.1 m
mean, ≤0.4 m max lateral drift on the val_2 example) — much closer than velocity
mode — while degrading gracefully from large off-path states.

### 2b. Pure-pursuit velocity steering (`--query-mode velocity`)

The original mode. Each frame the script:

1. Densifies the planned path to `PATH_STEP_M` (0.1 m) spacing.
2. Finds the **lookahead point** by the classic pure-pursuit rule
   (`pure_pursuit_target`): the forward intersection of the circle of radius
   `L = --speed × LOOKAHEAD_TIME_S` (default 1.0 s, so ≈1.0 m at 1.0 m/s — tied
   to the matcher's 1 s trajectory horizon) about the robot with the planned
   path. Scanning forward from the closest path point, it is the first point at
   distance ≥ L. If the robot has drifted >L off the path it targets the nearest
   path point (steer back on); near the goal it targets the final waypoint.
3. Sets `desiredVel` toward that lookahead (in the matcher frame via `T⁻¹`),
   scaled to `--speed`, and calls `matcher.step(desiredVel, 0)`. `desiredFace` is
   left zero so the robot faces its travel direction. The matcher's springs turn
   this velocity into the future-target query — the robot is never driven *to*
   the lookahead point; it only sets the instantaneous desired heading.

**Drift is larger and expected here** — velocity steering does not track the line
exactly. Either way the top-down panel draws **both** the planned path (orange
polyline) and the robot's *actual* position (red dot), so any gap is visible
rather than hidden.

### Control overlay — drawn in-scene, in the overhead chase

The target path and the motion-matching command are drawn as **real 3D geometry
inside the MuJoCo scene** (appended to the renderer's `MjvScene` via
`mjv_initGeom` / `mjv_connector` each frame), so they appear in the rendered
**overhead chase** view (top-right panel, labelled `Chase: target + MM command`)
— not as a flat overlay on the 2D map. The geoms are injected **only for the
chase camera**, so the ego RGB + depth streams stay clean (no markers polluting
the depth). Both renderer backends draw them: the OpenGL `mujoco.Renderer` and
the Filament renderer both rasterize the `MjvScene` via `mjr_render`.

- **orange floor strip** — the (rounded) planned path, a capsule chain on the
  floor (`PATH_Z` 0.04 m, subsampled to ~0.2 m segments).
- **green sphere** — the pure-pursuit **lookahead target** on the planned path.
- **yellow arrow** — the **command input** `desiredVel` handed to `matcher.step`
  (the direction the controller is told to go, ~0.8 m long).
- **magenta spheres + line** — the matcher's own **predicted command trajectory**
  (`matcher.Tpos`, its critically-damped spring prediction at the `HORIZONS`
  taps), mapped from the matcher frame into the scene. This is what the search
  query is built against, so it shows where the controller "thinks" it is headed.

The 2D **top-down map** panel (top-left) keeps just the baked planned path plus
the robot's actual pelvis position + heading marker. Colours of the in-scene
geoms are set in `run_mujoco_mm.py` (`PATH_RGBA` / `TARGET_RGBA` / `CMD_RGBA` /
`TPOS_RGBA`, RGBA 0–1); heights via `PATH_Z` / `TPOS_Z` / `TARGET_Z` / `CMD_Z`.

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
  motion_full.npz  motion_min.npz                       # saved motion (below)
```

## Saved motion: full sequence vs minimal control stream

Each run also saves the produced motion in **two** representations so you can
measure how much the motion-matching controller compresses a walk. The point:
the dense per-frame body pose is largely *implied* by a short path plus the few
motion-library segments the matcher stitched together.

**`motion_full.npz`** — the complete per-frame motion:

| key | shape / type | meaning |
|---|---|---|
| `qpos` | (T,36) float32 | scene-frame full-body pose per frame (7 root + 29 joints) |
| `command_vel` | (T,2) float32 | the `desiredVel` command fed to the matcher each frame |
| `db_index` | (T,) int64 | global motion-DB frame index each pose came from |
| `fps` | float32 | 30 |

**`motion_min.npz`** — the minimal control stream the full motion reconstructs from:

| key | shape / type | meaning |
|---|---|---|
| `trajectory` | (N,2) float32 | the A* path (raw waypoints) — *where to go* |
| `segments` | (S,4) int64 | `[clip_id, start_frame_in_clip, n_steps, is_jump]` per run |
| `clip_names` | (n_clips,) str | name for each `clip_id` |
| `dtheta`,`m0`,`s0`,`start_frame` | scalars/(2,) | anchor transform (matcher frame → scene) |
| `fps` | float32 | 30 |

`segments` is a **run-length encoding of `db_index`**: between searches the
matcher just advances its playhead by +1, so a contiguous same-clip run is
exactly "*play clip C from frame F and step forward N frames*". A search cut (or
a triggered jump) starts a new segment; `is_jump` flags jump-skill segments
(`every jump`). So a whole trajectory collapses to the path plus a handful of
`(clip, start-frame, n-steps-forward, jump?)` rows.

At the end of a run the script prints the comparison — logical (uncompressed
array bytes, the fair metric) and on-disk (`.npz`) sizes plus the ratio. For the
example 889-frame walk: full ≈ 125 KiB vs minimal ≈ 3.4 KiB logical → **~37×**
(≈ 51× on disk). The full `qpos` reconstructs by replaying the `segments`
against the motion library (joint angles come straight from the indexed DB
frames; the root re-integrates under the same command), placed into the scene by
the stored transform.

The `occupancy.npz` / `path.npz` are read straight from the existing
`nav_runs/<dataset>/<scene>/...`, so no occupancy/plan rebuild is needed.

## Tuning constants

| Constant | Default | Effect |
|---|---|---|
| `--query-mode` | `targets` | `targets` (path trajectory → `step_targets`, tight tracking) or `velocity` (pure-pursuit → `step`) |
| `--speed` / `WALK_SPEED` | 1.0 | desired travel speed fed to the matcher (m/s); sets target spacing (targets) or `desiredVel` magnitude (velocity) |
| `LOOKAHEAD_TIME_S` | 1.0 | pure-pursuit lookahead time (velocity mode only); circle radius L = `--speed` × this |
| `PATH_STEP_M` | 0.1 | densification spacing for the target/pursuit path |
| `CONVERGE_TIME_S` | 1.0 | targets mode: horizon over which an off-path lateral offset decays to zero (merge-onto-path rate); larger ⇒ gentler/more feasible cut-in |
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
- **Drift, not a bug** — in `velocity` mode the robot will not hug the planned
  line; the top-down panel intentionally shows planned (orange) vs actual (red).
  `targets` mode (the default) tracks the line tightly (≈0.1 m), but the same
  panel still shows both so any residual gap is visible.
- **Don't shadow `s0`** — the anchor translation is closed over by `m2s_xy` /
  `s2m_xy`; reusing the name for a local (e.g. an arc-length scalar) silently
  breaks the scene↔matcher map. The projection arc-length is `s_proj`.
- **Shared frame with the kinematic runtime** — occupancy + path are
  sim/runtime-agnostic, so the same `path.npz` drives `run_mujoco.py`,
  `run_mujoco_mm.py`, and `run_isaac.py` alike.

## Running it

```bash
# opengl (mlspaces-mujoco) — reuses the existing occupancy/path; NON-ceiling scene.
# Default --query-mode is `targets` (path trajectory -> step_targets, tight tracking).
conda run -n mlspaces-mujoco python scripts/navigation/run_mujoco_mm.py \
    --scene  <…/scenes/<dataset>/val_<N>.xml>            `# NON-ceiling variant` \
    --path   nav_runs/<dataset>/val_<N>/NN__.../path.npz \
    --occupancy nav_runs/<dataset>/val_<N>/occupancy.npz \
    --out-dir   nav_runs_mm/<dataset>/val_<N>/NN__... \
    [--speed 1.0] [--query-mode velocity]    `# add this to use pure-pursuit instead`

# filament (mlspaces) — better RGB
conda run -n mlspaces python scripts/navigation/run_mujoco_mm.py \
    --scene <…/val_<N>.xml> --path <…/path.npz> \
    --occupancy <…/occupancy.npz> --out-dir <nav_runs_mm/…> --renderer filament
```

Example generated so far: procthor-10k-val `val_2`, room-2 → room-11 (rounded
~25 m path) → `nav_runs_mm/procthor-10k-val/val_2/01__room-2__to__room-11/`. In
the default `targets` mode at `--speed 1.0` (with the off-path merge blend): 768
frames, 1280×960 H.264; 65 motion segments; lands 0.07 m from the goal with ≈0.1 m
mean (≤0.4 m max) path drift. (The older `velocity`-mode pure-pursuit run of the
same route was 889 frames / 99 segments with looser drift.)
