# Navigation pipeline

`scripts/navigation/` turns a scene into **cross-room navigation trajectories**:
a 2D occupancy map, an A* path between rooms, and a kinematic Unitree G1 robot
driven along that path while its egocentric RGB + depth cameras are rendered.
It runs unchanged on **mansion** floor exports and **procthor** scenes
(`procthor-10k`, `procthor-objaverse`).

This document is the design spec for the pipeline — detailed enough to
re-implement it from scratch. **Keep it in sync whenever you change
`scripts/navigation/`.**

## Pipeline at a glance

```
scene .xml ─▶ build_occupancy.py ─▶ occupancy.npz   2D occupancy + room labels
                                        │
occupancy.npz ─▶ plan.py (Planner) ─▶ path.npz       A* world-frame waypoints
                                        │
scene + path  ─▶ run_mujoco.py     ─▶ combined.mp4   G1 ego/chase video (MuJoCo)
              └▶ run_isaac.py      ─▶ isaac_combined.mp4  G1 ego/chase video (IsaacSim)

gen_trajectories.py orchestrates occupancy→plan→run_mujoco for N trajectories.
```

| Script | Role |
|---|---|
| `build_occupancy.py` | scene MJCF → 2D occupancy + room-label grid |
| `build_occupancy_usd.py` | scene **USD** → the same grid, USD-native (optional; no MJCF/MuJoCo) |
| `plan.py` | occupancy grid → clearance-aware A* path; reusable `Planner` class |
| `run_mujoco.py` | scene + path → kinematic G1 + egocentric/chase render (MuJoCo) |
| `run_mujoco_mm.py` | scene + path → **full-body motion-matched** G1 walk (MuJoCo); see [`navigation_motion_matching.md`](navigation_motion_matching.md) |
| `run_isaac.py` | USD scene + path → G1 egocentric RGB+depth + chase render (IsaacSim) |
| `gen_trajectories.py` | batch driver: N trajectories/scene, previews + index |

The stages are **sim-agnostic by design**: the occupancy grid and the A* path
are built in one world frame, so the same `path.npz` drives both the MuJoCo and
the IsaacSim runtime.

## Inputs and environments

- **Scenes (MJCF)** — `.xml` files driving `run_mujoco.py`. Mansion scenes come
  from `scripts/mansion/mansion_to_mjcf.py` (see `docs/mansion_to_mjcf.md`);
  procthor scenes are fetched with the molmospaces resource manager
  (`ResourceManager.install_packages("scenes", {...})`).
- **Scenes (USD)** — `.usda` files driving `run_isaac.py`:
  - mansion — `~/Projects/mansion/usd_export/<floorplan>/floor_<N>/scene.usda`
    (produced by `scripts/mansion/mansion_to_usd.py`, see `docs/mansion_to_usd.md`);
    e.g. `~/Projects/mansion/usd_export/public_healthcare_3f_300_fp001_0/floor_1/scene.usda`.
  - procthor — the molmospaces USD asset store at `~/.molmospaces/usd/scenes/`:
    `~/.molmospaces/usd/scenes/<procthor-10k-val|procthor-objaverse-val>/<version>/<scene>/scene.usda`
    (current version `20260128`).
- **Ceilinged variants** — both runtimes default to enclosed rooms (ceiling
  visible overhead). Pass the *_ceiling* variant of each scene:
  - **procthor** ships them per scene as `val_<N>_ceiling.xml` (MJCF) and
    `val_<N>_ceiling/scene.usda` (USD), with `ceiling_<roomid>_visual_0` meshes
    at the room's wall-top z. They share the room ids, doorways, and world
    frame of `val_<N>`, so the same `occupancy.npz` / `path.npz` drive both.
  - **mansion** has the ceiling baked into `scene.xml` / `scene.usda` since
    both converters emit `ceiling_<roomid>` per room — raise each `floorPolygon`
    to the global wall-top `y`. The USD ceilings are `doubleSided`.
- **Robot**: Unitree G1 — MJCF
  `~/Projects/CAMDM/PyTorch/visualize/assets/g1_29dof_rev_1_0.xml` (run_mujoco);
  USD `~/Projects/CAMDM/PyTorch/visualize/assets/g1_isaac/configuration/g1_base.usd`
  (run_isaac).
- **Conda envs**:
  - `mlspaces-mujoco` — occupancy building, planning, OpenGL rendering
    (stock MuJoCo + classic OpenGL renderer).
  - `mlspaces` — Filament rendering only (the Filament-built MuJoCo wheel
    lacks the classic OpenGL renderer).
  - `gen_trajectories.py` shells each sub-step to the correct env via `conda run`.

## Stage 1 — `build_occupancy.py`

Scene MJCF → `occupancy.npz` (a robot-agnostic 2D occupancy + room grid).

**Method:**
1. Compile the scene (`MjSpec.from_file().compile()`). Pass the scene path as
   `abspath`, **not** `resolve()` — procthor scenes are symlinked into the asset
   store and reference sibling dirs (`../../objects/thor`) via that layout;
   resolving the symlink breaks the relative mesh paths.
2. `classify_geoms` — split geoms by name:
   - **floor** = `is_floor(name)` AND `geom.contype == 0` (the *visual* floor
     geoms). `is_floor` matches at a name boundary — regex `floor($|[^a-z])` or
     `room[_|]\d` — so objaverse props like `floorlamp_*` / `room_divider_*` are
     not misread. procthor also has a *collision* `floor` geom (contype 8) which
     is deliberately excluded.
   - **door** = `"door"` in name; **wall** = `"wall"` in name.
3. `render_topdown` — an **orthographic top-down segmentation render** (camera
   elevation −90°, orthographic, framed to the floor geoms' AABB + 1 m buffer,
   `px_per_m` default 100). The segmentation image holds a geom id per pixel;
   pixels whose geom is a floor geom become free, everything else obstacle.
   Uses the molmospaces OpenGL renderer (`_get_renderer(use_filament=False)`).
4. `burn_walls` — vertical wall panels barely register in a straight-down
   render, and abutting room floors leave no gap, so interior walls are added
   explicitly: project each wall geom's vertices to the floor plane, take the
   principal-axis segment (SVD), draw it as a thick line (`WALL_THICKNESS_M`
   = 0.12 m) into the obstacle layer.
5. **Doors** — `classify_geoms` splits door geoms into **openings** (the static
   frame/threshold/lintel) and **leaves** (the swinging panel — a door geom whose
   body carries a hinge joint; procthor leaves default to ~90° **open**). The open
   **leaf** is burned in as an obstacle (`burn_walls(..., DOOR_LEAF_THICKNESS_M)`)
   so a path routes around the swung panel instead of clipping through it, while
   `carve_doors` draws a *free* line along each **opening** geom's footprint
   (`DOOR_CARVE_M` = 0.6 m) so the doorway itself stays passable. The leaf burn runs
   **after** the opening carve — otherwise the 0.6 m carve erases the part of the
   swung leaf nearest the hinge, leaving only a stub. Gated by **`--door-leaf-obstacles`**
   (`BooleanOptionalAction`, **default on**); `--no-door-leaf-obstacles` reverts to the
   legacy behaviour of carving every door geom (opening + leaf) fully free, so an open
   leaf leaves no obstacle and the kinematic, collision-free runtime lets the robot's
   body pass through it.
6. Build `room_map` (per-pixel room index from the floor-geom segmentation) and
   `room_names` (`clean_room_name` strips a `floor_` prefix, a floor tag like
   `F1_`, and the procthor `_visual_N` geom suffix).

The stored occupancy is **raw** — agent-radius clearance is *not* baked in; it
is applied later, at planning time, so one grid serves any robot radius.

### Stage 1 (USD) — `build_occupancy_usd.py`

`build_occupancy_usd.py` builds the *same* `occupancy.npz` straight from a
scene's USD geometry — an optional alternative to `build_occupancy.py` that
needs no MuJoCo and no MJCF. It reads the scene's `Geometry` scope and
rasterizes the `room_N_visual_0` floor meshes (free + room id), the
`wall_*_visual_0` wall meshes (obstacle), the furniture/decor Xform bboxes
(obstacle), and carves the `doorway_*` footprints back open. Pure
`pxr` + numpy + cv2 — no render. Run it in the `mlspaces-isaac` env; the
output is byte-compatible, so `plan.py` and `run_isaac.py` are unchanged.

Prefer the MJCF route (`build_occupancy.py`) when you can — one occupancy grid
then drives *both* sims. A procthor-objaverse MJCF only needs its objaverse
objects on disk first: they are not all bulk-downloaded by default, so install
them per scene with `install_scene_with_objects_and_grasps_from_path` (see
`docs/assets.md`). That clears the `Error opening file
.../objaverse/<uid>/<uid>_visual.obj` compile failure. Use
`build_occupancy_usd.py` for a USD-only workflow, or to skip MuJoCo entirely.

## Stage 2 — `plan.py` / `Planner`

`occupancy.npz` → `path.npz` (a clearance-aware A* path).

`Planner(occupancy_npz, agent_radius=0.2)` builds, once per (scene, radius):
1. **Clearance** — dilate obstacles by a circular kernel of radius
   `agent_radius * px_per_m`; a cell is free only if undilated.
2. **Downscale** by `DOWNSCALE` = 5 via min-pool (a coarse cell is free only if
   all 5×5 fine cells are free). `grid_spacing = DOWNSCALE / px_per_m`.
3. **Distance transform + graph** — `distance_transform_utils.make_distance_transform`
   then `make_grid_graph(grid, dt, weight_exp=2)`. The `weight_exp=2` edge
   weighting makes A* favour clearance, so paths keep away from walls.
4. **Rooms** — per-room navigable cells (downscaled `room_map` ∩ grid ∩ graph
   nodes); connected components give room-to-room reachability (`connectivity()`).

`plan(start_cell, goal_cell)` runs A* (`make_discrete_path`) and returns world
waypoints + length, or `None` if the cells are in different components.

**Path rounding** — raw grid A* produces a staircase of grid-aligned steps that
is rigid and awkward to follow. `save_path()` rounds it with
`smooth_waypoints()` — a clearance-checked moving average (`--smooth` window,
default 0.5 m): it densifies the path and averages each point over the window,
but **pulls any point that would land in the agent-dilated obstacle zone back
toward the raw path** until it is free again, so rounding never cuts a corner
through a wall. `path.npz` stores the **rounded** `waypoints` (what every runtime
follows) plus `waypoints_raw` (the original A* path); `path_debug.png` draws the
rounded path bold over the faint raw staircase. `--smooth 0` keeps the rigid
path. `gen_trajectories.py`'s `topdown.png` draws the same rounded path.

`save_path()` writes `path.npz` + a `path_debug.png` overlay.

CLI default: start = largest reachable room's anchor, goal = farthest reachable
room; `--start-room` / `--goal-room` override by name substring. `--resmooth
<path.npz>` re-rounds an *existing* path in place (using its `waypoints_raw`),
e.g. to re-round paths planned before smoothing was added or with a new window.

**Extra obstacle (`--obstacle`)** — `'x,y,sx,sy[,yaw_deg]'` stamps a (optionally
yaw-rotated) world rectangle (center xy, full sizes m) as occupied *before* the
clearance dilation, so A* must detour around it. Used to produce the navigation
"detour" variant: the same box the motion-matching jump variant leaps over
(`run_mujoco_mm.py --obstacle`) is stamped here so A* routes around it instead.
Align `yaw` with the local path heading and keep `sx` (depth along the path)
small so the box stays jumpable while `sy` (width across the path) blocks the
corridor. `Planner(..., obstacle=(x,y,sx,sy[,yaw_deg]))` exposes the same.

## Stage 3 — `run_mujoco.py`

Scene + `path.npz` → the egocentric/chase video.

1. **Merge the G1** — load scene and G1 as `MjSpec`s, absolutise the G1 mesh
   paths, then `scene.worldbody.add_frame().attach_body(g1_root, "g1_", "")`.
   Add an `ego` camera on `g1_torso_link` at the **Realsense D435 mount
   position** taken from the G1 URDF (`d435_link` fixed joint off
   `torso_link`), but with the URDF's downward pitch **removed** so the
   camera looks forward — the real D435 tilts down 47.6° for manipulation,
   but a forward-facing view is far more useful for navigation (the agent
   sees what's ahead instead of staring at its own arms and the floor).

   URDF reference:

   ```xml
   <!-- d435 -->
   <link name="d435_link"></link>
   <joint name="d435_joint" type="fixed">
     <origin xyz="0.0576235 0.01753 0.42987" rpy="0 0.8307767239493009 0"/>
     <parent link="torso_link"/>
     <child link="d435_link"/>
   </joint>
   ```

   We keep `xyz = [0.0576235, 0.01753, 0.42987]` (relative to `torso_link`)
   and drop the rpy. MuJoCo: `cam.pos = [0.0576235, 0.01753, 0.42987]`,
   `cam.quat = [0.5, 0.5, -0.5, -0.5]` (looks +x forward, +z world-up).
   Isaac: `EGO_LOCAL = Gf.Matrix4d(0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0,
   0.0576235, 0.01753, 0.42987, 1)` — same translation, identical horizontal
   orientation. The MuJoCo G1 MJCF (`g1_29dof_rev_1_0.xml`) and the Isaac G1
   USD both strip `d435_link` (the URDF→MJCF/USD conversion loses fixed-only
   links; the MJCF still declares `head_link.STL` as an unused mesh asset),
   so the camera mounts directly on `torso_link` with this offset baked in.

   For the Filament backend also mount a forward fill light on the torso at
   the same position, pointed slightly down (`lamp.dir = [1.0, 0, -0.15]`) —
   Filament ignores the MuJoCo headlight, so an interior ego view would
   otherwise be near-black.
2. **Path → motion** — `resample_path` densifies the polyline to
   `SPEED/STEP_HZ` spacing (SPEED 1.0 m/s, STEP_HZ 30); `smooth_path` is a
   moving average (`SMOOTH_WINDOW` 45); `compute_yaws` takes the
   central-difference tangent and applies a wrap-aware low-pass (`YAW_ALPHA`
   0.2). Together these give smooth, non-snapping camera turns.
3. **Kinematic step loop** — per step set the G1 floating-base qpos to
   `[x, y, PELVIS_Z=0.793, *yaw_quat(yaw)]`, `mj_forward`, then render the
   ego RGB, ego depth, and a chase view. The chase is a free
   (`mjCAMERA_FREE`) camera repositioned each frame so it sits close behind
   the robot and below the ceiling (`CHASE_BACK` 1.3 m, `CHASE_Z` 2.3 m,
   `CHASE_LOOK_Z` 0.9 m); this stays inside the ceilinged room and matches
   run_isaac. No physics — the base is placed directly.
4. **Renderer** — `--renderer opengl` uses `mujoco.Renderer` (run in
   `mlspaces-mujoco`); `--renderer filament` uses the molmospaces
   `MjFilamentRenderer` (run in `mlspaces`, physically based lighting).
5. **Outputs** — `combined.mp4` (2×2: top-down map | chase | ego RGB | ego
   depth) plus `ego_rgb.mp4`, `ego_depth.mp4`, `follow.mp4`,
   `combined_montage.png`. `write_video` encodes **H.264** (libx264, yuv420p,
   `+faststart`) by piping frames to `ffmpeg` — OpenCV's `mp4v` is MPEG-4
   Part 2 whose Simple Profile caps near 1280×720 and fails to play on the web
   for the 1280×960 panel. Falls back to `mp4v` only if `ffmpeg` is absent.

### Stage 3 (full-body motion matching) — `run_mujoco_mm.py`

A drop-in alternative to `run_mujoco.py` that renders a **walking gait** (all 36
G1 qpos articulated, driven by the motion-matching controller in
`~/Projects/motionmatching-g1`) instead of the frozen-legged kinematic slide. It
reuses the same Stage-1/2 `occupancy.npz` / `path.npz` and writes to a separate
`nav_runs_mm/` tree. Its full design spec lives in
[`navigation_motion_matching.md`](navigation_motion_matching.md).

## Stage 3 (IsaacSim) — `run_isaac.py`

The IsaacSim counterpart of `run_mujoco.py`: drives a Unitree G1 along the same
`path.npz` through a **USD** scene and renders the robot's egocentric RGB +
depth and a chase view. See `docs/isaac_navigation_log.md` for the full
diagnostic history.

1. **GUI mode, not headless** — `SimulationApp(headless=False)`. IsaacSim's
   headless camera-sensor API crashes here (`IRenderSettings ... stage-id`);
   GUI mode keeps the render path alive. It therefore needs a display — run
   with the Chrome Remote Desktop virtual display `:20`
   (`DISPLAY=:20 XAUTHORITY=$HOME/.Xauthority`), in the `mlspaces-isaac` env.
2. **Open the USD scene + add the G1** — `omni.usd` `open_stage`; the G1
   (`g1_isaac/g1.usd`, imported from MJCF with the `MJCFCreate*` kit commands)
   is referenced at the **stage root** `/g1`. The path is reused as-is: the
   mansion USD shares the mansion MJCF world frame.
3. **Stage-root rule** — the G1 and both cameras live at the stage root, never
   under `/World`: a converted scene's `/World` can carry a det-1 reflection
   transform (the mansion handedness fix) that IsaacSim's `XFormPrim` pose math
   rejects (`scipy Rotation.from_matrix`, non-positive determinant).
4. **Cameras — identical placement to run_mujoco.** The ego `Camera` sensor is
   a child of `/g1/pelvis/torso_link` with a fixed local transform (0.12 m
   forward + 0.42 m up, looking along the robot's +x), so it rides the robot
   like run_mujoco's torso-mounted camera. The chase `Camera` is repositioned
   each frame via `set_camera_view` to sit behind the robot and below the
   ceiling (`CHASE_BACK` 1.3 m, `CHASE_Z` 2.3 m, `CHASE_LOOK_Z` 0.9 m), which
   matches run_mujoco's interior chase exactly. Both cameras use a 45°
   vertical FOV (MuJoCo's default camera fovy). The G1 base sits at
   `G1_GROUND_Z` = 0.315 m so its soles rest on the z=0 floor (mansion and
   procthor both place the floor at z=0).
5. **Drive + capture** — per smoothed pose, place `/g1` (translate + Z-rotate),
   aim `/chase_cam` via `set_camera_view`, `world.step(render=True)`, then read
   `get_rgba()` and the `distance_to_image_plane` depth from each sensor.
6. **Assemble** — the 2×2 combined panel (top-down map | chase | ego RGB | ego
   depth) is built per frame, mirroring run_mujoco's `combined.mp4`. All
   streams are encoded to H.264 via the system `/usr/bin/ffmpeg` (with
   `LD_LIBRARY_PATH` stripped — the isaacsim env's libs break it), **before**
   `app.close()` (fast-shutdown can hard-exit the process).

Output: `<out-dir>/isaac_{ego,depth,follow,combined}.mp4` (+
`isaac_combined_montage.png`). The `isaac_` prefix sets them apart from the
MuJoCo renders at a glance.

**Per-scene tuning** (applied in code, via the session layer — the USD asset is
never modified):
- *Light taming* — procthor USD scenes ship a 1000-intensity DomeLight +
  DistantLight that wash the render out; `tame_lights()` clamps them
  (`--dome-max`, `--distant-max`). The mansion USD has no lights of its own.
- *Object references* — procthor USD scenes resolve furniture from
  `usd/scenes/objects/{thor,objaverse}`, which must be symlinked to
  `usd/objects/<src>/<version>` (filesystem only).

## Driver — `gen_trajectories.py`

Batches the three stages for one scene; run once per scene.

1. Build occupancy once (skipped if `occupancy.npz` already exists); construct a
   `Planner`.
2. `sample_trajectories(count, seed)` — from the largest mutually-reachable room
   group, form candidate **ordered room pairs** (distinct rooms; or
   within-room if the scene has a single navigable room). A reshuffled,
   refilling stream yields pairs — distinct pairs are preferred and repeats are
   allowed once exhausted. For each pair, sample random start/goal *positions*
   from the rooms' cells until A* succeeds and the path is ≥ `MIN_LENGTH_M`
   (2.0 m). The same room pair may recur with different positions.
3. **Two passes** (so previews land before the slow renders):
   - **Pass 1** — per trajectory write `path.npz`, a `topdown.png` preview
     (room-tinted map + trajectory + room labels), and append an index record;
     then write the per-dataset `index.json`.
   - **Pass 2** — render each trajectory's video via `run_mujoco.py`.
4. Continues past a failing scene; re-running a scene refreshes its index
   entries.

## Data formats

`occupancy.npz`
| key | shape / type | meaning |
|---|---|---|
| `occupancy` | (H,W) bool | True = free / navigable, raw (not dilated) |
| `room_map` | (H,W) int32 | per-pixel room index 1..R, 0 = none |
| `room_names` | (R,) str | `room_names[i-1]` names room index `i` |
| `world_to_map` | (2,4) float | `[x,y,z,1] → [row,col]` |
| `map_to_world` | (2,3) float | `[row,col,1] → [x,y]` |
| `px_per_m` | float | grid resolution |

`path.npz` — `waypoints` (N,2) world (x,y) start→goal, **rounded** (clearance-
checked); `waypoints_raw` (M,2) the raw grid A* path before rounding;
`start_room`, `goal_room`.

`index.json` (one per dataset) — `{dataset, updated, trajectories: [...]}`; each
trajectory: `id, scene, scene_xml, dir, start_room, goal_room, start_xy,
goal_xy, length_m, n_waypoints, cross_room, renderer, agent_radius,
topdown_png, combined_mp4`.

Output tree:
```
nav_runs/<dataset>/
  index.json                          # gen_trajectories: one per dataset
  <scene>/
    occupancy.npz  occupancy_debug.png
    NN__<start>__to__<goal>/          # gen_trajectories: one dir per trajectory
      path.npz  path_debug.png  topdown.png
      combined.mp4  ego_rgb.mp4  ego_depth.mp4  follow.mp4  combined_montage.png
```
`nav_runs/` lives at the repo root and is gitignored. The motion-matching
runtime writes to a parallel `nav_runs_mm/` tree — see
[`navigation_motion_matching.md`](navigation_motion_matching.md).

`run_isaac.py` writes `isaac_{ego,depth,follow,combined}.mp4` +
`isaac_combined_montage.png` into its `--out-dir`. It is invoked directly (not
by `gen_trajectories.py`), so placement follows `--out-dir`: in the batches run
so far that is each `NN__.../` trajectory subdir for mansion (one render per
trajectory, beside the MuJoCo files) and the `<scene>/` dir itself for procthor
(one render per scene, sharing that scene's `occupancy.npz` / `path.npz`).

## Tuning constants

| Constant | File | Default | Effect |
|---|---|---|---|
| `--px-per-m` | build_occupancy | 100 | occupancy grid resolution |
| `WALL_THICKNESS_M` | build_occupancy | 0.12 | burned-in wall width |
| `DOOR_CARVE_M` | build_occupancy | 0.6 | doorway opening width |
| `--agent-radius` | plan / gen | 0.2 | obstacle dilation; keep modest so doorways stay open |
| `--smooth` | plan / gen | 0.5 | path-rounding window (m); 0 = rigid grid A* path |
| `DOWNSCALE` | plan | 5 | occupancy cells per A* cell |
| `SPEED`, `STEP_HZ` | run_mujoco | 1.0, 30 | travel speed, render rate |
| `SMOOTH_WINDOW` | run_mujoco | 45 | path moving-average window |
| `YAW_ALPHA` | run_mujoco | 0.2 | heading low-pass (smaller = smoother) |
| `CHASE_BACK / CHASE_Z / CHASE_LOOK_Z` | run_mujoco, run_isaac | 1.3, 2.3, 0.9 | interior chase camera offset (must stay below the ceiling) |
| `G1_GROUND_Z` | run_isaac | 0.315 | G1 base height so soles rest on z=0 |
| `MIN_LENGTH_M` | gen_trajectories | 2.0 | reject degenerate samples |

## Gotchas

- **Symlinked scenes** — load with `abspath`, never `resolve()`; procthor
  scenes resolve `../../objects/thor` relative to the asset-store layout.
- **Floor detection** is name-boundary aware and requires `contype == 0` (the
  visual floor geoms); the procthor collision `floor` geom is excluded.
- **Walls / doors** must be burned in / carved out — a straight-down render
  alone misses vertical walls and seals doorways.
- **Path rounding ≠ runtime smoothing** — the rigid look of the *raw* grid A*
  path is rounded once at plan time (`save_path` → `path.npz` `waypoints`), so
  `path_debug.png` / `topdown.png` and every runtime follow the same rounded
  line. `run_mujoco.py` additionally moving-average-smooths at runtime (it
  re-smooths the already-rounded path, harmlessly); `run_mujoco_mm.py` follows
  the rounded `path.npz` directly. Rounding is clearance-checked, so it never
  cuts through a wall.
- **Filament ignores the MuJoCo headlight** — the run uses a robot-mounted fill
  light for the Filament backend.
- **Video codec** — emit H.264, not OpenCV `mp4v`; the 1280×960 panel exceeds
  MPEG-4 Simple Profile limits and will not play on the web / Notion.
- **Envs** — occupancy, planning and OpenGL rendering need `mlspaces-mujoco`;
  Filament rendering needs `mlspaces`. The pipeline never calls molmospaces
  `get_thormap`, so the Filament THORMAP segfault (issue #79) does not apply.
- **Single navigable room** — the driver falls back to within-room sampling
  rather than erroring; such trajectories are flagged `cross_room: false`.
- **Ceilings vs occupancy** — `build_occupancy.py` ignores geoms outside the
  floor / wall / door categories, so a `ceiling_*` mesh does not become an
  obstacle; one occupancy grid serves both ceilinged and uncovered variants of
  a scene. The chase camera height (`CHASE_Z` 2.3 m) is the only constant tied
  to ceiling height — raise it if a scene's ceiling sits above 2.9 m.
- **procthor-objaverse lazy install** — `val_<N>_ceiling.xml` is a symlink and
  may be missing under `scenes/procthor-objaverse-val/` after a cache prune.
  Call `install_scene_with_objects_and_grasps_from_path(val_<N>.xml)` to
  re-create it (the helper also pulls the scene's objaverse + grasp assets).

## Running it

Batch (one scene), from a conda-equipped shell:
```bash
python scripts/navigation/gen_trajectories.py \
    --scene <scene.xml> --dataset <label> --count <N> [--renderer opengl|filament]
```

Interactive viewer (needs `DISPLAY=:20` under Chrome Remote Desktop):
```bash
# MuJoCo viewer, optionally with the G1 merged in
conda activate mlspaces-mujoco
DISPLAY=:20 XAUTHORITY=$HOME/.Xauthority \
  python scripts/navigation/open_mujoco_gui.py <scene.xml> [--g1 [X Y]]

# IsaacSim viewer, optionally referencing the G1 USD
conda activate mlspaces-isaac
DISPLAY=:20 XAUTHORITY=$HOME/.Xauthority \
  python scripts/navigation/open_isaac_gui.py <scene.usda> [--g1 [X Y]]
```
`--g1` (no args) spawns the robot at the scene origin; pass `X Y` to place it
at an arbitrary xy. The MuJoCo viewer can switch to the `ego` camera (drop-down
in the side panel) to preview what `run_mujoco.py` records; the IsaacSim
viewport defaults to a free-fly camera you can drive around the scene.

Manual, stage by stage, in `mlspaces-mujoco`:
```bash
python scripts/navigation/build_occupancy.py --scene <scene.xml> --out <dir>/occupancy.npz
python scripts/navigation/plan.py --occupancy <dir>/occupancy.npz --out <dir>/path.npz \
    [--start-room NAME --goal-room NAME]          # inspect <dir>/path_debug.png
python scripts/navigation/run_mujoco.py --scene <scene.xml> --path <dir>/path.npz \
    --occupancy <dir>/occupancy.npz --out-dir <dir>
```

For the full-body **motion-matched** walk runtime (`run_mujoco_mm.py`), see
[`navigation_motion_matching.md`](navigation_motion_matching.md).

IsaacSim render — `run_isaac.py` consumes the **USD** scene and the *same*
sim-agnostic `path.npz` (occupancy + plan stages are unchanged). Run it in the
`mlspaces-isaac` env, with the `:20` display:
```bash
DISPLAY=:20 XAUTHORITY=$HOME/.Xauthority \
  conda run -n mlspaces-isaac python scripts/navigation/run_isaac.py \
    --scene <scene.usda> --path <dir>/path.npz --out-dir <dir>
```
`--occupancy` is auto-found next to `path.npz` (it feeds the combined map
panel). occupancy + path are sim-agnostic: a scene's MJCF-derived
`occupancy.npz` / `path.npz` (Stages 1–2) drive run_isaac directly, so MuJoCo
and IsaacSim share one grid. If a procthor-objaverse MJCF errors with a missing
`objaverse/<uid>` file, install the scene's objects first (see Stage 1 /
`docs/assets.md`). For a USD-only workflow, build the occupancy from the USD
instead (same `mlspaces-isaac` env, byte-compatible output):
```bash
conda run -n mlspaces-isaac python scripts/navigation/build_occupancy_usd.py \
    --scene <scene.usda> --out <dir>/occupancy.npz
```
