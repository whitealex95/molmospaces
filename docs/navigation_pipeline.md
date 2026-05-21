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
              └▶ run_isaac.py      ─▶ ego_isaac.mp4  egocentric video (IsaacSim)

gen_trajectories.py orchestrates occupancy→plan→run_mujoco for N trajectories.
```

| Script | Role |
|---|---|
| `build_occupancy.py` | scene MJCF → 2D occupancy + room-label grid |
| `build_occupancy_usd.py` | scene **USD** → the same grid (for scenes whose MJCF won't compile) |
| `plan.py` | occupancy grid → clearance-aware A* path; reusable `Planner` class |
| `run_mujoco.py` | scene + path → kinematic G1 + egocentric/chase render (MuJoCo) |
| `run_isaac.py` | USD scene + path → G1 egocentric RGB+depth + chase render (IsaacSim) |
| `gen_trajectories.py` | batch driver: N trajectories/scene, previews + index |

The stages are **sim-agnostic by design**: the occupancy grid and the A* path
are built in one world frame, so the same `path.npz` drives both the MuJoCo and
the IsaacSim runtime.

## Inputs and environments

- **Scenes** are MJCF `.xml` files. Mansion scenes come from
  `scripts/mansion/mansion_to_mjcf.py` (see `docs/mansion_to_mjcf.md`);
  procthor scenes are fetched with the molmospaces resource manager
  (`ResourceManager.install_packages("scenes", {...})`).
- **Robot**: Unitree G1 — `~/Projects/CAMDM/PyTorch/visualize/assets/g1_29dof_rev_1_0.xml`.
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
5. `carve_doors` — a doorway has a lintel above it, so it reads as obstacle and
   gets sealed by the wall burn-in, but it is passable. Draw a *free* line along
   each door geom's footprint (`DOOR_CARVE_M` = 0.6 m wide).
6. Build `room_map` (per-pixel room index from the floor-geom segmentation) and
   `room_names` (`clean_room_name` strips a `floor_` prefix, a floor tag like
   `F1_`, and the procthor `_visual_N` geom suffix).

The stored occupancy is **raw** — agent-radius clearance is *not* baked in; it
is applied later, at planning time, so one grid serves any robot radius.

### Stage 1 (USD) — `build_occupancy_usd.py`

Some procthor-objaverse scenes have an incomplete MJCF (the objaverse MJCF
object library is a partial subset, so a missing object breaks compilation)
but a complete USD. `build_occupancy_usd.py` builds the *same* `occupancy.npz`
straight from the USD geometry: it reads the scene's `Geometry` scope, then
rasterizes the `room_N_visual_0` floor meshes (free + room id), the
`wall_*_visual_0` wall meshes (obstacle), the furniture/decor Xform bboxes
(obstacle), and carves the `doorway_*` footprints back open. Pure
`pxr` + numpy + cv2 — no render. Run it in the `mlspaces-isaac` env; the
output is byte-compatible, so `plan.py` and `run_isaac.py` are unchanged.

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
`save_path()` writes `path.npz` + a `path_debug.png` overlay.

CLI default: start = largest reachable room's anchor, goal = farthest reachable
room; `--start-room` / `--goal-room` override by name substring.

## Stage 3 — `run_mujoco.py`

Scene + `path.npz` → the egocentric/chase video.

1. **Merge the G1** — load scene and G1 as `MjSpec`s, absolutise the G1 mesh
   paths, then `scene.worldbody.add_frame().attach_body(g1_root, "g1_", "")`.
   Add an `ego` camera on `g1_torso_link` (pos `[0.12,0,0.42]`, quat
   `[0.5,0.5,-0.5,-0.5]` → looks +x forward, +z world-up). For the Filament
   backend also mount a forward fill light on the torso — Filament ignores the
   MuJoCo headlight, so an interior ego view would otherwise be near-black.
2. **Path → motion** — `resample_path` densifies the polyline to
   `SPEED/STEP_HZ` spacing (SPEED 1.0 m/s, STEP_HZ 30); `smooth_path` is a
   moving average (`SMOOTH_WINDOW` 45); `compute_yaws` takes the
   central-difference tangent and applies a wrap-aware low-pass (`YAW_ALPHA`
   0.2). Together these give smooth, non-snapping camera turns.
3. **Kinematic step loop** — per step set the G1 floating-base qpos to
   `[x, y, PELVIS_Z=0.793, *yaw_quat(yaw)]`, `mj_forward`, then render the
   ego RGB, ego depth, and a chase view (`mjCAMERA_TRACKING` on `g1_pelvis`).
   No physics — the base is placed directly.
4. **Renderer** — `--renderer opengl` uses `mujoco.Renderer` (run in
   `mlspaces-mujoco`); `--renderer filament` uses the molmospaces
   `MjFilamentRenderer` (run in `mlspaces`, physically based lighting).
5. **Outputs** — `combined.mp4` (2×2: top-down map | chase | ego RGB | ego
   depth) plus `ego_rgb.mp4`, `ego_depth.mp4`, `follow.mp4`,
   `combined_montage.png`. `write_video` encodes **H.264** (libx264, yuv420p,
   `+faststart`) by piping frames to `ffmpeg` — OpenCV's `mp4v` is MPEG-4
   Part 2 whose Simple Profile caps near 1280×720 and fails to play on the web
   for the 1280×960 panel. Falls back to `mp4v` only if `ffmpeg` is absent.

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
4. **Cameras** — the ego `Camera` sensor is a child of `/g1/pelvis/torso_link`
   with a fixed local transform (0.12 m forward + 0.42 m up, looking along the
   robot's +x), so it rides the robot like run_mujoco's torso-mounted camera.
   The chase `Camera` is at the root and re-aimed per frame.
5. **Drive + capture** — per smoothed pose, place `/g1` (translate + Z-rotate),
   aim `/chase_cam` via `set_camera_view`, `world.step(render=True)`, then read
   `get_rgba()` and the `distance_to_image_plane` depth from each sensor.
6. **Assemble** — H.264 via the system `/usr/bin/ffmpeg` (with
   `LD_LIBRARY_PATH` stripped — the isaacsim env's libs break it), **before**
   `app.close()` (fast-shutdown can hard-exit the process).

Output: `<out-dir>/{ego_isaac,depth_isaac,follow_isaac}.mp4`. A 2×2 combined
panel is not yet replicated for IsaacSim.

**Per-scene tuning** (applied in code, via the session layer — the USD asset is
never modified):
- *Light taming* — procthor USD scenes ship a 1000-intensity DomeLight +
  DistantLight that wash the render out; `tame_lights()` clamps them
  (`--dome-max`, `--distant-max`). The mansion USD has no lights of its own.
- *Chase camera* — `--chase-z` / `--chase-back`; procthor scenes have a
  ceiling, so the chase sits lower (e.g. `--chase-z 2.1 --chase-back 2.6`);
  the ceiling-less mansion uses the defaults (3.0 / 4.5).
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

`path.npz` — `waypoints` (N,2) world (x,y) start→goal; `start_room`, `goal_room`.

`index.json` (one per dataset) — `{dataset, updated, trajectories: [...]}`; each
trajectory: `id, scene, scene_xml, dir, start_room, goal_room, start_xy,
goal_xy, length_m, n_waypoints, cross_room, renderer, agent_radius,
topdown_png, combined_mp4`.

Output tree:
```
nav_runs/<dataset>/
  index.json
  <scene>/
    occupancy.npz  occupancy_debug.png
    NN__<start>__to__<goal>/
      path.npz  path_debug.png  topdown.png
      combined.mp4  ego_rgb.mp4  ego_depth.mp4  follow.mp4  combined_montage.png
```
`nav_runs/` lives at the repo root and is gitignored.

## Tuning constants

| Constant | File | Default | Effect |
|---|---|---|---|
| `--px-per-m` | build_occupancy | 100 | occupancy grid resolution |
| `WALL_THICKNESS_M` | build_occupancy | 0.12 | burned-in wall width |
| `DOOR_CARVE_M` | build_occupancy | 0.6 | doorway opening width |
| `--agent-radius` | plan / gen | 0.2 | obstacle dilation; keep modest so doorways stay open |
| `DOWNSCALE` | plan | 5 | occupancy cells per A* cell |
| `SPEED`, `STEP_HZ` | run_mujoco | 1.0, 30 | travel speed, render rate |
| `SMOOTH_WINDOW` | run_mujoco | 45 | path moving-average window |
| `YAW_ALPHA` | run_mujoco | 0.2 | heading low-pass (smaller = smoother) |
| `MIN_LENGTH_M` | gen_trajectories | 2.0 | reject degenerate samples |

## Gotchas

- **Symlinked scenes** — load with `abspath`, never `resolve()`; procthor
  scenes resolve `../../objects/thor` relative to the asset-store layout.
- **Floor detection** is name-boundary aware and requires `contype == 0` (the
  visual floor geoms); the procthor collision `floor` geom is excluded.
- **Walls / doors** must be burned in / carved out — a straight-down render
  alone misses vertical walls and seals doorways.
- **Filament ignores the MuJoCo headlight** — the run uses a robot-mounted fill
  light for the Filament backend.
- **Video codec** — emit H.264, not OpenCV `mp4v`; the 1280×960 panel exceeds
  MPEG-4 Simple Profile limits and will not play on the web / Notion.
- **Envs** — occupancy, planning and OpenGL rendering need `mlspaces-mujoco`;
  Filament rendering needs `mlspaces`. The pipeline never calls molmospaces
  `get_thormap`, so the Filament THORMAP segfault (issue #79) does not apply.
- **Single navigable room** — the driver falls back to within-room sampling
  rather than erroring; such trajectories are flagged `cross_room: false`.

## Running it

Batch (one scene), from a conda-equipped shell:
```bash
python scripts/navigation/gen_trajectories.py \
    --scene <scene.xml> --dataset <label> --count <N> [--renderer opengl|filament]
```

Manual, stage by stage, in `mlspaces-mujoco`:
```bash
python scripts/navigation/build_occupancy.py --scene <scene.xml> --out <dir>/occupancy.npz
python scripts/navigation/plan.py --occupancy <dir>/occupancy.npz --out <dir>/path.npz \
    [--start-room NAME --goal-room NAME]          # inspect <dir>/path_debug.png
python scripts/navigation/run_mujoco.py --scene <scene.xml> --path <dir>/path.npz \
    --occupancy <dir>/occupancy.npz --out-dir <dir>
```
