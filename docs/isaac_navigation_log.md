# IsaacSim navigation — trial log

Working log for getting the navigation pipeline (see `navigation_pipeline.md`)
rendering in IsaacSim. Updated as attempts are made, so the effort can resume
after a session ends or the user steps away.

## Goal

An IsaacSim runtime (`run_isaac.py`) that loads a scene, drives a camera (and
later a G1 robot) along an A* path (`path.npz`), and produces an egocentric
navigation video — the IsaacSim counterpart of `run_mujoco.py`.

## Constraints / environment facts

- GPU (RTX 4090) is shared and will **not** be freed — a ~12 GB training job
  runs concurrently. Every IsaacSim run is wrapped in a hard `timeout`
  (~4–5 min); a run exceeding that is treated as failed.
- IsaacSim env: `mlspaces-isaac` conda env.
- Headless camera-sensor rendering crashed in earlier sessions
  (`IRenderSettings::getRenderSettings failed getting a stage-id`).
- GUI mode works: `open_isaac_gui.py` launched fine (~8 s to startup).
- Display: Chrome Remote Desktop virtual X display is **`:20`** (1600×1200,
  auth `/home/jkim3662/.Xauthority`). It persists whether or not the user is
  connected via CRD.
- **Screen recording of `:20` works** — `ffmpeg -f x11grab -i :20` produces
  valid H.264 and the frames are readable. This is the fallback capture path.
- Mansion USD: `~/Projects/mansion/usd_export/public_healthcare_3f_300_fp001_0/floor_1/scene.usda`
  (same world frame as the mansion MJCF, so the existing `path.npz` aligns).
- procthor USD: `~/.molmospaces/usd/scenes/{procthor-10k-val,procthor-objaverse-val,ithor}`.
- No G1 USD on disk. Plan: for the egocentric view, move a *camera* along the
  path (no robot mesh needed); add a robot mesh only for a chase view.

## Strategy — capture path, in priority order

1. GUI mode + per-frame viewport capture (`capture_viewport_to_file`).
2. GUI mode + camera-sensor API.
3. GUI mode + `ffmpeg x11grab` screen recording of `:20`  ← most robust fallback.

## Trials

| # | Date | Approach | Result | Notes |
|---|------|----------|--------|-------|
| 0 | 2026-05-21 | Verify `ffmpeg x11grab` of `:20` | ✅ works | 1600×1200 H.264, frames readable |
| 1 | 2026-05-21 | GUI: load mansion USD, frame camera, screen-record `:20` | ✅ works | mansion USD loaded (1844 prims, bbox 19×17×3.5 m), viewport renders, screen-record captures it. No crash. |
| 2 | 2026-05-21 | GUI: fly egocentric camera along A* path, `capture_viewport_to_file` per frame | ✅ works | 173 ego frames captured in 16 s, assembled to H.264 video. Frames 0/160 are clean egocentric room views; ~a few frames clip a door panel (kinematic camera, no collision — same artifact as run_mujoco). |
| 3 | 2026-05-21 | Promote to `scripts/navigation/run_isaac.py` (self-contained: scene + path → `ego_isaac.mp4`) | ✅ works | Two bugs found + fixed: (a) ffmpeg ran after `app.close()`, which IsaacSim fast-shutdown hard-exits past — assemble *before* `close()`; (b) ffmpeg is not on the isaacsim env PATH and conda libs break the system binary — call `/usr/bin/ffmpeg` with `LD_LIBRARY_PATH` stripped. Verified on mansion trajectories 01 (173 frames) and 02 (441 frames). |
| 4 | 2026-05-21 | Batch all 10 mansion trajectories | ✅ works | 10/10 `ego_isaac.mp4` rendered (165–538 frames each, ~15–44 s capture). |
| 5 | 2026-05-21 | procthor `val_308` (USD + MJCF both on disk) — build occupancy/plan from MJCF, render in IsaacSim from USD | ✅ works (render washed-out) | procthor MJCF and USD **do share the world frame** — the ego camera flies the path correctly inside the building (floor / walls / windows / doors all in place). 438 frames, `ego_isaac.mp4` written. But the procthor USD renders **washed-out / over-bright** (pale hazy walls) where the mansion USD rendered cleanly — a procthor-USD lighting/materials issue, independent of `run_isaac.py`. |

## Key findings — IsaacSim rendering SOLVED

- IsaacSim **GUI mode on `:20` works** and renders USD scenes (startup ~4 s).
  The headless camera-sensor crash is dodged entirely by using GUI mode.
- **`capture_viewport_to_file` is the capture method** — clean per-frame PNGs,
  sim-locked, no UI chrome, ~0.1 s/frame. No screen recording needed after all
  (it stays as a proven fallback only).
- Approach that works: `SimulationApp(headless=False)` on `:20` → `open_stage`
  → per A* pose `set_camera_view(eye, target)` + `app.update()` +
  `capture_viewport_to_file` → assemble PNGs to H.264 with ffmpeg.
- Minor artifact: the kinematic camera clips door panels for a few frames when
  crossing a doorway (no collision). Acceptable — same as run_mujoco.

## Status — SOLVED for mansion

`scripts/navigation/run_isaac.py` produces IsaacSim egocentric navigation
videos. Run it with the display env set:

```
DISPLAY=:20 XAUTHORITY=$HOME/.Xauthority \
  conda run -n mlspaces-isaac python scripts/navigation/run_isaac.py \
    --scene <scene.usda> --path <path.npz> --out-dir <dir>
```

Verified: `nav_runs/mansion/public_healthcare_3f_floor1/{01,02}__*/ego_isaac.mp4`.

## Remaining work / next steps

- **procthor render quality**: procthor works and is frame-aligned, but the
  procthor USD renders washed-out/over-bright (pale hazy walls). Likely a
  too-strong dome/environment light or unloaded wall materials in the procthor
  USD. Needs investigation in the USD's lighting/material prims — independent
  of `run_isaac.py`. Note: procthor USD scenes on disk (`val_308`, `val_565`,
  …) do **not** overlap the downloaded procthor MJCF set (`val_0`–`4`); a
  procthor IsaacSim run needs a scene present in both formats.
- **Depth / chase / combined panel**: `run_isaac.py` renders ego RGB only;
  run_mujoco's 2×2 combined video (map | chase | ego RGB | ego depth) is not
  yet replicated for IsaacSim.
- **Driver wiring**: `gen_trajectories.py` does not yet have a `--sim isaac`
  option.

## Working files

- `scripts/navigation/run_isaac.py` — the IsaacSim runtime (committed).
- `/tmp/run_isaac.py`, `/tmp/isaac_record.sh` — superseded probe scripts.
