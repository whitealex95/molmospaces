#!/usr/bin/env python3
"""IsaacSim egocentric navigation runtime.

Flies an egocentric camera along an A* path (``path.npz`` from plan.py) through
a USD scene and renders the walk-through to an H.264 video -- the IsaacSim
counterpart of run_mujoco.py. The occupancy grid and path are sim-agnostic, so
the same ``path.npz`` drives both runtimes (provided the USD scene shares the
MJCF world frame -- true for mansion, since both converters use one convention).

Capture uses IsaacSim's GUI viewport (``capture_viewport_to_file``), which
renders reliably where the headless camera-sensor API crashed. It therefore
needs a display -- run with the Chrome Remote Desktop virtual display:

    DISPLAY=:20 XAUTHORITY=$HOME/.Xauthority \\
      conda run -n mlspaces-isaac python run_isaac.py \\
        --scene scene.usda --path path.npz --out-dir OUT

Output: ``<out-dir>/ego_isaac.mp4``.

See docs/navigation_pipeline.md and docs/isaac_navigation_log.md.
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--scene", required=True, help="USD scene (.usd/.usda)")
ap.add_argument("--path", required=True, help="path.npz from plan.py")
ap.add_argument("--out-dir", required=True, help="output directory")
ap.add_argument("--eye-z", type=float, default=1.3, help="camera eye height (m)")
ap.add_argument("--fps", type=int, default=30, help="output video frame rate")
ap.add_argument("--max-frames", type=int, default=300, help="cap on captured frames")
ap.add_argument("--keep-frames", action="store_true", help="keep the PNG frames")
args = ap.parse_args()

# SimulationApp must be created before importing any omni.* module.
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": False, "width": 1600, "height": 1200})

import omni.usd  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402


def resample(wp, step):
    """Densify a polyline to ~step-spaced points."""
    out = [wp[0].astype(float)]
    for a, b in zip(wp[:-1], wp[1:]):
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            continue
        d = seg / length
        n = max(1, int(round(length / step)))
        for i in range(1, n + 1):
            out.append(a + d * (length * i / n))
    return np.array(out)


def smooth(p, w=45):
    """Moving-average smoothing; endpoints fixed."""
    n = len(p)
    if n < 3:
        return p.astype(float)
    k = w // 2
    o = p.astype(float).copy()
    for i in range(1, n - 1):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        o[i] = p[lo:hi].mean(axis=0)
    return o


def compute_yaws(poses, alpha=0.2):
    """Central-difference tangent heading with a wrap-aware low-pass filter."""
    n = len(poses)
    raw = np.zeros(n)
    for i in range(n):
        d = poses[min(i + 1, n - 1)] - poses[max(i - 1, 0)]
        raw[i] = np.arctan2(d[1], d[0]) if np.linalg.norm(d) > 1e-6 else raw[i - 1]
    y, out = raw[0], np.zeros(n)
    for i in range(n):
        y += alpha * np.arctan2(np.sin(raw[i] - y), np.cos(raw[i] - y))
        out[i] = y
    return out


def main() -> int:
    out_dir = Path(args.out_dir)
    frame_dir = out_dir / "isaac_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    wp = np.load(args.path, allow_pickle=True)["waypoints"].astype(float)
    poses = smooth(resample(wp, 1.0 / args.fps), 45)
    yaws = compute_yaws(poses)
    step = max(1, len(poses) // args.max_frames)
    idx = list(range(0, len(poses), step))
    print(f"path: {len(poses)} smoothed poses -> {len(idx)} capture frames", flush=True)

    ctx = omni.usd.get_context()
    print(f"opening stage: {args.scene}", flush=True)
    ctx.open_stage(args.scene)
    for _ in range(120):  # let the stage + meshes settle
        app.update()
    print(f"stage loaded prims={len(list(ctx.get_stage().Traverse()))}", flush=True)

    vp = get_active_viewport()
    print("READY", flush=True)

    t0 = time.time()
    for fi, pi in enumerate(idx):
        x, y = poses[pi]
        a = float(yaws[pi])
        set_camera_view(
            eye=[float(x), float(y), args.eye_z],
            target=[float(x + np.cos(a)), float(y + np.sin(a)), args.eye_z - 0.15],
            camera_prim_path="/OmniverseKit_Persp",
        )
        app.update()
        app.update()
        capture_viewport_to_file(vp, str(frame_dir / f"frame_{fi:04d}.png"))
        app.update()
        if fi % 40 == 0:
            print(f"  frame {fi}/{len(idx)}  t={time.time() - t0:.1f}s", flush=True)
    for _ in range(40):  # flush the last async captures
        app.update()

    frames = sorted(frame_dir.glob("frame_*.png"))
    print(f"captured {len(frames)} frames in {time.time() - t0:.1f}s", flush=True)

    rc = 0
    if not frames:
        print("ERROR: no frames captured", flush=True)
        rc = 1
    else:
        # Assemble BEFORE app.close(): IsaacSim fast-shutdown can hard-exit the
        # process, so code after close() may never run. Use the system ffmpeg
        # by absolute path with LD_LIBRARY_PATH stripped -- the isaacsim conda
        # env neither ships ffmpeg nor system-compatible libraries.
        out_mp4 = out_dir / "ego_isaac.mp4"
        ffmpeg = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg"
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(args.fps),
                    "-pattern_type",
                    "glob",
                    "-i",
                    str(frame_dir / "frame_*.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(out_mp4),
                ],
                check=True,
                env=env,
            )
            print(f"wrote {out_mp4}", flush=True)
            if not args.keep_frames:
                for f in frames:
                    f.unlink()
                frame_dir.rmdir()
        except Exception as e:  # noqa: BLE001
            print(f"ffmpeg assembly failed: {e}  (frames kept at {frame_dir})", flush=True)
            rc = 1

    app.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
