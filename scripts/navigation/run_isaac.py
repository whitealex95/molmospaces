#!/usr/bin/env python3
"""IsaacSim navigation runtime -- egocentric RGB + depth and a chase view.

Drives a Unitree G1 along an A* path (``path.npz`` from plan.py) through a USD
scene and renders, per step, the robot's egocentric RGB + depth and a chase
view -- the IsaacSim counterpart of run_mujoco.py. The occupancy grid and path
are sim-agnostic, so the same ``path.npz`` drives both runtimes.

Capture uses the ``isaacsim.sensors.camera`` Camera sensor in GUI mode (the
headless camera-sensor path crashes here). The G1 and all cameras are created
at the **stage root** -- a converted scene's ``/World`` prim can carry a det-1
reflection transform (the mansion handedness fix) that IsaacSim's XFormPrim
pose math rejects. Needs a display:

    DISPLAY=:20 XAUTHORITY=$HOME/.Xauthority \\
      conda run -n mlspaces-isaac python run_isaac.py \\
        --scene scene.usda --path path.npz --out-dir OUT

Output: ``<out-dir>/{ego_isaac,depth_isaac,follow_isaac}.mp4``.

See docs/navigation_pipeline.md and docs/isaac_navigation_log.md.
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np

G1_USD = Path.home() / "Projects/CAMDM/PyTorch/visualize/assets/g1_isaac/configuration/g1_base.usd"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--scene", required=True, help="USD scene (.usd/.usda)")
ap.add_argument("--path", required=True, help="path.npz from plan.py")
ap.add_argument("--out-dir", required=True, help="output directory")
ap.add_argument("--g1", default=str(G1_USD), help="G1 USD (geometry layer)")
ap.add_argument("--robot-z", type=float, default=0.79, help="G1 root height (m)")
ap.add_argument("--fps", type=int, default=30, help="output video frame rate")
ap.add_argument("--max-frames", type=int, default=300, help="cap on rendered frames")
ap.add_argument("--dome-max", type=float, default=180.0, help="clamp scene DomeLight intensity")
ap.add_argument(
    "--distant-max", type=float, default=500.0, help="clamp scene DistantLight intensity"
)
ap.add_argument("--chase-z", type=float, default=3.0, help="chase camera height (m)")
ap.add_argument("--chase-back", type=float, default=4.5, help="chase camera distance behind (m)")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": False, "width": 1280, "height": 960})

import cv2  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.camera")

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

CAM_W, CAM_H = 1280, 960
# ego camera local transform on the G1 torso: 0.12 m forward + 0.42 m up, and a
# rotation so the camera looks along the robot's +x (forward), +z up. Rows are
# the camera's X/Y/Z axes expressed in the torso frame, then the translation.
EGO_LOCAL = Gf.Matrix4d(0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0.12, 0, 0.42, 1)


def resample(wp, step):
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


def tame_lights(stage, dome_max, distant_max):
    """Clamp over-bright scene lights. The procthor USD ships a 1000-intensity
    DomeLight + DistantLight that wash the render out; the mansion USD has no
    lights at all. This is a per-scene, code-side fix applied to an in-memory
    session-layer override -- the USD asset on disk is never modified."""
    for prim in stage.Traverse():
        t = str(prim.GetTypeName())
        if t not in ("DomeLight", "DistantLight"):
            continue
        cap = dome_max if t == "DomeLight" else distant_max
        a = prim.GetAttribute("inputs:intensity")
        cur = a.Get() if a and a.HasAuthoredValue() else None
        if cur is not None and cur > cap:
            a.Set(float(cap))
            print(f"  tamed {t} {prim.GetName()}: {cur} -> {cap}", flush=True)


def colorize_depth(depth, near=0.1, far=8.0):
    d = np.clip(np.nan_to_num(np.asarray(depth), nan=far, posinf=far), near, far)
    norm = ((d - near) / (far - near) * 255).astype(np.uint8)
    return cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)


def write_video(path, frames, fps):
    """Encode BGR uint8 frames to H.264 via the system ffmpeg (the isaacsim
    env's libs break it, so call /usr/bin/ffmpeg with LD_LIBRARY_PATH stripped)."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    ffmpeg = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg"
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    proc = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
        env=env,
    )
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()


def main() -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wp = np.load(args.path, allow_pickle=True)["waypoints"].astype(float)
    poses = smooth(resample(wp, 1.0 / args.fps), 45)
    yaws = compute_yaws(poses)
    step = max(1, len(poses) // args.max_frames)
    idx = list(range(0, len(poses), step))
    print(f"path: {len(poses)} smoothed poses -> {len(idx)} frames", flush=True)

    ctx = omni.usd.get_context()
    print(f"opening stage: {args.scene}", flush=True)
    ctx.open_stage(args.scene)
    for _ in range(120):
        app.update()
    stage = ctx.get_stage()
    # all edits (G1, cameras, light overrides) go to the in-memory session
    # layer -- the scene USD file on disk is never modified
    stage.SetEditTarget(stage.GetSessionLayer())
    tame_lights(stage, args.dome_max, args.distant_max)

    # G1 at the stage root (referenced geometry layer -> static, no physics)
    g1 = stage.DefinePrim("/g1", "Xform")
    g1.GetReferences().AddReference(args.g1)
    g1x = UsdGeom.Xformable(g1)
    g1x.ClearXformOpOrder()
    g1_t = g1x.AddTranslateOp()
    g1_r = g1x.AddRotateZOp()
    print(f"G1 referenced: {args.g1}", flush=True)

    world = World()
    world.reset()

    # ego camera rigidly mounted on the G1 torso (like run_mujoco) -- it is a
    # child of the torso prim with a fixed local transform, so it rides the
    # robot. The G1's internal frames are det+1, so a camera under it is safe.
    torso_path = str(g1.GetPath())
    for prim in Usd.PrimRange(g1):
        if prim.GetName() == "torso_link":
            torso_path = str(prim.GetPath())
            break
    print(f"ego camera mount: {torso_path}", flush=True)
    ego = Camera(prim_path=f"{torso_path}/ego_cam", resolution=(CAM_W, CAM_H))
    ego_xf = UsdGeom.Xformable(stage.GetPrimAtPath(f"{torso_path}/ego_cam"))
    ego_xf.ClearXformOpOrder()
    ego_xf.AddTransformOp().Set(EGO_LOCAL)
    ego.initialize()
    ego.add_distance_to_image_plane_to_frame()
    chase = Camera(prim_path="/chase_cam", resolution=(CAM_W, CAM_H))
    chase.initialize()
    for _ in range(40):
        world.step(render=True)
    print("READY", flush=True)

    ego_rgb, ego_depth, follow = [], [], []
    t0 = time.time()
    for fi, pi in enumerate(idx):
        x, y = poses[pi]
        a = float(yaws[pi])
        ca, sa = np.cos(a), np.sin(a)

        # drive the G1; the ego camera is a child of the torso and rides along
        g1_t.Set(Gf.Vec3d(float(x), float(y), args.robot_z))
        g1_r.Set(float(np.degrees(a)))

        set_camera_view(
            eye=[x - args.chase_back * ca, y - args.chase_back * sa, args.chase_z],
            target=[x, y, 1.0],
            camera_prim_path="/chase_cam",
        )

        for _ in range(5):
            world.step(render=True)

        er = np.asarray(ego.get_rgba())
        cr = np.asarray(chase.get_rgba())
        dep = ego.get_current_frame().get("distance_to_image_plane")
        if er.size > 1:
            ego_rgb.append(er[..., :3][..., ::-1].copy())
        if cr.size > 1:
            follow.append(cr[..., :3][..., ::-1].copy())
        if dep is not None and np.asarray(dep).size > 1:
            ego_depth.append(colorize_depth(dep))
        if fi % 40 == 0:
            print(f"  frame {fi}/{len(idx)}  t={time.time() - t0:.1f}s", flush=True)
    print(
        f"captured ego_rgb={len(ego_rgb)} depth={len(ego_depth)} follow={len(follow)} "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )

    write_video(out_dir / "ego_isaac.mp4", ego_rgb, args.fps)
    write_video(out_dir / "depth_isaac.mp4", ego_depth, args.fps)
    write_video(out_dir / "follow_isaac.mp4", follow, args.fps)
    print(f"wrote ego_isaac.mp4 / depth_isaac.mp4 / follow_isaac.mp4 -> {out_dir}", flush=True)

    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
