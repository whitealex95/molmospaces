#!/usr/bin/env python3
"""IsaacSim navigation runtime -- egocentric RGB + depth, a chase view, and a
2x2 combined panel.

Drives a Unitree G1 along an A* path (``path.npz`` from plan.py) through a USD
scene and renders, per step, the robot's egocentric RGB + depth and a chase
view -- the IsaacSim counterpart of run_mujoco.py. The occupancy grid and path
are sim-agnostic, so the same ``path.npz`` drives both runtimes.

Camera placement matches run_mujoco.py: the ego camera is rigidly mounted on
the G1 torso (0.12 m forward, 0.42 m up, looking down robot +x); the chase
camera is an interior follow camera -- behind the robot and below the ceiling,
looking at its mid-body. Both cameras use a 45 deg vertical FOV.

Capture uses the ``isaacsim.sensors.camera`` Camera sensor in GUI mode (the
headless camera-sensor path crashes here). The G1 and all cameras are created
at the **stage root** -- a converted scene's ``/World`` prim can carry a det-1
reflection transform (the mansion handedness fix) that IsaacSim's XFormPrim
pose math rejects. Needs a display:

    DISPLAY=:20 XAUTHORITY=$HOME/.Xauthority \\
      conda run -n mlspaces-isaac python run_isaac.py \\
        --scene scene.usda --path path.npz --out-dir OUT

Output: ``<out-dir>/isaac_{ego,depth,follow,combined}.mp4`` (+ a montage png).

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
ap.add_argument(
    "--occupancy",
    default=None,
    help="occupancy.npz for the combined map panel (default: found near path.npz)",
)
ap.add_argument("--g1", default=str(G1_USD), help="G1 USD (geometry layer)")
ap.add_argument(
    "--robot-z", type=float, default=None, help="G1 base height (m); default: auto (feet on z=0)"
)
ap.add_argument("--fps", type=int, default=30, help="output video frame rate")
ap.add_argument("--max-frames", type=int, default=300, help="cap on rendered frames")
ap.add_argument("--dome-max", type=float, default=180.0, help="clamp scene DomeLight intensity")
ap.add_argument(
    "--distant-max", type=float, default=500.0, help="clamp scene DistantLight intensity"
)
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
# chase camera: an interior follow camera -- close behind and above the robot,
# below the ceiling, looking down at its mid-body. A high external camera
# cannot see into a ceilinged room; a far-behind camera clips through walls in
# tight rooms, so it sits near-overhead. Same rule as run_mujoco.py.
CHASE_BACK = 1.3  # m behind the robot, opposite its heading
CHASE_Z = 2.3  # m camera height -- below the ~2.9 m procthor ceiling
CHASE_LOOK_Z = 0.9  # m look-at height on the robot
EGO_VFOV_DEG = 45.0  # MuJoCo's default camera fovy (vertical FOV)
# g1_base.usd places its origin at the pelvis; the lowest geometry (the soles)
# is 0.315 m below that -- measured from the asset's mesh extents. The mansion
# and procthor converters both put the scene floor at z=0, so a base height of
# 0.315 m rests the soles on the floor. (BBoxCache is unreliable on this
# instanced USD -- it returns an empty bound -- so the value is a constant.)
G1_GROUND_Z = 0.315


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


def set_camera_fov(stage, cam_path, vfov_deg, w, h):
    """Set a USD camera's vertical FOV to match a MuJoCo camera fovy. focal
    length and aperture are a ratio, so any consistent unit works."""
    cam = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))
    f = 24.0
    v_ap = 2.0 * f * float(np.tan(np.radians(vfov_deg) / 2.0))
    cam.CreateFocalLengthAttr(f)
    cam.CreateVerticalApertureAttr(v_ap)
    cam.CreateHorizontalApertureAttr(v_ap * w / h)


def colorize_depth(depth, near=0.1, far=8.0):
    d = np.clip(np.nan_to_num(np.asarray(depth), nan=far, posinf=far), near, far)
    norm = ((d - near) / (far - near) * 255).astype(np.uint8)
    return cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)


def build_map_panel(occ_path, poses, pw, ph):
    """Static 2D top-down layout panel (rooms tinted, trajectory drawn).
    Returns the panel image and a world->panel-pixel function. Identical to
    run_mujoco.build_map_panel so the combined panels line up."""
    o = np.load(occ_path, allow_pickle=True)
    occupancy = o["occupancy"]
    room_map = o["room_map"]
    room_names = o["room_names"]
    w2m = o["world_to_map"]

    img = np.where(occupancy[..., None], 235, 45).astype(np.uint8).repeat(3, axis=2)
    n = max(len(room_names), 1)
    hsv = np.stack(
        [
            np.linspace(0, 179, n, endpoint=False).astype(np.uint8),
            np.full(n, 70, np.uint8),
            np.full(n, 255, np.uint8),
        ],
        axis=1,
    )
    colors = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]
    for i in range(1, n + 1):
        img[room_map == i] = colors[i - 1]

    traj = np.array([w2m @ np.array([x, y, 0.0, 1.0]) for x, y in poses])  # [row,col]
    cv2.polylines(
        img,
        [np.stack([traj[:, 1], traj[:, 0]], 1).astype(np.int32)],
        False,
        (0, 150, 255),
        5,
        cv2.LINE_AA,
    )

    h, w = img.shape[:2]
    s = min(pw / w, ph / h)
    rw, rh = max(1, int(w * s)), max(1, int(h * s))
    panel = np.full((ph, pw, 3), 255, np.uint8)
    ox, oy = (pw - rw) // 2, (ph - rh) // 2
    panel[oy : oy + rh, ox : ox + rw] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)

    def world_to_panel(x, y):
        rc = w2m @ np.array([x, y, 0.0, 1.0])
        return (int(rc[1] * s + ox), int(rc[0] * s + oy))

    return panel, world_to_panel


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


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

    # occupancy.npz drives the combined map panel. nav_runs puts it next to
    # path.npz (procthor) or one level up, shared by a scene's trajectories
    # (mansion) -- accept either.
    if args.occupancy:
        occ_path = Path(args.occupancy)
    else:
        near = Path(args.path).parent / "occupancy.npz"
        occ_path = near if near.exists() else Path(args.path).parent.parent / "occupancy.npz"
    do_combined = occ_path.exists()
    if not do_combined:
        print(f"  no occupancy.npz near {args.path}; combined panel skipped", flush=True)

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

    # match MuJoCo's 45 deg vertical FOV on both cameras
    set_camera_fov(stage, f"{torso_path}/ego_cam", EGO_VFOV_DEG, CAM_W, CAM_H)
    set_camera_fov(stage, "/chase_cam", EGO_VFOV_DEG, CAM_W, CAM_H)

    # warm up the renderer / camera sensors before capture
    for _ in range(40):
        world.step(render=True)

    # base height that rests the G1 soles on the scene floor (see G1_GROUND_Z)
    robot_z = args.robot_z if args.robot_z is not None else G1_GROUND_Z
    print(f"READY  (G1 base z = {robot_z:.3f})", flush=True)

    map_base, world_to_panel = (None, None)
    if do_combined:
        map_base, world_to_panel = build_map_panel(occ_path, poses, 640, 480)

    ego_rgb, ego_depth, follow, combined = [], [], [], []
    t0 = time.time()
    for fi, pi in enumerate(idx):
        x, y = poses[pi]
        a = float(yaws[pi])

        # drive the G1; the ego camera is a child of the torso and rides along
        g1_t.Set(Gf.Vec3d(float(x), float(y), robot_z))
        g1_r.Set(float(np.degrees(a)))

        # chase: interior follow camera, behind the robot and below the ceiling
        ca, sa = np.cos(a), np.sin(a)
        set_camera_view(
            eye=[float(x - CHASE_BACK * ca), float(y - CHASE_BACK * sa), CHASE_Z],
            target=[float(x), float(y), CHASE_LOOK_Z],
            camera_prim_path="/chase_cam",
        )

        for _ in range(5):
            world.step(render=True)

        er = np.asarray(ego.get_rgba())
        cr = np.asarray(chase.get_rgba())
        dep = ego.get_current_frame().get("distance_to_image_plane")
        if er.size <= 1 or cr.size <= 1 or dep is None or np.asarray(dep).size <= 1:
            if fi % 40 == 0:
                print(f"  frame {fi}/{len(idx)}: incomplete capture, skipped", flush=True)
            continue
        rgb = er[..., :3][..., ::-1].copy()
        chs = cr[..., :3][..., ::-1].copy()
        dpt = colorize_depth(dep)
        ego_rgb.append(rgb)
        follow.append(chs)
        ego_depth.append(dpt)

        if do_combined:
            mp = map_base.copy()
            dot = world_to_panel(x, y)
            head = world_to_panel(x + 0.6 * np.cos(a), y + 0.6 * np.sin(a))
            cv2.line(mp, dot, head, (40, 40, 40), 3, cv2.LINE_AA)
            cv2.circle(mp, dot, 8, (0, 0, 230), -1, cv2.LINE_AA)
            p6 = (640, 480)
            top = np.hstack(
                [
                    label(mp, "Top-down map"),
                    label(cv2.resize(chs, p6, interpolation=cv2.INTER_AREA), "Chase"),
                ]
            )
            bot = np.hstack(
                [
                    label(cv2.resize(rgb, p6, interpolation=cv2.INTER_AREA), "Ego RGB"),
                    label(cv2.resize(dpt, p6, interpolation=cv2.INTER_AREA), "Ego depth"),
                ]
            )
            combined.append(np.vstack([top, bot]))

        if fi % 40 == 0:
            print(f"  frame {fi}/{len(idx)}  t={time.time() - t0:.1f}s", flush=True)
    print(f"captured {len(ego_rgb)} frames in {time.time() - t0:.1f}s", flush=True)

    write_video(out_dir / "isaac_ego.mp4", ego_rgb, args.fps)
    write_video(out_dir / "isaac_depth.mp4", ego_depth, args.fps)
    write_video(out_dir / "isaac_follow.mp4", follow, args.fps)
    write_video(out_dir / "isaac_combined.mp4", combined, args.fps)

    if combined:
        sel = np.linspace(0, len(combined) - 1, 6).astype(int)
        tiles = [cv2.resize(combined[i], (640, 480)) for i in sel]
        sheet = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
        cv2.imwrite(str(out_dir / "isaac_combined_montage.png"), sheet)

    made = ["isaac_ego", "isaac_depth", "isaac_follow"] + (["isaac_combined"] if combined else [])
    print(f"wrote {'/'.join(made)}.mp4 -> {out_dir}", flush=True)

    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
