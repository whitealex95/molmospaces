#!/usr/bin/env python3
"""MuJoCo runtime: kinematically drive a Unitree G1 along an A* path through a
scene and render its egocentric RGB + depth at 30 Hz.

The G1 is merged into the scene MJCF and its floating base is set along a
*smoothed* path each step (kinematic -- no physics). Path positions are
moving-average smoothed and the heading is low-pass filtered, so the cameras
turn gradually instead of snapping at A* waypoints.

Output (in <scene_dir>/nav_run/):
  combined.mp4   -- 2x2: top-down map | chase | ego RGB | ego depth
  ego_rgb.mp4 / ego_depth.mp4 / follow.mp4  -- the individual streams

Renderer (--renderer):
  opengl    -- stock MuJoCo OpenGL; run from the ``mlspaces-mujoco`` env.
  filament  -- physically based lighting + shadows; run from ``mlspaces``
               (needs the Filament mujoco wheel) for noticeably better RGB.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import mujoco
import numpy as np

G1_XML = (
    Path.home() / "Projects" / "CAMDM" / "PyTorch" / "visualize" / "assets" / "g1_29dof_rev_1_0.xml"
)
STEP_HZ = 30.0
SPEED = 1.0  # m/s base travel speed
PELVIS_Z = 0.793  # G1 standing pelvis height above the floor
EGO_W, EGO_H = 640, 480
SMOOTH_WINDOW = 45  # path moving-average window (in resampled steps); collision-verified
YAW_ALPHA = 0.2  # heading low-pass factor (smaller = smoother / more lag)
# chase camera: interior follow camera -- close behind and above the robot,
# below the ceiling, looking down at its mid-body (same rule as run_isaac.py).
# Kept near-overhead so it does not clip through walls in tight rooms.
CHASE_BACK = 1.3  # m behind the robot
CHASE_Z = 2.3  # m camera height (below the ~2.9 m ceiling)
CHASE_LOOK_Z = 0.9  # m look-at height on the robot
CHASE_DIST = float(np.hypot(CHASE_BACK, CHASE_Z - CHASE_LOOK_Z))
CHASE_ELEV = -float(np.degrees(np.arctan2(CHASE_Z - CHASE_LOOK_Z, CHASE_BACK)))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def resample_path(waypoints: np.ndarray, step: float) -> np.ndarray:
    """Densify a polyline to ~`step`-spaced points."""
    out = [waypoints[0].astype(float)]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            continue
        d = seg / length
        n = max(1, int(round(length / step)))
        for i in range(1, n + 1):
            out.append(a + d * (length * i / n))
    return np.array(out)


def smooth_path(pts: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing of a dense polyline; endpoints kept fixed."""
    n = len(pts)
    if n < 3 or window < 3:
        return pts.astype(float)
    k = window // 2
    out = pts.astype(float).copy()
    for i in range(1, n - 1):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        out[i] = pts[lo:hi].mean(axis=0)
    return out


def compute_yaws(poses: np.ndarray, alpha: float) -> np.ndarray:
    """Per-step heading: central-difference tangent of the (smoothed) path,
    then a wrap-aware low-pass filter -- no abrupt turns."""
    n = len(poses)
    raw = np.zeros(n)
    for i in range(n):
        d = poses[min(i + 1, n - 1)] - poses[max(i - 1, 0)]
        raw[i] = np.arctan2(d[1], d[0]) if np.linalg.norm(d) > 1e-6 else raw[i - 1]
    yaws = np.zeros(n)
    y = raw[0]
    for i in range(n):
        y += alpha * np.arctan2(np.sin(raw[i] - y), np.cos(raw[i] - y))
        yaws[i] = y
    return yaws


def build_model(scene_xml: Path, g1_xml: Path, backend: str = "opengl"):
    """Merge the G1 into the scene MJCF; add a head-mounted 'ego' camera.

    For the Filament backend a forward fill light is also mounted on the torso.
    Filament ignores the MuJoCo headlight, so an interior ego view would
    otherwise be lit by scene lights alone and read near-black."""
    scene = mujoco.MjSpec.from_file(str(scene_xml))
    g1 = mujoco.MjSpec.from_file(str(g1_xml))

    g1_dir = Path(g1_xml).parent
    sub = g1.meshdir or ""
    for m in g1.meshes:
        if not os.path.isabs(m.file):
            m.file = str((g1_dir / sub / m.file).resolve())

    frame = scene.worldbody.add_frame()
    frame.attach_body(g1.worldbody.bodies[0], "g1_", "")

    torso = scene.body("g1_torso_link")
    cam = torso.add_camera()
    cam.name = "ego"
    cam.pos = [0.12, 0.0, 0.42]
    cam.quat = [0.5, 0.5, -0.5, -0.5]  # look along +x (forward), +z world-up

    if backend == "filament":
        lamp = torso.add_light()
        lamp.name = "ego_lamp"
        lamp.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        lamp.pos = [0.12, 0.0, 0.42]
        lamp.dir = [1.0, 0.0, -0.15]  # robot-forward, slightly down
        lamp.diffuse = [2.5, 2.5, 2.5]  # bright: Filament has no headlight fill
        lamp.specular = [0.1, 0.1, 0.1]
        lamp.castshadow = 0

    return scene.compile()


def colorize_depth(depth: np.ndarray, near=0.1, far=8.0) -> np.ndarray:
    d = np.clip(np.nan_to_num(depth, nan=far, posinf=far), near, far)
    norm = ((d - near) / (far - near) * 255).astype(np.uint8)
    return cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)


def make_renderer(model, height: int, width: int, backend: str):
    """Build an offscreen renderer; returns ``render(data, camera, depth=False)``.

    opengl   -- stock ``mujoco.Renderer`` (run in the mlspaces-mujoco env).
    filament -- molmospaces ``MjFilamentRenderer``: physically based lighting
                + shadows; needs the Filament mujoco wheel (mlspaces env).
    """
    if backend == "filament":
        from molmo_spaces.env.mj_extensions import MjModelBindings
        from molmo_spaces.renderer.filament_rendering import MjFilamentRenderer

        r = MjFilamentRenderer(MjModelBindings(model), height=height, width=width)
        update = r.update
    else:
        r = mujoco.Renderer(model, height, width)
        update = r.update_scene

    def render(data, camera, depth: bool = False) -> np.ndarray:
        if depth:
            r.enable_depth_rendering()
            update(data, camera)
            out = np.array(r.render())
            r.disable_depth_rendering()
            return out
        r.disable_depth_rendering()
        update(data, camera)
        return np.array(r.render())

    return render


def build_map_panel(occ_path: Path, poses: np.ndarray, pw: int, ph: int):
    """Static 2D top-down layout panel (rooms tinted, trajectory drawn).
    Returns the panel image and a world->panel-pixel function."""
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


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--path", type=Path, required=True, help="path.npz from plan.py")
    ap.add_argument(
        "--occupancy", type=Path, default=None, help="occupancy.npz (default: next to scene)"
    )
    ap.add_argument("--g1", type=Path, default=G1_XML)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--renderer",
        choices=["opengl", "filament"],
        default="opengl",
        help="opengl (mlspaces-mujoco env) or filament (mlspaces env, better RGB)",
    )
    args = ap.parse_args()

    occ_path = args.occupancy or args.scene.parent / "occupancy.npz"
    out_dir = (args.out_dir or args.scene.parent / "nav_run").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    p = np.load(args.path, allow_pickle=True)
    waypoints = p["waypoints"].astype(float)
    print(f"path: {p['start_room']} -> {p['goal_room']}  ({len(waypoints)} waypoints)")

    poses = smooth_path(resample_path(waypoints, SPEED / STEP_HZ), SMOOTH_WINDOW)
    yaws = compute_yaws(poses, YAW_ALPHA)
    print(
        f"merging G1; {len(poses)} smoothed kinematic steps @ {STEP_HZ:.0f} Hz"
        f"  [{args.renderer} renderer]"
    )

    model = build_model(args.scene, args.g1, args.renderer)
    data = mujoco.MjData(model)
    base_adr = model.joint("g1_floating_base_joint").qposadr[0]

    render = make_renderer(model, EGO_H, EGO_W, args.renderer)
    # interior follow camera: behind the robot, below the ceiling (free camera;
    # lookat + azimuth are set per frame so it tracks the robot's heading)
    followcam = mujoco.MjvCamera()
    followcam.type = mujoco.mjtCamera.mjCAMERA_FREE
    followcam.distance = CHASE_DIST
    followcam.elevation = CHASE_ELEV

    map_base, world_to_panel = build_map_panel(occ_path, poses, EGO_W, EGO_H)

    rgb_frames, depth_frames, follow_frames, combined = [], [], [], []
    for i, (x, y) in enumerate(poses):
        yaw = yaws[i]
        data.qpos[base_adr : base_adr + 7] = [x, y, PELVIS_Z, *yaw_quat(yaw)]
        mujoco.mj_forward(model, data)

        rgb = render(data, "ego")[:, :, ::-1].copy()
        rgb_frames.append(rgb)

        depth = colorize_depth(render(data, "ego", depth=True))
        depth_frames.append(depth)

        followcam.lookat[:] = [x, y, CHASE_LOOK_Z]
        followcam.azimuth = float(np.degrees(yaw))
        follow = render(data, followcam)[:, :, ::-1].copy()
        follow_frames.append(follow)

        # 2D map panel: static layout + a marker at the robot pose
        mp = map_base.copy()
        dot = world_to_panel(x, y)
        head = world_to_panel(x + 0.6 * np.cos(yaw), y + 0.6 * np.sin(yaw))
        cv2.line(mp, dot, head, (40, 40, 40), 3, cv2.LINE_AA)
        cv2.circle(mp, dot, 8, (0, 0, 230), -1, cv2.LINE_AA)

        top = np.hstack([label(mp, "Top-down map"), label(follow, "Chase")])
        bot = np.hstack([label(rgb, "Ego RGB"), label(depth, "Ego depth")])
        combined.append(np.vstack([top, bot]))

    def write_video(path, frames, fps=STEP_HZ):
        """Encode BGR uint8 frames to an H.264 mp4 via ffmpeg.

        H.264 + yuv420p is what browsers and Notion can play. OpenCV's mp4v
        is MPEG-4 Part 2, whose Simple Profile caps out near 1280x720 -- the
        1280x960 combined panel exceeds it and fails to render online. Falls
        back to OpenCV mp4v only if ffmpeg is not on PATH."""
        h, w = frames[0].shape[:2]
        if shutil.which("ffmpeg"):
            proc = subprocess.Popen(
                [
                    "ffmpeg",
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
                    "-crf",
                    "20",
                    "-movflags",
                    "+faststart",
                    str(path),
                ],
                stdin=subprocess.PIPE,
            )
            for f in frames:
                proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed encoding {path}")
            return
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()

    write_video(out_dir / "combined.mp4", combined)
    write_video(out_dir / "ego_rgb.mp4", rgb_frames)
    write_video(out_dir / "ego_depth.mp4", depth_frames)
    write_video(out_dir / "follow.mp4", follow_frames)

    # quick-look montage of the combined stream
    idx = np.linspace(0, len(combined) - 1, 6).astype(int)
    tiles = [cv2.resize(combined[i], (640, 480)) for i in idx]
    sheet = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
    cv2.imwrite(str(out_dir / "combined_montage.png"), sheet)

    print(f"wrote {len(combined)} frames -> {out_dir}/combined.mp4 (+ individual streams)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
