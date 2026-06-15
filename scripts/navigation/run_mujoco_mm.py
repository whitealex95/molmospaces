#!/usr/bin/env python3
"""MuJoCo runtime: drive a Unitree G1 with FULL-BODY motion-matched locomotion
along an A* path through a scene and render its egocentric RGB + depth at 30 Hz.

Unlike ``run_mujoco.py`` (which kinematically slides the floating base only, with
the legs frozen), this script merges the *Menagerie* G1 and sets **all 36 qpos**
each frame from the motion-matching controller in ``~/Projects/motionmatching-g1``
(``mm_g1.controller.MotionMatcher``). The result is a walking gait -- swinging
legs, arm sway, pelvis bob -- that actually traverses the planned route.

How the A* path drives the matcher
----------------------------------
The matcher is a *velocity*-controlled character: ``matcher.step(desiredVel,
desiredFace)`` integrates its own world root from a motion database, it cannot be
teleported onto an arbitrary path. So we:

1. Anchor a rigid 2D transform ``T`` (z-rotation + xy-translation) that maps the
   matcher's start frame onto the path's first waypoint with the path's initial
   heading. The matcher then runs in its own frame and every output pose is
   mapped back into the scene frame by ``T``.
2. Steer with **pure pursuit**: each frame aim a desired velocity at a lookahead
   point along the densified path (expressed in the matcher frame via ``T^-1``).
   The matcher drifts slightly off the line -- that is expected and realistic for
   motion matching.

The target path and the motion-matching command (lookahead target, desiredVel
arrow, the matcher's predicted Tpos trajectory) are drawn as real 3D geometry in
the scene -- appended to the renderer's MjvScene for the CHASE camera only, so
they show in the overhead chase view while the ego RGB/depth stay clean.

The Menagerie G1 (``assets/unitree_g1/g1.xml`` in the motion-matching repo) is the
exact model the matcher's 36-D qpos is authored for (root 7 + 29 joints, same
joint order). It shares the ``pelvis`` / ``floating_base_joint`` / ``torso_link``
body names that ``run_mujoco.py`` already mounts the ego camera on, so the camera
rig is unchanged.

Output (in --out-dir, default nav_runs_mm/...):
  combined.mp4   -- 2x2: top-down map | chase (target + MM command) | ego RGB | ego depth
  ego_rgb.mp4 / ego_depth.mp4 / follow.mp4  -- the individual streams
  motion_full.npz / motion_min.npz          -- full vs minimal saved motion

Renderer (--renderer):
  opengl    -- stock MuJoCo OpenGL; run from the ``mlspaces-mujoco`` env.
  filament  -- physically based lighting + shadows; run from ``mlspaces``.

The motion-matching controller (numpy + scipy + mujoco only) runs in either env.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import mujoco
import numpy as np

# Motion-matching repo: provides mm_g1 (controller + motion library) and the
# Menagerie G1 the matcher's qpos is authored for.
MM_ROOT = Path.home() / "Projects" / "motionmatching-g1"
G1_MM_XML = MM_ROOT / "assets" / "unitree_g1" / "g1.xml"

STEP_HZ = 30.0  # matcher data + render rate (the matcher is fixed at 30 fps)
WALK_SPEED = 1.0  # m/s desired travel speed fed to the matcher's velocity springs
EGO_W, EGO_H = 640, 480
# Pure-pursuit lookahead distance L = speed * LOOKAHEAD_TIME_S. Tied to the
# matcher's 1 s trajectory horizon: the lookahead point is the forward
# intersection of the circle of radius L about the robot with the planned path.
LOOKAHEAD_TIME_S = 1.0
ARRIVE_TOL_M = 0.4  # within this of the final waypoint counts as arrived
SETTLE_FRAMES = 45  # extra frames (desiredVel=0) after arrival so the gait settles
PATH_STEP_M = 0.1  # densification spacing for the pure-pursuit target path
# targets mode: when the robot is off the path, its residual lateral offset is added
# to the sampled centerline targets and decayed linearly to zero over this horizon, so
# the targets describe a smooth merge back onto the line (over ~CONVERGE_TIME_S worth of
# travel) instead of demanding an infeasible instant lateral snap. = the far horizon tap
# (1 s) by default, so the offset is fully closed by the farthest target.
CONVERGE_TIME_S = 1.0

# Overhead chase camera: deliberately MORE top-down than run_mujoco.py so the
# full-body gait and the route are both clearly visible. Use a NON-ceiling scene
# variant (val_<N>.xml, not val_<N>_ceiling.xml) so nothing occludes this view.
CHASE_BACK = 1.5  # m behind the robot
CHASE_Z = 5.0  # m camera height (well above the robot)
CHASE_LOOK_Z = 0.9  # m look-at height on the robot
CHASE_DIST = float(np.hypot(CHASE_BACK, CHASE_Z - CHASE_LOOK_Z))
CHASE_ELEV = -float(np.degrees(np.arctan2(CHASE_Z - CHASE_LOOK_Z, CHASE_BACK)))


def yaw_quat(yaw: float) -> np.ndarray:
    """wxyz quaternion of a rotation about +z by `yaw`."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions (a applied after b)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def resample_path(waypoints: np.ndarray, step: float) -> np.ndarray:
    """Densify a polyline to ~`step`-spaced points (for the pursuit target)."""
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


def path_arclength(planned):
    """Cumulative arc-length (m) at each densified path point; arclen[0] = 0."""
    seg = np.linalg.norm(np.diff(planned, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def sample_path_at(planned, arclen, s):
    """Point + unit tangent on the densified path at arc-length `s` (clamped to
    the path). Used by the future-targets query mode to read targets *along* the
    trajectory at fixed arc-lengths ahead of the robot's projection."""
    s = float(np.clip(s, 0.0, arclen[-1]))
    j = int(np.clip(np.searchsorted(arclen, s), 1, len(planned) - 1))
    seg = planned[j] - planned[j - 1]
    seglen = arclen[j] - arclen[j - 1]
    frac = (s - arclen[j - 1]) / seglen if seglen > 1e-9 else 0.0
    pos = planned[j - 1] + seg * frac
    tan = seg / (np.linalg.norm(seg) or 1.0)
    return pos, tan


def pure_pursuit_target(planned, robot_xy, closest, lookahead):
    """Classic pure-pursuit lookahead: the forward intersection of the circle of
    radius `lookahead` about the robot with the planned path.

    Scans the path from the robot's closest point forward and returns the first
    point that lies on/just past the circle (distance >= lookahead) -- a point
    ~`lookahead` away in the travel direction, so steering anticipates the path
    `lookahead` metres ahead. If the rest of the path is all within the circle,
    returns the final waypoint; if the robot has drifted >`lookahead` from the
    path, returns the nearest path point so it steers back on. Returns
    (target_xy, index)."""
    n = len(planned)
    if np.linalg.norm(planned[closest] - robot_xy) >= lookahead:
        return planned[closest], closest  # drifted far off -> head back to path
    j = closest
    while j < n and np.linalg.norm(planned[j] - robot_xy) < lookahead:
        j += 1
    j = min(j, n - 1)
    return planned[j], j


# --- In-scene debug geometry -------------------------------------------------
# Appended to the renderer's MjvScene each frame so the target path + the
# motion-matching command show up as real 3D geometry in the render (the
# overhead chase view), not as a flat 2D overlay on the map panel. RGBA is
# 0..1 (NOT the BGR the cv2 map overlay used).
PATH_RGBA = (1.0, 0.55, 0.0, 1.0)  # planned path, orange floor strip
TARGET_RGBA = (0.1, 0.9, 0.2, 1.0)  # pure-pursuit lookahead target, green
CMD_RGBA = (1.0, 0.85, 0.0, 1.0)  # desiredVel command arrow, yellow
TPOS_RGBA = (0.85, 0.2, 0.8, 1.0)  # matcher predicted command trajectory, magenta
PATH_Z = 0.04  # heights (m) above the z=0 floor for each marker
TPOS_Z = 0.08
TARGET_Z = 0.12
CMD_Z = 0.12


def _decor_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(pos, float),
        np.eye(3).ravel(),
        np.asarray(rgba, np.float32),
    )
    g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1


def _decor_connector(scene, gtype, width, frm, to, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, gtype, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32)
    )
    mujoco.mjv_connector(g, gtype, width, np.asarray(frm, float), np.asarray(to, float))
    g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1


def build_model(scene_xml: Path, g1_xml: Path, backend: str = "opengl", obstacle=None):
    """Merge the Menagerie G1 into the scene MJCF; add a head-mounted 'ego'
    camera (identical placement to run_mujoco.py). For Filament also mount a
    forward fill light on the torso (Filament ignores the MuJoCo headlight).

    obstacle: optional (x, y, sx, sy, sz[, yaw_deg]) box (center xy, full sizes, m,
    optional yaw about +z) added as a static red geom resting on the z=0 floor -- the
    thing the robot jumps over (jump variant) or the A* detour routes around. Purely
    visual: the runtime is kinematic (no collision), so it never physically blocks the
    robot."""
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
    # Realsense D435 mount position from the G1 URDF (d435_link fixed joint off
    # torso_link) with the URDF's 47.6 deg downward pitch REMOVED -> looks +x
    # forward, +z world-up. See run_mujoco.py / docs/navigation_pipeline.md.
    cam.pos = [0.0576235, 0.01753, 0.42987]
    cam.quat = [0.5, 0.5, -0.5, -0.5]

    if backend == "filament":
        lamp = torso.add_light()
        lamp.name = "ego_lamp"
        lamp.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        lamp.pos = [0.0576235, 0.01753, 0.42987]
        lamp.dir = [1.0, 0.0, -0.15]
        lamp.diffuse = [2.5, 2.5, 2.5]
        lamp.specular = [0.1, 0.1, 0.1]
        lamp.castshadow = 0

    if obstacle is not None:
        ox, oy, sx, sy, sz = obstacle[:5]
        yaw = np.radians(obstacle[5]) if len(obstacle) > 5 else 0.0
        box = scene.worldbody.add_geom()
        box.name = "nav_obstacle"
        box.type = mujoco.mjtGeom.mjGEOM_BOX
        box.size = [sx / 2, sy / 2, sz / 2]
        box.pos = [ox, oy, sz / 2]  # rest on the z=0 floor
        box.quat = yaw_quat(yaw)  # align with the path so it stays jumpably shallow
        box.rgba = [0.85, 0.18, 0.18, 1.0]

    return scene.compile()


def colorize_depth(depth: np.ndarray, near=0.1, far=8.0) -> np.ndarray:
    d = np.clip(np.nan_to_num(depth, nan=far, posinf=far), near, far)
    norm = ((d - near) / (far - near) * 255).astype(np.uint8)
    return cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)


def make_renderer(model, height: int, width: int, backend: str):
    """Offscreen renderer; returns ``render(data, camera, depth=False)``."""
    if backend == "filament":
        from molmo_spaces.env.mj_extensions import MjModelBindings
        from molmo_spaces.renderer.filament_rendering import MjFilamentRenderer

        r = MjFilamentRenderer(MjModelBindings(model), height=height, width=width)
        update = r.update
    else:
        r = mujoco.Renderer(model, height, width)
        update = r.update_scene

    def render(data, camera, depth: bool = False, decorate=None) -> np.ndarray:
        if depth:
            r.enable_depth_rendering()
            update(data, camera)
            out = np.array(r.render())
            r.disable_depth_rendering()
            return out
        r.disable_depth_rendering()
        update(data, camera)
        if decorate is not None:  # append in-scene debug geoms after the scene
            decorate(r.scene)  # is built, before it is rasterized
        return np.array(r.render())

    return render


def build_map_panel(occ_path: Path, planned: np.ndarray, pw: int, ph: int, obstacle=None):
    """Static 2D top-down panel (rooms tinted, PLANNED path drawn, plus the
    obstacle box footprint if given). Returns the panel image and a
    world->panel-pixel function. The robot's ACTUAL position is drawn per frame
    by the caller (motion matching drifts off the planned line)."""
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

    traj = np.array([w2m @ np.array([x, y, 0.0, 1.0]) for x, y in planned])  # [row,col]
    cv2.polylines(
        img,
        [np.stack([traj[:, 1], traj[:, 0]], 1).astype(np.int32)],
        False,
        (0, 150, 255),
        5,
        cv2.LINE_AA,
    )

    # Obstacle footprint (the box the jump leaps / the detour routes around): a filled
    # red rectangle (matching the in-scene box). obstacle is (x,y,sx,sy,sz[,yaw_deg]) --
    # index 4 is the height sz, the optional yaw is index 5.
    if obstacle is not None:
        ox_, oy_, sx, sy = obstacle[:4]
        yaw = np.radians(obstacle[5]) if len(obstacle) > 5 else 0.0
        cs, sn = np.cos(yaw), np.sin(yaw)
        corners_w = [
            (ox_ + cs * dx - sn * dy, oy_ + sn * dx + cs * dy)
            for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))
        ]
        poly = np.array(
            [(w2m @ np.array([x, y, 0.0, 1.0]))[::-1] for x, y in corners_w], np.int32
        )  # (col, row) for cv2
        cv2.fillConvexPoly(img, poly, (46, 46, 217))  # BGR ~ the box's red rgba

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


def load_matcher(mm_root: Path):
    """Import mm_g1 from the motion-matching repo and build the matcher."""
    if not (mm_root / "mm_g1").is_dir():
        raise SystemExit(f"motion-matching repo not found at {mm_root} (no mm_g1/ package)")
    sys.path.insert(0, str(mm_root))
    from mm_g1.controller import MotionMatcher
    from mm_g1.data import load_library

    lib = load_library()  # builds data/motion_lib.npz on first use, then caches
    return MotionMatcher(lib)


def write_video(path, frames, fps=STEP_HZ):
    """Encode BGR uint8 frames to an H.264 mp4 via ffmpeg (browser/Notion safe).
    Falls back to OpenCV mp4v only if ffmpeg is not on PATH."""
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


def build_segments(idx_log, clip_id, frame_in_clip, skill):
    """Run-length-encode the per-frame DB index into motion-matching segments.

    Between searches the matcher just advances its playhead by +1, so a run of
    contiguous, same-clip DB indices is exactly "play clip C from frame F and
    step forward N frames". A non-contiguous jump (an inertialized search cut, or
    a triggered jump) starts a new segment. Returns (S,4) int64 rows of
    [clip_id, start_frame_in_clip, n_steps, is_jump]."""
    idx = np.asarray(idx_log)
    segs = []
    start = 0
    for i in range(1, len(idx) + 1):
        contiguous = (
            i < len(idx) and idx[i] == idx[i - 1] + 1 and clip_id[idx[i]] == clip_id[idx[i - 1]]
        )
        if not contiguous:
            g = idx[start]
            segs.append([int(clip_id[g]), int(frame_in_clip[g]), i - start, int(skill[g])])
            start = i
    return np.array(segs, np.int64).reshape(-1, 4)


def save_motion(
    out_dir, qpos, idx_log, cmd_log, clip_id, frame_in_clip, skill, clip_names, traj, meta
):
    """Save the FULL per-frame motion and the MINIMAL motion-matching control
    stream, then print a size comparison (the compression the matcher buys).

    full   -- motion_full.npz: qpos (T,36) world poses + command + DB index/frame.
    minimal -- motion_min.npz: the A* trajectory + RLE motion-index segments
               (clip, start-frame, n-steps-forward, is-jump) + the anchor
               transform. The full pose stream reconstructs from these by
               replaying the segments against the motion library."""
    T = len(qpos)
    segments = build_segments(idx_log, clip_id, frame_in_clip, skill)
    n_jumps = int((segments[:, 3] == 1).sum()) if len(segments) else 0

    np.savez_compressed(
        out_dir / "motion_full.npz",
        qpos=qpos.astype(np.float32),  # (T,36) scene-frame full body pose
        command_vel=cmd_log.astype(np.float32),  # (T,2) desiredVel fed to the matcher
        db_index=idx_log.astype(np.int64),  # (T,) global motion-DB frame per pose
        fps=np.float32(meta["fps"]),
    )
    np.savez_compressed(
        out_dir / "motion_min.npz",
        trajectory=traj.astype(np.float32),  # (N,2) the A* path (the "where to go")
        segments=segments,  # (S,4) [clip_id, start_frame_in_clip, n_steps, is_jump]
        clip_names=np.array(clip_names, object),
        dtheta=np.float32(meta["dtheta"]),  # anchor transform (matcher frame -> scene)
        m0=np.asarray(meta["m0"], np.float32),
        s0=np.asarray(meta["s0"], np.float32),
        start_frame=np.int64(meta["start_frame"]),
        fps=np.float32(meta["fps"]),
    )

    # Logical (uncompressed) representation sizes -- the fair compression metric.
    full_logical = qpos.astype(np.float32).nbytes
    min_logical = segments.nbytes + traj.astype(np.float32).nbytes + 64
    full_disk = (out_dir / "motion_full.npz").stat().st_size
    min_disk = (out_dir / "motion_min.npz").stat().st_size
    print(
        f"motion: {T} frames in {len(segments)} segment(s), {n_jumps} jump(s)\n"
        f"  full    (qpos {T}x36 f32): {full_logical / 1024:8.1f} KiB logical | "
        f"{full_disk / 1024:7.1f} KiB on disk -> motion_full.npz\n"
        f"  minimal (traj + segments): {min_logical / 1024:8.1f} KiB logical | "
        f"{min_disk / 1024:7.1f} KiB on disk -> motion_min.npz\n"
        f"  compression: {full_logical / max(min_logical, 1):.1f}x logical, "
        f"{full_disk / max(min_disk, 1):.1f}x on disk"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--scene",
        type=Path,
        required=True,
        help="scene MJCF (use the NON-ceiling variant for the top-down view)",
    )
    ap.add_argument("--path", type=Path, required=True, help="path.npz from plan.py")
    ap.add_argument(
        "--occupancy", type=Path, default=None, help="occupancy.npz (default: next to path)"
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--g1", type=Path, default=G1_MM_XML, help="Menagerie G1 MJCF the matcher targets"
    )
    ap.add_argument("--mm-root", type=Path, default=MM_ROOT, help="motion-matching repo root")
    ap.add_argument("--speed", type=float, default=WALK_SPEED, help="desired travel speed (m/s)")
    ap.add_argument(
        "--query-mode",
        choices=["velocity", "targets"],
        default="targets",
        help="how the matcher is driven (default: targets): 'targets' = future "
        "targets sampled along the path are fed to the matcher's query directly "
        "(matcher.step_targets); 'velocity' = pure-pursuit desiredVel and the "
        "matcher's springs predict the future-target query (matcher.step)",
    )
    ap.add_argument(
        "--converge-time",
        type=float,
        default=CONVERGE_TIME_S,
        help="targets mode: horizon (s) over which an off-path lateral offset decays "
        "to zero (merge-onto-path rate). 0 disables the blend (pure-centerline targets)",
    )
    ap.add_argument(
        "--obstacle",
        type=str,
        default=None,
        help="place a box obstacle 'x,y,sx,sy,sz' (center xy, full sizes m) on the floor; "
        "rendered in all cameras. Used by the jump variant (leap over) and the detour "
        "variant (A* routes around it). Visual only -- the kinematic runtime has no collision",
    )
    ap.add_argument(
        "--jump",
        action="store_true",
        help="trigger the matcher's jump skill to leap the --obstacle (a fixed scripted "
        "leap; fire it on a straight, clear segment so the arc clears the box and lands free)",
    )
    ap.add_argument(
        "--jump-lead",
        type=float,
        default=1.5,
        help="run-up lead distance (m) before the obstacle at which the jump is triggered",
    )
    ap.add_argument(
        "--max-frames", type=int, default=0, help="hard frame cap (0 = auto from path length)"
    )
    ap.add_argument(
        "--renderer",
        choices=["opengl", "filament"],
        default="opengl",
        help="opengl (mlspaces-mujoco env) or filament (mlspaces env, better RGB)",
    )
    args = ap.parse_args()

    occ_path = args.occupancy or args.path.parent / "occupancy.npz"
    out_dir = (args.out_dir or args.path.parent / "nav_run_mm").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    obstacle = None
    if args.obstacle:
        obstacle = tuple(float(v) for v in args.obstacle.split(","))
        if len(obstacle) not in (5, 6):
            raise SystemExit("--obstacle must be 'x,y,sx,sy,sz' or 'x,y,sx,sy,sz,yaw_deg'")
    if args.jump and obstacle is None:
        raise SystemExit("--jump needs --obstacle (the box to leap over)")

    p = np.load(args.path, allow_pickle=True)
    waypoints = p["waypoints"].astype(float)
    print(f"path: {p['start_room']} -> {p['goal_room']}  ({len(waypoints)} waypoints)")

    # Densified planned path for pure pursuit + the top-down reference line.
    planned = resample_path(waypoints, PATH_STEP_M)
    path_len = float(np.sum(np.linalg.norm(np.diff(planned, axis=0), axis=1)))
    # Subsampled (~0.2 m) copy used to draw the path as a 3D floor strip in-scene.
    stride = max(1, int(round(0.2 / PATH_STEP_M)))
    path3d = planned[::stride]
    if not np.array_equal(path3d[-1], planned[-1]):
        path3d = np.vstack([path3d, planned[-1]])
    arclen = path_arclength(planned)  # for the targets-query mode (sample along path)

    # Jump trigger point: arc-length where the obstacle sits on the path, minus the
    # run-up lead. The jump is fired once when the robot's projection passes this -- on
    # the straight, clear segment the obstacle was placed on, so the leap clears it.
    jump_arclen = None
    if args.jump:
        obs_xy = np.array(obstacle[:2])
        obs_idx = int(np.argmin(np.linalg.norm(planned - obs_xy, axis=1)))
        jump_arclen = max(0.0, arclen[obs_idx] - args.jump_lead)

    matcher = load_matcher(args.mm_root)
    # Motion-library index tables for the compact ("minimal") representation:
    # which DB frame each output pose came from, its clip, and its frame-in-clip.
    start_frame = int(matcher.animFrame)  # the matcher's reset playhead (global DB index)
    clip_id = np.asarray(matcher.lib["clip_id"])
    skill = np.asarray(matcher.skill)
    if "frame_in_clip" in matcher.lib:
        frame_in_clip = np.asarray(matcher.lib["frame_in_clip"])
    else:  # derive from clip-id run boundaries
        frame_in_clip = np.zeros(len(clip_id), np.int32)
        for ci in np.unique(clip_id):
            frame_in_clip[clip_id == ci] = np.arange(int(np.sum(clip_id == ci)))
    clip_names = [str(s) for s in matcher.lib["clip_names"]] if "clip_names" in matcher.lib else []

    # --- Rigid transform: matcher frame -> scene frame, anchored at path start.
    # The matcher resets to some world pose from its motion DB; map that onto the
    # path's first waypoint with the path's initial heading. dtheta rotates the
    # matcher's heading to the scene-frame initial tangent.
    m0 = matcher.rootPos[:2].copy()
    my0 = float(matcher.rootYaw)
    s0 = planned[0].copy()
    seg0 = planned[min(8, len(planned) - 1)] - planned[0]
    sy0 = float(np.arctan2(seg0[1], seg0[0])) if np.linalg.norm(seg0) > 1e-6 else my0
    dtheta = sy0 - my0
    R = rot2d(dtheta)
    Rinv = rot2d(-dtheta)
    dquat = yaw_quat(dtheta)

    def m2s_xy(p_m):  # matcher xy -> scene xy
        return R @ (np.asarray(p_m) - m0) + s0

    def s2m_xy(p_s):  # scene xy -> matcher xy
        return Rinv @ (np.asarray(p_s) - s0) + m0

    model = build_model(args.scene, args.g1, args.renderer, obstacle=obstacle)
    data = mujoco.MjData(model)
    base_adr = model.joint("g1_floating_base_joint").qposadr[0]

    render = make_renderer(model, EGO_H, EGO_W, args.renderer)
    followcam = mujoco.MjvCamera()
    followcam.type = mujoco.mjtCamera.mjCAMERA_FREE
    followcam.distance = CHASE_DIST
    followcam.elevation = CHASE_ELEV

    map_base, world_to_panel = build_map_panel(occ_path, planned, EGO_W, EGO_H, obstacle=obstacle)

    if args.max_frames > 0:
        max_frames = args.max_frames
    else:
        # generous cap: travel at ~0.7x the commanded speed, then 2x slack.
        est = path_len / max(0.3, 0.7 * args.speed) * STEP_HZ
        max_frames = int(est * 2 + 120)

    lookahead = args.speed * LOOKAHEAD_TIME_S  # pure-pursuit circle radius (m)
    horizon_s = matcher.Ttimes  # [1/3, 2/3, 1] s -- the matcher's future-target taps
    print(
        f"merging Menagerie G1; full-body motion matching @ {STEP_HZ:.0f} Hz, "
        f"speed {args.speed:.1f} m/s, query-mode={args.query_mode}, "
        f"lookahead {lookahead:.2f} m, path {path_len:.1f} m  [{args.renderer} renderer]"
    )

    rgb_frames, depth_frames, follow_frames, combined = [], [], [], []
    qpos_log, idx_log, cmd_log = [], [], []  # full pose, DB index, command velocity (scene)
    settle = 0
    nframes = 0
    jump_fired = False
    while nframes < max_frames:
        nframes += 1
        robot_s = m2s_xy(matcher.rootPos[:2])
        closest = int(np.argmin(np.linalg.norm(planned - robot_s, axis=1)))
        arrived = (
            closest >= len(planned) - 2 and np.linalg.norm(planned[-1] - robot_s) < ARRIVE_TOL_M
        )
        if arrived:
            settle += 1

        # Jump trigger: once the robot's projection reaches the run-up point before the
        # obstacle, request the leap. The matcher enters it on the next step (step_targets
        # honours the trigger); while airborne it rides the jump clip and ignores targets,
        # then the merge blend pulls the post-landing drift back onto the path.
        if jump_arclen is not None and not jump_fired and arclen[closest] >= jump_arclen:
            matcher.trigger_jump()
            jump_fired = True
            print(f"  jump triggered at arclen {arclen[closest]:.1f} m "
                  f"(obstacle ~{arclen[closest] + args.jump_lead:.1f} m)")

        if args.query_mode == "targets":
            # --- Future-targets query: sample the path at fixed arc-lengths
            # AHEAD OF THE ROBOT'S PROJECTION (speed * [1/3, 2/3, 1] s), then add the
            # robot's residual lateral offset decayed to zero across the horizon so the
            # targets describe a smooth MERGE back onto the line. Sampling ahead of the
            # projection (arclen[closest]) -- not the robot itself -- keeps the targets
            # on the path ahead; the decaying-offset blend below makes the near targets
            # sit on the diagonal approach toward the line rather than demanding an
            # instant (infeasible) lateral snap when the robot has drifted off.
            z0 = float(matcher.rootPos[2])
            if arrived:
                pts = [planned[-1]] * len(horizon_s)
                tans = [planned[-1] - planned[-2]] * len(horizon_s)
                desired_vel_s = np.zeros(2)
            else:
                # arc-length of the robot's projection; sample targets ahead of it.
                # NB: must NOT be named `s0` -- that is the anchor translation closed
                # over by m2s_xy/s2m_xy; shadowing it corrupts the scene<->matcher map.
                s_proj = arclen[closest]
                err = robot_s - planned[closest]  # cross-track offset: projection -> robot
                base = [sample_path_at(planned, arclen, s_proj + args.speed * t) for t in horizon_s]
                # Add the residual offset, weight w: 1 (near) -> 0 (far over CONVERGE_TIME_S),
                # and derive each facing from the resulting approach chain (robot -> targets)
                # so the heading curves diagonally in and straightens onto the path.
                conv = args.converge_time
                pts, tans = [], []
                prev = robot_s
                for (b, ptan), t in zip(base, horizon_s):
                    if conv > 0:  # decaying-offset merge blend
                        p = b + err * max(0.0, 1.0 - t / conv)
                        step = p - prev
                        tans.append(step if np.linalg.norm(step) > 1e-6 else ptan)
                    else:  # blend disabled: pure-centerline targets + path tangents
                        p = b
                        tans.append(ptan)
                    pts.append(p)
                    prev = p
                fd = pts[0] - robot_s
                nrm = np.linalg.norm(fd)
                desired_vel_s = fd / nrm * args.speed if nrm > 1e-6 else np.zeros(2)
            target_s = pts[-1]  # the 1 s-horizon target (green sphere)
            tpos_m = np.array([[*s2m_xy(p), z0] for p in pts])
            tdir_m = np.array([[*(Rinv @ (t / (np.linalg.norm(t) or 1.0))), 0.0] for t in tans])
            qm = matcher.step_targets(tpos_m, tdir_m)
        else:
            # --- Velocity query: pure-pursuit lookahead -> desiredVel; the
            # matcher's springs predict the future-target query from it.
            if arrived:
                target_s = planned[-1]
                desired_vel_s = np.zeros(2)  # stop -> matcher settles to idle
                desired_vel_m = np.zeros(3)
            else:
                target_s, _ = pure_pursuit_target(planned, robot_s, closest, lookahead)
                dir_s = target_s - robot_s
                nrm = np.linalg.norm(dir_s)
                unit_s = dir_s / nrm if nrm > 1e-6 else np.zeros(2)
                desired_vel_s = unit_s * args.speed  # command, scene frame (viz/log)
                desired_vel_m = np.array([*(Rinv @ unit_s), 0.0]) * args.speed  # matcher frame
            qm = matcher.step(desired_vel_m, [0.0, 0.0, 0.0])  # face = follow velocity

        # --- map matcher-frame pose into the scene frame, set the FULL qpos
        scene_xy = m2s_xy(qm[0:2])
        qscene = qm.copy()
        qscene[0:2] = scene_xy
        qscene[3:7] = quat_mul(dquat, qm[3:7])
        data.qpos[base_adr : base_adr + 36] = qscene
        # Position-only pipeline (FK + camera/light placement), NOT mj_forward:
        # we place the robot kinematically and only render, so we never want the
        # collision/constraint solver -- and a transient wall penetration would
        # otherwise make it fail with "FactorizeHessian: rank-deficient ...".
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_camlight(model, data)

        qpos_log.append(qscene.astype(np.float32))
        idx_log.append(int(matcher.animFrame))  # which DB frame produced this pose
        cmd_log.append(desired_vel_s.astype(np.float32))

        x, y = float(qscene[0]), float(qscene[1])
        yaw = matcher.rootYaw + dtheta

        # Predicted command trajectory (matcher.Tpos), matcher frame -> scene xy.
        tpred = [m2s_xy(tp[:2]) for tp in matcher.Tpos]

        def decorate(scene, x=x, y=y, target_s=target_s, dvs=desired_vel_s, tpred=tpred):
            """Draw the target path + the motion-matching command as real 3D
            geometry in the render (so it shows in the overhead chase, not on a
            flat map). Called only for the chase camera, so ego RGB/depth are
            left clean."""
            # planned path: orange capsule strip on the floor
            for a, b in zip(path3d[:-1], path3d[1:]):
                _decor_connector(
                    scene,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    0.03,
                    [a[0], a[1], PATH_Z],
                    [b[0], b[1], PATH_Z],
                    PATH_RGBA,
                )
            # matcher's predicted command trajectory (magenta): robot -> Tpos
            chain = [(x, y), *[(t[0], t[1]) for t in tpred]]
            for a, b in zip(chain[:-1], chain[1:]):
                _decor_connector(
                    scene,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    0.02,
                    [a[0], a[1], TPOS_Z],
                    [b[0], b[1], TPOS_Z],
                    TPOS_RGBA,
                )
            for t in tpred:
                _decor_sphere(scene, [t[0], t[1], TPOS_Z], 0.05, TPOS_RGBA)
            # pure-pursuit lookahead target (green sphere)
            _decor_sphere(scene, [target_s[0], target_s[1], TARGET_Z], 0.12, TARGET_RGBA)
            # command-velocity arrow (yellow): desiredVel direction, ~0.8 m long
            if np.linalg.norm(dvs) > 1e-6:
                u = dvs / np.linalg.norm(dvs)
                _decor_connector(
                    scene,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.05,
                    [x, y, CMD_Z],
                    [x + 0.8 * u[0], y + 0.8 * u[1], CMD_Z],
                    CMD_RGBA,
                )

        rgb = render(data, "ego")[:, :, ::-1].copy()  # ego stays clean (no decor)
        rgb_frames.append(rgb)
        depth = colorize_depth(render(data, "ego", depth=True))
        depth_frames.append(depth)

        followcam.lookat[:] = [x, y, CHASE_LOOK_Z]
        followcam.azimuth = float(np.degrees(yaw))
        follow = render(data, followcam, decorate=decorate)[:, :, ::-1].copy()
        follow_frames.append(follow)

        # top-down map: just the planned path (baked in) + the robot pose marker.
        mp = map_base.copy()
        dot = world_to_panel(x, y)
        head = world_to_panel(x + 0.6 * np.cos(yaw), y + 0.6 * np.sin(yaw))
        cv2.line(mp, dot, head, (40, 40, 40), 3, cv2.LINE_AA)
        cv2.circle(mp, dot, 8, (0, 0, 230), -1, cv2.LINE_AA)

        top = np.hstack(
            [
                label(mp, "Top-down map"),
                label(follow, f"Chase ({args.query_mode}): target + command"),
            ]
        )
        bot = np.hstack([label(rgb, "Ego RGB"), label(depth, "Ego depth")])
        combined.append(np.vstack([top, bot]))

        if settle >= SETTLE_FRAMES:
            break

    if nframes >= max_frames and settle < SETTLE_FRAMES:
        print(f"  note: hit max-frames cap ({max_frames}) before arriving at the goal.")

    write_video(out_dir / "combined.mp4", combined)
    write_video(out_dir / "ego_rgb.mp4", rgb_frames)
    write_video(out_dir / "ego_depth.mp4", depth_frames)
    write_video(out_dir / "follow.mp4", follow_frames)

    idx = np.linspace(0, len(combined) - 1, 6).astype(int)
    tiles = [cv2.resize(combined[i], (640, 480)) for i in idx]
    sheet = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
    cv2.imwrite(str(out_dir / "combined_montage.png"), sheet)

    save_motion(
        out_dir,
        np.array(qpos_log),
        np.array(idx_log, np.int64),
        np.array(cmd_log),
        clip_id,
        frame_in_clip,
        skill,
        clip_names,
        p["waypoints_raw"].astype(np.float32)
        if "waypoints_raw" in p
        else waypoints.astype(np.float32),
        dict(dtheta=dtheta, m0=m0, s0=s0, start_frame=start_frame, fps=STEP_HZ),
    )

    print(f"wrote {len(combined)} frames -> {out_dir}/combined.mp4 (+ individual streams)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
