#!/usr/bin/env python3
"""Build a sim-agnostic 2D occupancy + room-label grid from a scene MJCF.

The scene is rendered top-down (orthographic segmentation): floor geoms become
navigable space, everything else becomes obstacle. Wall geometry is then burned
into the obstacle layer separately -- vertical wall panels barely register in a
straight-down render, and abutting room floors leave no gap between rooms, so
the segmentation alone misses interior walls.

Output: an ``occupancy.npz`` next to the scene:
  occupancy     (H,W) bool   -- True = free / navigable, raw (NOT agent-dilated)
  room_map      (H,W) int32  -- per-pixel room index (0 = none); navigable only
  room_names    (R,)  str    -- room_names[i-1] is the name of room index i
  world_to_map  (2,4) float  -- [x,y,z,1] -> [row,col]
  map_to_world  (2,3) float  -- [row,col,1] -> [x,y]
  px_per_m      float

Works on any MJCF scene -- mansion exports and procthor (10k / objaverse)
scenes alike. The grid is sim-agnostic (one shared world frame backs both the
MuJoCo and the IsaacSim runtime). Agent-radius clearance is applied later, at
planning time, so the stored grid stays robot-agnostic.

Run from the ``mlspaces-mujoco`` env (OpenGL renderer).
"""

import argparse
import glob
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import mujoco
import numpy as np

from molmo_spaces.utils.linalg_utils import inverse_homogeneous_matrix
from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb
from molmo_spaces.utils.scene_maps import _get_renderer

MANSION_EXPORT_ROOT = Path.home() / "Projects" / "mansion" / "mjcf_export"
WALL_THICKNESS_M = 0.12  # nominal thickness used when burning walls into the grid
DOOR_CARVE_M = 0.6  # opening carved back through a wall at each door geom


def find_latest_scene() -> Path:
    hits = glob.glob(str(MANSION_EXPORT_ROOT / "**" / "scene.xml"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"No scene.xml found under {MANSION_EXPORT_ROOT}")
    return Path(max(hits, key=os.path.getmtime))


def geom_name(model, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""


def is_floor(name: str) -> bool:
    n = name.lower()
    # mansion: "floor_F1_*"  |  procthor: "room_N_*" / "room|N".
    # Match at a name boundary so objaverse objects ("floorlamp_*",
    # "room_divider_*") are not misread as floor geoms.
    return bool(re.match(r"floor($|[^a-z])", n)) or bool(re.match(r"room[_|]\d", n))


def clean_room_name(name: str) -> str:
    s = re.sub(r"^floor_", "", name, flags=re.I)
    s = re.sub(r"^[a-z]?\d+_", "", s, flags=re.I)  # strip a floor tag like "F1_"
    s = re.sub(r"_visual(_\d+)?$", "", s, flags=re.I)  # strip procthor geom suffix
    return s.replace("|", " ").replace("_", " ").strip() or name


def classify_geoms(model):
    floors, walls, doors = [], [], []
    for g in range(model.ngeom):
        n = geom_name(model, g)
        if not n:
            continue
        nl = n.lower()
        if is_floor(n) and model.geom(g).contype == 0:
            floors.append(g)
        elif "door" in nl:
            doors.append(g)
        elif "wall" in nl:
            walls.append(g)
    return floors, walls, doors


def geom_world_verts(model, data, gid):
    """World-frame vertices of a geom (mesh vertices, or box corners as fallback)."""
    R = data.geom_xmat[gid].reshape(3, 3)
    p = data.geom_xpos[gid]
    did = int(model.geom_dataid[gid])
    if did >= 0 and int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        mv = model.mesh_vert
        mv = mv.reshape(-1, 3) if mv.ndim == 1 else mv
        a, n = int(model.mesh_vertadr[did]), int(model.mesh_vertnum[did])
        v = mv[a : a + n].astype(np.float64)
    else:
        s = model.geom_size[gid].astype(np.float64)
        v = np.array(
            [[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])]
        )
    return v @ R.T + p


def render_topdown(model, data, floor_ids, px_per_m, device_id):
    """Orthographic top-down segmentation render. Returns seg (geom-id image),
    world_to_map (2x4), map_to_world (2x3) and the rounding-corrected px_per_m."""
    center, size = geom_aabb(model, data, floor_ids, tight_mesh=False)
    size = size + np.array([2.0, 2.0, 0.0])  # 1 m buffer per side

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.distance = 5.0
    cam.azimuth = 0
    cam.elevation = -90
    cam.orthographic = 1

    h = round(px_per_m * size[0])
    w = round(px_per_m * size[1])
    px = h / size[0]  # absorb the round()

    renderer = _get_renderer(model, width=w, height=h, device_id=device_id, use_filament=False)
    renderer.update(data, cam)
    for sc in renderer.scene.camera:
        sc.orthographic = 1
        sc.frustum_bottom = -size[0] / 2
        sc.frustum_top = size[0] / 2

    cam_to_world = np.eye(4)
    cam_to_world[:3, 3] = renderer.scene.camera[0].pos
    x_ax = np.cross(renderer.scene.camera[0].up, -renderer.scene.camera[0].forward)
    cam_to_world[:3, :3] = np.column_stack(
        (x_ax, renderer.scene.camera[0].up, -renderer.scene.camera[0].forward)
    )
    assert np.allclose(cam_to_world[:3, 2], [0, 0, 1]), "camera must look straight down"

    renderer.enable_segmentation_rendering()
    seg = renderer.render()[..., 0].astype(np.int32)
    renderer.close()

    cam_to_map = np.array([[0, -px, 0, h / 2], [px, 0, 0, w / 2]])
    world_to_map = cam_to_map @ inverse_homogeneous_matrix(cam_to_world)
    map_to_centered = np.array([[0, 1, -w / 2], [-1, 0, h / 2], [0, 0, 1]])
    centered_to_cam = np.array([[1 / px, 0, 0], [0, 1 / px, 0], [0, 0, 1]])
    cam_to_world_floor = cam_to_world[:-1, [0, 1, 3]].copy()
    cam_to_world_floor[2, 2] = 0
    map_to_world = cam_to_world_floor @ centered_to_cam @ map_to_centered
    return seg, world_to_map, map_to_world, px


def burn_walls(obstacle, model, data, wall_ids, world_to_map, px_per_m):
    """Rasterize wall geometry into ``obstacle`` (uint8, 1 = obstacle).

    Walls are vertical panels: project each to the floor plane, take the
    principal-axis segment, and draw it as a thick line."""
    thick = max(3, int(round(WALL_THICKNESS_M * px_per_m)))
    for g in wall_ids:
        xy = geom_world_verts(model, data, g)[:, :2]
        if len(xy) < 2:
            continue
        centroid = xy.mean(axis=0)
        d = xy - centroid
        axis = np.linalg.svd(d, full_matrices=False)[2][0]
        t = d @ axis
        ends = np.array(
            [
                [*(centroid + axis * t.min()), 0.0, 1.0],
                [*(centroid + axis * t.max()), 0.0, 1.0],
            ]
        )
        rc = ends @ world_to_map.T  # (2, 2) -> [row, col]
        cv2.line(
            obstacle,
            (int(rc[0, 1]), int(rc[0, 0])),
            (int(rc[1, 1]), int(rc[1, 0])),
            1,
            thick,
        )


def _principal_segment(model, data, gid):
    """The geom's xy footprint as a principal-axis segment ((r0,c0,..),(r1,..))
    style world endpoints; None if degenerate."""
    xy = geom_world_verts(model, data, gid)[:, :2]
    if len(xy) < 2:
        return None
    centroid = xy.mean(axis=0)
    d = xy - centroid
    axis = np.linalg.svd(d, full_matrices=False)[2][0]
    t = d @ axis
    return centroid + axis * t.min(), centroid + axis * t.max()


def carve_doors(obstacle, model, data, door_ids, world_to_map, px_per_m):
    """Punch free openings back through walls at door geoms.

    A doorway has wall above it (lintel) and a frame, so it reads as obstacle
    in a top-down render and gets sealed by the wall burn-in -- but it is
    passable. Carve the opening back along each door geom's footprint."""
    thick = max(5, int(round(DOOR_CARVE_M * px_per_m)))
    for g in door_ids:
        seg = _principal_segment(model, data, g)
        if seg is None:
            continue
        a, b = seg
        rc0 = world_to_map @ np.array([a[0], a[1], 0.0, 1.0])
        rc1 = world_to_map @ np.array([b[0], b[1], 0.0, 1.0])
        cv2.line(
            obstacle,
            (int(rc0[1]), int(rc0[0])),
            (int(rc1[1]), int(rc1[0])),
            0,
            thick,
        )


def build(model_path: Path, px_per_m: int, device_id):
    spec = mujoco.MjSpec.from_file(str(model_path))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    floors, walls, doors = classify_geoms(model)
    if not floors:
        raise RuntimeError("No floor geoms found (visual geoms named floor_* / room|*)")

    seg, world_to_map, map_to_world, px = render_topdown(model, data, floors, px_per_m, device_id)

    # obstacle: 1 where the pixel is not a floor geom; burn walls in, then carve
    # doorways back open (they are passable but read as obstacle top-down).
    obstacle = np.ones(seg.shape, np.uint8)
    for fid in floors:
        obstacle[seg == fid] = 0
    burn_walls(obstacle, model, data, walls, world_to_map, px)
    carve_doors(obstacle, model, data, doors, world_to_map, px)

    occupancy = obstacle == 0  # True = free / navigable

    # per-pixel room index (1..R); 0 where not navigable or not a room.
    room_map = np.zeros(seg.shape, np.int32)
    room_names = []
    for idx, fid in enumerate(floors, start=1):
        room_map[seg == fid] = idx
        room_names.append(clean_room_name(geom_name(model, fid)))
    room_map[~occupancy] = 0

    return {
        "occupancy": occupancy,
        "room_map": room_map,
        "room_names": np.array(room_names),
        "world_to_map": world_to_map,
        "map_to_world": map_to_world,
        "px_per_m": float(px),
        "n_walls": len(walls),
        "n_doors": len(doors),
    }


def debug_image(occupancy, room_map, room_names) -> np.ndarray:
    """White = free, black = obstacle, rooms tinted by index."""
    img = np.where(occupancy[..., None], 255, 30).astype(np.uint8).repeat(3, axis=2)
    n = max(len(room_names), 1)
    hues = np.linspace(0, 179, n, endpoint=False).astype(np.uint8)
    hsv = np.stack([hues, np.full(n, 90, np.uint8), np.full(n, 255, np.uint8)], axis=1)
    colors = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]
    for i in range(1, n + 1):
        img[room_map == i] = colors[i - 1]
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scene", type=Path, default=None, help="scene.xml (default: latest mansion export)"
    )
    ap.add_argument("--px-per-m", type=int, default=100, help="grid resolution (pixels per metre)")
    ap.add_argument("--device-id", type=int, default=None, help="EGL render device id")
    ap.add_argument(
        "--out", type=Path, default=None, help="output .npz (default: <scene_dir>/occupancy.npz)"
    )
    args = ap.parse_args()

    # abspath (NOT resolve): keep symlinks intact. Scenes symlinked into the
    # asset store reference sibling dirs (procthor: ../../objects/thor) via the
    # asset-store layout; resolving the symlink into the resources store breaks
    # those relative mesh paths.
    scene = Path(os.path.abspath(args.scene or find_latest_scene()))
    out = args.out or scene.parent / "occupancy.npz"
    print(f"Scene: {scene}")

    r = build(scene, args.px_per_m, args.device_id)
    occ = r["occupancy"]

    np.savez(
        out,
        occupancy=occ,
        room_map=r["room_map"],
        room_names=r["room_names"],
        world_to_map=r["world_to_map"],
        map_to_world=r["map_to_world"],
        px_per_m=r["px_per_m"],
    )
    debug_png = out.with_name(out.stem + "_debug.png")
    cv2.imwrite(str(debug_png), debug_image(occ, r["room_map"], r["room_names"]))

    print(f"Grid:  {occ.shape[1]} x {occ.shape[0]} px  @ {r['px_per_m']:.1f} px/m")
    print(
        f"Free:  {occ.mean():.1%} of the grid  |  walls: {r['n_walls']}  doors carved: {r['n_doors']}"
    )
    print(f"Rooms ({len(r['room_names'])}):")
    for i, name in enumerate(r["room_names"], start=1):
        px_count = int((r["room_map"] == i).sum())
        print(f"  {i}. {name}  ({px_count} px)")
    print(f"\nWrote: {out}")
    print(f"       {debug_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
