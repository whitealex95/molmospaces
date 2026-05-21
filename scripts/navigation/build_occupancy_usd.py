#!/usr/bin/env python3
"""Build the navigation occupancy grid directly from a USD scene.

The MJCF route (build_occupancy.py) needs the scene's MJCF to compile, but the
objaverse MJCF object library is an incomplete subset -- many procthor-objaverse
scenes' MJCFs fail on a missing object. This script produces the same
``occupancy.npz`` from the USD geometry instead: it rasterizes the room-floor,
wall and doorway prims (procthor USD names them ``room_N_visual_0`` /
``wall_*_visual_0`` / ``doorway_*`` -- the same scheme as the MJCF).

Output is byte-compatible with build_occupancy.py's, so plan.py and
run_isaac.py consume it unchanged.

Run from the ``mlspaces-isaac`` env (needs ``pxr``).
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from pxr import Gf, Usd, UsdGeom

DOOR_MARGIN_M = 0.25  # extra carve around each doorway footprint


def is_floor(n):
    return bool(re.match(r"room[_|]\d", n.lower()))


def is_wall(n):
    nl = n.lower()
    return "wall" in nl and "visual" in nl and "collision" not in nl


def clean_room_name(n):
    s = re.sub(r"_visual(_\d+)?$", "", n, flags=re.I)
    return s.replace("_", " ").strip() or n


def mesh_world_tris(prim, xcache):
    """World-space triangles (T,3,3) of a UsdGeom.Mesh prim, or None."""
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    idx = mesh.GetFaceVertexIndicesAttr().Get()
    if not pts or not counts or not idx:
        return None
    p = np.array([[v[0], v[1], v[2], 1.0] for v in pts])
    m = np.array(xcache.GetLocalToWorldTransform(prim))  # 4x4, USD row-vector
    world = (p @ m)[:, :3]
    tris, o = [], 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((idx[o], idx[o + k], idx[o + k + 1]))
        o += c
    if not tris:
        return None
    return world[np.array(tris)]


def fill_tris(grid, tris_xy, value):
    """Rasterize world-xy triangles into a grid via the world->pixel map."""
    for t in tris_xy:
        cv2.fillConvexPoly(grid, t.astype(np.int32), value, cv2.LINE_8)


def build_from_usd(usd_path, px_per_m):
    stage = Usd.Stage.Open(str(usd_path))
    xcache = UsdGeom.XformCache()
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )

    geo = None
    for r in stage.GetPseudoRoot().GetChildren():
        g = r.GetChild("Geometry")
        if g and g.IsValid():
            geo = g
            break
    if geo is None:
        raise RuntimeError("no /<scene>/Geometry scope in the USD")

    # direct children of the scene Geometry scope: room_*/wall_* are Meshes,
    # doorways and furniture/decor are Xforms, the ceiling (if any) is a Mesh
    # that matches neither floor nor wall and is correctly ignored.
    floor_prims, wall_prims, door_prims, furn_prims = {}, [], [], []
    for c in geo.GetChildren():
        n = c.GetName()
        t = str(c.GetTypeName())
        if t == "Mesh" and is_floor(n):
            floor_prims.setdefault(n, []).append(c)
        elif t == "Mesh" and is_wall(n):
            wall_prims.append(c)
        elif n.lower().startswith("doorway_"):
            door_prims.append(c)
        elif t == "Xform":
            furn_prims.append(c)  # furniture / decor objects
    if not floor_prims:
        raise RuntimeError("no room_*_visual_* floor meshes found in the USD")

    # world-xy extent of all floor geometry -> grid + transforms
    floor_tris = {
        name: [mesh_world_tris(p, xcache) for p in prims] for name, prims in floor_prims.items()
    }
    floor_tris = {n: [t for t in ts if t is not None] for n, ts in floor_tris.items()}
    allf = np.concatenate([np.concatenate(ts) for ts in floor_tris.values() if ts])
    xmin, ymin = allf[..., 0].min() - 1.0, allf[..., 1].min() - 1.0
    xmax, ymax = allf[..., 0].max() + 1.0, allf[..., 1].max() + 1.0
    px = float(px_per_m)
    w = int(round((xmax - xmin) * px))
    h = int(round((ymax - ymin) * px))
    world_to_map = np.array([[0, px, 0, -ymin * px], [px, 0, 0, -xmin * px]])
    map_to_world = np.array([[0, 1 / px, xmin], [1 / px, 0, ymin]])

    def to_px(tris):
        # (T,3,3) world -> (T,3,2) pixel (col,row) for cv2
        col = (tris[..., 0] - xmin) * px
        row = (tris[..., 1] - ymin) * px
        return np.stack([col, row], axis=-1)

    room_map = np.zeros((h, w), np.int32)
    room_names = []
    for idx, (name, ts) in enumerate(sorted(floor_tris.items()), start=1):
        room_names.append(clean_room_name(name))
        for t in ts:
            fill_tris(room_map, to_px(t), idx)

    free = (room_map > 0).astype(np.uint8)  # 1 = free / navigable

    # walls + furniture -> obstacle (carve them back out of the floor)
    obstacle = np.zeros((h, w), np.uint8)
    for p in wall_prims:
        t = mesh_world_tris(p, xcache)
        if t is not None:
            fill_tris(obstacle, to_px(t), 1)
    n_furn = 0
    for p in furn_prims:
        rng = bbox_cache.ComputeWorldBound(p).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        mn, mx = rng.GetMin(), rng.GetMax()
        col = (np.array([mn[0], mx[0], mx[0], mn[0]]) - xmin) * px
        row = (np.array([mn[1], mn[1], mx[1], mx[1]]) - ymin) * px
        cv2.fillConvexPoly(obstacle, np.stack([col, row], axis=-1).astype(np.int32), 1)
        n_furn += 1
    free[obstacle == 1] = 0
    room_map[obstacle == 1] = 0

    # doorways -> carve free (rooms must stay connected through openings)
    for p in door_prims:
        rng = bbox_cache.ComputeWorldBound(p).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        mn, mx = rng.GetMin(), rng.GetMax()
        m = DOOR_MARGIN_M
        col = (np.array([mn[0] - m, mx[0] + m, mx[0] + m, mn[0] - m]) - xmin) * px
        row = (np.array([mn[1] - m, mn[1] - m, mx[1] + m, mx[1] + m]) - ymin) * px
        cv2.fillConvexPoly(free, np.stack([col, row], axis=-1).astype(np.int32), 1)

    return {
        "occupancy": free.astype(bool),
        "room_map": room_map,
        "room_names": np.array(room_names),
        "world_to_map": world_to_map,
        "map_to_world": map_to_world,
        "px_per_m": px,
        "n_walls": len(wall_prims),
        "n_doors": len(door_prims),
        "n_furniture": n_furn,
    }


def debug_image(occ, room_map, room_names):
    img = np.where(occ[..., None], 255, 30).astype(np.uint8).repeat(3, axis=2)
    n = max(len(room_names), 1)
    hsv = np.stack(
        [
            np.linspace(0, 179, n, endpoint=False).astype(np.uint8),
            np.full(n, 90, np.uint8),
            np.full(n, 255, np.uint8),
        ],
        axis=1,
    )
    colors = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]
    for i in range(1, n + 1):
        img[room_map == i] = colors[i - 1]
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, required=True, help="USD scene (.usd/.usda)")
    ap.add_argument("--px-per-m", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True, help="output occupancy.npz")
    args = ap.parse_args()

    r = build_from_usd(args.scene, args.px_per_m)
    occ = r["occupancy"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        occupancy=occ,
        room_map=r["room_map"],
        room_names=r["room_names"],
        world_to_map=r["world_to_map"],
        map_to_world=r["map_to_world"],
        px_per_m=r["px_per_m"],
    )
    cv2.imwrite(
        str(args.out.with_name(args.out.stem + "_debug.png")),
        debug_image(occ, r["room_map"], r["room_names"]),
    )
    print(f"Grid:  {occ.shape[1]} x {occ.shape[0]} px @ {r['px_per_m']:.0f} px/m", flush=True)
    print(
        f"Free:  {occ.mean():.1%}  |  walls: {r['n_walls']}  doorways: {r['n_doors']}  "
        f"furniture: {r['n_furniture']}",
        flush=True,
    )
    print(f"Rooms ({len(r['room_names'])}): {list(r['room_names'])}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
