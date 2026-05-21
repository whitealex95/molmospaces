#!/usr/bin/env python3
"""Clearance-aware A* path planning on a sim-agnostic occupancy grid.

Loads ``occupancy.npz`` (from build_occupancy.py), dilates obstacles by the
agent radius, runs A* (molmo_spaces.utils.distance_transform_utils -- the
edge weights favour clearance, so the path keeps away from walls), and writes
a world-frame waypoint path that both the MuJoCo and IsaacSim runtimes follow.

Output: ``path.npz`` next to the occupancy grid:
  waypoints   (N,2) float -- world (x,y) waypoints, start -> goal
  start_room / goal_room  -- room names

Run from the ``mlspaces-mujoco`` env.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

import molmo_spaces.utils.distance_transform_utils as dtu

DOWNSCALE = 5  # occupancy-grid cells per A* graph cell


def circular_kernel(r: int) -> np.ndarray:
    r = max(int(r), 1)
    k = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
    cv2.circle(k, (r, r), r, 1, -1)
    return k


def world_to_px(world_to_map, x, y):
    return world_to_map @ np.array([x, y, 0.0, 1.0])  # -> (row, col)


def px_to_world(map_to_world, row, col):
    return (map_to_world @ np.array([row, col, 1.0]))[:2]  # -> (x, y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--occupancy", type=Path, required=True, help="occupancy.npz from build_occupancy.py"
    )
    ap.add_argument(
        "--agent-radius",
        type=float,
        default=0.2,
        help="robot radius (m) for obstacle dilation; keep modest so doorways stay open",
    )
    ap.add_argument("--start-room", type=str, default=None, help="start room name substring")
    ap.add_argument("--goal-room", type=str, default=None, help="goal room name substring")
    ap.add_argument(
        "--out", type=Path, default=None, help="output path.npz (default: next to occupancy)"
    )
    args = ap.parse_args()

    o = np.load(args.occupancy, allow_pickle=True)
    occ = o["occupancy"]  # bool, True = free
    room_map = o["room_map"]
    room_names = [str(s) for s in o["room_names"]]
    world_to_map = o["world_to_map"]
    map_to_world = o["map_to_world"]
    px_per_m = float(o["px_per_m"])

    # agent-radius clearance, then downscale (min-pool: a coarse cell is free
    # only if every fine cell is free).
    rad_px = max(1, int(round(args.agent_radius * px_per_m)))
    free = cv2.dilate((~occ).astype(np.uint8), circular_kernel(rad_px)) == 0
    ds = DOWNSCALE
    hd, wd = free.shape[0] // ds, free.shape[1] // ds
    grid = free[: hd * ds, : wd * ds].reshape(hd, ds, wd, ds).min(axis=(1, 3))
    grid_spacing = ds / px_per_m

    dt = dtu.make_distance_transform(grid, grid_spacing)
    graph = dtu.make_grid_graph(grid, dt, weight_exp=2)
    print(f"A* graph: {hd}x{wd} cells, {graph.number_of_nodes()} navigable nodes")

    # Per-room anchor = the most-open navigable coarse cell of that room.
    def coarse_mask(room_idx):
        m = (room_map == room_idx)[: hd * ds, : wd * ds]
        return m.reshape(hd, ds, wd, ds).any(axis=(1, 3)) & grid

    anchors, sizes = {}, {}
    for idx, name in enumerate(room_names, start=1):
        m = coarse_mask(idx)
        if not m.any():
            continue
        r, c = np.unravel_index(np.argmax(np.where(m, dt, -1.0)), m.shape)
        if (int(r), int(c)) in graph:
            anchors[name] = (int(r), int(c))
            sizes[name] = int(m.sum())
    if len(anchors) < 2:
        raise SystemExit(f"need >=2 navigable rooms; got {list(anchors)}")

    # Connectivity: rooms whose anchors fall in the same A* component are
    # mutually reachable -- i.e. any pair of them is a valid room->room
    # scenario. Disconnected groups mean a doorway got sealed (over-dilated).
    comp_of = {}
    for ci, comp in enumerate(nx.connected_components(graph)):
        for node in comp:
            comp_of[node] = ci
    groups = defaultdict(list)
    for name, a in anchors.items():
        groups[comp_of[a]].append(name)
    groups = sorted(groups.values(), key=len, reverse=True)
    print(f"Room connectivity ({len(groups)} group(s); rooms in a group are mutually reachable):")
    for grp in groups:
        print(f"  - {grp}")
    reachable = set(groups[0])
    if len(groups) > 1:
        print(
            f"  NOTE: {len(anchors) - len(reachable)} room(s) disconnected -- "
            f"reduce --agent-radius or widen DOOR_CARVE_M if a doorway sealed."
        )

    def pick(sub):
        return next((n for n in anchors if sub.lower() in n.lower()), None)

    if args.start_room:
        start_name = pick(args.start_room)
        if start_name is None:
            raise SystemExit(f"--start-room '{args.start_room}' matched nothing in {list(anchors)}")
    else:
        start_name = max(reachable, key=lambda n: sizes[n])

    reach = nx.single_source_dijkstra_path_length(graph, anchors[start_name])
    if args.goal_room:
        goal_name = pick(args.goal_room)
        if goal_name is None:
            raise SystemExit(f"--goal-room '{args.goal_room}' matched nothing in {list(anchors)}")
        if anchors[goal_name] not in reach:
            raise SystemExit(
                f"'{goal_name}' is not reachable from '{start_name}' (different group)"
            )
    else:  # farthest reachable room
        cands = {n: reach[a] for n, a in anchors.items() if n != start_name and a in reach}
        if not cands:
            raise SystemExit("no room reachable from the start room")
        goal_name = max(cands, key=cands.get)

    sr, sc = anchors[start_name]
    gr, gc = anchors[goal_name]
    waypoints_px, _, cost = dtu.make_discrete_path(graph, sr, sc, gr, gc, dt, 2, grid_spacing, 0.6)
    wp_full = np.array(waypoints_px, dtype=float) * ds  # coarse -> full-res px
    waypoints = np.array([px_to_world(map_to_world, r, c) for r, c in wp_full])
    length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
    print(f"path: {start_name} -> {goal_name}  |  {len(waypoints)} waypoints, {length:.1f} m")

    out = (args.out or args.occupancy.with_name("path.npz")).resolve()
    np.savez(out, waypoints=waypoints, start_room=start_name, goal_room=goal_name)

    # debug overlay
    vis = np.where(occ[..., None], 235, 40).astype(np.uint8).repeat(3, axis=2)
    pts = np.array([world_to_px(world_to_map, x, y)[::-1] for x, y in waypoints], np.int32)
    cv2.polylines(vis, [pts], False, (0, 140, 255), max(2, rad_px // 3), cv2.LINE_AA)
    cv2.circle(vis, tuple(pts[0]), rad_px, (0, 200, 0), -1, cv2.LINE_AA)
    cv2.circle(vis, tuple(pts[-1]), rad_px, (0, 0, 230), -1, cv2.LINE_AA)
    dbg = out.with_name("path_debug.png")
    cv2.imwrite(str(dbg), vis)

    print(f"wrote: {out}")
    print(f"       {dbg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
