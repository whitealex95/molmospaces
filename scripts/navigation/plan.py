#!/usr/bin/env python3
"""Clearance-aware A* path planning on a sim-agnostic occupancy grid.

Loads ``occupancy.npz`` (from build_occupancy.py), dilates obstacles by the
agent radius, and runs A* whose edge weights favour clearance (the path keeps
away from walls). The ``Planner`` class is reusable -- built once per scene it
plans any number of start/goal cell pairs; gen_trajectories.py uses it to
batch-sample trajectories. Run as a script it plans one room->room path.

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


class Planner:
    """Clearance-aware A* over a downscaled occupancy grid.

    Built once per (scene, agent_radius); ``plan()`` any number of start/goal
    coarse-cell pairs. Shared by this script's CLI and gen_trajectories.py.
    """

    def __init__(self, occupancy_npz, agent_radius: float = 0.2):
        o = np.load(occupancy_npz, allow_pickle=True)
        self.occ = o["occupancy"]  # bool, True = free
        self.room_map = o["room_map"]
        self.room_names = [str(s) for s in o["room_names"]]
        self.world_to_map = o["world_to_map"]
        self.map_to_world = o["map_to_world"]
        self.px_per_m = float(o["px_per_m"])
        self.agent_radius = agent_radius

        # agent-radius clearance, then downscale (min-pool: a coarse cell is
        # free only if every fine cell is free).
        self.rad_px = max(1, int(round(agent_radius * self.px_per_m)))
        free = cv2.dilate((~self.occ).astype(np.uint8), circular_kernel(self.rad_px)) == 0
        ds = DOWNSCALE
        hd, wd = free.shape[0] // ds, free.shape[1] // ds
        self.grid = free[: hd * ds, : wd * ds].reshape(hd, ds, wd, ds).min(axis=(1, 3))
        self.ds = ds
        self.grid_spacing = ds / self.px_per_m
        self.dt = dtu.make_distance_transform(self.grid, self.grid_spacing)
        self.graph = dtu.make_grid_graph(self.grid, self.dt, weight_exp=2)

        # navigable A* cells per room index (1..R)
        self._room_cells = {}
        for idx in range(1, len(self.room_names) + 1):
            m = (self.room_map == idx)[: hd * ds, : wd * ds]
            m = m.reshape(hd, ds, wd, ds).any(axis=(1, 3)) & self.grid
            rs, cs = np.where(m)
            cells = [(int(r), int(c)) for r, c in zip(rs, cs) if (int(r), int(c)) in self.graph]
            if cells:
                self._room_cells[idx] = cells

        # A* connected component of every node
        self.comp_of = {}
        for ci, comp in enumerate(nx.connected_components(self.graph)):
            for node in comp:
                self.comp_of[node] = ci

    @property
    def room_indices(self) -> list[int]:
        return sorted(self._room_cells)

    def room_name(self, idx: int) -> str:
        return self.room_names[idx - 1]

    def room_cells(self, idx: int) -> list[tuple[int, int]]:
        """Navigable A* cells (r,c) belonging to room ``idx``."""
        return self._room_cells.get(idx, [])

    def room_anchor(self, idx: int):
        """The most-open navigable cell of a room (maximum clearance)."""
        cells = self.room_cells(idx)
        return max(cells, key=lambda rc: float(self.dt[rc])) if cells else None

    def connectivity(self) -> list[list[int]]:
        """Room-index groups; rooms in a group are mutually reachable.
        Sorted largest group first."""
        groups = defaultdict(list)
        for idx in self.room_indices:
            a = self.room_anchor(idx)
            if a is not None and a in self.comp_of:
                groups[self.comp_of[a]].append(idx)
        return sorted(groups.values(), key=len, reverse=True)

    def plan(self, start_cell, goal_cell):
        """A* between two coarse cells. Returns (waypoints_world (N,2),
        length_m) or None if the cells are not mutually reachable."""
        if start_cell not in self.graph or goal_cell not in self.graph:
            return None
        if self.comp_of.get(start_cell) != self.comp_of.get(goal_cell):
            return None
        sr, sc = start_cell
        gr, gc = goal_cell
        wp_px, _, _ = dtu.make_discrete_path(
            self.graph, sr, sc, gr, gc, self.dt, 2, self.grid_spacing, 0.6
        )
        wp_full = np.array(wp_px, dtype=float) * self.ds  # coarse -> full-res px
        waypoints = np.array([px_to_world(self.map_to_world, r, c) for r, c in wp_full])
        length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
        return waypoints, length

    def save_path(self, out, waypoints, start_room: str, goal_room: str) -> None:
        """Write ``path.npz`` and a ``*_debug.png`` overlay next to it."""
        out = Path(out)
        np.savez(out, waypoints=waypoints, start_room=start_room, goal_room=goal_room)
        vis = np.where(self.occ[..., None], 235, 40).astype(np.uint8).repeat(3, axis=2)
        pts = np.array([world_to_px(self.world_to_map, x, y)[::-1] for x, y in waypoints], np.int32)
        cv2.polylines(vis, [pts], False, (0, 140, 255), max(2, self.rad_px // 3), cv2.LINE_AA)
        cv2.circle(vis, tuple(pts[0]), self.rad_px, (0, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, tuple(pts[-1]), self.rad_px, (0, 0, 230), -1, cv2.LINE_AA)
        cv2.imwrite(str(out.with_name(out.stem + "_debug.png")), vis)


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

    planner = Planner(args.occupancy, args.agent_radius)
    print(f"A* graph: {planner.graph.number_of_nodes()} navigable nodes")

    groups = planner.connectivity()
    print(f"Room connectivity ({len(groups)} group(s); rooms in a group are mutually reachable):")
    for grp in groups:
        print(f"  - {[planner.room_name(i) for i in grp]}")
    if not groups or len(groups[0]) < 2:
        raise SystemExit("need >=2 mutually reachable rooms for a room->room path")
    reachable = groups[0]

    def pick(sub):
        return next((i for i in reachable if sub.lower() in planner.room_name(i).lower()), None)

    if args.start_room:
        start_idx = pick(args.start_room)
        if start_idx is None:
            raise SystemExit(f"--start-room '{args.start_room}' matched no reachable room")
    else:
        start_idx = max(reachable, key=lambda i: len(planner.room_cells(i)))

    if args.goal_room:
        goal_idx = pick(args.goal_room)
        if goal_idx is None:
            raise SystemExit(f"--goal-room '{args.goal_room}' matched no reachable room")
    else:  # farthest reachable room from the start anchor
        reach = nx.single_source_dijkstra_path_length(planner.graph, planner.room_anchor(start_idx))
        cands = {
            i: reach[planner.room_anchor(i)]
            for i in reachable
            if i != start_idx and planner.room_anchor(i) in reach
        }
        if not cands:
            raise SystemExit("no room reachable from the start room")
        goal_idx = max(cands, key=cands.get)

    res = planner.plan(planner.room_anchor(start_idx), planner.room_anchor(goal_idx))
    if res is None:
        raise SystemExit("planning failed: start and goal not mutually reachable")
    waypoints, length = res
    start_name, goal_name = planner.room_name(start_idx), planner.room_name(goal_idx)
    print(f"path: {start_name} -> {goal_name}  |  {len(waypoints)} waypoints, {length:.1f} m")

    out = (args.out or args.occupancy.with_name("path.npz")).resolve()
    planner.save_path(out, waypoints, start_name, goal_name)
    print(f"wrote: {out}")
    print(f"       {out.with_name(out.stem + '_debug.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
