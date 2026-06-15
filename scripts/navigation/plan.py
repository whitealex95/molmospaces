#!/usr/bin/env python3
"""Clearance-aware A* path planning on a sim-agnostic occupancy grid.

Loads ``occupancy.npz`` (from build_occupancy.py), dilates obstacles by the
agent radius, and runs A* whose edge weights favour clearance (the path keeps
away from walls). The ``Planner`` class is reusable -- built once per scene it
plans any number of start/goal cell pairs; gen_trajectories.py uses it to
batch-sample trajectories. Run as a script it plans one room->room path.

Output: ``path.npz`` next to the occupancy grid:
  waypoints      (N,2) float -- ROUNDED world (x,y) waypoints, start -> goal
  waypoints_raw  (M,2) float -- the raw grid A* path before rounding
  start_room / goal_room     -- room names

A clearance-checked moving average rounds the staircase grid A* path so it is
easier to follow (``--smooth`` window; 0 keeps the rigid path). The rounding
never cuts a corner through a wall.

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


def resample_polyline(pts: np.ndarray, step: float) -> np.ndarray:
    """Densify a polyline to ~`step`-spaced points (endpoints preserved)."""
    pts = np.asarray(pts, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            continue
        n = max(1, int(round(length / step)))
        for i in range(1, n + 1):
            out.append(a + seg * (i / n))
    return np.array(out)


class Planner:
    """Clearance-aware A* over a downscaled occupancy grid.

    Built once per (scene, agent_radius); ``plan()`` any number of start/goal
    coarse-cell pairs. Shared by this script's CLI and gen_trajectories.py.
    """

    def __init__(
        self,
        occupancy_npz,
        agent_radius: float = 0.2,
        smooth_strength: float = 0.5,
        obstacle=None,
    ):
        o = np.load(occupancy_npz, allow_pickle=True)
        self.occ = o["occupancy"]  # bool, True = free
        self.room_map = o["room_map"]
        self.room_names = [str(s) for s in o["room_names"]]
        self.world_to_map = o["world_to_map"]
        self.map_to_world = o["map_to_world"]
        self.px_per_m = float(o["px_per_m"])
        self.agent_radius = agent_radius
        self.smooth_strength = smooth_strength  # path-rounding window (m); 0 disables

        # Optional extra obstacle (x, y, sx, sy[, yaw_deg]): mark a (optionally rotated)
        # world rectangle -- center xy, full sizes m, yaw about +z -- as occupied before
        # clearance dilation, so A* must route around it. Aligning yaw with the local path
        # heading lets the box be shallow along the path (jumpable) yet wide across it.
        if obstacle is not None:
            ox, oy, sx, sy = obstacle[:4]
            yaw = np.radians(obstacle[4]) if len(obstacle) > 4 else 0.0
            self.occ = self.occ.copy()
            cs, sn = np.cos(yaw), np.sin(yaw)
            corners_w = [
                (ox + cs * dx - sn * dy, oy + sn * dx + cs * dy)
                for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))
            ]
            poly = np.array(
                [world_to_px(self.world_to_map, x, y)[::-1] for x, y in corners_w], np.int32
            )  # (col, row) for cv2
            mask = np.zeros(self.occ.shape, np.uint8)
            cv2.fillConvexPoly(mask, poly, 1)
            self.occ[mask == 1] = False

        # agent-radius clearance, then downscale (min-pool: a coarse cell is
        # free only if every fine cell is free).
        self.rad_px = max(1, int(round(agent_radius * self.px_per_m)))
        free = cv2.dilate((~self.occ).astype(np.uint8), circular_kernel(self.rad_px)) == 0
        self.free_px = free  # agent-dilated free mask (full px res); used by smoothing
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

    def _free_at(self, x: float, y: float) -> bool:
        """True if world point (x,y) is inside the agent-dilated free space."""
        rc = self.world_to_map @ np.array([x, y, 0.0, 1.0])
        r, c = int(round(rc[0])), int(round(rc[1]))
        h, w = self.free_px.shape
        return 0 <= r < h and 0 <= c < w and bool(self.free_px[r, c])

    def smooth_waypoints(self, waypoints) -> np.ndarray:
        """Round off the grid-staircase A* path with a clearance-checked moving
        average, so the path is easier to follow (and looks round, not rigid).

        Endpoints are pinned. Any point a smoothing pass would push into the
        agent-dilated obstacle zone is pulled back toward the raw path until it
        is free again -- so rounding never cuts a corner through a wall."""
        waypoints = np.asarray(waypoints, float)
        if self.smooth_strength <= 0 or len(waypoints) < 3:
            return waypoints
        step = self.grid_spacing * 0.5
        pts = resample_polyline(waypoints, step)
        if len(pts) < 3:
            return pts
        k = max(1, int(round(self.smooth_strength / step)))
        out = pts.copy()
        for _ in range(2):  # two passes -> noticeably rounder
            cur = out.copy()
            for i in range(1, len(out) - 1):
                lo, hi = max(0, i - k), min(len(out), i + k + 1)
                cand = cur[lo:hi].mean(axis=0)
                if self._free_at(cand[0], cand[1]):
                    out[i] = cand
                    continue
                for t in (0.66, 0.33):  # blend back toward the raw point
                    c2 = t * cand + (1 - t) * cur[i]
                    if self._free_at(c2[0], c2[1]):
                        out[i] = c2
                        break
            out[0], out[-1] = pts[0], pts[-1]
        return out

    def save_path(self, out, waypoints, start_room: str, goal_room: str) -> None:
        """Write ``path.npz`` and a ``*_debug.png`` overlay next to it.

        ``waypoints`` in the npz is the **rounded** path (what runtimes follow);
        the raw grid A* path is kept as ``waypoints_raw`` for reference."""
        out = Path(out)
        raw = np.asarray(waypoints, float)
        smooth = self.smooth_waypoints(raw)
        np.savez(
            out,
            waypoints=smooth,
            waypoints_raw=raw,
            start_room=start_room,
            goal_room=goal_room,
        )
        vis = np.where(self.occ[..., None], 235, 40).astype(np.uint8).repeat(3, axis=2)
        raw_pts = np.array([world_to_px(self.world_to_map, x, y)[::-1] for x, y in raw], np.int32)
        sm_pts = np.array([world_to_px(self.world_to_map, x, y)[::-1] for x, y in smooth], np.int32)
        # raw A* path faint grey underneath, rounded path bold orange on top.
        cv2.polylines(vis, [raw_pts], False, (120, 120, 120), max(1, self.rad_px // 5), cv2.LINE_AA)
        cv2.polylines(vis, [sm_pts], False, (0, 140, 255), max(2, self.rad_px // 3), cv2.LINE_AA)
        cv2.circle(vis, tuple(sm_pts[0]), self.rad_px, (0, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, tuple(sm_pts[-1]), self.rad_px, (0, 0, 230), -1, cv2.LINE_AA)
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
    ap.add_argument(
        "--smooth",
        type=float,
        default=0.5,
        help="path-rounding window in metres (0 = rigid grid A* path)",
    )
    ap.add_argument(
        "--obstacle",
        type=str,
        default=None,
        help="extra blocking box 'x,y,sx,sy[,yaw_deg]' (center xy, full sizes m, optional "
        "yaw about +z) stamped occupied before planning, so A* detours around it (e.g. the "
        "box the MM jump variant leaps; align yaw with the path so it stays jumpably shallow)",
    )
    ap.add_argument(
        "--resmooth",
        type=Path,
        default=None,
        help="round an EXISTING path.npz in place (uses its waypoints_raw if present) and exit",
    )
    args = ap.parse_args()

    obstacle = None
    if args.obstacle:
        obstacle = tuple(float(v) for v in args.obstacle.split(","))
        if len(obstacle) not in (4, 5):
            raise SystemExit("--obstacle must be 'x,y,sx,sy' or 'x,y,sx,sy,yaw_deg'")
    planner = Planner(
        args.occupancy, args.agent_radius, smooth_strength=args.smooth, obstacle=obstacle
    )
    print(f"A* graph: {planner.graph.number_of_nodes()} navigable nodes")

    if args.resmooth is not None:
        p = np.load(args.resmooth, allow_pickle=True)
        raw = p["waypoints_raw"] if "waypoints_raw" in p else p["waypoints"]
        planner.save_path(args.resmooth, raw, str(p["start_room"]), str(p["goal_room"]))
        print(f"re-smoothed (window {args.smooth} m) -> {args.resmooth}")
        return 0

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
