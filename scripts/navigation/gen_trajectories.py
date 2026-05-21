#!/usr/bin/env python3
"""Batch-generate cross-room navigation trajectories for one scene.

Builds the occupancy grid once, samples ``--count`` start/goal *positions*
(distinct rooms when possible; within-room for single-room scenes -- the same
room pair may recur with different positions), then in two passes:
  1. writes each path + a top-down preview PNG, and the per-dataset index.json
     (fast -- so you can see what every trajectory covers up front);
  2. renders the G1 ego/chase video for each (slow).

Layout:
  nav_runs/<dataset>/<scene>/occupancy.npz
  nav_runs/<dataset>/<scene>/NN__<start>__to__<goal>/
      path.npz  topdown.png  combined.mp4  ego_rgb.mp4  ego_depth.mp4  follow.mp4
  nav_runs/<dataset>/index.json

Run from any conda-equipped shell; the occupancy build and rendering are
shelled out to the right env (mlspaces-mujoco / mlspaces). The driver itself
needs the mlspaces (or mlspaces-mujoco) env for the planner imports.
"""

import argparse
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from plan import Planner, world_to_px

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NAV_RUNS = REPO / "nav_runs"
MIN_LENGTH_M = 2.0  # reject degenerate near-zero-length samples


def sanitize(name: str) -> str:
    s = "".join(c if c.isalnum() else "-" for c in name.lower())
    return "-".join(p for p in s.split("-") if p) or "room"


def conda_run(env: str, *args: str) -> None:
    subprocess.run(["conda", "run", "--no-capture-output", "-n", env, "python", *args], check=True)


def save_topdown(planner: Planner, waypoints, out_path, title: str) -> None:
    """Top-down preview PNG: room-tinted occupancy + trajectory + room labels.
    Lets you see what a trajectory covers before rendering its video."""
    occ, room_map, names = planner.occ, planner.room_map, planner.room_names
    img = np.where(occ[..., None], 235, 50).astype(np.uint8).repeat(3, axis=2)
    n = max(len(names), 1)
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

    pts = np.array([world_to_px(planner.world_to_map, x, y)[::-1] for x, y in waypoints], np.int32)
    cv2.polylines(img, [pts], False, (30, 90, 240), 6, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[0]), 13, (0, 180, 0), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[-1]), 13, (0, 0, 220), -1, cv2.LINE_AA)
    for i in range(1, n + 1):
        m = room_map == i
        if m.any():
            rs, cs = np.where(m)
            cv2.putText(
                img,
                names[i - 1],
                (int(cs.mean()) - 34, int(rs.mean())),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (15, 15, 15),
                2,
                cv2.LINE_AA,
            )

    rs, cs = np.where(occ)  # crop to the building footprint + margin
    if len(rs):
        mg = 30
        img = img[max(0, rs.min() - mg) : rs.max() + mg, max(0, cs.min() - mg) : cs.max() + mg]
    if img.shape[1] > 1000:  # downscale wide grids
        s = 1000 / img.shape[1]
        img = cv2.resize(img, (1000, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    bar = np.zeros((34, img.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([bar, img]))


def sample_trajectories(planner: Planner, count: int, seed: int) -> list[dict]:
    """Sample ``count`` trajectories (each a dict with start/goal room indices,
    waypoints, length). Distinct room pairs are preferred and reused -- with
    fresh positions -- once exhausted; single-room scenes sample within-room
    start/goal positions."""
    rng = random.Random(seed)
    groups = planner.connectivity()
    if not groups:
        raise SystemExit("no navigable rooms in the scene")
    rooms = groups[0]  # largest mutually-reachable group
    if len(rooms) >= 2:
        base_pairs = [(a, b) for a in rooms for b in rooms if a != b]
        print(f"sampling cross-room trajectories from {len(rooms)} rooms")
    else:
        base_pairs = [(rooms[0], rooms[0])]
        print("single navigable room -- sampling within-room start/goal positions")

    def pairs_stream():
        while True:
            pool = base_pairs[:]
            rng.shuffle(pool)
            yield from pool

    stream = pairs_stream()
    trajs: list[dict] = []
    attempts = 0
    while len(trajs) < count and attempts < count * 200:
        attempts += 1
        si, gi = next(stream)
        scells, gcells = planner.room_cells(si), planner.room_cells(gi)
        if not scells or not gcells:
            continue
        for _ in range(25):
            sc = rng.choice(scells)
            gc = rng.choice(gcells)
            if sc == gc:
                continue
            res = planner.plan(sc, gc)
            if res is not None and res[1] >= MIN_LENGTH_M:
                waypoints, length = res
                trajs.append(
                    {
                        "start_idx": si,
                        "goal_idx": gi,
                        "waypoints": waypoints,
                        "length": length,
                    }
                )
                break
    if len(trajs) < count:
        print(f"warning: only sampled {len(trajs)}/{count} valid trajectories")
    return trajs


def update_index(index_path: Path, dataset: str, scene_name: str, records: list[dict]) -> None:
    """Merge ``records`` into the per-dataset index.json, replacing any prior
    entries for the same scene."""
    data: dict = {"dataset": dataset, "trajectories": []}
    if index_path.exists():
        data = json.loads(index_path.read_text())
    kept = [t for t in data.get("trajectories", []) if t.get("scene") != scene_name]
    data["dataset"] = dataset
    data["updated"] = datetime.now().isoformat(timespec="seconds")
    data["trajectories"] = kept + records
    index_path.write_text(json.dumps(data, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, required=True, help="scene .xml")
    ap.add_argument(
        "--dataset", required=True, help="dataset label, e.g. mansion / procthor-10k-val"
    )
    ap.add_argument("--count", type=int, required=True, help="number of trajectories")
    ap.add_argument("--renderer", choices=["opengl", "filament"], default="opengl")
    ap.add_argument("--agent-radius", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene-name", default=None, help="output sub-dir name (default: file stem)")
    args = ap.parse_args()

    scene = Path(os.path.abspath(args.scene))  # abspath: keep symlinks intact
    # generic "scene.xml" (mansion exports) -> name the run after its folder
    scene_name = args.scene_name or (scene.parent.name if scene.stem == "scene" else scene.stem)
    scene_root = NAV_RUNS / args.dataset / scene_name
    scene_root.mkdir(parents=True, exist_ok=True)
    occ = scene_root / "occupancy.npz"

    if occ.exists():
        print(f"reusing occupancy: {occ}")
    else:
        print(f"building occupancy -> {occ}")
        conda_run(
            "mlspaces-mujoco",
            str(HERE / "build_occupancy.py"),
            "--scene",
            str(scene),
            "--out",
            str(occ),
        )

    planner = Planner(occ, args.agent_radius)
    trajs = sample_trajectories(planner, args.count, args.seed)
    if not trajs:
        raise SystemExit("no trajectories sampled")

    index_path = NAV_RUNS / args.dataset / "index.json"

    # pass 1 -- path + top-down preview + index (fast)
    items, records = [], []
    for i, t in enumerate(trajs, start=1):
        sname = planner.room_name(t["start_idx"])
        gname = planner.room_name(t["goal_idx"])
        tdir = scene_root / f"{i:02d}__{sanitize(sname)}__to__{sanitize(gname)}"
        tdir.mkdir(exist_ok=True)
        planner.save_path(tdir / "path.npz", t["waypoints"], sname, gname)
        save_topdown(
            planner,
            t["waypoints"],
            tdir / "topdown.png",
            f"{scene_name}  |  {sname} -> {gname}  |  {t['length']:.1f} m",
        )
        start_xy, goal_xy = t["waypoints"][0], t["waypoints"][-1]
        rel = tdir.relative_to(NAV_RUNS / args.dataset)
        records.append(
            {
                "id": i,
                "scene": scene_name,
                "scene_xml": str(scene),
                "dir": str(rel),
                "start_room": sname,
                "goal_room": gname,
                "start_xy": [round(float(start_xy[0]), 3), round(float(start_xy[1]), 3)],
                "goal_xy": [round(float(goal_xy[0]), 3), round(float(goal_xy[1]), 3)],
                "length_m": round(t["length"], 2),
                "n_waypoints": int(len(t["waypoints"])),
                "cross_room": t["start_idx"] != t["goal_idx"],
                "renderer": args.renderer,
                "agent_radius": args.agent_radius,
                "topdown_png": str(rel / "topdown.png"),
                "combined_mp4": str(rel / "combined.mp4"),
            }
        )
        items.append((tdir, t, sname, gname))
    update_index(index_path, args.dataset, scene_name, records)
    print(f"{len(items)} previews + index ready -> {scene_root}")

    # pass 2 -- render the videos (slow)
    render_env = "mlspaces" if args.renderer == "filament" else "mlspaces-mujoco"
    for i, (tdir, t, sname, gname) in enumerate(items, start=1):
        print(
            f"[{i}/{len(items)}] render {sname} -> {gname}  {t['length']:.1f} m  ({args.renderer})"
        )
        conda_run(
            render_env,
            str(HERE / "run_mujoco.py"),
            "--scene",
            str(scene),
            "--path",
            str(tdir / "path.npz"),
            "--occupancy",
            str(occ),
            "--renderer",
            args.renderer,
            "--out-dir",
            str(tdir),
        )

    print(f"\n{len(records)} trajectories -> {scene_root}")
    print(f"index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
