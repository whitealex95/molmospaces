#!/usr/bin/env python3
"""Side-by-side overview of a MansionWorld MJCF scene.

Renders a single PNG with two panels:
  * left  -- top-down RGB render of the scene
  * right -- floor layout: rooms colour-coded and labelled, plus walls and doors

Both panels use the +x-right / +y-up convention and carry an orientation
widget (+x / +y arrows and an N/E/S/W compass; North == +y, East == +x).

The render comes out of MuJoCo in an arbitrary axis-aligned orientation; the
image is reoriented with a dihedral (transpose/flip) transform derived from the
world->map matrix so the saved PNG always has +x right and +y up.

Run from the ``mlspaces-mujoco`` conda env (OpenGL renderer; the Filament wheel
segfaults on this render path -- see CLAUDE.md issue #79).
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
FONT = cv2.FONT_HERSHEY_SIMPLEX

WALL_COLOR = (70, 70, 70)  # BGR
DOOR_COLOR = (0, 165, 255)
OTHER_COLOR = (205, 205, 205)  # furniture / windows / misc
BG_COLOR = (255, 255, 255)  # outside the building


def find_latest_scene() -> Path:
    hits = glob.glob(str(MANSION_EXPORT_ROOT / "**" / "scene.xml"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"No scene.xml found under {MANSION_EXPORT_ROOT}")
    return Path(max(hits, key=os.path.getmtime))


def geom_name(model, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""


def classify_geoms(model):
    floors, walls, doors = [], [], []
    for g in range(model.ngeom):
        n = geom_name(model, g).lower()
        if not n:
            continue
        if "floor" in n and model.geom(g).contype == 0:
            floors.append(g)
        elif "door" in n:
            doors.append(g)
        elif "wall" in n:
            walls.append(g)
    return floors, walls, doors


def clean_room_name(name: str) -> str:
    s = re.sub(r"^floor_", "", name, flags=re.I)
    s = re.sub(r"^[a-z]?\d+_", "", s, flags=re.I)  # strip a leading floor tag like "F1_"
    return s.replace("_", " ").strip()


def render_topdown(model_path: Path, px_per_m: int, device_id):
    """Orthographic top-down RGB + segmentation render of the scene."""
    spec = mujoco.MjSpec.from_file(str(model_path))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    floors, _, _ = classify_geoms(model)
    if not floors:
        raise RuntimeError("No floor geoms found (visual geoms with 'floor' in the name)")

    center, size = geom_aabb(model, data, floors, tight_mesh=False)
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
    px = h / size[0]

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

    rgb = renderer.render().copy()  # (h, w, 3) uint8, RGB
    renderer.enable_segmentation_rendering()
    seg = renderer.render()[..., 0].astype(np.int32)  # (h, w) geom ids, background == -1
    renderer.close()

    cam_to_map = np.array([[0, -px, 0, h / 2], [px, 0, 0, w / 2]])
    world_to_map = cam_to_map @ inverse_homogeneous_matrix(cam_to_world)
    return model, data, rgb, seg, world_to_map


def compute_dihedral(world_to_map: np.ndarray) -> np.ndarray:
    """Integer 2x2 mapping render (row,col) -> output (row,col) so that, in the
    output image, world +x is rightward (+col) and world +y is upward (-row)."""
    L = world_to_map[:2, :2]
    px = np.abs(L).max()
    target = np.array([[0.0, -px], [px, 0.0]])  # out_row = -px*y, out_col = +px*x
    D = np.rint(target @ np.linalg.inv(L)).astype(int)
    assert abs(round(float(np.linalg.det(D)))) == 1, f"unexpected render orientation: D={D}"
    return D


def apply_dihedral(img: np.ndarray, D: np.ndarray) -> np.ndarray:
    D = D.copy()
    if D[0, 1] != 0:  # output row comes from render col -> transpose
        img = np.swapaxes(img, 0, 1)
        D = D[:, ::-1]
    if D[0, 0] < 0:
        img = img[::-1]
    if D[1, 1] < 0:
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


def dihedral_affine(D: np.ndarray, render_shape) -> np.ndarray:
    """2x3 affine mapping render (row,col,1) -> output (row,col), matching apply_dihedral."""
    h, w = int(render_shape[0]), int(render_shape[1])
    M = np.eye(3)
    D = D.copy()
    if D[0, 1] != 0:  # transpose
        M = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]]) @ M
        h, w = w, h
        D = D[:, ::-1]
    if D[0, 0] < 0:  # flip rows
        M = np.array([[-1.0, 0, h - 1], [0, 1, 0], [0, 0, 1]]) @ M
    if D[1, 1] < 0:  # flip cols
        M = np.array([[1.0, 0, 0], [0, -1, w - 1], [0, 0, 1]]) @ M
    return M[:2]


def room_palette(n: int):
    n = max(n, 1)
    hsv = np.stack(
        [
            np.linspace(0, 179, n, endpoint=False).astype(np.uint8),
            np.full(n, 110, np.uint8),
            np.full(n, 240, np.uint8),
        ],
        axis=1,
    )
    bgr = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]
    return [tuple(int(c) for c in row) for row in bgr]


def build_layout(seg: np.ndarray, model):
    """Colour rooms (and furniture) from the segmentation render.

    Walls and doors are added separately by ``draw_wall_door_lines`` -- vertical
    panels barely register in a straight-down render.
    """
    floors, walls, doors = classify_geoms(model)
    lut = np.full((model.ngeom, 3), OTHER_COLOR, np.uint8)  # furniture / misc
    rooms = []
    for color, g in zip(room_palette(len(floors)), floors):
        lut[g] = color
        rooms.append((g, clean_room_name(geom_name(model, g)), color))

    layout = np.full((*seg.shape, 3), BG_COLOR, np.uint8)
    valid = seg >= 0
    layout[valid] = lut[np.clip(seg[valid], 0, model.ngeom - 1)]
    return layout, rooms, walls, doors


def geom_world_verts(model, data, gid):
    """World-frame vertices of a geom (mesh vertices, or box corners as a fallback)."""
    R = data.geom_xmat[gid].reshape(3, 3)
    p = data.geom_xpos[gid]
    did = int(model.geom_dataid[gid])
    if did >= 0 and int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        mv = model.mesh_vert
        if mv.ndim == 1:
            mv = mv.reshape(-1, 3)
        a = int(model.mesh_vertadr[did])
        n = int(model.mesh_vertnum[did])
        v = mv[a : a + n].astype(np.float64)
    else:
        s = model.geom_size[gid].astype(np.float64)
        v = np.array(
            [[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])]
        )
    return v @ R.T + p


def draw_wall_door_lines(layout, model, data, walls, doors, world_to_out):
    """Draw walls and doors as thick line segments on the floor plane.

    Vertical wall/door panels project to ~zero area in a straight-down view, so
    they barely appear in the segmentation render. Instead, project each geom's
    vertices onto the xy-plane and draw its principal-axis segment.
    """
    scale = float(np.abs(world_to_out[:2, :2]).max())  # output pixels per metre

    def draw(gids, color, world_thickness):
        thick = max(3, int(round(world_thickness * scale)))
        for g in gids:
            xy = geom_world_verts(model, data, g)[:, :2]
            if len(xy) < 2:
                continue
            centroid = xy.mean(axis=0)
            d = xy - centroid
            axis = np.linalg.svd(d, full_matrices=False)[2][0]  # principal direction
            t = d @ axis
            ends = np.array(
                [
                    [*(centroid + axis * t.min()), 0.0, 1.0],
                    [*(centroid + axis * t.max()), 0.0, 1.0],
                ]
            )
            rc = ends @ world_to_out.T  # (2, 2) -> [row, col]
            cv2.line(
                layout,
                (int(rc[0, 1]), int(rc[0, 0])),
                (int(rc[1, 1]), int(rc[1, 0])),
                color,
                thick,
                cv2.LINE_AA,
            )

    draw(walls, WALL_COLOR, 0.14)
    draw(doors, DOOR_COLOR, 0.16)


def draw_text(img, text, org, scale, color, anchor="bl"):
    """Draw text with a white halo, clamped to stay within the image.

    anchor: combine h{l,c,r} and v{t,m,b}.
    """
    th = max(1, int(round(scale * 2)))
    (tw, tht), _ = cv2.getTextSize(text, FONT, scale, th)
    x, y = org
    if "c" in anchor:
        x -= tw // 2
    elif "r" in anchor:
        x -= tw
    if "m" in anchor:
        y += tht // 2
    elif "t" in anchor:
        y += tht
    h, w = img.shape[:2]
    x = int(np.clip(x, 3, max(3, w - tw - 3)))
    y = int(np.clip(y, tht + 3, h - 3))
    cv2.putText(img, text, (x, y), FONT, scale, (255, 255, 255), th + 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, th, cv2.LINE_AA)


def label_rooms(layout, seg, rooms, scale):
    for g, name, _ in rooms:
        ys, xs = np.where(seg == g)
        if xs.size == 0:
            continue
        draw_text(layout, name, (int(xs.mean()), int(ys.mean())), scale, (15, 15, 15), "cm")


def draw_orientation_widget(img):
    """+x / +y arrows and an N/E/S/W compass in the bottom-right corner."""
    h, w = img.shape[:2]
    S = max(190, int(0.24 * h))
    m = int(0.035 * h)
    x0, y0 = w - S - m, h - S - m
    x1, y1 = x0 + S, y0 + S

    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (0.38 * roi + 0.62 * 255).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (140, 140, 140), 2)

    cx, cy = x0 + S // 2, y0 + S // 2
    arm = int(0.28 * S)
    lab = int(0.40 * S)
    dark = (40, 40, 40)
    pos = (0, 95, 215)

    cv2.line(img, (cx, cy), (cx, cy + arm), dark, 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy), (cx - arm, cy), dark, 2, cv2.LINE_AA)
    cv2.arrowedLine(img, (cx, cy), (cx, cy - arm), pos, 3, cv2.LINE_AA, tipLength=0.32)
    cv2.arrowedLine(img, (cx, cy), (cx + arm, cy), pos, 3, cv2.LINE_AA, tipLength=0.32)
    cv2.circle(img, (cx, cy), 3, dark, -1, cv2.LINE_AA)

    fc = max(0.55, 0.0017 * h)
    fa = fc * 0.7
    draw_text(img, "N", (cx, cy - lab), fc, dark, "cm")
    draw_text(img, "S", (cx, cy + lab), fc, dark, "cm")
    draw_text(img, "E", (cx + lab, cy), fc, dark, "cm")
    draw_text(img, "W", (cx - lab, cy), fc, dark, "cm")
    draw_text(img, "+y", (cx + int(0.11 * S), cy - arm + int(0.06 * S)), fa, pos, "lm")
    draw_text(img, "+x", (cx + arm - int(0.04 * S), cy + int(0.12 * S)), fa, pos, "cm")


def draw_legend(img):
    """Small wall/door colour key in the top-left corner."""
    h = img.shape[0]
    items = [("wall", WALL_COLOR), ("door", DOOR_COLOR)]
    fs = max(0.55, 0.0017 * h)
    sw = int(0.030 * h)
    pad = int(0.018 * h)
    row = sw + pad
    bw = int(0.22 * h)
    bh = pad + row * len(items)
    x0, y0 = int(0.022 * h), int(0.022 * h)

    roi = img[y0 : y0 + bh, x0 : x0 + bw].astype(np.float32)
    img[y0 : y0 + bh, x0 : x0 + bw] = (0.4 * roi + 0.6 * 255).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (140, 140, 140), 2)
    for i, (name, color) in enumerate(items):
        sy = y0 + pad + i * row
        cv2.rectangle(img, (x0 + pad, sy), (x0 + pad + sw, sy + sw), color, -1)
        cv2.rectangle(img, (x0 + pad, sy), (x0 + pad + sw, sy + sw), (90, 90, 90), 1)
        draw_text(img, name, (x0 + 2 * pad + sw, sy + sw // 2), fs, (20, 20, 20), "lm")


def titled_panel(img, title, strip_h):
    h, w = img.shape[:2]
    (tw, tht), _ = cv2.getTextSize(title, FONT, 1.0, 2)
    scale = min(0.92 * w / tw, 0.55 * strip_h / tht)
    strip = np.full((strip_h, w, 3), 248, np.uint8)
    draw_text(strip, title, (w // 2, strip_h // 2), scale, (25, 25, 25), "cm")
    cv2.line(strip, (0, strip_h - 1), (w, strip_h - 1), (170, 170, 170), 2)
    return np.vstack([strip, img])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scene", type=Path, default=None, help="scene.xml (default: latest mansion export)"
    )
    ap.add_argument("--px-per-m", type=int, default=60, help="render resolution (pixels per metre)")
    ap.add_argument("--device-id", type=int, default=None, help="EGL render device id")
    ap.add_argument(
        "--out", type=Path, default=None, help="output PNG (default: <scene>_overview.png)"
    )
    args = ap.parse_args()

    scene = (args.scene or find_latest_scene()).resolve()
    out = (args.out or scene.parent / f"{scene.stem}_overview.png").resolve()
    print(f"Scene: {scene}")

    model, data, rgb, seg, world_to_map = render_topdown(scene, args.px_per_m, args.device_id)

    D = compute_dihedral(world_to_map)
    world_to_out = dihedral_affine(D, seg.shape) @ np.vstack([world_to_map, [0, 0, 0, 1]])
    rgb = apply_dihedral(np.ascontiguousarray(rgb[..., ::-1]), D)  # RGB -> BGR, reorient
    seg = apply_dihedral(seg, D)

    layout, rooms, walls, doors = build_layout(seg, model)
    draw_wall_door_lines(layout, model, data, walls, doors, world_to_out)
    label_rooms(layout, seg, rooms, max(0.5, 0.00085 * layout.shape[0]))
    draw_legend(layout)

    for panel in (rgb, layout):
        draw_orientation_widget(panel)

    strip_h = max(74, int(0.06 * rgb.shape[0]))
    left = titled_panel(rgb, "Top-Down View", strip_h)
    right = titled_panel(layout, "Floor Layout  (rooms / walls / doors)", strip_h)
    sep = np.full((left.shape[0], 8, 3), 255, np.uint8)
    composite = np.hstack([left, sep, right])

    cv2.imwrite(str(out), composite)
    print(f"Rooms detected: {len(rooms)}")
    for _, name, _ in rooms:
        print(f"  - {name}")
    print(f"Wrote: {out}  ({composite.shape[1]} x {composite.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
