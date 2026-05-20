"""MansionWorld floor JSON → MJCF converter.

Sibling of ``scripts/mansion/mansion_to_usd.py`` — same source data, same
coverage (85/85 unique assetIds for `floor_1.json`), but emits MJCF for
loading in MuJoCo (`mlspaces-mujoco` env).

v1 scope (visual only, no physics):
- Floors as ear-clip triangulated polygon meshes (one per room; concave-safe)
- Walls as triangulated polygon meshes (one per wall)
- Objects (objathor `.pkl.gz` / mansion_patch `.json` / molmospaces THOR `.xml`)
  placed at their Unity pose
- All geoms have ``contype=0 conaffinity=0`` (visual only — no collisions)

See ``docs/mansion_to_mjcf.md`` for the full pipeline and known gaps.

Run::

    conda activate mlspaces-mujoco   # or mlspaces — both have mujoco
    python scripts/mansion/mansion_to_mjcf.py \\
        --scene-json /path/to/floor_1.json \\
        --output-dir /path/to/mjcf_export
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import pickle
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG = logging.getLogger("mansion_to_mjcf")

OBJATHOR_UID_RE = re.compile(r"^[0-9a-f]{32}$")


# ---------------------------------------------------------------------------
# Asset resolution
# ---------------------------------------------------------------------------


@dataclass
class AssetSource:
    """Where a given assetId resolves and which source kind it is."""

    kind: str  # "objathor_pkl" | "objathor_json" | "thor_mjcf"
    data_path: Path  # .pkl.gz / .json / .xml
    asset_dir: Path  # parent dir (textures and meshes live here)


def _index_thor_mjcf(thor_mjcf_dir: Path) -> dict[str, Path]:
    """Build a {stem -> xml_path} index of molmospaces THOR MJCF files,
    ignoring the ``_mesh.xml`` / ``_prim.xml`` / ``_old.xml`` variants."""

    out: dict[str, Path] = {}
    for p in thor_mjcf_dir.rglob("*.xml"):
        name = p.name.lower()
        if name.endswith("_mesh.xml") or name.endswith("_prim.xml") or name.endswith("_old.xml"):
            continue
        out[p.stem] = p
    return out


def locate_asset(
    asset_id: str,
    objathor_dir: Path,
    thor_mjcf_index: dict[str, Path],
) -> AssetSource | None:
    cand = objathor_dir / asset_id
    if cand.is_dir():
        pkl = cand / f"{asset_id}.pkl.gz"
        if pkl.is_file():
            return AssetSource("objathor_pkl", pkl, cand)
        js = cand / f"{asset_id}.json"
        if js.is_file():
            return AssetSource("objathor_json", js, cand)

    xml = thor_mjcf_index.get(asset_id)
    if xml is not None:
        return AssetSource("thor_mjcf", xml, xml.parent)
    return None


def load_objathor_dict(src: AssetSource) -> dict[str, Any]:
    if src.kind == "objathor_pkl":
        with gzip.open(src.data_path, "rb") as f:
            return pickle.load(f)
    if src.kind == "objathor_json":
        return json.loads(src.data_path.read_text())
    raise ValueError(f"unsupported objathor source kind: {src.kind}")


# ---------------------------------------------------------------------------
# OBJ writer
# ---------------------------------------------------------------------------


def write_obj(
    out_path: Path,
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    normals: list[tuple[float, float, float]] | None = None,
    uvs: list[tuple[float, float]] | None = None,
) -> None:
    """Write a Wavefront .obj. Indices in ``triangles`` are 0-based; we shift
    to 1-based for OBJ format."""

    lines: list[str] = []
    for x, y, z in vertices:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    if uvs is not None:
        for u, v in uvs:
            lines.append(f"vt {u:.6f} {v:.6f}")
    if normals is not None:
        for nx, ny, nz in normals:
            lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")

    has_vt = uvs is not None
    has_vn = normals is not None
    for a, b, c in triangles:
        i, j, k = a + 1, b + 1, c + 1  # OBJ is 1-indexed
        if has_vt and has_vn:
            lines.append(f"f {i}/{i}/{i} {j}/{j}/{j} {k}/{k}/{k}")
        elif has_vn:
            lines.append(f"f {i}//{i} {j}//{j} {k}//{k}")
        elif has_vt:
            lines.append(f"f {i}/{i} {j}/{j} {k}/{k}")
        else:
            lines.append(f"f {i} {j} {k}")

    out_path.write_text("\n".join(lines) + "\n")


def _dicts_to_xyz(seq: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    return [(float(p["x"]), float(p["y"]), float(p["z"])) for p in seq]


def _dicts_to_uv(seq: list[dict[str, Any]]) -> list[tuple[float, float]]:
    return [(float(p["x"]), float(p["y"])) for p in seq]


# ---------------------------------------------------------------------------
# Asset baking results
# ---------------------------------------------------------------------------


@dataclass
class BakedAsset:
    """Mesh/material/texture defs and per-geom bindings for one assetId."""

    asset_id: str
    # textures: list of (name, file_relpath)
    textures: list[tuple[str, str]] = field(default_factory=list)
    # materials: list of (name, attrs_dict)
    materials: list[tuple[str, dict[str, str]]] = field(default_factory=list)
    # meshes: list of (name, file_relpath)
    meshes: list[tuple[str, str]] = field(default_factory=list)
    # per-geom binding for instances: list of {mesh, material?}
    geoms: list[dict[str, str]] = field(default_factory=list)
    # Anchor offset (Unity frame). For objathor / patched assets, the mesh
    # origin is at bbox-BOTTOM-center while mansion's JSON position is the
    # bbox CENTER, so we subtract (ymax+ymin)/2 from the body's y. THOR-MJCF
    # assets use their own conventions baked into the source XML, so offset=0.
    anchor_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Asset-intrinsic rotation offset about the up axis, in degrees, from the
    # ``yRotOffset`` field in the objathor pkl/json. Mansion's per-instance
    # ``rotation.y`` is added to this; without it, assets whose natural
    # orientation differs from the canonical pose (e.g. waiting_bench-0 with
    # an L-shape needing 45°) appear rotated wrong. THOR-MJCF assets have
    # this baked into their source XML, so default 0 is correct for them.
    y_rot_offset_deg: float = 0.0


def bake_objathor(
    asset_id: str, src: AssetSource, out_assets: Path
) -> BakedAsset | None:
    """Convert an objathor pkl/json asset to <out_assets>/<id>/<id>.obj plus
    copied albedo.jpg."""

    try:
        d = load_objathor_dict(src)
    except Exception as exc:
        LOG.warning("failed to load %s: %s", asset_id, exc)
        return None

    verts = _dicts_to_xyz(d["vertices"])
    raw_tris = list(d["triangles"])
    if len(raw_tris) % 3 != 0:
        LOG.warning("%s: triangle count %d not divisible by 3", asset_id, len(raw_tris))
        return None
    triangles = [tuple(raw_tris[i : i + 3]) for i in range(0, len(raw_tris), 3)]
    normals = _dicts_to_xyz(d["normals"]) if d.get("normals") else None
    uvs = _dicts_to_uv(d["uvs"]) if d.get("uvs") else None

    # Compute bbox center — mansion's JSON position.y is the bbox CENTER height
    # in world coords, but the mesh origin is at the bbox BOTTOM-center.
    # So body.pos.y must be shifted by -bbox_center.y to land the bbox correctly.
    # x/z are typically symmetric around 0 already; we still compute to be safe.
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    bbox_center = (
        (min(xs) + max(xs)) / 2.0,
        (min(ys) + max(ys)) / 2.0,
        (min(zs) + max(zs)) / 2.0,
    )

    safe_id = _safe_mjcf_name(asset_id)
    asset_subdir = out_assets / safe_id
    asset_subdir.mkdir(parents=True, exist_ok=True)
    obj_path = asset_subdir / f"{safe_id}.obj"
    write_obj(obj_path, verts, triangles, normals, uvs)

    # copy albedo if present — MuJoCo 3.x only loads .png natively, so convert.
    tex_relpath: str | None = None
    src_albedo = src.asset_dir / "albedo.jpg"
    if src_albedo.is_file():
        dst = asset_subdir / "albedo.png"
        if not dst.exists():
            from PIL import Image
            Image.open(src_albedo).convert("RGB").save(dst, "PNG")
        tex_relpath = f"{safe_id}/albedo.png"

    # objathor pkl/json stores ``yRotOffset`` (in degrees) as an asset-intrinsic
    # rotation offset about the up axis. It must be added to mansion's per-instance
    # rotation.y at placement time. Default to 0 if absent.
    y_rot_offset = float(d.get("yRotOffset", 0.0))
    mesh_name = f"{safe_id}__mesh"
    baked = BakedAsset(asset_id=asset_id, anchor_offset=bbox_center,
                       y_rot_offset_deg=y_rot_offset)
    baked.meshes.append((mesh_name, f"{safe_id}/{safe_id}.obj"))
    geom_attrs = {"mesh": mesh_name}
    if tex_relpath is not None:
        tex_name = f"{safe_id}__tex"
        mat_name = f"{safe_id}__mat"
        baked.textures.append((tex_name, tex_relpath))
        baked.materials.append(
            (
                mat_name,
                {
                    "texture": tex_name,
                    "specular": "0.1",
                    "shininess": "0.1",
                },
            )
        )
        geom_attrs["material"] = mat_name
    baked.geoms.append(geom_attrs)
    return baked


def bake_thor_mjcf(
    asset_id: str, src: AssetSource, out_assets: Path
) -> BakedAsset | None:
    """Parse a molmospaces THOR MJCF, copy its meshes/textures, and emit
    prefixed asset definitions + per-geom bindings (visual mesh only)."""

    try:
        tree = ET.parse(src.data_path)
    except Exception as exc:
        LOG.warning("failed to parse THOR MJCF %s: %s", src.data_path, exc)
        return None

    root = tree.getroot()
    safe_id = _safe_mjcf_name(asset_id)
    asset_subdir = out_assets / safe_id
    asset_subdir.mkdir(parents=True, exist_ok=True)

    baked = BakedAsset(asset_id=asset_id)

    def _copy_into(rel: str) -> str:
        src_file = src.asset_dir / rel
        if not src_file.is_file():
            raise FileNotFoundError(src_file)
        # MuJoCo 3.x only loads .png natively for textures; convert .jpg/.jpeg
        # to .png on copy. Other file types (.obj meshes, etc.) just copy.
        sfx = src_file.suffix.lower()
        if sfx in (".jpg", ".jpeg"):
            rel_png = rel[: -len(src_file.suffix)] + ".png"
            dst = asset_subdir / rel_png
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                from PIL import Image
                Image.open(src_file).convert("RGB").save(dst, "PNG")
            return f"{safe_id}/{rel_png}"
        dst = asset_subdir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src_file, dst)
        return f"{safe_id}/{rel}"

    # --- textures
    tex_name_map: dict[str, str] = {}  # old name -> new name
    for t in root.iter("texture"):
        old = t.get("name")
        if not old:
            continue
        new = f"{safe_id}__tex_{_safe_mjcf_name(old)}"
        tex_name_map[old] = new
        file_attr = t.get("file")
        if file_attr:
            try:
                rel_in_out = _copy_into(file_attr)
            except FileNotFoundError as exc:
                LOG.warning("texture missing: %s (%s)", exc, asset_id)
                continue
            baked.textures.append((new, rel_in_out))
        # textures defined inline (builtin/rgb*) are skipped — emit attrs as-is later if needed

    # --- materials (rgba and/or texture ref)
    mat_name_map: dict[str, str] = {}
    for m in root.iter("material"):
        old = m.get("name")
        if not old:
            continue
        new = f"{safe_id}__mat_{_safe_mjcf_name(old)}"
        mat_name_map[old] = new
        attrs = {}
        rgba = m.get("rgba")
        if rgba is not None:
            attrs["rgba"] = rgba
        old_tex = m.get("texture")
        if old_tex is not None and old_tex in tex_name_map:
            attrs["texture"] = tex_name_map[old_tex]
        specular = m.get("specular")
        shininess = m.get("shininess")
        if specular is not None:
            attrs["specular"] = specular
        if shininess is not None:
            attrs["shininess"] = shininess
        baked.materials.append((new, attrs))

    # --- meshes
    mesh_name_map: dict[str, str] = {}
    for me in root.iter("mesh"):
        old = me.get("name")
        if not old:
            continue
        new = f"{safe_id}__mesh_{_safe_mjcf_name(old)}"
        mesh_name_map[old] = new
        file_attr = me.get("file")
        if not file_attr:
            continue
        try:
            rel_in_out = _copy_into(file_attr)
        except FileNotFoundError as exc:
            LOG.warning("mesh missing: %s (%s)", exc, asset_id)
            continue
        baked.meshes.append((new, rel_in_out))

    # --- per-geom bindings + per-geom transform relative to asset root
    # Walk the source XML body tree starting at the asset root (= first <body>
    # under <worldbody>), composing each parent body's pos+quat. Capture each
    # visual <geom>'s cumulative transform so the door / window asset's
    # bbox-center anchor lands at the instance position (not bbox-bottom).
    worldbody = root.find("worldbody")
    if worldbody is None:
        LOG.warning("THOR asset %s has no <worldbody>", asset_id)
        return baked
    asset_root = next(iter(worldbody.findall("body")), None)
    if asset_root is None:
        LOG.warning("THOR asset %s has no root <body> under <worldbody>", asset_id)
        return baked

    def _parse_pos(s: str | None) -> tuple[float, float, float]:
        if not s:
            return (0.0, 0.0, 0.0)
        parts = s.split()
        return (float(parts[0]), float(parts[1]), float(parts[2]))

    def _parse_quat(s: str | None) -> tuple[float, float, float, float]:
        # MuJoCo quaternion order: (w, x, y, z). Identity = (1, 0, 0, 0).
        if not s:
            return (1.0, 0.0, 0.0, 0.0)
        parts = s.split()
        return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))

    def walk(body: ET.Element, parent_pos, parent_quat) -> None:
        local_pos = _parse_pos(body.get("pos"))
        local_quat = _parse_quat(body.get("quat"))
        cum_pos, cum_quat = _compose_transform(parent_pos, parent_quat,
                                                local_pos, local_quat)
        for g in body.findall("geom"):
            if g.get("type") != "mesh":
                continue
            cls = (g.get("class") or "").upper()
            if "VISUAL" not in cls and cls:
                continue  # skip colliders
            old_mesh = g.get("mesh")
            if old_mesh is None or old_mesh not in mesh_name_map:
                continue

            g_pos = _parse_pos(g.get("pos"))
            g_quat = _parse_quat(g.get("quat"))
            geom_pos, geom_quat = _compose_transform(cum_pos, cum_quat,
                                                      g_pos, g_quat)

            binding: dict[str, str] = {"mesh": mesh_name_map[old_mesh]}
            old_mat = g.get("material")
            if old_mat is not None and old_mat in mat_name_map:
                binding["material"] = mat_name_map[old_mat]
            rgba = g.get("rgba")
            if rgba is not None and "material" not in binding:
                binding["rgba"] = rgba
            if any(abs(v) > 1e-9 for v in geom_pos):
                binding["pos"] = f"{geom_pos[0]:.6f} {geom_pos[1]:.6f} {geom_pos[2]:.6f}"
            if not _is_identity_quat(geom_quat):
                binding["quat"] = f"{geom_quat[0]:.6f} {geom_quat[1]:.6f} {geom_quat[2]:.6f} {geom_quat[3]:.6f}"
            baked.geoms.append(binding)
        for child in body.findall("body"):
            walk(child, cum_pos, cum_quat)

    # Start at asset root with identity transform — the asset root's own pos /
    # quat (typically zero / identity) is "where the asset anchor sits in the
    # instance body's frame," so the instance body itself effectively occupies
    # that anchor. We therefore include the asset root's own transform too.
    walk(asset_root, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

    if not baked.geoms:
        LOG.warning("THOR asset %s produced no visual geoms", asset_id)
    return baked


# ---------------------------------------------------------------------------
# Quaternion math (MuJoCo convention: (w, x, y, z))
# ---------------------------------------------------------------------------


def _quat_mul(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_rotate(
    q: tuple[float, float, float, float], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    w, x, y, z = q
    vx, vy, vz = v
    # v_rot = q * v_q * q_conj where v_q = (0, vx, vy, vz)
    # Direct expansion (avoids building intermediate quats):
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _compose_transform(
    p1: tuple[float, float, float],
    q1: tuple[float, float, float, float],
    p2: tuple[float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Apply T2 = (p2, q2) after T1 = (p1, q1). New pos = p1 + R(q1) * p2; new quat = q1 * q2."""

    rp2 = _quat_rotate(q1, p2)
    new_pos = (p1[0] + rp2[0], p1[1] + rp2[1], p1[2] + rp2[2])
    new_quat = _quat_mul(q1, q2)
    return new_pos, new_quat


def _is_identity_quat(q: tuple[float, float, float, float], tol: float = 1e-9) -> bool:
    return abs(q[0] - 1.0) < tol and abs(q[1]) < tol and abs(q[2]) < tol and abs(q[3]) < tol


def _safe_mjcf_name(name: str) -> str:
    """MJCF names must be unique and avoid certain chars."""

    s = re.sub(r"[^0-9A-Za-z_-]", "_", name)
    if not s:
        s = "x"
    return s


# ---------------------------------------------------------------------------
# Polygon mesh helpers (rooms / walls)
# ---------------------------------------------------------------------------


def _triangulate_polygon(
    pts: list[tuple[float, float, float]],
) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple, possibly concave, planar polygon.

    Returns 0-based ``(i, j, k)`` index triples wound in the polygon's vertex
    order. Correct for concave rooms (e.g. L-shaped floors); a fan would spill
    triangles across the concave notch and fill the convex hull instead.

    Raises ``ValueError`` if the polygon is degenerate or self-intersecting: a
    non-simple polygon has no valid triangulation, so the caller skips and logs
    it rather than emit a silently-wrong mesh.
    """
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    # Project onto the polygon's plane: Newell's normal -> drop the dominant
    # axis. Handles horizontal floors and vertical walls alike.
    nx = ny = nz = 0.0
    for i in range(n):
        cx, cy, cz = pts[i]
        ax, ay, az = pts[(i + 1) % n]
        nx += (cy - ay) * (cz + az)
        ny += (cz - az) * (cx + ax)
        nz += (cx - ax) * (cy + ay)
    drop = max(range(3), key=lambda k: abs((nx, ny, nz)[k]))
    u, v = (k for k in range(3) if k != drop)
    p2 = [(pt[u], pt[v]) for pt in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signed = sum(
        p2[i][0] * p2[(i + 1) % n][1] - p2[(i + 1) % n][0] * p2[i][1]
        for i in range(n)
    )
    ccw = signed > 0.0

    def is_convex(a, b, c):
        cr = cross(p2[a], p2[b], p2[c])
        return cr > 0.0 if ccw else cr < 0.0

    def in_triangle(p, a, b, c):
        d1 = cross(p2[a], p2[b], p2[p])
        d2 = cross(p2[b], p2[c], p2[p])
        d3 = cross(p2[c], p2[a], p2[p])
        neg = d1 < 0.0 or d2 < 0.0 or d3 < 0.0
        pos = d1 > 0.0 or d2 > 0.0 or d3 > 0.0
        return not (neg and pos)

    remaining = list(range(n))
    tris: list[tuple[int, int, int]] = []
    guard = 0
    while len(remaining) > 3 and guard < n * n:
        guard += 1
        m = len(remaining)
        clipped = False
        for k in range(m):
            a = remaining[(k - 1) % m]
            b = remaining[k]
            c = remaining[(k + 1) % m]
            if not is_convex(a, b, c):
                continue
            if any(j not in (a, b, c) and in_triangle(j, a, b, c) for j in remaining):
                continue
            tris.append((a, b, c))
            remaining.pop(k)
            clipped = True
            break
        if not clipped:
            break

    if len(remaining) == 3:
        tris.append((remaining[0], remaining[1], remaining[2]))

    if len(tris) != n - 2:
        raise ValueError(
            f"cannot triangulate polygon (n={n}): not a simple polygon "
            f"(self-intersecting or degenerate) — ear clipping produced "
            f"{len(tris)} of {n - 2} triangles"
        )
    return tris


def write_polygon_obj(
    out_path: Path, polygon: list[dict[str, float]]
) -> None:
    pts = _dicts_to_xyz(polygon)
    if len(pts) < 3:
        raise ValueError("polygon < 3 points")
    tris = _triangulate_polygon(pts)
    write_obj(out_path, pts, tris)


def _wall_local_frame(
    polygon: list[dict[str, float]],
    kind: str = "window",
) -> tuple[
    tuple[float, float, float],  # origin
    tuple[float, float, float],  # x_axis (along bottom edge, in xz plane)
    tuple[float, float, float],  # y_axis (world up = Unity +y)
    float,                       # width  (bottom edge length)
    float,                       # height (max_y - min_y)
] | None:
    """Detect the wall's 2D-local frame.

    ``kind`` matches the convention used by ``compute_wall_aligned_pose``:
    doors use the predecessor (BR-start), windows use the successor (BL-start).
    See bug log §L. Default ``"window"`` because cutouts called without an
    explicit kind are most often window-shaped exterior holes."""

    ys = [float(p["y"]) for p in polygon]
    min_y = min(ys)
    max_y = max(ys)
    bottom_idxs = [i for i, y in enumerate(ys) if abs(y - min_y) < 1e-6]
    if len(bottom_idxs) != 2:
        return None
    n = len(polygon)
    if (bottom_idxs[0] + 1) % n == bottom_idxs[1]:
        if kind == "door":
            i_start, i_end = bottom_idxs[0], bottom_idxs[1]
        else:
            i_start, i_end = bottom_idxs[1], bottom_idxs[0]
    elif (bottom_idxs[1] + 1) % n == bottom_idxs[0]:
        if kind == "door":
            i_start, i_end = bottom_idxs[1], bottom_idxs[0]
        else:
            i_start, i_end = bottom_idxs[0], bottom_idxs[1]
    else:
        return None
    p_start = polygon[i_start]
    p_end = polygon[i_end]
    dx = float(p_end["x"]) - float(p_start["x"])
    dz = float(p_end["z"]) - float(p_start["z"])
    width = math.hypot(dx, dz)
    if width < 1e-9:
        return None
    ux = (dx / width, 0.0, dz / width)
    uy = (0.0, 1.0, 0.0)
    origin = (float(p_start["x"]), min_y, float(p_start["z"]))
    return origin, ux, uy, width, max_y - min_y


def _subtract_holes_from_rect(
    rect: tuple[float, float, float, float],
    holes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Decompose ``rect`` minus axis-aligned rectangular ``holes`` into a list
    of axis-aligned sub-rectangles (no overlaps, no holes). Algorithm: for
    each hole, split each remaining rect into up to 4 pieces (top, bottom,
    left, right of the hole's clipped portion). Assumes holes don't overlap
    each other (true for doors/windows on the same wall)."""

    pieces: list[tuple[float, float, float, float]] = [rect]
    for hx0, hy0, hx1, hy1 in holes:
        next_pieces: list[tuple[float, float, float, float]] = []
        for rx0, ry0, rx1, ry1 in pieces:
            cx0 = max(rx0, hx0); cy0 = max(ry0, hy0)
            cx1 = min(rx1, hx1); cy1 = min(ry1, hy1)
            if cx0 >= cx1 or cy0 >= cy1:
                next_pieces.append((rx0, ry0, rx1, ry1))
                continue
            # top strip
            if cy1 < ry1:
                next_pieces.append((rx0, cy1, rx1, ry1))
            # bottom strip
            if cy0 > ry0:
                next_pieces.append((rx0, ry0, rx1, cy0))
            # left strip (only within the hole's y-range, to avoid overlap with top/bottom)
            if cx0 > rx0:
                next_pieces.append((rx0, cy0, cx0, cy1))
            # right strip
            if cx1 < rx1:
                next_pieces.append((cx1, cy0, rx1, cy1))
        pieces = next_pieces
    return pieces


def write_wall_with_holes_obj(
    out_path: Path,
    polygon: list[dict[str, float]],
    holes_local: list[tuple[float, float, float, float]],
) -> bool:
    """Write a wall .obj with rectangular cutouts at the given wall-local 2D
    rectangles. Returns True on success, False on fall-through (caller should
    fall back to the simple full-polygon writer)."""

    frame = _wall_local_frame(polygon)
    if frame is None:
        return False
    origin, ux, uy, width, height = frame

    # Clip holes to the wall rectangle just in case
    clipped: list[tuple[float, float, float, float]] = []
    for hx0, hy0, hx1, hy1 in holes_local:
        cx0 = max(0.0, hx0); cy0 = max(0.0, hy0)
        cx1 = min(width, hx1); cy1 = min(height, hy1)
        if cx1 - cx0 > 1e-6 and cy1 - cy0 > 1e-6:
            clipped.append((cx0, cy0, cx1, cy1))

    rects = _subtract_holes_from_rect((0.0, 0.0, width, height), clipped)
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    for rx0, ry0, rx1, ry1 in rects:
        base = len(verts)
        # Project each corner from wall-local 2D into world 3D
        for lx, ly in ((rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)):
            verts.append((
                origin[0] + lx * ux[0] + ly * uy[0],
                origin[1] + lx * ux[1] + ly * uy[1],
                origin[2] + lx * ux[2] + ly * uy[2],
            ))
        tris.append((base + 0, base + 1, base + 2))
        tris.append((base + 0, base + 2, base + 3))

    if not verts:
        # Wall is entirely consumed by holes — emit a degenerate placeholder
        return False
    write_obj(out_path, verts, tris)
    return True


# ---------------------------------------------------------------------------
# Door / window pose from wall + assetPosition
# ---------------------------------------------------------------------------


def compute_pose_from_segment(
    item: dict[str, Any],
    walls_by_id: dict[str, dict[str, Any]],
) -> tuple[tuple[float, float, float], float] | None:
    """Cleaner placement using mansion's ``doorSegment`` / ``windowSegment``
    (world-coordinate ground truth). Position = segment midpoint, height from
    ``assetPosition.y``, rotation from segment direction.

    Falls back to None if the item has no segment (caller should then try
    :func:`compute_wall_aligned_pose`).
    """

    seg = item.get("doorSegment") or item.get("windowSegment")
    if not seg or len(seg) < 2:
        return None
    # Endpoints are [x, z] in world coordinates (at the wall's floor).
    (x0, z0), (x1, z1) = seg[0], seg[1]
    mx = (float(x0) + float(x1)) / 2.0
    mz = (float(z0) + float(z1)) / 2.0
    ap = item.get("assetPosition") or {"y": 0.0}
    my = float(ap.get("y", 0.0))

    dx = float(x1) - float(x0)
    dz = float(z1) - float(z0)
    # Rotation matches compute_wall_aligned_pose: θ = atan2(-dz, dx) so the
    # asset's local +x axis aligns with the wall direction in Unity coords.
    theta_deg = math.degrees(math.atan2(-dz, dx))
    return (mx, my, mz), theta_deg


def world_hole_corners_from_segment(
    item: dict[str, Any],
    walls_by_id: dict[str, dict[str, Any]],
) -> list[tuple[float, float, float]] | None:
    """Compute the hole's 4 world corners from ``doorSegment`` / ``windowSegment``
    + ``holePolygon`` y-range. Endpoints of the segment are the hole's bottom
    corners (at the wall's floor y); the holePolygon's y range gives the
    vertical extent above the floor."""

    seg = item.get("doorSegment") or item.get("windowSegment")
    if not seg or len(seg) < 2:
        return None
    hp = item.get("holePolygon")
    if not hp or len(hp) < 2:
        return None

    # Wall floor y: use the referenced wall's min y if available, else 0.
    wid = item.get("wall0") or item.get("wall1")
    wall = walls_by_id.get(wid) if wid else None
    floor_y = 0.0
    if wall is not None:
        ys = [float(p["y"]) for p in (wall.get("polygon") or [])]
        if ys:
            floor_y = min(ys)

    ys_hp = [float(p["y"]) for p in hp]
    y_lo = min(ys_hp) + floor_y
    y_hi = max(ys_hp) + floor_y

    (x0, z0), (x1, z1) = seg[0], seg[1]
    return [
        (float(x0), y_lo, float(z0)),  # bottom corner at seg[0]
        (float(x1), y_lo, float(z1)),  # bottom corner at seg[1]
        (float(x1), y_hi, float(z1)),  # top corner at seg[1]
        (float(x0), y_hi, float(z0)),  # top corner at seg[0]
    ]


def compute_wall_aligned_pose(
    item: dict[str, Any],
    walls_by_id: dict[str, dict[str, Any]],
    kind: str = "door",
) -> tuple[tuple[float, float, float], float] | None:
    """Compute (world_position, world_y_rotation_degrees) for a door or window.

    Mansion doors/windows have no top-level ``position``/``rotation`` — only
    ``assetPosition`` in the wall's local frame plus ``wall0``/``wall1`` IDs.
    See bug log §K (initial mistake), then §L: doors and windows use
    **different** corner conventions because mansion's authoring places doors
    relative to the destination room's wall frame (wall1) while windows are
    authored relative to wall0. Operationally, that means:
      - ``kind == "door"``    → start at predecessor (the bottom-edge corner
                                that comes earlier in the polygon's traversal).
                                Equivalent to "first bot vertex in poly order".
      - ``kind == "window"``  → start at successor (the bottom-edge corner
                                that comes later in the polygon's traversal).

    The wall's polygon ordering is NOT consistent across mansion (some
    walls use ``[TL, TR, BR, BL]``, others ``[BL, TL, TR, BR]``), so we
    detect the bottom edge by min-y and pick start by polygon-traversal
    direction.
    """

    asset_pos = item.get("assetPosition")
    if asset_pos is None:
        return None
    wall_id = item.get("wall0") or item.get("wall1")
    wall = walls_by_id.get(wall_id) if wall_id else None
    if wall is None:
        return None
    poly = wall.get("polygon") or []
    if len(poly) < 4:
        return None

    ys = [float(p["y"]) for p in poly]
    min_y = min(ys)
    bottom_idxs = [i for i, y in enumerate(ys) if abs(y - min_y) < 1e-6]
    if len(bottom_idxs) != 2:
        return None

    n = len(poly)
    # See docstring: doors → predecessor; windows → successor.
    if (bottom_idxs[0] + 1) % n == bottom_idxs[1]:
        if kind == "door":
            i_start, i_end = bottom_idxs[0], bottom_idxs[1]
        else:
            i_start, i_end = bottom_idxs[1], bottom_idxs[0]
    elif (bottom_idxs[1] + 1) % n == bottom_idxs[0]:
        if kind == "door":
            i_start, i_end = bottom_idxs[1], bottom_idxs[0]
        else:
            i_start, i_end = bottom_idxs[0], bottom_idxs[1]
    else:
        return None  # bottom verts not adjacent — unexpected

    p_start = poly[i_start]
    p_end = poly[i_end]
    dx = float(p_end["x"]) - float(p_start["x"])
    dz = float(p_end["z"]) - float(p_start["z"])
    norm = math.hypot(dx, dz)
    if norm < 1e-9:
        return None
    ux, uz = dx / norm, dz / norm  # wall's "along" axis in the world x-z plane

    ax = float(asset_pos.get("x", 0.0))
    ay = float(asset_pos.get("y", 0.0))
    az_offset = float(asset_pos.get("z", 0.0))  # almost always 0 in mansion

    world_x = float(p_start["x"]) + ax * ux + az_offset * (-uz)
    world_y = min_y + ay
    world_z = float(p_start["z"]) + ax * uz + az_offset * ux

    # The door's local +x axis (Unity) maps to the wall's along-axis.
    # Rotation about Unity +y by θ takes (1,0,0) → (cos θ, 0, -sin θ),
    # so we want cos θ = ux and -sin θ = uz, hence θ = atan2(-uz, ux).
    theta_deg = math.degrees(math.atan2(-uz, ux))
    return (world_x, world_y, world_z), theta_deg


# ---------------------------------------------------------------------------
# MJCF assembly
# ---------------------------------------------------------------------------


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _format_attrs(attrs: dict[str, str]) -> str:
    return " ".join(f'{k}="{_xml_escape(v)}"' for k, v in attrs.items())


@dataclass
class ConversionReport:
    scene_json: str
    output_dir: str
    n_objects_total: int = 0
    n_objects_placed: int = 0
    n_doors_total: int = 0
    n_doors_placed: int = 0
    n_windows_total: int = 0
    n_windows_placed: int = 0
    n_rooms: int = 0
    n_walls: int = 0
    unique_assetids: int = 0
    baked_objathor: int = 0
    baked_thor_mjcf: int = 0
    missing_assetids: list[str] = field(default_factory=list)
    skipped_objects: list[dict[str, str]] = field(default_factory=list)


def convert(
    scene_json: Path,
    output_dir: Path,
    objathor_dir: Path,
    thor_mjcf_dir: Path,
) -> Path:
    scene = json.loads(scene_json.read_text())

    out_root = output_dir / scene_json.parent.name.replace("#", "_") / scene_json.stem
    out_assets = out_root / "assets"
    out_rooms = out_assets / "rooms"
    out_walls = out_assets / "walls"
    for d in (out_assets, out_rooms, out_walls):
        d.mkdir(parents=True, exist_ok=True)
    LOG.info("output directory: %s", out_root)

    LOG.info("indexing %s ...", thor_mjcf_dir)
    thor_index = _index_thor_mjcf(thor_mjcf_dir)
    LOG.info("  %d THOR MJCF entries indexed", len(thor_index))

    # ------------------------------------------------------------------- pass 1
    asset_sources: dict[str, AssetSource] = {}
    missing: list[str] = []
    for src in ("objects", "doors", "windows"):
        for o in scene.get(src) or []:
            aid = o.get("assetId")
            if not aid or aid in asset_sources or aid in missing:
                continue
            located = locate_asset(aid, objathor_dir, thor_index)
            if located is None:
                missing.append(aid)
            else:
                asset_sources[aid] = located

    LOG.info(
        "asset coverage: %d resolved, %d missing", len(asset_sources), len(missing)
    )
    for aid in missing:
        LOG.warning("missing asset: %s", aid)

    # ------------------------------------------------------------------- pass 2
    baked: dict[str, BakedAsset] = {}
    for aid, src in asset_sources.items():
        try:
            if src.kind in ("objathor_pkl", "objathor_json"):
                ba = bake_objathor(aid, src, out_assets)
            elif src.kind == "thor_mjcf":
                ba = bake_thor_mjcf(aid, src, out_assets)
            else:
                ba = None
        except Exception as exc:
            LOG.exception("bake failed for %s: %s", aid, exc)
            ba = None
        if ba is None or not ba.meshes:
            missing.append(aid)
            continue
        baked[aid] = ba
        LOG.info("baked %s (%s)  meshes=%d  textures=%d  materials=%d",
                 aid, src.kind, len(ba.meshes), len(ba.textures), len(ba.materials))

    # ------------------------------------------------------------------- pass 3
    # write per-room and per-wall polygon meshes
    room_meshes: list[tuple[str, str]] = []  # (name, file_relpath)
    for room in scene.get("rooms") or []:
        rid = _safe_mjcf_name(room.get("id", "room"))
        poly = room.get("floorPolygon") or []
        if len(poly) < 3:
            continue
        try:
            write_polygon_obj(out_rooms / f"{rid}.obj", poly)
            room_meshes.append((f"floor_{rid}", f"rooms/{rid}.obj"))
        except Exception as exc:
            LOG.warning("room %s skipped: %s", rid, exc)

    # Pre-index walls by id (used for hole projection below and door/window
    # placement in the instances pass).
    walls_by_id: dict[str, dict[str, Any]] = {
        w["id"]: w for w in (scene.get("walls") or []) if w.get("id")
    }

    # Build holes_by_wall by projecting each hole through WORLD space.
    # holePolygon is in wall0's local frame. Since wall1 (the coincident other
    # side of the same physical wall) often has a polygon vertex order with
    # the OPPOSITE bottom-edge direction and a DIFFERENT bottom-edge start
    # vertex, we cannot reuse the same (xmin..xmax, ymin..ymax) tuple for
    # both walls. Instead:
    #   1. Build wall0's local frame.
    #   2. Compute the hole's 4 corners in world coords via wall0's frame.
    #   3. For each wall (wall0 AND wall1), project the world corners into
    #      that wall's own local frame; take the 2D bbox of the projection.
    holes_by_wall: dict[str, list[tuple[float, float, float, float]]] = {}

    def _project_to_wall_local(
        world_pt: tuple[float, float, float], frame
    ) -> tuple[float, float]:
        origin, ux, uy, _width, _height = frame
        dx = world_pt[0] - origin[0]
        dy = world_pt[1] - origin[1]
        dz = world_pt[2] - origin[2]
        lx = dx * ux[0] + dy * ux[1] + dz * ux[2]
        ly = dx * uy[0] + dy * uy[1] + dz * uy[2]
        return lx, ly

    for src in ("doors", "windows"):
        kind = "door" if src == "doors" else "window"
        for item in scene.get(src) or []:
            hp = item.get("holePolygon") or []
            if len(hp) < 2:
                continue
            wid0 = item.get("wall0")
            wall0 = walls_by_id.get(wid0) if wid0 else None
            if wall0 is None:
                continue
            # Prefer world-coordinate hole corners from the segment.
            corners_world = world_hole_corners_from_segment(item, walls_by_id)
            if corners_world is None:
                # Fall back to projecting holePolygon (wall-local 2D) through
                # wall0's frame using the polygon-corner rule (see §L).
                frame0 = _wall_local_frame(wall0.get("polygon") or [], kind=kind)
                if frame0 is None:
                    continue
                origin0, ux0, uy0, _w0, _h0 = frame0
                xs = [float(p["x"]) for p in hp]
                ys = [float(p["y"]) for p in hp]
                lo_x, lo_y = min(xs), min(ys)
                hi_x, hi_y = max(xs), max(ys)
                corners_local = [(lo_x, lo_y), (hi_x, lo_y), (hi_x, hi_y), (lo_x, hi_y)]
                corners_world = [
                    (
                        origin0[0] + lx * ux0[0] + ly * uy0[0],
                        origin0[1] + lx * ux0[1] + ly * uy0[1],
                        origin0[2] + lx * ux0[2] + ly * uy0[2],
                    )
                    for (lx, ly) in corners_local
                ]

            for wid_key in ("wall0", "wall1"):
                wid = item.get(wid_key)
                if not wid:
                    continue
                wall = walls_by_id.get(wid)
                if wall is None:
                    continue
                # The wall frame for the CUTOUT-MESH-LOCAL coords can use
                # either rule — the world-projection step makes the rectangle
                # land at the right world position either way. Use "window"
                # rule by default (matches the wall-mesh writer below).
                frame = _wall_local_frame(wall.get("polygon") or [])
                if frame is None:
                    continue
                lxs, lys = zip(*(_project_to_wall_local(p, frame) for p in corners_world))
                rect = (min(lxs), min(lys), max(lxs), max(lys))
                holes_by_wall.setdefault(wid, []).append(rect)

    wall_meshes: list[tuple[str, str]] = []
    walls_cut = 0
    for i, wall in enumerate(scene.get("walls") or []):
        wid_raw = wall.get("id", f"wall_{i}")
        wid = _safe_mjcf_name(wid_raw)
        poly = wall.get("polygon") or []
        if len(poly) < 3:
            continue
        try:
            holes = holes_by_wall.get(wid_raw, [])
            ok = False
            if holes:
                ok = write_wall_with_holes_obj(out_walls / f"{wid}.obj", poly, holes)
                if ok:
                    walls_cut += 1
            if not ok:
                write_polygon_obj(out_walls / f"{wid}.obj", poly)
            wall_meshes.append((f"wall_{wid}", f"walls/{wid}.obj"))
        except Exception as exc:
            LOG.warning("wall %s skipped: %s", wid, exc)
    LOG.info("walls written: %d (%d with cutouts)", len(wall_meshes), walls_cut)

    # ------------------------------------------------------------------- pass 4
    # build scene.xml
    scene_xml = out_root / "scene.xml"
    report = ConversionReport(
        scene_json=str(scene_json),
        output_dir=str(out_root),
        unique_assetids=len(asset_sources) + len(missing),
        baked_objathor=sum(1 for ba in baked.values()
                            if asset_sources[ba.asset_id].kind != "thor_mjcf"),
        baked_thor_mjcf=sum(1 for ba in baked.values()
                             if asset_sources[ba.asset_id].kind == "thor_mjcf"),
        missing_assetids=sorted(set(missing)),
    )

    lines: list[str] = []
    lines.append(f'<mujoco model="mansion_{_safe_mjcf_name(scene_json.stem)}">')
    # balanceinertia: flat (planar) room/wall meshes have a degenerate inertia
    # where A+B == C; floating-point rounding can tip it to A+B < C and fail the
    # compile. The scene is visual-only, so balancing these inertias is harmless.
    lines.append('  <compiler angle="degree" autolimits="true" '
                 'balanceinertia="true" meshdir="assets" texturedir="assets"/>')
    lines.append('  <option gravity="0 0 -9.81"/>')
    lines.append('  <visual>')
    lines.append('    <headlight diffuse="0.55 0.55 0.55" '
                 'ambient="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>')
    lines.append('    <global azimuth="120" elevation="-25"/>')
    lines.append('    <quality shadowsize="2048"/>')
    lines.append('  </visual>')

    # --- assets
    lines.append('  <asset>')

    # textures
    seen_tex: set[str] = set()
    for ba in baked.values():
        for name, relfile in ba.textures:
            if name in seen_tex:
                continue
            seen_tex.add(name)
            lines.append(f'    <texture type="2d" name="{_xml_escape(name)}" '
                         f'file="{_xml_escape(relfile)}"/>')
    # materials
    seen_mat: set[str] = set()
    for ba in baked.values():
        for name, attrs in ba.materials:
            if name in seen_mat:
                continue
            seen_mat.add(name)
            lines.append(f'    <material name="{_xml_escape(name)}" {_format_attrs(attrs)}/>')
    # asset meshes — inertia="shell" so MuJoCo doesn't reject thin / non-watertight
    # meshes during the volume-based inertia computation (we're visual-only anyway).
    seen_mesh: set[str] = set()
    for ba in baked.values():
        for name, relfile in ba.meshes:
            if name in seen_mesh:
                continue
            seen_mesh.add(name)
            lines.append(f'    <mesh name="{_xml_escape(name)}" '
                         f'file="{_xml_escape(relfile)}" inertia="shell"/>')
    # room/wall meshes (flat polygons — inertia="shell" is required, they have zero volume)
    for name, relfile in room_meshes:
        lines.append(f'    <mesh name="{_xml_escape(name)}" '
                     f'file="{_xml_escape(relfile)}" inertia="shell"/>')
    for name, relfile in wall_meshes:
        lines.append(f'    <mesh name="{_xml_escape(name)}" '
                     f'file="{_xml_escape(relfile)}" inertia="shell"/>')
    lines.append('  </asset>')

    # --- worldbody
    lines.append('  <worldbody>')
    lines.append('    <light pos="0 0 8" dir="0 0 -1" '
                 'diffuse="0.7 0.7 0.7" specular="0.05 0.05 0.05"/>')
    # Unity y-up → MuJoCo z-up: rotate root +90° about x
    lines.append('    <body name="scene_root" euler="90 0 0">')

    # rooms
    lines.append('      <body name="rooms">')
    for name, _ in room_meshes:
        lines.append(f'        <geom name="{_xml_escape(name)}" type="mesh" '
                     f'mesh="{_xml_escape(name)}" rgba="0.85 0.82 0.78 1" '
                     f'contype="0" conaffinity="0"/>')
        report.n_rooms += 1
    lines.append('      </body>')

    # walls
    lines.append('      <body name="walls">')
    for name, _ in wall_meshes:
        lines.append(f'        <geom name="{_xml_escape(name)}" type="mesh" '
                     f'mesh="{_xml_escape(name)}" rgba="0.92 0.92 0.92 1" '
                     f'contype="0" conaffinity="0"/>')
        report.n_walls += 1
    lines.append('      </body>')

    # instances
    lines.append('      <body name="instances">')
    instance_counts: dict[str, int] = {}

    def _place(category: str, item: dict[str, Any]) -> bool:
        aid = item.get("assetId")
        if not aid:
            return False
        ba = baked.get(aid)
        if ba is None:
            report.skipped_objects.append(
                {"id": item.get("id", ""), "assetId": aid or "", "reason": "asset missing"}
            )
            return False
        idx = instance_counts.get(aid, 0)
        instance_counts[aid] = idx + 1

        rx = ry = rz = 0.0
        if category in ("door", "window"):
            # Prefer the world-coordinate ground truth (doorSegment /
            # windowSegment). Fall back to assetPosition + wall frame for
            # items lacking a segment field. See bug log §L for context.
            wp = compute_pose_from_segment(item, walls_by_id)
            if wp is None:
                wp = compute_wall_aligned_pose(item, walls_by_id, kind=category)
            if wp is None:
                report.skipped_objects.append(
                    {"id": item.get("id", ""), "assetId": aid,
                     "reason": "could not resolve wall for door/window"}
                )
                return False
            (x, y, z), ry = wp
        else:
            pos = item.get("position") or {"x": 0, "y": 0, "z": 0}
            rot = item.get("rotation") or {"x": 0, "y": 0, "z": 0}
            x = float(pos["x"]); y = float(pos["y"]); z = float(pos["z"])
            rx = float(rot.get("x", 0.0))
            ry = float(rot.get("y", 0.0)) + ba.y_rot_offset_deg
            rz = float(rot.get("z", 0.0))

        # Apply the asset's anchor offset (objathor meshes use bbox-bottom-center
        # origin while mansion's position.y is bbox-center height — subtract
        # bbox_center to land the geometry correctly). For symmetric meshes
        # the x/z offsets are ~0; we still subtract for completeness.
        ox, oy, oz = ba.anchor_offset
        # Rotate the offset by ry around Unity +y so it's still correct after
        # the per-instance rotation (otherwise rotated bodies drift in xz).
        cos_r = math.cos(math.radians(ry))
        sin_r = math.sin(math.radians(ry))
        rot_ox = ox * cos_r + oz * sin_r
        rot_oz = -ox * sin_r + oz * cos_r
        x -= rot_ox
        y -= oy
        z -= rot_oz

        body_name = _safe_mjcf_name(f"{category}__{aid}__{idx}")
        # MuJoCo body euler is intrinsic XYZ in compiler angle="degree"
        lines.append(
            f'        <body name="{_xml_escape(body_name)}" '
            f'pos="{x:.6f} {y:.6f} {z:.6f}" euler="{rx:.6f} {ry:.6f} {rz:.6f}">'
        )
        for i, g in enumerate(ba.geoms):
            gname = _safe_mjcf_name(f"{body_name}_g{i}")
            attrs = {"name": gname, "type": "mesh"}
            attrs.update(g)
            attrs["contype"] = "0"
            attrs["conaffinity"] = "0"
            lines.append(f'          <geom {_format_attrs(attrs)}/>')
        lines.append('        </body>')
        return True

    for o in scene.get("objects") or []:
        report.n_objects_total += 1
        if _place("obj", o):
            report.n_objects_placed += 1
    for o in scene.get("doors") or []:
        report.n_doors_total += 1
        if _place("door", o):
            report.n_doors_placed += 1
    for o in scene.get("windows") or []:
        report.n_windows_total += 1
        if _place("window", o):
            report.n_windows_placed += 1

    lines.append('      </body>')  # /instances
    lines.append('    </body>')    # /scene_root
    lines.append('  </worldbody>')
    lines.append('</mujoco>')

    scene_xml.write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------- pass 5
    # report
    (out_root / "conversion_report.json").write_text(
        json.dumps(report.__dict__, indent=2)
    )
    LOG.info(
        "wrote %s  (rooms=%d walls=%d objects=%d/%d doors=%d/%d windows=%d/%d)",
        scene_xml,
        report.n_rooms,
        report.n_walls,
        report.n_objects_placed,
        report.n_objects_total,
        report.n_doors_placed,
        report.n_doors_total,
        report.n_windows_placed,
        report.n_windows_total,
    )
    return scene_xml


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-json", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--objathor-dir",
        type=Path,
        default=Path.home() / ".objathor-assets" / "2023_09_23" / "assets",
    )
    p.add_argument(
        "--thor-mjcf-dir",
        type=Path,
        default=None,
        help="Defaults to molmospaces THOR MJCF cache "
        "(ASSETS_DIR/objects/thor from molmo_spaces_constants).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.thor_mjcf_dir is None:
        from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
        args.thor_mjcf_dir = ASSETS_DIR / "objects" / "thor"

    if not args.scene_json.is_file():
        LOG.error("scene JSON not found: %s", args.scene_json)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    convert(
        scene_json=args.scene_json,
        output_dir=args.output_dir,
        objathor_dir=args.objathor_dir,
        thor_mjcf_dir=args.thor_mjcf_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
