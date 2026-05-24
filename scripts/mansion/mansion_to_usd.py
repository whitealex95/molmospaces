"""MansionWorld floor JSON → USDA converter.

Bridges the mansion-side asset caches (objathor + mansion_patch) and the
molmospaces USD library directly into a single ``scene.usda`` that opens
in IsaacSim.

See ``docs/mansion_to_usd.md`` for the full pipeline rationale, scope, and
known gaps. v1 scope: floors (one quad per room polygon), walls (quads),
and objects (objathor UID, patched OBJ, or molmospaces USD-thor reference).
Doors/windows are placed as objects but do not punch holes in walls.

Run in the ``mlspaces-isaac`` conda env (provides ``pxr``)::

    conda activate mlspaces-isaac
    python scripts/mansion/mansion_to_usd.py \\
        --scene-json /path/to/floor_1.json \\
        --output-dir /path/to/usd_export
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

LOG = logging.getLogger("mansion_to_usd")

OBJATHOR_UID_RE = re.compile(r"^[0-9a-f]{32}$")


# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------


@dataclass
class AssetSource:
    """Where a given assetId resolves and which source kind it is."""

    kind: str  # "objathor_pkl" | "objathor_json" | "usd_reference"
    data_path: Path  # .pkl.gz / .json / .usda
    asset_dir: Path  # the parent dir (contains textures)


@dataclass
class BakedAsset:
    """Per-asset bake result with anchor info needed for instance placement.

    Mirrors the MJCF converter's BakedAsset. See ``docs/mansion_to_mjcf.md``
    bug log §A (anchor offset) and §H (yRotOffset) for the rationale."""

    asset_id: str
    usda_path: Path | None = None  # local <id>.usda; None for external usd_reference
    external_ref: Path | None = None  # absolute path to external molmospaces USD asset
    # Anchor offset in mesh-local coords (Unity frame). Objathor meshes have
    # origin at bbox-BOTTOM-center but mansion's `position` is the bbox CENTER;
    # we subtract this offset (rotated by per-instance ry) at placement.
    # THOR USD references already anchor at bbox-center in mansion's convention,
    # so this stays (0, 0, 0).
    anchor_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Asset-intrinsic rotation about up axis (degrees) from objathor's
    # `yRotOffset` field. Mansion's per-instance rotation.y is added to this.
    y_rot_offset_deg: float = 0.0


def locate_asset(
    asset_id: str,
    objathor_dir: Path,
    thor_usd_dir: Path,
    objaverse_usd_dir: Path,
) -> AssetSource | None:
    """Find ``asset_id`` in the available caches; return None if missing."""

    cand = objathor_dir / asset_id
    if cand.is_dir():
        pkl = cand / f"{asset_id}.pkl.gz"
        if pkl.is_file():
            return AssetSource("objathor_pkl", pkl, cand)
        js = cand / f"{asset_id}.json"
        if js.is_file():
            return AssetSource("objathor_json", js, cand)

    # Molmospaces USD library — look for ``<asset_id>.usda`` inside the per-asset dir.
    for usd_root in (thor_usd_dir, objaverse_usd_dir):
        cand = usd_root / asset_id
        if cand.is_dir():
            usda = cand / f"{asset_id}.usda"
            if usda.is_file():
                return AssetSource("usd_reference", usda, cand)
            # Some molmospaces assets ship as ``<id>_mesh.usda``.
            mesh_usda = cand / f"{asset_id}_mesh.usda"
            if mesh_usda.is_file():
                return AssetSource("usd_reference", mesh_usda, cand)

    return None


def load_objathor_dict(src: AssetSource) -> dict[str, Any]:
    if src.kind == "objathor_pkl":
        with gzip.open(src.data_path, "rb") as f:
            return pickle.load(f)
    if src.kind == "objathor_json":
        return json.loads(src.data_path.read_text())
    raise ValueError(f"unsupported objathor source kind: {src.kind}")


# ---------------------------------------------------------------------------
# USD asset baking (objathor / patched → <id>.usda)
# ---------------------------------------------------------------------------


def _arr_xyz(seq: list[dict[str, Any]]) -> np.ndarray:
    """Convert a list of ``{x,y,z}`` dicts to an (N, 3) float32 array."""

    return np.asarray(
        [[float(v["x"]), float(v["y"]), float(v["z"])] for v in seq], dtype=np.float32
    )


def _arr_xy(seq: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(v["x"]), float(v["y"])] for v in seq], dtype=np.float32)


def bake_objathor_asset(
    asset_id: str,
    src: AssetSource,
    out_assets_dir: Path,
) -> BakedAsset:
    """Bake an objathor pkl/json asset into ``<out_assets_dir>/<id>.usda``
    and return a ``BakedAsset`` carrying the anchor offset and yRotOffset.

    Also copies ``albedo.jpg`` (if present) into ``<out_assets_dir>/<id>/``
    so the resulting USDA can be moved without losing textures.
    """

    asset_dict = load_objathor_dict(src)

    verts = _arr_xyz(asset_dict["vertices"])
    tri_indices = np.asarray(asset_dict["triangles"], dtype=np.int32)
    normals = _arr_xyz(asset_dict["normals"]) if asset_dict.get("normals") else None
    uvs = _arr_xy(asset_dict["uvs"]) if asset_dict.get("uvs") else None

    if tri_indices.size % 3 != 0:
        raise ValueError(
            f"{asset_id}: triangle index count {tri_indices.size} is not a multiple of 3"
        )
    n_tris = tri_indices.size // 3
    face_vertex_counts = np.full(n_tris, 3, dtype=np.int32)

    # Bbox center (mesh-local) — used at placement to subtract from instance
    # position so the bbox-center lands at mansion's `position` (see MJCF §A).
    if verts.shape[0] > 0:
        bbox_center = (
            float((verts[:, 0].min() + verts[:, 0].max()) / 2.0),
            float((verts[:, 1].min() + verts[:, 1].max()) / 2.0),
            float((verts[:, 2].min() + verts[:, 2].max()) / 2.0),
        )
    else:
        bbox_center = (0.0, 0.0, 0.0)
    y_rot_offset = float(asset_dict.get("yRotOffset", 0.0))

    asset_usda = out_assets_dir / f"{asset_id}.usda"
    asset_tex_dir = out_assets_dir / asset_id
    asset_tex_dir.mkdir(parents=True, exist_ok=True)

    # Copy texture(s) so the asset is self-contained
    albedo_local: Path | None = None
    for tex_name in ("albedo.jpg",):
        src_tex = src.asset_dir / tex_name
        if src_tex.is_file():
            dst = asset_tex_dir / tex_name
            if not dst.exists():
                shutil.copy2(src_tex, dst)
            albedo_local = dst

    stage = Usd.Stage.CreateNew(str(asset_usda))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)  # objathor meshes are y-up
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_path = Sdf.Path(f"/{_safe_prim_name(asset_id)}")
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, root_path.AppendChild("Mesh"))
    mesh.CreatePointsAttr(verts.tolist())
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts.tolist())
    mesh.CreateFaceVertexIndicesAttr(tri_indices.tolist())
    if normals is not None:
        mesh.CreateNormalsAttr(normals.tolist())
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    if uvs is not None:
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        st = primvars_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        st.Set(uvs.tolist())

    if albedo_local is not None:
        _bind_albedo_material(stage, mesh, root_path, asset_id, albedo_local.name)

    stage.GetRootLayer().Save()
    return BakedAsset(
        asset_id=asset_id,
        usda_path=asset_usda,
        anchor_offset=bbox_center,
        y_rot_offset_deg=y_rot_offset,
    )


def _bind_albedo_material(
    stage: Usd.Stage,
    mesh: UsdGeom.Mesh,
    root_path: Sdf.Path,
    asset_id: str,
    albedo_relpath_inside_dir: str,
) -> None:
    """Attach a simple UsdPreviewSurface + albedo texture to ``mesh``."""

    mat_path = root_path.AppendChild("Material")
    material = UsdShade.Material.Define(stage, mat_path)

    shader_path = mat_path.AppendChild("Surface")
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    tex_path = mat_path.AppendChild("Albedo")
    tex = UsdShade.Shader.Define(stage, tex_path)
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        f"./{asset_id}/{albedo_relpath_inside_dir}"
    )
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    # UV reader feeding the texture's "st" input
    reader_path = mat_path.AppendChild("UVReader")
    reader = UsdShade.Shader.Define(stage, reader_path)
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    UsdShade.MaterialBindingAPI(mesh).Bind(material)


def _safe_prim_name(name: str) -> str:
    """Make a string safe to use as a USD prim name."""

    s = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not s or s[0].isdigit():
        s = "x_" + s
    return s


# ---------------------------------------------------------------------------
# Scene-level geometry: floors and walls
# ---------------------------------------------------------------------------


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
    """Detect a wall's 2D local frame from its 3D polygon.

    Mirrors the MJCF converter (see ``docs/mansion_to_mjcf.md`` §C, §K, §L).
    Bottom edge is found by min-y vertices; ``kind`` selects which corner is
    the local origin. Default ``"window"`` (successor of the other bottom
    vertex in polygon order); ``"door"`` picks the predecessor instead.
    The cleaner segment-based code path bypasses this entirely — this
    helper remains for cutout-mesh-local 2D coords and as fallback.
    """

    ys = [float(p["y"]) for p in polygon]
    min_y = min(ys); max_y = max(ys)
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
    p_start, p_end = polygon[i_start], polygon[i_end]
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
    of axis-aligned sub-rectangles via strip decomposition (same as MJCF §D).
    """

    pieces: list[tuple[float, float, float, float]] = [rect]
    for hx0, hy0, hx1, hy1 in holes:
        next_pieces: list[tuple[float, float, float, float]] = []
        for rx0, ry0, rx1, ry1 in pieces:
            cx0 = max(rx0, hx0); cy0 = max(ry0, hy0)
            cx1 = min(rx1, hx1); cy1 = min(ry1, hy1)
            if cx0 >= cx1 or cy0 >= cy1:
                next_pieces.append((rx0, ry0, rx1, ry1))
                continue
            if cy1 < ry1:
                next_pieces.append((rx0, cy1, rx1, ry1))     # top strip
            if cy0 > ry0:
                next_pieces.append((rx0, ry0, rx1, cy0))     # bottom strip
            if cx0 > rx0:
                next_pieces.append((rx0, cy0, cx0, cy1))     # left strip
            if cx1 < rx1:
                next_pieces.append((cx1, cy0, rx1, cy1))     # right strip
        pieces = next_pieces
    return pieces


def _add_wall_with_holes_mesh(
    stage: Usd.Stage,
    prim_path: Sdf.Path,
    polygon: list[dict[str, float]],
    holes_local: list[tuple[float, float, float, float]],
    color: tuple[float, float, float] = (0.92, 0.92, 0.92),
) -> bool:
    """Add a wall mesh with rectangular cutouts. Returns True on success
    (caller falls back to the full-polygon writer on False)."""

    frame = _wall_local_frame(polygon)
    if frame is None:
        return False
    origin, ux, uy, width, height = frame

    clipped: list[tuple[float, float, float, float]] = []
    for hx0, hy0, hx1, hy1 in holes_local:
        cx0 = max(0.0, hx0); cy0 = max(0.0, hy0)
        cx1 = min(width, hx1); cy1 = min(height, hy1)
        if cx1 - cx0 > 1e-6 and cy1 - cy0 > 1e-6:
            clipped.append((cx0, cy0, cx1, cy1))

    rects = _subtract_holes_from_rect((0.0, 0.0, width, height), clipped)
    if not rects:
        return False
    points: list[tuple[float, float, float]] = []
    face_counts: list[int] = []
    face_indices: list[int] = []
    for rx0, ry0, rx1, ry1 in rects:
        base = len(points)
        for lx, ly in ((rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)):
            points.append((
                origin[0] + lx * ux[0] + ly * uy[0],
                origin[1] + lx * ux[1] + ly * uy[1],
                origin[2] + lx * ux[2] + ly * uy[2],
            ))
        # Two triangles per sub-rect
        face_counts.extend([3, 3])
        face_indices.extend([base + 0, base + 1, base + 2,
                              base + 0, base + 2, base + 3])

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    UsdGeom.Gprim(mesh).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return True


# ---------------------------------------------------------------------------
# Door/window pose: segment-based (primary) + polygon-corner (fallback)
# See docs/mansion_to_mjcf.md §B, §K, §L for the full story.
# ---------------------------------------------------------------------------


def compute_pose_from_segment(
    item: dict[str, Any],
    walls_by_id: dict[str, dict[str, Any]],
) -> tuple[tuple[float, float, float], float] | None:
    """Compute (world_position, world_y_rotation_degrees) from mansion's
    ``doorSegment`` / ``windowSegment`` world-coordinate fields. Primary
    placement path; falls back to :func:`compute_wall_aligned_pose` when
    segment is absent.
    """

    seg = item.get("doorSegment") or item.get("windowSegment")
    if not seg or len(seg) < 2:
        return None
    (x0, z0), (x1, z1) = seg[0], seg[1]
    mx = (float(x0) + float(x1)) / 2.0
    mz = (float(z0) + float(z1)) / 2.0
    ap = item.get("assetPosition") or {"y": 0.0}
    my = float(ap.get("y", 0.0))
    dx = float(x1) - float(x0)
    dz = float(z1) - float(z0)
    theta_deg = math.degrees(math.atan2(-dz, dx))
    return (mx, my, mz), theta_deg


def world_hole_corners_from_segment(
    item: dict[str, Any],
    walls_by_id: dict[str, dict[str, Any]],
) -> list[tuple[float, float, float]] | None:
    """Hole's 4 world corners from segment endpoints + holePolygon y range.
    See docs/mansion_to_mjcf.md §L."""

    seg = item.get("doorSegment") or item.get("windowSegment")
    if not seg or len(seg) < 2:
        return None
    hp = item.get("holePolygon")
    if not hp or len(hp) < 2:
        return None

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
        (float(x0), y_lo, float(z0)),
        (float(x1), y_lo, float(z1)),
        (float(x1), y_hi, float(z1)),
        (float(x0), y_hi, float(z0)),
    ]


def compute_wall_aligned_pose(
    item: dict[str, Any],
    walls_by_id: dict[str, dict[str, Any]],
    kind: str = "door",
) -> tuple[tuple[float, float, float], float] | None:
    """Fallback pose computation from wall polygon + assetPosition (for items
    lacking a doorSegment/windowSegment). Doors and windows use different
    polygon-corner rules — see docs/mansion_to_mjcf.md §L."""

    asset_pos = item.get("assetPosition")
    if asset_pos is None:
        return None
    wid = item.get("wall0") or item.get("wall1")
    wall = walls_by_id.get(wid) if wid else None
    if wall is None:
        return None
    poly = wall.get("polygon") or []
    if len(poly) < 4:
        return None
    frame = _wall_local_frame(poly, kind=kind)
    if frame is None:
        return None
    origin, ux, _uy, _w, _h = frame
    ax = float(asset_pos.get("x", 0.0))
    ay = float(asset_pos.get("y", 0.0))
    az_offset = float(asset_pos.get("z", 0.0))
    world_x = origin[0] + ax * ux[0] + az_offset * (-ux[2])
    world_y = origin[1] + ay
    world_z = origin[2] + ax * ux[2] + az_offset * ux[0]
    theta_deg = math.degrees(math.atan2(-ux[2], ux[0]))
    return (world_x, world_y, world_z), theta_deg


def _triangulate_polygon(
    pts: list[tuple[float, float, float]],
) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple, possibly concave, planar polygon.

    Returns 0-based ``(i, j, k)`` index triples wound in the polygon's vertex
    order. A single concave n-gon face (or a fan) would be triangulated across
    the concave notch by the renderer and fill the convex hull instead of the
    true polygon — wrong for L-shaped rooms.

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


def _add_polygon_mesh(
    stage: Usd.Stage,
    prim_path: Sdf.Path,
    polygon: list[dict[str, float]],
    color: tuple[float, float, float] = (0.6, 0.6, 0.6),
) -> UsdGeom.Mesh:
    """Add a triangulated polygon mesh (concave-safe) at ``prim_path``.

    The polygon points are interpreted as Unity (y-up) coordinates and stored
    verbatim — the world-level ``rotateX=90`` xform handles the up-axis swap.
    """

    pts = [(float(p["x"]), float(p["y"]), float(p["z"])) for p in polygon]
    if len(pts) < 3:
        raise ValueError(f"polygon at {prim_path} has only {len(pts)} points")
    tris = _triangulate_polygon(pts)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([3] * len(tris))
    mesh.CreateFaceVertexIndicesAttr([idx for tri in tris for idx in tri])
    UsdGeom.Gprim(mesh).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return mesh


# ---------------------------------------------------------------------------
# Lights: proceduralParameters.lights -> UsdLux
# ---------------------------------------------------------------------------

# Unity light intensity is unitless / renderer-defined; USD's UsdLux intensity
# is in nits (DistantLight) or candelas (SphereLight) when the renderer
# interprets them physically. These multipliers are empirical — the 1000x for
# directional matches the procthor converter's DEFAULT_DIR_LIGHT_INTENSITY
# (molmo_spaces_isaac/.../assets/utils/lights.py), and 30000x for sphere lights
# is tuned so a ceiling fixture at unity_intensity=0.75 reads as a typical room
# light in IsaacSim. Tweak via the helpers if the result looks dim/blown out.
_DIR_LIGHT_INTENSITY_MULT = 1000.0
_POINT_LIGHT_INTENSITY_MULT = 1000000.0 # 300000.0
# Synthetic ambient dome authored when `proceduralParameters.skyboxId` is set
# (mansion ships the name but no texture). Matches procthor's authored
# DomeLight intensity; run_isaac.py's tame_lights clamps this to --dome-max
# (default 180) at render time -- same end result as procthor scenes.
_DOME_LIGHT_INTENSITY = 1000.0


def _light_rgb(item: dict[str, Any]) -> Gf.Vec3f:
    rgb = item.get("rgb") or {"r": 1.0, "g": 1.0, "b": 1.0}
    return Gf.Vec3f(float(rgb.get("r", 1.0)), float(rgb.get("g", 1.0)), float(rgb.get("b", 1.0)))


def _author_directional_light(
    stage: Usd.Stage,
    parent_path: Sdf.Path,
    idx: int,
    light_dict: dict[str, Any],
) -> None:
    """Author a Unity directional light as ``UsdLux.DistantLight``.

    Position is ignored (parallel rays — only direction matters). Unity Euler
    angles use ZXY local-frame order; we mirror that with ``AddRotateZXYOp``.
    USD distant lights emit along local **-Z** while Unity directionals emit
    along local **+Z**, so we add a ``RotateY 180`` to flip the convention.
    The light lives under ``/World`` so the y<->z reflection (M) applies
    naturally to its world-space direction — same as scene geometry.
    """
    name = _safe_prim_name(f"directional_{idx}")
    path = parent_path.AppendChild(name)
    light = UsdLux.DistantLight.Define(stage, path)

    intensity = float(light_dict.get("intensity", 1.0)) * _DIR_LIGHT_INTENSITY_MULT
    light.CreateIntensityAttr().Set(intensity)
    light.CreateColorAttr().Set(_light_rgb(light_dict))

    rot = light_dict.get("rotation") or {}
    rx = float(rot.get("x", 0.0))
    ry = float(rot.get("y", 0.0))
    rz = float(rot.get("z", 0.0))

    xf = UsdGeom.Xformable(light)
    # Unity ZXY order -> AddRotateZXYOp. (USD applies Rz, then Rx, then Ry to
    # local axes after each prior rotation — matches Unity's eulerAngles.)
    rot_op = xf.AddRotateZXYOp(UsdGeom.XformOp.PrecisionDouble)
    rot_op.Set(Gf.Vec3d(rx, ry, rz))
    # +Z (Unity forward) -> -Z (USD distant-light forward).
    flip_op = xf.AddRotateYOp(UsdGeom.XformOp.PrecisionDouble)
    flip_op.Set(180.0)


def _author_point_light(
    stage: Usd.Stage,
    parent_path: Sdf.Path,
    idx: int,
    light_dict: dict[str, Any],
) -> None:
    """Author a Unity point light as ``UsdLux.SphereLight``.

    Position is in Unity coords; ``/World``'s y<->z reflection handles the
    swap to USD's z-up frame. Unity's ``range`` (hard distance cutoff) has no
    direct USD equivalent — USD uses physical inverse-square falloff. We set a
    small visible ``radius`` (purely cosmetic in the viewport).
    """
    raw_id = str(light_dict.get("id", f"point_{idx}"))
    name = _safe_prim_name(f"point_{idx}__{raw_id}")
    path = parent_path.AppendChild(name)
    light = UsdLux.SphereLight.Define(stage, path)

    pos = light_dict.get("position") or {"x": 0.0, "y": 0.0, "z": 0.0}
    xf = UsdGeom.Xformable(light)
    tx = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    tx.Set(Gf.Vec3d(float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))))

    intensity = float(light_dict.get("intensity", 1.0)) * _POINT_LIGHT_INTENSITY_MULT
    light.CreateIntensityAttr().Set(intensity)
    light.CreateColorAttr().Set(_light_rgb(light_dict))
    light.CreateRadiusAttr().Set(0.1)


def _author_ambient_dome(
    stage: Usd.Stage,
    parent_path: Sdf.Path,
    skybox_id: str,
    skybox_textures_dir: Path | None,
    out_root: Path,
) -> str | None:
    """Author a ``UsdLux.DomeLight`` for ambient + sky IBL.

    If ``<skybox_textures_dir>/<skybox_id>.png`` exists, copy it into
    ``<out_root>/Textures/`` and wire it onto the dome as a textured IBL --
    matching the procthor convention (see ``~/.molmospaces/usd/scenes/
    procthor-10k-val/<version>/val_*/Payload/Contents.usda``: ``DomeLight
    "scene_skybox_light"`` with ``inputs:texture:file =
    @./Textures/SkyAlbany.png@``). A textured dome provides directional
    HDR sky illumination -- "daylight pours in through windows" -- which is
    why procthor's renders look brighter than ours did with a neutral-white
    dome at the same intensity.

    Falls back to a neutral-white untextured dome if no matching PNG is found.
    Returns the copied texture filename, or ``None`` if untextured.
    """
    path = parent_path.AppendChild("ambient_dome")
    light = UsdLux.DomeLight.Define(stage, path)
    light.CreateIntensityAttr().Set(_DOME_LIGHT_INTENSITY)

    tex_src: Path | None = None
    if skybox_textures_dir is not None:
        cand = skybox_textures_dir / f"{skybox_id}.png"
        if cand.is_file():
            tex_src = cand
        else:
            LOG.warning(
                "dome texture for skyboxId=%s not found in %s; "
                "falling back to neutral-white dome",
                skybox_id, skybox_textures_dir,
            )

    if tex_src is None:
        light.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        return None

    out_textures = out_root / "Textures"
    out_textures.mkdir(parents=True, exist_ok=True)
    dst = out_textures / tex_src.name
    if not dst.exists():
        shutil.copy2(tex_src, dst)
    light.CreateTextureFileAttr().Set(Sdf.AssetPath(f"./Textures/{tex_src.name}"))
    light.CreateTextureFormatAttr().Set("automatic")
    return tex_src.name


def create_lights(
    stage: Usd.Stage,
    scene: dict[str, Any],
    report: ConversionReport,
    skybox_textures_dir: Path | None = None,
    out_root: Path | None = None,
) -> None:
    """Author ``proceduralParameters.lights`` into ``/World/Lights``.

    Reads Unity-style light dicts (directional / point) and emits
    ``UsdLux.DistantLight`` / ``UsdLux.SphereLight`` respectively. Spot lights
    are warned and skipped (no instances seen in audited mansion floors;
    extend ``_author_spot_light`` when one appears).

    Also authors a ``UsdLux.DomeLight`` when ``proceduralParameters.skyboxId``
    is set; the dome is textured if a matching ``<skyboxId>.png`` is found
    under ``skybox_textures_dir`` (see :func:`_author_ambient_dome`).
    """
    pp = scene.get("proceduralParameters") or {}
    lights_in = pp.get("lights") or []
    report.n_lights_total = len(lights_in)
    skybox_id = pp.get("skyboxId")
    if not lights_in and not skybox_id:
        return

    lights_root = UsdGeom.Xform.Define(stage, "/World/Lights")
    parent_path = lights_root.GetPath()

    for i, L in enumerate(lights_in):
        typ = (L.get("type") or "").lower()
        try:
            if typ == "directional":
                _author_directional_light(stage, parent_path, i, L)
            elif typ == "point":
                _author_point_light(stage, parent_path, i, L)
            else:
                LOG.warning("light %d: unsupported type %r, skipping", i, typ)
                continue
            report.n_lights_authored += 1
        except Exception as exc:
            LOG.warning("light %d (%s): failed to author (%s)", i, L.get("id", "?"), exc)

    if skybox_id and out_root is not None:
        tex = _author_ambient_dome(
            stage, parent_path, skybox_id, skybox_textures_dir, out_root
        )
        report.has_ambient_dome = True
        report.dome_texture = tex
        LOG.info(
            "ambient dome authored (skyboxId=%s, texture=%s)",
            skybox_id, tex or "<neutral-white fallback>",
        )

    LOG.info(
        "lights authored: %d/%d  ambient_dome=%s  dome_texture=%s",
        report.n_lights_authored,
        report.n_lights_total,
        report.has_ambient_dome,
        report.dome_texture or "-",
    )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


@dataclass
class ConversionReport:
    scene_json: str
    output_dir: str
    n_objects_total: int = 0
    n_objects_placed: int = 0
    n_objects_skipped: int = 0
    n_doors_total: int = 0
    n_doors_placed: int = 0
    n_windows_total: int = 0
    n_windows_placed: int = 0
    n_rooms: int = 0
    n_walls: int = 0
    n_ceilings: int = 0
    n_lights_total: int = 0
    n_lights_authored: int = 0
    has_ambient_dome: bool = False
    dome_texture: str | None = None
    unique_assetids: int = 0
    baked_assets: int = 0
    referenced_assets: int = 0
    missing_assetids: list[str] = field(default_factory=list)
    skipped_objects: list[dict[str, str]] = field(default_factory=list)


def convert(
    scene_json: Path,
    output_dir: Path,
    objathor_dir: Path,
    thor_usd_dir: Path,
    objaverse_usd_dir: Path,
    skybox_textures_dir: Path | None = None,
) -> Path:
    scene = json.loads(scene_json.read_text())

    out_root = output_dir / scene_json.parent.name.replace("#", "_") / scene_json.stem
    out_assets = out_root / "assets"
    out_assets.mkdir(parents=True, exist_ok=True)
    LOG.info("output directory: %s", out_root)

    # ------------------------------------------------------------------- pass 1
    # collect unique assetIds, resolve them to a source
    asset_sources: dict[str, AssetSource] = {}
    missing: list[str] = []
    for src in ("objects", "doors", "windows"):
        for o in scene.get(src) or []:
            aid = o.get("assetId")
            if not aid or aid in asset_sources or aid in missing:
                continue
            located = locate_asset(aid, objathor_dir, thor_usd_dir, objaverse_usd_dir)
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
    # bake objathor / patched assets and capture per-asset anchor info
    baked: dict[str, BakedAsset] = {}  # asset_id -> BakedAsset (with anchor_offset, y_rot_offset)
    for aid, src in asset_sources.items():
        if src.kind in ("objathor_pkl", "objathor_json"):
            try:
                baked[aid] = bake_objathor_asset(aid, src, out_assets)
                LOG.info(
                    "baked %s -> %s  (anchor=%s, yRotOffset=%.2f°)",
                    aid, baked[aid].usda_path.name,
                    tuple(round(v, 3) for v in baked[aid].anchor_offset),
                    baked[aid].y_rot_offset_deg,
                )
            except Exception as exc:
                LOG.exception("failed to bake %s: %s", aid, exc)
                missing.append(aid)
        elif src.kind == "usd_reference":
            baked[aid] = BakedAsset(
                asset_id=aid,
                external_ref=src.data_path,
                # External USD assets from molmospaces already anchor at
                # bbox-center in mansion's convention — no offset needed.
            )
            LOG.info("reference %s -> %s", aid, src.data_path)

    # ------------------------------------------------------------------- pass 3
    # build scene.usda (ASCII, debug-friendly)
    scene_usda = out_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_usda))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    # Convert Unity (left-handed, y-up) -> USD (right-handed, z-up) with the
    # y<->z swap M: (x,y,z) -> (x,z,y). A swap is a reflection (det -1), which
    # is *required* to fix handedness -- the old RotateX(90) is a pure rotation,
    # so it fixed only the up-axis and left the whole scene mirrored. USD
    # xformOp:transform accepts a reflection matrix directly, so a single root
    # op converts everything; geometry inside /World stays in raw Unity coords.
    xform_op = world.AddXformOp(
        UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble
    )
    xform_op.Set(Gf.Matrix4d(1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1))

    report = ConversionReport(
        scene_json=str(scene_json),
        output_dir=str(out_root),
        unique_assetids=len(asset_sources) + len(missing),
        baked_assets=sum(1 for ba in baked.values() if ba.usda_path is not None),
        referenced_assets=sum(1 for ba in baked.values() if ba.external_ref is not None),
        missing_assetids=missing,
    )

    # Pre-index walls by id (used for hole projection + door/window placement).
    walls_by_id: dict[str, dict[str, Any]] = {
        w["id"]: w for w in (scene.get("walls") or []) if w.get("id")
    }

    # Build holes_by_wall: project each door/window hole through WORLD space
    # so it lands at the same world position regardless of wall1's polygon
    # ordering. See docs/mansion_to_mjcf.md §F.
    def _project_world_to_local(world_pt, frame):
        origin, ux, uy, _w, _h = frame
        dx = world_pt[0] - origin[0]; dy = world_pt[1] - origin[1]; dz = world_pt[2] - origin[2]
        return (dx * ux[0] + dy * ux[1] + dz * ux[2],
                dx * uy[0] + dy * uy[1] + dz * uy[2])

    holes_by_wall: dict[str, list[tuple[float, float, float, float]]] = {}
    for src_key in ("doors", "windows"):
        kind = "door" if src_key == "doors" else "window"
        for item in scene.get(src_key) or []:
            # Prefer world-coordinate corners from the segment.
            corners_world = world_hole_corners_from_segment(item, walls_by_id)
            if corners_world is None:
                hp = item.get("holePolygon") or []
                if len(hp) < 2:
                    continue
                wid0 = item.get("wall0")
                wall0 = walls_by_id.get(wid0) if wid0 else None
                if wall0 is None:
                    continue
                frame0 = _wall_local_frame(wall0.get("polygon") or [], kind=kind)
                if frame0 is None:
                    continue
                o0, ux0, uy0, _w0, _h0 = frame0
                xs = [float(p["x"]) for p in hp]; ys = [float(p["y"]) for p in hp]
                lo_x, lo_y, hi_x, hi_y = min(xs), min(ys), max(xs), max(ys)
                corners_world = [
                    (o0[0] + lx * ux0[0] + ly * uy0[0],
                     o0[1] + lx * ux0[1] + ly * uy0[1],
                     o0[2] + lx * ux0[2] + ly * uy0[2])
                    for lx, ly in ((lo_x, lo_y), (hi_x, lo_y), (hi_x, hi_y), (lo_x, hi_y))
                ]
            for wid_key in ("wall0", "wall1"):
                wid = item.get(wid_key)
                if not wid:
                    continue
                wall = walls_by_id.get(wid)
                if wall is None:
                    continue
                frame = _wall_local_frame(wall.get("polygon") or [])
                if frame is None:
                    continue
                lxs, lys = zip(*(_project_world_to_local(p, frame) for p in corners_world))
                holes_by_wall.setdefault(wid, []).append(
                    (min(lxs), min(lys), max(lxs), max(lys))
                )

    # rooms (floor quads)
    rooms_root = UsdGeom.Xform.Define(stage, "/World/Rooms")
    for room in scene.get("rooms") or []:
        rid = _safe_prim_name(room.get("id", "room"))
        poly = room.get("floorPolygon") or []
        if len(poly) < 3:
            continue
        try:
            _add_polygon_mesh(
                stage, rooms_root.GetPath().AppendChild(rid), poly, color=(0.85, 0.82, 0.78)
            )
            report.n_rooms += 1
        except Exception as exc:
            LOG.warning("room %s skipped: %s", rid, exc)

    # walls (with cutouts at door/window holes; see docs/mansion_to_mjcf.md §D, §F)
    walls_root = UsdGeom.Xform.Define(stage, "/World/Walls")
    walls_cut = 0
    for i, wall in enumerate(scene.get("walls") or []):
        wid_raw = wall.get("id", f"wall_{i}")
        wid = _safe_prim_name(wid_raw)
        poly = wall.get("polygon") or []
        if len(poly) < 3:
            continue
        try:
            holes = holes_by_wall.get(wid_raw, [])
            ok = False
            if holes:
                ok = _add_wall_with_holes_mesh(
                    stage, walls_root.GetPath().AppendChild(wid), poly, holes,
                    color=(0.92, 0.92, 0.92),
                )
                if ok:
                    walls_cut += 1
            if not ok:
                _add_polygon_mesh(
                    stage, walls_root.GetPath().AppendChild(wid), poly,
                    color=(0.92, 0.92, 0.92),
                )
            report.n_walls += 1
        except Exception as exc:
            LOG.warning("wall %s skipped: %s", wid, exc)
    LOG.info("walls authored: %d (%d with cutouts)", report.n_walls, walls_cut)

    # ceilings: per-room floor polygons raised to the wall-top y so the
    # rooms are enclosed. Mirrors procthor's per-room `ceiling_<roomid>_visual_0`
    # convention. Set doubleSided so the underside is visible from inside.
    ceil_y = max(
        (float(p["y"]) for w in (scene.get("walls") or []) for p in (w.get("polygon") or [])),
        default=0.0,
    )
    if ceil_y > 0.0:
        ceilings_root = UsdGeom.Xform.Define(stage, "/World/Ceilings")
        for room in scene.get("rooms") or []:
            rid = _safe_prim_name(room.get("id", "room"))
            poly = room.get("floorPolygon") or []
            if len(poly) < 3:
                continue
            raised = [{"x": p["x"], "y": ceil_y, "z": p["z"]} for p in poly]
            cid = _safe_prim_name(f"ceiling_{room.get('id', 'room')}")
            try:
                mesh = _add_polygon_mesh(
                    stage, ceilings_root.GetPath().AppendChild(cid), raised,
                    color=(0.95, 0.95, 0.95),
                )
                mesh.CreateDoubleSidedAttr(True)
                report.n_ceilings += 1
            except Exception as exc:
                LOG.warning("ceiling %s skipped: %s", cid, exc)
        LOG.info("ceilings authored: %d (y=%.3f)", report.n_ceilings, ceil_y)

    # lights from proceduralParameters.lights
    create_lights(
        stage, scene, report,
        skybox_textures_dir=skybox_textures_dir,
        out_root=out_root,
    )

    # objects + doors + windows
    inst_root = UsdGeom.Xform.Define(stage, "/World/Instances")
    instance_counts: dict[str, int] = {}

    def _place(category: str, item: dict[str, Any]) -> bool:
        aid = item.get("assetId")
        if not aid:
            return False
        ba = baked.get(aid)
        if ba is None:
            report.skipped_objects.append(
                {"id": item.get("id", ""), "assetId": aid, "reason": "asset missing"}
            )
            return False

        rx_val = 0.0; rz_val = 0.0
        if category in ("door", "window"):
            # Doors / windows have no top-level `position` — use mansion's
            # world-coord segment (primary) or wall + assetPosition fallback.
            # See docs/mansion_to_mjcf.md §B, §L.
            wp = compute_pose_from_segment(item, walls_by_id)
            if wp is None:
                wp = compute_wall_aligned_pose(item, walls_by_id, kind=category)
            if wp is None:
                report.skipped_objects.append(
                    {"id": item.get("id", ""), "assetId": aid,
                     "reason": "could not resolve wall for door/window"}
                )
                return False
            (x, y, z), ry_val = wp
        else:
            pos = item.get("position") or {"x": 0, "y": 0, "z": 0}
            rot = item.get("rotation") or {"x": 0, "y": 0, "z": 0}
            x = float(pos["x"]); y = float(pos["y"]); z = float(pos["z"])
            rx_val = float(rot.get("x", 0.0))
            ry_val = float(rot.get("y", 0.0))
            rz_val = float(rot.get("z", 0.0))

        # Add the asset's intrinsic yRotOffset (docs/mansion_to_mjcf.md §H).
        ry_val += ba.y_rot_offset_deg

        # Subtract the bbox-center anchor offset, rotated by the FINAL ry
        # so rotated bodies don't drift in xz (docs/mansion_to_mjcf.md §A).
        ox, oy, oz = ba.anchor_offset
        cos_r = math.cos(math.radians(ry_val))
        sin_r = math.sin(math.radians(ry_val))
        rot_ox = ox * cos_r + oz * sin_r
        rot_oz = -ox * sin_r + oz * cos_r
        x -= rot_ox; y -= oy; z -= rot_oz

        idx = instance_counts.get(aid, 0)
        instance_counts[aid] = idx + 1
        inst_name = _safe_prim_name(f"{category}__{aid}__{idx}")
        xf_path = inst_root.GetPath().AppendChild(inst_name)
        xf = UsdGeom.Xform.Define(stage, xf_path)

        tx = xf.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble)
        tx.Set(Gf.Vec3d(x, y, z))
        ry_op = xf.AddXformOp(UsdGeom.XformOp.TypeRotateY, UsdGeom.XformOp.PrecisionDouble)
        ry_op.Set(ry_val)
        if abs(rx_val) > 1e-9:
            rx_op = xf.AddXformOp(UsdGeom.XformOp.TypeRotateX, UsdGeom.XformOp.PrecisionDouble)
            rx_op.Set(rx_val)
        if abs(rz_val) > 1e-9:
            rz_op = xf.AddXformOp(UsdGeom.XformOp.TypeRotateZ, UsdGeom.XformOp.PrecisionDouble)
            rz_op.Set(rz_val)

        ref_path = f"./assets/{aid}.usda" if ba.usda_path is not None else str(ba.external_ref)
        xf.GetPrim().GetReferences().AddReference(ref_path)
        return True

    for o in scene.get("objects") or []:
        report.n_objects_total += 1
        if _place("obj", o):
            report.n_objects_placed += 1
        else:
            report.n_objects_skipped += 1
    for o in scene.get("doors") or []:
        report.n_doors_total += 1
        if _place("door", o):
            report.n_doors_placed += 1
    for o in scene.get("windows") or []:
        report.n_windows_total += 1
        if _place("window", o):
            report.n_windows_placed += 1

    stage.GetRootLayer().Save()

    # ------------------------------------------------------------------- pass 4
    # report
    (out_root / "conversion_report.json").write_text(
        json.dumps(report.__dict__, indent=2)
    )
    LOG.info(
        "wrote %s  (rooms=%d walls=%d objects=%d/%d doors=%d/%d windows=%d/%d lights=%d/%d)",
        scene_usda,
        report.n_rooms,
        report.n_walls,
        report.n_objects_placed,
        report.n_objects_total,
        report.n_doors_placed,
        report.n_doors_total,
        report.n_windows_placed,
        report.n_windows_total,
        report.n_lights_authored,
        report.n_lights_total,
    )
    return scene_usda


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
        "--thor-usd-dir",
        type=Path,
        default=Path("/home/jkim3662/Projects/molmospaces/assets/usd/objects/thor"),
    )
    p.add_argument(
        "--objaverse-usd-dir",
        type=Path,
        default=Path("/home/jkim3662/Projects/molmospaces/assets/usd/objects/objaverse"),
    )
    p.add_argument(
        "--skybox-textures-dir",
        type=Path,
        default=Path.home() / ".molmospaces" / "usd" / "scenes" / "procthor-10k-val"
        / "20260128" / "val_1_ceiling" / "Payload" / "Textures",
        help="dir to search for `<skyboxId>.png` to use as DomeLight IBL "
        "(matches procthor's textured-dome lighting). Falls back to neutral-white "
        "dome if the file is missing.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.scene_json.is_file():
        LOG.error("scene JSON not found: %s", args.scene_json)
        return 1
    if not args.objathor_dir.is_dir():
        LOG.warning(
            "objathor dir not found (%s) — objathor-backed assets will be reported missing",
            args.objathor_dir,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    convert(
        scene_json=args.scene_json,
        output_dir=args.output_dir,
        objathor_dir=args.objathor_dir,
        thor_usd_dir=args.thor_usd_dir,
        objaverse_usd_dir=args.objaverse_usd_dir,
        skybox_textures_dir=args.skybox_textures_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
