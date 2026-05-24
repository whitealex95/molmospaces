# Mansion → MJCF conversion

How to convert a single floor JSON from the **MansionWorld** dataset
(`~/Projects/mansion/mansionworld/<building>/floor_<n>.json`) into an
**MJCF** scene that opens in MuJoCo.

> **When working in this repo:** read this doc before re-deriving the
> mansion-to-MJCF pipeline. Sibling of `docs/mansion_to_usd.md`; same source
> data, same coverage (85/85 unique assetIds), different output target.

---

## 1. What this conversion does

Bridges two source caches into a MuJoCo-loadable `scene.xml`:

| Source | Used for | How it's baked |
|---|---|---|
| `~/.objathor-assets/2023_09_23/assets/<uid>/<uid>.pkl.gz` | 65 objathor furniture meshes | Read vertices/triangles/normals/UVs from the pickle, write `<uid>.obj` + `albedo.png` |
| `~/.objathor-assets/2023_09_23/assets/<name>/<name>.json` | 2 mansion_patch assets (`small_stair`, `toilet-suite`) | Same schema as `.pkl.gz`, baked identically |
| `${MLSPACES_ASSETS_DIR}/objects/thor/**/<id>.xml` | 18 named THOR doors / windows / objects (e.g. `Doorway_2`, `Tissue_Box_1`, `Cloth_12`) | Parse the molmospaces THOR MJCF, copy its meshes/textures, emit prefixed asset entries + per-geom visual bindings |

Result for `public_healthcare_3f_300_fp001#0/floor_1.json`: **100 %
coverage** — 85 / 85 unique assetIds resolve, 0 skipped.

### Why a separate path from the USD converter

MuJoCo doesn't load USD natively. The USD converter (`mansion_to_usd.py`)
uses `pxr` to write a USD stage; this converter writes MJCF XML and OBJ /
PNG asset files. Both read the same upstream caches (`.pkl.gz` and
molmospaces THOR), so coverage is identical.

The molmospaces objaverse MJCF cache (`${MLSPACES_ASSETS_DIR}/objects/objaverse/`)
turned out to be **metadata-only on disk** — a 129 647-entry lazy-download
manifest with zero mesh files extracted. Even with all archives downloaded
in bulk, only 23 / 65 of mansion's objathor UIDs are in molmospaces's
curated subset. So going through molmospaces's housegen + MJCF assets path
caps at ~48 % coverage. The objathor `.pkl.gz` → `.obj` direct bake gets us
the missing 42 UIDs.

---

## 2. Prerequisites

Same as `docs/mansion_to_usd.md` Section 2:

1. Mansion repo set up with objathor assets + `setup_mansion.py` patch.
2. Molmospaces THOR MJCF library populated. Triggered automatically by:
   ```bash
   python -m molmo_spaces.molmo_spaces_constants
   ```
   from the `mlspaces` (or `mlspaces-mujoco`) env. The script downloads
   `${MLSPACES_ASSETS_DIR}/objects/thor/**/*.xml` (~2.6 GB).

### Conda env

Run the converter from either `mlspaces` or `mlspaces-mujoco`. Loading and
viewing the result must happen in **`mlspaces-mujoco`** (the Filament wheel
in `mlspaces` is missing the classic OpenGL UI symbols, see
`CLAUDE.md` § Visualizing a scene).

```bash
conda activate mlspaces-mujoco
```

`mansion_to_mjcf.py` depends only on `mujoco`, `Pillow`, and the standard
library — no `pxr` / `usdex` required.

---

## 3. Quick start

```bash
conda activate mlspaces-mujoco
cd ~/Projects/molmospaces

python scripts/mansion/mansion_to_mjcf.py \
    --scene-json /home/jkim3662/Projects/mansion/mansionworld/public_healthcare_3f_300_fp001#0/floor_1.json \
    --output-dir /home/jkim3662/Projects/mansion/mjcf_export
```

Outputs:

```
~/Projects/mansion/mjcf_export/<scene_dir>/<floor_stem>/
├── scene.xml                 # the MJCF to open in MuJoCo
├── assets/
│   ├── <safe_id>/<safe_id>.obj   # per-asset baked mesh (objathor / patched)
│   ├── <safe_id>/albedo.png      # PNG-converted albedo texture
│   ├── <safe_id>/<rel>.obj       # for THOR-MJCF: original mesh files copied
│   ├── rooms/<room_id>.obj       # per-room floor polygon
│   └── walls/<wall_id>.obj       # per-wall polygon
└── conversion_report.json
```

Open in the MuJoCo viewer:

```bash
python -m mujoco.viewer --mjcf ~/Projects/mansion/mjcf_export/.../floor_1/scene.xml
```

Or programmatically:

```python
import mujoco as mj
model = mj.MjModel.from_xml_path(".../scene.xml")
data = mj.MjData(model)
mj.mj_forward(model, data)
```

---

## 4. Observed output (this user, `floor_1.json`)

```
$ du -sh .../floor_1/
107M    .../floor_1/

$ cat .../floor_1/conversion_report.json
{
  "n_objects_total": 144, "n_objects_placed": 144,
  "n_doors_total": 7,     "n_doors_placed": 7,
  "n_windows_total": 13,  "n_windows_placed": 13,
  "n_rooms": 7, "n_walls": 82, "n_ceilings": 7,
  "unique_assetids": 85,
  "baked_objathor": 67, "baked_thor_mjcf": 18,
  "missing_assetids": [], "skipped_objects": []
}
```

After loading in MuJoCo 3.5.0:

```
bodies:  169
geoms:   342
meshes:  222
textures: 89
materials: 111
body xpos bbox: min=[ 0.0, -15.95,  0.0 ]  max=[ 18.96,  0.0,  2.60 ]
```

A test render confirmed walls, floor, doorways, and furniture are all
visible and correctly positioned (reception area with red benches, rooms
populated with their objathor furniture).

---

## 5. What v1 of the converter does

| Feature | v1 status | Notes |
|---|---|---|
| Floor per room | ✅ | Fan-triangulated polygon mesh, `inertia="shell"` |
| Ceiling per room | ✅ | Each room's `floorPolygon` re-emitted at `y = max(wall.polygon.y)` as `ceilings/<roomid>.obj` and `ceiling_<roomid>` geom; same triangulation + `inertia="shell"` as the floor |
| Walls | ✅ | Fan-triangulated polygon mesh, `inertia="shell"` |
| Objects (objathor `.pkl.gz` / patched `.json`) | ✅ | `<id>.obj` with v/vt/vn/f + PNG albedo. Anchor offset applied to land bbox on the floor (objathor mesh origin is bbox-bottom-center while mansion `position.y` is bbox-center). |
| Objects (THOR-MJCF) | ✅ | Mesh files copied as-is, prefixed names, visual geoms only |
| Doors | ⚠️ static | Placed at the correct world pose computed from `wall0` polygon + `assetPosition` in wall-local frame. **Not articulated** (no hinges). Wall mesh is cut at the door's `holePolygon` (visible doorway). |
| Windows | ⚠️ static | Same as doors — placed via wall + assetPosition, with wall cutout at the window's `holePolygon`. Wall polygon vertex order is inconsistent across the dataset (interior vs exterior walls use different orderings) — the converter detects the bottom edge by min-y. |
| Wall cutouts at doors / windows | ✅ | For each wall, hole rectangles from any door/window with `wall0` or `wall1` = wall.id are subtracted from the wall mesh via "rectangle minus rectangles" strip decomposition, producing up to 4 sub-rectangles per hole that get triangulated and emitted. Wall1 side may be mirrored — see bug log §B. |
| THOR-MJCF asset body transforms | ✅ | Each visual mesh's cumulative `(pos, quat)` is composed from its chain of parent `<body>` elements in the source XML and emitted as `<geom pos quat ...>` inside the instance body. This is required because mansion's `assetPosition` is the bbox-CENTER while the molmospaces THOR XMLs use inner bodies with non-trivial offsets to position the mesh's bbox-center at the asset root. |
| Lights | ❌ | Single fixed `<light>` from above; no per-scene lights from `proceduralParameters` |
| Skybox | ❌ | Ignored |
| Materials beyond albedo | ⚠️ | Albedo only for objathor; for THOR-MJCF whatever the source XML had — but `normal`/`emission`/`metallic_smoothness` from objathor are skipped |
| Collisions / physics | ❌ | Every geom has `contype=0 conaffinity=0` (visual only). No `<inertial>` overrides; no `<contact>` excludes |
| Joints (e.g. door hinges) | ❌ | THOR-MJCF source has joints; we drop them in v1 |

The script writes every skipped asset to `conversion_report.json`.

### Coordinate convention

Mansion JSON uses **Unity** (left-handed, `+y` up); MuJoCo is **right-handed,
`+z` up**. The conversion is the y↔z swap

    M: (x, y, z) -> (x, z, y)

which matches procthor's `unity_to_mj_pos`. M is a **reflection** (det -1),
and a reflection is *required*: converting left-handed Unity to right-handed
MuJoCo cannot be done with a rotation — a pure rotation fixes only the
up-axis and leaves the whole scene mirrored (see bug log §M).

A MuJoCo geom/body transform cannot encode a reflection, so M is baked in
decomposed form, `M = ROT90X ∘ DIAG` with `DIAG = diag(1, 1, -1)`:

- Every `<mesh>` gets `scale="1 1 -1"` (the DIAG half — MuJoCo reflects the
  mesh and its normals natively for a negative-scale axis).
- Room / wall geoms keep raw Unity world-coord verts in their `.obj`; the
  `rooms` and `walls` bodies carry `euler="90 0 0"` (the ROT90X half), and
  `ROT90X ∘ DIAG = M`.
- Placed objects: the body `pos` / `quat` are baked via `unity_to_mj_pos`
  and `unity_to_mj_quat` (`R_mj = M @ R_unity @ DIAG`).
- THOR-asset per-geom bindings are reflected by DIAG-conjugation
  (`reflect_nested_pos` / `reflect_nested_quat`).

`scene_root` is now just a plain grouping container — there is no
scene-wide rotation node.

### Anchor offset for objathor / patched assets

Objathor `.pkl.gz` meshes have their vertices stored with the local origin
at the **bbox-bottom-center** (y vertices range over `[0, height]`), while
mansion's JSON `position.y` is the **bbox-center** height in world coords.
The converter computes `bbox_center` at bake time and stores it in
`BakedAsset.anchor_offset`; at placement we subtract that offset from the
body position so the bbox lands on the floor. x/z are typically near-zero
because meshes are symmetric, but we apply them anyway (rotated by the
instance's y-rotation first, so rotated bodies don't drift in xz).

### Door / window pose from wall + assetPosition

Doors and windows have no top-level `position`/`rotation` in mansion JSON;
only `assetPosition` in the wall's local frame and `wall0`/`wall1` IDs.

The converter:
1. Looks up `wall0` (preferred) in the walls table.
2. Finds the wall's bottom edge: the two vertices with `y == min(polygon.y)`.
   They are always adjacent in the polygon's traversal order; we pick the
   one whose `(i + 1) % n` neighbor is the other bottom vertex as the start.
3. World position = `start + assetPosition.x · bottom_edge_dir + assetPosition.y · up`.
4. Y-rotation (Unity +y / MuJoCo +z) = `atan2(-dz, dx)` of the bottom-edge
   direction, so the door's local +x aligns with the wall.

This handles the inconsistent wall polygon ordering in the dataset
(interior walls use `[TL, TR, BR, BL]`, exterior walls use `[BL, TL, TR, BR]`).

---

## 6. Reproducing from scratch on a different scene

1. **Pick a scene JSON**:
   ```bash
   ls ~/Projects/mansion/mansionworld | head
   ```
2. **Audit coverage** (same script works for MJCF or USD path):
   ```bash
   PYTHONPATH=. python scripts/mansion/audit_assetids.py \
       --scene-json ~/Projects/mansion/mansionworld/<building>/floor_<n>.json
   ```
   If `missing_assetids` is non-empty, the converter's report will show
   exactly what was skipped.

3. **Convert**:
   ```bash
   conda activate mlspaces-mujoco
   python scripts/mansion/mansion_to_mjcf.py \
       --scene-json ~/Projects/mansion/mansionworld/<building>/floor_<n>.json \
       --output-dir ~/Projects/mansion/mjcf_export
   ```

4. **View**:
   ```bash
   python -m mujoco.viewer --mjcf .../floor_<n>/scene.xml
   ```

---

## 7. Known issues / TODO

- ~~**`wall1`-side cutout mirroring.**~~ **FIXED in bug log §F** — holes are
  now projected through world space and re-projected into each wall's own
  local frame, which handles mirrored bottom-edge directions automatically.
- ~~**Objects clipping through walls due to placer not respecting depth.**~~
  **FIXED in bug log §H.** The protrusion wasn't a mansion placer issue —
  the L-shape bench was rotated 45° wrong because the converter ignored the
  asset's `yRotOffset` field. After applying it, the bench fits inside the
  room. Other previously-suspicious assets (12 objathor UIDs with non-zero
  `yRotOffset`) are now also correctly oriented.
- **No articulated doors / handles.** THOR-MJCF source has body+joint+door
  chains (e.g. `Doorway_2` has a hinge joint and a handle joint). v1
  emits the body-tree transforms but strips joints, producing static
  visual geoms. Re-introduce joints by emitting `<body>`-with-`<joint>`
  blocks where the source had them.
- **No collisions.** All geoms have `contype=0 conaffinity=0`. For robot
  navigation tasks, switch walls and floor to `contype=1 conaffinity=1` and
  set a coarser collider mesh (or `<geom type="box">` from wall extents).
- **No physics inertias for movable objects.** `kinematic=true` mansion
  objects could become movable bodies by adding `<freejoint/>` and
  removing the mesh `inertia="shell"` (require watertight collider mesh).
- **`.jpg` → `.png` is lossless re-encode.** Costs ~5–10 s extra wall time
  on first bake; subsequent runs reuse the cached PNGs (skip-if-exists).
- **Wall + floor z-fighting.** Walls and floor live at `y=0` (Unity), so
  their bases coincide and z-fight in the viewer. Either thicken walls
  into boxes or lift them by a few mm.

---

## 8. Bug log

A running record of issues hit while iterating on this converter. Future
sessions: read this BEFORE debugging similar-looking problems.

### M. Whole scene mirrored — rotation used where a reflection was needed

**Symptom:** mansion MJCF (and USD) scenes were left-right mirror images of
the source layout. Internally consistent, but flipped relative to procthor
scenes and to reality — easy to miss without an asymmetric reference.

**Cause:** Unity is left-handed; MuJoCo and z-up USD are right-handed.
Converting left- to right-handed *requires* an orientation-reversing
transform (a reflection, det -1). Both converters used a pure **rotation** —
MJCF `scene_root euler="90 0 0"`, USD `/World` `RotateX 90` — which fixes the
up-axis but preserves handedness, so every scene came out mirrored. A body or
xform rotation node can never fix this: rotations are det +1 by definition.

**Fix:** bake the y↔z swap `M: (x,y,z) -> (x,z,y)` (a reflection, matching
procthor's `unity_to_mj_pos`) into the geometry. MJCF: `M = ROT90X ∘
diag(1,1,-1)` — `diag` on every `<mesh scale>`, `ROT90X` on the `rooms` /
`walls` bodies, object and THOR placements baked via `unity_to_mj_pos` /
`unity_to_mj_quat` / `reflect_nested_*`; the `scene_root euler` is removed.
USD: `/World` carries an `xformOp:transform` set to the M matrix (USD accepts
a reflection directly); geometry inside stays raw Unity. See the "Coordinate
convention" section above.

**Verified:** every room floor vertex lands at exactly `M(unity_json_vertex)`
— 7/7 rooms in both MJCF and USD — not the mirrored `(x,-z,y)`.

### A. Objects floating ~½ object-height above the floor

**Symptom:** every objathor object renders with its base lifted by roughly
half its bounding-box height.

**Root cause:** objathor `.pkl.gz` meshes have local origin at the
**bbox-bottom-center** (vertices `y ∈ [0, height]`), but mansion's
`position.y` for an object is the **bbox-center** height in world coords
(AI2-THOR / Holodeck prefab anchor convention). Placing the body at
`position.y` puts the mesh-bottom at the bbox-center height → floats.

**Fix:** at bake time, compute `anchor_offset = bbox_center` per asset.
At placement, subtract `R(ry) · anchor_offset` from body pos (rotated by
the instance's y-rotation, so rotated bodies don't drift in xz).

**Verification:** for all 65 objathor UIDs in `floor_1.json`,
`|bbox_center.x|` and `|bbox_center.z|` are both < 1 cm (meshes are
symmetric in x/z), so only y materially shifts — but the converter handles
all three axes for safety.

### B. Doors and windows piled at the world origin

**Symptom:** all 7 doors and all 13 windows appear stacked at (0, 0, 0)
with no rotation.

**Root cause:** mansion's doors/windows have no top-level
`position`/`rotation` field — only `assetPosition` *in the wall's local
frame* plus `wall0` / `wall1` IDs. The first version of `_place()` looked
for `position`, found none, and defaulted to (0, 0, 0).

**Fix:** new helper `compute_wall_aligned_pose(item, walls_by_id)`:
1. Look up `wall0` (preferred) and its polygon.
2. Detect bottom edge (see bug §C).
3. World pos = `start + assetPosition.x · bottom_dir + assetPosition.y · up`.
4. Y-rotation = `atan2(-dz, dx)` of the bottom-edge direction, so the
   door's local +x axis aligns with the wall.

**Known incomplete:** the `holePolygon` is registered against both `wall0`
and `wall1` for cutouts (§D), but mansion may store the coordinates only
in `wall0`'s frame. If `wall1`'s bottom edge is reversed, that side's
cutout is mirrored — visible in coincident interior walls where one side
has a door-shaped hole and the other has the door panel.

### C. 11 / 13 windows skipped due to "could not resolve wall"

**Symptom:** after fixing bug §B, doors placed cleanly but most windows
were skipped with `reason: "could not resolve wall for door/window"`.

**Root cause:** wall polygon vertex ordering is **inconsistent across the
mansion dataset**. Interior walls use `[TL, TR, BR, BL]` (so polygon[3] →
polygon[2] is the bottom edge). Exterior walls use `[BL, TL, TR, BR]`
(so polygon[0] → polygon[3] is the bottom edge — polygon[3] → polygon[2]
is a *vertical* left edge with zero xz-span, hence the early return
when `hypot(dx, dz) < 1e-9`).

**Fix:** in both `compute_wall_aligned_pose` and `_wall_local_frame`,
detect the bottom edge by finding the two vertices with `y == min y`,
then pick the CCW ordering where `(i + 1) % n` gives the other bottom
vertex.

### D. Doors visually overlap with walls (no opening)

**Symptom:** doors render in the correct world pose but the wall mesh
behind them is intact, so the doorway is not visible — it looks like a
solid wall with a door panel stuck on its face.

**Root cause:** the v1 wall writer emitted each wall as a single-face
polygon mesh, ignoring `holePolygon` data. No subtraction.

**Fix:** `write_wall_with_holes_obj` — compute the wall's 2D local frame
(bottom-edge origin, x along bottom edge, y up), collect all `holePolygon`
rectangles registered against this wall (via doors/windows' `wall0` /
`wall1`), run rectangle-minus-rectangles strip decomposition, triangulate
each sub-rectangle, project back to 3D. 40 / 82 walls in `floor_1.json`
got cutouts.

### H. Bench protruding past wall in consultation_room_2 — REVISED ROOT CAUSE

**Symptom:** `waiting_bench-0` (assetId `1b701029…`) in
`F1_consultation_room_2` has parts of its mesh protruding through the
room's north wall.

**First (wrong) diagnosis:** I initially blamed mansion's procedural
placer for not leaving clearance for a deep bench. The bbox is square
(1.44 × 1.44 m) so I assumed the bench was square. WRONG: the mesh is
actually an **L-shape** (corner bench) — vertex distribution shows three
of four xz-quadrants populated, the (x<0, z>0) quadrant has zero
vertices. The square bbox is just the L's outer envelope.

**Actual root cause:** the objathor `.pkl.gz` file carries a
``yRotOffset`` field — an asset-intrinsic rotation about the up axis (in
degrees) that's meant to be **added to mansion's per-instance
`rotation.y`** at placement time. Mansion's JSON `rotation.y=180°` for
this bench is "180° from the asset's natural pose"; the natural pose
itself is rotated 45° (yRotOffset). v1 of the converter silently ignored
`yRotOffset`, so the L-shape's arms were oriented 45° off from where
mansion intended, sending one arm through the wall.

12 of the 65 objathor UIDs in `floor_1.json` have non-zero `yRotOffset`,
ranging from −12° to +45° (visitor chairs, trash bins, and other corner /
asymmetric assets).

**Fix:** read `yRotOffset` at bake time into `BakedAsset.y_rot_offset_deg`;
at placement, use `ry = item.rotation.y + ba.y_rot_offset_deg`. Verified
empirically: after the fix, `waiting_bench-0`'s world bbox shrinks from
y ∈ [−16.29, −14.85] (protruding past the y=−16 wall) to
y ∈ [−15.92, −15.22] (cleanly inside the room).

THOR-MJCF assets don't use `yRotOffset` — their source XMLs bake any
required rotation into nested body transforms (see bug §E). Leave the
default `y_rot_offset_deg = 0` for those.

### I. "Floating" wall cabinets — false alarm

**Symptom:** Two black, TV-shaped objects in `F1_consultation_room_2`
appear to float at chest/eye height with no visible support.

**Investigation:** these are `wall_cabinet-0` / `wall_cabinet-1` (assetId
`b4deb4ce…`). They're at Unity y=2.02, rotated 90° / 270°, mounted against
the west wall (x=13) and east wall (x=19) respectively. Their world
x-extents `[13.00, 13.40]` and `[18.60, 19.00]` both touch the walls
exactly — they are properly attached.

**Verdict:** **not a bug.** Wall-mounted cabinets are supposed to hang in
mid-air. The visual "floating" impression is normal for wall-mounted
geometry. If you'd expect the cabinet to look like a piece of furniture
standing on the floor, you're conflating this asset with a free-standing
cabinet — the asset's name (`wall_cabinet`) is the giveaway.

### L. Two polygon-corner rules needed, depending on wall polygon order

**Symptom (after the §K fix landed):** windows looked aligned correctly,
but doors that were previously at the right position were now off by 1–6 m
along their walls.

**Initial (misleading) framing:** I first explained this as "doors are
authored in wall1's frame, windows in wall0's frame". That sounds like
two different conventions in mansion's authoring — but it's wrong. AI2-THOR
opens these scenes with a single, consistent rule, and so does mansion's
authoring code. The user pushed back: if mansion had two conventions, the
AI2-THOR engine wouldn't render them correctly either, but it does.

**Actual root cause:** mansion uses ONE consistent rule for placing
doors and windows. Verified by reading mansion's ``doorSegment`` and
``windowSegment`` fields, which store each item's footprint in **world
coordinates** directly. The midpoint of those segments is the item's
intended world position — no wall frame needed:

| item | doorSegment midpoint | converter output |
|---|---|---|
| Doorway_Double_5 (records, 9 m wall) | (7.85, 12.00) | (7.85, 12.00) ✓ |
| Doorway_2 (utility, 7 m wall) | (5.85, 0.00)  | (5.85, 0.00)  ✓ |
| Window_Hung_32x60 #1 (restrooms) | (19.00, 0.59) | (19.00, 0.59) ✓ |

(... matches for all 7 doors and all 13 windows.)

The reason my converter needs *different polygon-index arithmetic* for
doors vs windows is purely an artifact of how mansion stores polygons:
- **Interior walls** (where the 7 doors all live in this scene) are
  stored as ``[TL, TR, BR, BL]``.
- **Outer walls** (where most of the 13 windows live) are stored as
  ``[BL, TL, TR, BR]``.
The two "outer walls with interior-style ``[TL,TR,BR,BL]`` ordering"
(``F1_stair|outer|0``, ``F1_utility_room|outer|4``) have **windows** on
them — and using the "window rule" (successor) on those gives the
correct positions too. So the rule depends on **polygon vertex order**,
not on item type. My code's ``kind="door"`` / ``kind="window"`` labels
just happen to track polygon order because of how this scene is wired.

**Fix:** add a ``kind`` parameter to ``compute_wall_aligned_pose`` and
``_wall_local_frame``:
```python
if (bottom_idxs[0]+1) % n == bottom_idxs[1]:
    if kind == "door":
        i_start, i_end = bottom_idxs[0], bottom_idxs[1]   # predecessor
    else:
        i_start, i_end = bottom_idxs[1], bottom_idxs[0]   # successor
```
``_place()`` passes ``kind=category`` (the existing `"door"`/`"window"`
string), and the hole-projection loop matches so cutouts stay paired with
their items.

**Cleaner version (implemented as the primary path; kind-aware code kept
as fallback):** ``compute_pose_from_segment()`` and
``world_hole_corners_from_segment()`` use mansion's ``doorSegment`` /
``windowSegment`` (world-coordinate ground truth) directly. Position =
segment midpoint, rotation = ``atan2(-dz, dx)`` of segment direction,
hole world corners = segment endpoints (bottom) plus ``holePolygon`` y
range. This bypasses every polygon-corner question.

Verified bit-identical to the polygon-corner code: regenerated the
entire scene with the segment path disabled (forced fallback), diffed
the resulting ``scene.xml`` and all 82 wall ``.obj`` files — zero
differences, and a fresh render is pixel-for-pixel identical. So the
two implementations are equivalent on this dataset. The segment path
runs first so any future scene where mansion's polygon conventions vary
still works; the polygon-corner fallback covers cases where a segment
field is missing.

**Empirical confirmation:** for door interior|15 in `floor_1.json`, the
predecessor rule gives world x=7.85 (matches doorSegment midpoint) and
the successor rule gives 1.15 (mansion would NOT have rendered this in
AI2-THOR). Line of sight when opening the door: aligns with
``Window_Slider_60x48 #1`` at x=7.46 on the north outer wall — NOT with
``#0`` at x=1.41 (the other window on the same wall, west end).

### K. Door/window positions mirrored about wall midpoint

**Symptom:** doors and windows that should be near one corner of a wall
appear at the opposite corner. For the user-reported case in
`F1_reception_and_records`: the room's south wall has a door near the
WEST end (intended visible-through-doorway line of sight to a window on
the room's north wall, also near the west end). v1 of the converter
placed the door at the EAST end of the wall, breaking the line of sight.

**Root cause:** in `compute_wall_aligned_pose` and `_wall_local_frame`,
the bottom-edge "start" vertex was picked as the **predecessor** of the
other bottom vertex in polygon order (the rule
`(bottom_idxs[0]+1)%n == bottom_idxs[1] → start = bottom_idxs[0]`). That
consistently selected **BR** as the start. But mansion's authoring
convention measures `assetPosition.x` from **BL**, with the x-axis
pointing toward BR. So my "start at BR with direction toward BL" mirrors
every door/window position about the wall midpoint.

Items at the wall midpoint barely move under mirroring (e.g.
`Doorframe_Double_9` shifted only 0.025 m); items near a corner mirror
dramatically (e.g. `Doorway_Double_5` shifted 6.71 m across a 9 m wall).

**Fix:** swap the rule — pick the **successor** instead of the
predecessor:
```python
if (bottom_idxs[0]+1) % n == bottom_idxs[1]:
    i_start, i_end = bottom_idxs[1], bottom_idxs[0]   # was: 0, 1
elif (bottom_idxs[1]+1) % n == bottom_idxs[0]:
    i_start, i_end = bottom_idxs[0], bottom_idxs[1]   # was: 1, 0
```
Apply to both `compute_wall_aligned_pose` (placement) and
`_wall_local_frame` (cutout) so they stay consistent.

**Why I didn't notice this earlier:** for most doors (which mansion
roughly centers on their wall) the mirror error is small. The
`Doorframe_Double_9` test I used while developing §B was a centered door
(2 m wall, door at x=1.01 vs midpoint at x=1.00) — mirror gave x=0.99,
within visual noise. The user-found case (Doorway_Double_5 at x=1.15 on
a 9 m wall) made the bug obvious.

**Audit:** 18 of 20 doors+windows in `floor_1.json` shifted by ≥ 0.2 m
after the fix; max shift 6.71 m for the long-wall door.

### J. Layout PNG y-axis was inverted

**Symptom:** room labels placed at the wrong vertical positions on the
labeled top-down (`public_restrooms` label at bottom-right, but the toilet
fixtures visible at top-right).

**Root cause:** the helper `w2p` (in `scripts/mansion/diagnostic`-style
ad-hoc render scripts) used `py = H/2 + (wy - lookat.y) / half_h * (H/2)`,
assuming image-y increases with world-y. But MuJoCo's free camera at
`azimuth=90, elevation=-89.5°` has image-up = world **+y** (NOT -y), so
the formula should be `py = H/2 - (wy - lookat.y) / half_h * (H/2)`.

**Fix:** flip the sign in `w2p`. Top-down image now reads: top = south
(Unity z=0), bottom = north (Unity z=16), right = east (Unity x=19),
left = west (Unity x=0). Render path: see `/tmp/render_correct_layout.py`.

### G. Object placement audit summary (updated 2026-05-19)

After bug §A landed, I audited all 65 objathor UIDs in `floor_1.json` for
mesh-asymmetry and 5 representative assets (3 floor-standing, 3 wall-mounted)
for the bbox-center anchor convention:

| asset (placement)                          | pos.y  | bbox.y range  | bbox_center.y | match? |
|---|---|---|---|---|
| `waiting_bench_cluster-0` (floor)          | 0.43   | [0.00, 0.86]  | 0.43          | YES    |
| `self_checkin_station-0`   (floor)         | 0.37   | [0.00, 0.74]  | 0.37          | YES    |
| `notice_board-0`   (wall, high)            | 1.81   | [0.00, 1.01]  | 0.507         | NO ⚠️  |
| `health_education_poster-0` (wall)         | 2.07   | [0.00, 1.33]  | 0.663         | NO ⚠️  |
| `wayfinding_sign_panel-0`   (wall)         | 1.94   | [0.00, 2.10]  | 1.052         | NO ⚠️  |

The "NO" wall-mounted entries are NOT broken: `pos.y` for them is the
world-space y where the bbox-center should land. The mismatch column above
just means `pos.y > bbox_center.y` because these are mounted high on the
wall — bbox-center ends up at world y > 1m. Bug §A's
`body.y = pos.y - bbox_center.y` formula puts the bbox-center at exactly
`pos.y` for both classes; the convention is uniform.

So: **all sampled assets in floor_1.json follow `pos.y = bbox_center_world_y`**
and bug §A's fix handles them correctly. If a specific scene still has a
"floating" object, it's likely either:
- A different asset class (e.g. ceiling-mounted lights) not yet sampled.
- A THOR-MJCF asset whose source XML body tree expects a different anchor
  than mansion's `pos.y` (bug §E partial fix may not cover all THOR assets).
- A visual artifact of the wall being on the opposite side (no occlusion).

### F. Wall cutouts misaligned with door / window meshes on coincident walls

**Symptom:** windows being transparent revealed that the wall cutouts and
the actual door / window meshes weren't at the same world position — the
cutout was offset from the asset by some amount along the wall.

**Root cause:** `holePolygon` is authored in `wall0`'s local 2D frame.
v1 of the cutout pipeline reused the same `(xmin, ymin, xmax, ymax)`
rectangle for both `wall0` and `wall1` (the coincident other-side wall).
But mansion's wall pairs frequently have **opposite bottom-edge directions
and different bottom-edge start vertices** — e.g. for `window[0]`'s wall0
the bottom edge runs from `z=3.5` to `z=0` (dz=-3.5), while wall1 runs
from `z=0` to `z=3.5` (dz=+3.5). So a hole at wall-local x=0.708 ends up
at world `z=2.792` on wall0 but at world `z=0.708` on wall1 — same local
number, completely different world position. The wall1 cutout ends up at
the mirrored location across the wall.

**Fix:** project the hole through world space.
1. Build `wall0`'s 2D local frame.
2. Compute the hole's 4 corners in world coords by mapping the 4 corners
   of `(xmin..xmax, ymin..ymax)` through wall0's frame.
3. For each of `wall0` and `wall1`, project the 4 world corners back into
   that wall's own local frame; take the axis-aligned bbox of the
   projection as the local cutout rectangle.

This is invariant to polygon vertex order — wall1's mirrored frame
automatically produces a mirrored local x, which is exactly what's needed
to put its cutout at the correct world position.

### E. THOR door/window meshes shifted upward by ~half their height

**Symptom:** after fixes §A and §B, the door's bottom appears floating
above the floor inside the doorway.

**Root cause:** molmospaces THOR XMLs anchor each mesh via a chain of
nested `<body>` elements with non-trivial `pos` and `quat`. For example
in `Doorway_2.xml`:
```
<body name="Doorway_2_doorway_2" pos="0 -1.03702 -0.00345" quat="0 0 -1 0">
  <geom mesh="Doorway_2_doorway_2"/>
```
The mesh itself is bbox-bottom-center anchored (y ∈ [0, 2.073]), but the
parent body's pos.y = -1.037 and 180° quat together put the *mesh's
bbox-center* exactly at the asset-root origin — which is what mansion's
`assetPosition` (door center) expects.

v1 of `bake_thor_mjcf` collected `<geom>` elements via `root.iter("geom")`,
which traverses *all* descendants flatly and silently drops every parent
body transform. So meshes ended up at their bbox-bottom-center, anchored
to the asset root → shifted up by half-height.

**Fix:** rewrite the geom collection as a recursive walk over the source
`<worldbody>` tree, composing each parent body's pos/quat as
`(cumulative_pos, cumulative_quat)` using MuJoCo's quaternion convention
`(w, x, y, z)`. Each visual `<geom>` records its cumulative pos+quat
(plus the geom's own pos/quat composed in) and emits them as `pos quat`
attributes on the `<geom>` element in the output MJCF. Quaternion helpers:
`_quat_mul`, `_quat_rotate`, `_compose_transform`.

---

## 9. Why we don't reuse `MlSpacesSceneBuilder.load_from_json()`

molmospaces's housegen builder reads MJCF mesh files from
`${MLSPACES_ASSETS_DIR}/objects/{thor,objaverse}/`. The objaverse cache
contains a 129 647-entry manifest but zero extracted meshes (lazy download
pattern). After full bulk-download, only **23 / 65** of mansion's objathor
UIDs are in that manifest at all — molmospaces curates a subset of
objaverse that doesn't cover mansion's selection. So the builder route
caps at ~48 % coverage. Direct `.pkl.gz` → `.obj` baking gets the missing
42 UIDs.
