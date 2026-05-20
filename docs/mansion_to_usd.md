# Mansion → USDA conversion

How to convert a single floor JSON from the **MansionWorld** dataset
(`~/Projects/mansion/mansionworld/<building>/floor_<n>.json`) into a
USDA scene that opens in IsaacSim.

> **When working in this repo:** always read this doc before re-deriving the
> mansion conversion pipeline from source. It records the audited asset
> coverage, the exact commands that work, and the gotchas already hit.

---

## 1. What this conversion is (and isn't)

The MansionWorld floor JSON is an **AI2-THOR / Holodeck-style** scene file
(top-level keys `rooms`, `walls`, `doors`, `windows`, `objects`,
`proceduralParameters`) using **Unity coordinates** (`y` is up).

The molmospaces `MlSpacesSceneBuilder` + `ms-convert-houses` pipeline
**cannot convert mansion scenes end-to-end** as-is. Auditing one floor
(`public_healthcare_3f_300_fp001#0/floor_1.json`, 85 unique assetIds) against
the molmospaces asset library found:

| Library | Coverage |
|---|---|
| molmospaces MJCF `objects/thor` | 18 / 85 (doors, windows, a few named THOR objects) |
| molmospaces MJCF `objects/objaverse` | **0 / 85** |
| molmospaces USD `objects/thor` | 18 / 85 |
| molmospaces USD `objects/objaverse` | 5 / 85 |
| `~/.objathor-assets/2023_09_23/assets/` (mansion-side cache) | **67 / 85** |
| Total asset coverage (mansion-side + molmospaces) | **85 / 85** ✅ |

Conclusion: every mansion assetId is resolvable, but only via the
**mansion-side asset caches** (objathor + mansion_patch).
Reference: `scripts/mansion/audit_assetids.py` (run any time you change scenes).

This doc therefore uses a **direct objathor / mansion_patch → USD bridge**,
implemented in `scripts/mansion/mansion_to_usd.py`. It bypasses the
molmospaces housegen builder entirely.

---

## 2. Prerequisites (one-time)

### 2.1 Mansion dataset & patch (already done for this user)

Follow the mansion repo's README (`~/Projects/mansion/README.md`):

```bash
# 1. objathor base data + assets + annotations + features
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
python -m objathor.dataset.download_assets             --version 2023_09_23
python -m objathor.dataset.download_annotations        --version 2023_09_23
python -m objathor.dataset.download_features           --version 2023_09_23

# 2. mansion_patch.zip (from huggingface.co/datasets/superbigsaw/MansionWorld)
#    place next to setup_mansion.py and run
python setup_mansion.py
```

Confirm the resulting layout:

```
~/.objathor-assets/2023_09_23/
├── assets/                       # 50k+ folders, each with <uid>.pkl.gz, albedo.jpg, normal.jpg, emission.jpg
│   ├── 00000054c36d44a2a483bdbff31d8edf/
│   │   ├── 00000054c36d44a2a483bdbff31d8edf.pkl.gz
│   │   ├── albedo.jpg
│   │   ├── normal.jpg
│   │   └── emission.jpg
│   ├── small_stair/              # patched (.obj/.json instead of .pkl.gz)
│   ├── toilet-suite/
│   └── elevator_panel_*/
├── annotations.json.gz
└── holodeck/2023_09_23/doors/door-database.json

~/.ai2thor/releases/thor-Linux64-local/                # patched local AI2-THOR binary
```

The patched asset folders (`small_stair`, `toilet-suite`, `elevator_panel_*`)
contain `.obj` + texture jpgs, **not** `.pkl.gz` — the converter handles both.

### 2.2 Molmospaces USD library (for THOR-named doors/windows)

Used for the 18 named THOR assets (doors, windows, Tissue_Box_1, etc.) that
ship in molmospaces's curated USD set:

```bash
conda activate mlspaces-isaac
cd ~/Projects/molmospaces/molmo_spaces_isaac
ms-download --type usd --install-dir ../assets/usd --assets thor
```

(You already have this — `~/Projects/molmospaces/assets/usd/objects/thor/`.)

### 2.3 Conda env

The converter uses Pixar `pxr` (USD bindings); run it in **`mlspaces-isaac`**.

```bash
conda activate mlspaces-isaac
```

---

## 3. Quick start

```bash
conda activate mlspaces-isaac
cd ~/Projects/molmospaces

python scripts/mansion/mansion_to_usd.py \
    --scene-json /home/jkim3662/Projects/mansion/mansionworld/public_healthcare_3f_300_fp001#0/floor_1.json \
    --output-dir /home/jkim3662/Projects/mansion/usd_export \
    --objathor-dir ~/.objathor-assets/2023_09_23/assets \
    --thor-usd-dir ~/Projects/molmospaces/assets/usd/objects/thor
```

Outputs:

```
~/Projects/mansion/usd_export/<scene_stem>/
├── scene.usda            # the file to drag into IsaacSim
├── assets/
│   ├── <uid>.usda        # one per unique objathor mesh referenced by this scene
│   ├── <uid>/albedo.jpg  # texture copies (so scene is self-contained)
│   └── ...
└── conversion_report.json   # what got rendered, what got skipped, why
```

Open `scene.usda` in IsaacSim (`File → Open`, or drag into Stage).

---

## 4. What the converter does

The placement logic was iteratively debugged in the MJCF sibling
(`docs/mansion_to_mjcf.md`, bug log §A–§L) and the same fixes have been
ported here. Numerical verification: all 20 door/window world positions
in `floor_1.json` match mansion's `doorSegment` / `windowSegment` midpoints
to 5 decimal places (max Δ = 0.00000 m).

| Feature | status | Notes |
|---|---|---|
| Floor per room | ✅ | Single n-gon `UsdGeom.Mesh` face per `rooms[].floorPolygon` |
| Walls | ✅ | Triangulated mesh per `walls[].polygon`; **cut at door/window holes** via rectangle-minus-rectangles strip decomposition (see mjcf doc §D, §F) |
| Objects (objathor UID, `.pkl.gz`) | ✅ | Baked to `<uid>.usda` with mesh + albedo texture |
| Objects (mansion_patch, `.json`) | ✅ | `small_stair`, `toilet-suite`, `elevator_panel_*` baked the same way |
| Objects (named THOR, e.g. `Tissue_Box_1`) | ✅ | External `Sdf.Reference` to molmospaces USD-thor |
| Doors | ✅ | World pose derived from mansion's `doorSegment` (primary) or wall + `assetPosition` (fallback). Static visual only — no hinge joints. |
| Windows | ✅ | Same — from `windowSegment` or wall + `assetPosition`. |
| Wall cutouts | ✅ | Hole world corners computed from segment endpoints + `holePolygon` y range, projected into each wall's frame so coincident interior walls cut at the same world position. |
| Objathor anchor offset | ✅ | bbox-center subtracted (rotated by per-instance ry + yRotOffset) so meshes sit on the floor instead of floating by half-height. |
| `yRotOffset` | ✅ | Read from objathor pkl and added to per-instance `rotation.y` (12 of 65 objathor assets in this scene have non-zero values). |
| Lights | ❌ | No lights authored; viewer fallback only |
| Skybox / dome | ❌ | `proceduralParameters.skyboxId` ignored |
| Materials beyond albedo | ❌ | `normal.jpg`, `emission.jpg`, `metallic_smoothness.jpg` are unused |
| Collisions / physics | ❌ | Visual mesh only; no `UsdPhysics.CollisionAPI` |

The script writes every skipped asset to `conversion_report.json` so you know
what's missing in the resulting USDA. See `docs/mansion_to_mjcf.md` for the
full bug log; the same fixes apply here.

### First-run results (this user's floor_1.json)

Running the Section 3 command on
`public_healthcare_3f_300_fp001#0/floor_1.json` produces:

```
~/Projects/mansion/usd_export/public_healthcare_3f_300_fp001_0/floor_1/
├── scene.usda                  ~92 KB (1829 prims total)
├── assets/                     ~85 MB, 67 baked .usda + 67 texture subdirs
└── conversion_report.json
```

Summary line printed at the end:

```
wrote .../scene.usda  (rooms=7 walls=82 objects=144/144 doors=7/7 windows=13/13)
```

`conversion_report.json`:

```json
{
  "n_objects_total": 144, "n_objects_placed": 144, "n_objects_skipped": 0,
  "n_doors_total": 7,     "n_doors_placed": 7,
  "n_windows_total": 13,  "n_windows_placed": 13,
  "n_rooms": 7, "n_walls": 82,
  "unique_assetids": 85, "baked_assets": 67, "referenced_assets": 18,
  "missing_assetids": [], "skipped_objects": []
}
```

A non-empty `missing_assetids` list on a future scene means an objathor
download was incomplete (re-run `python -m objathor.dataset.download_assets`)
or `setup_mansion.py` didn't install a patched asset folder you need.

### Coordinate convention

Mansion JSON uses **Unity** (left-handed, `+y` up); USD / IsaacSim is
**right-handed, `+z` up**. The conversion is the y↔z swap

    M: (x, y, z) -> (x, z, y)

a **reflection** (det -1) — required to convert left-handed Unity to
right-handed USD. A pure rotation (the old `RotateX 90`) only fixes the
up-axis and leaves the scene mirrored — see `docs/mansion_to_mjcf.md` §M.

The converter applies M once, as an `xformOp:transform` on `/World` (USD
accepts a reflection matrix directly). A single root op converts the whole
scene; all geometry and placements inside `/World` stay in raw Unity coords.

---

## 5. Reproducing from scratch

Step-by-step for a different building / floor:

1. **Verify prerequisites are in place** (Section 2).

2. **Pick a scene JSON**:
   ```bash
   ls ~/Projects/mansion/mansionworld | head
   # e.g. office_corporate_hq_10f_300_fp001#0/
   ```

3. **Audit asset coverage** (cheap, ~1 s):
   ```bash
   PYTHONPATH=. python scripts/mansion/audit_assetids.py \
       --scene-json ~/Projects/mansion/mansionworld/<building>/floor_<n>.json
   ```
   This prints how many unique assetIds resolve in each library and lists the
   missing ones. If something is missing from the objathor cache, re-run
   `python -m objathor.dataset.download_assets --version 2023_09_23` or check
   that `setup_mansion.py` completed.

4. **Convert**:
   ```bash
   conda activate mlspaces-isaac
   python scripts/mansion/mansion_to_usd.py \
       --scene-json ~/Projects/mansion/mansionworld/<building>/floor_<n>.json \
       --output-dir ~/Projects/mansion/usd_export
   ```

5. **Inspect**:
   ```bash
   ls ~/Projects/mansion/usd_export/<building>__floor_<n>/
   # scene.usda  assets/  conversion_report.json
   cat ~/Projects/mansion/usd_export/<building>__floor_<n>/conversion_report.json
   ```

6. **Open in IsaacSim**: drag `scene.usda` into the Stage panel.

---

## 6. Known issues / TODO

- **Doors and windows don't punch holes in walls.** They're placed as floating
  geometry inside the wall plane, so doorways will visually overlap the wall.
  Fix would replicate molmospaces's `make_hole_in_wall` logic
  (`molmo_spaces/housegen/utils.py`) for USD walls.
- **Materials are albedo-only.** Objathor packages ship `albedo.jpg`,
  `normal.jpg`, `emission.jpg` (and `metallic_smoothness.jpg` for patched
  assets). v1 only wires albedo into `UsdPreviewSurface.diffuseColor`. The
  other textures are not currently copied either — only `albedo.jpg` is.
- **No physics.** Add `UsdPhysics.CollisionAPI` per mesh (or a per-asset
  collider mesh from the `colliders` field already inside the `.pkl.gz`) to
  make scenes usable with PhysX/Newton.
- **No lights.** `proceduralParameters.lights` carries an array of
  Unity point lights; convert to `UsdLux.SphereLight`. `skyboxId` could map
  to a `UsdLux.DomeLight` texture (mansion stores no skybox texture in the
  patch, so this needs a separate asset source).
- **Walls and floors share z=0**, which causes z-fighting along the wall
  bases. Either thicken walls into boxes (preferred) or offset the floor
  slightly.
- **External USD references use absolute paths.** Moving `usd_export/` to
  another machine without `~/Projects/molmospaces/assets/usd/objects/thor/`
  at the same path will break the 18 THOR-named references. Either pre-copy
  the referenced asset dirs into `usd_export/<scene>/assets/` and rewrite
  refs to relative, or bake them inline like the objathor ones.

If you extend the converter, also update the v1 table in Section 4 and the
`conversion_report.json` schema.

### Cross-reference: the MJCF sibling has the canonical bug log

The MJCF converter (`docs/mansion_to_mjcf.md` Section 8) went through
several iterations to land doors / windows / objects at correct world
poses and to add wall cutouts. **All of those fixes have been ported here:**

1. **§A** — Objathor anchor offset: bbox-center subtracted at placement,
   rotated by per-instance `ry + yRotOffset`.
2. **§B / §C / §K** — Doors / windows derived from wall + assetPosition
   with min-y bottom-edge detection (fallback path).
3. **§D / §F** — Wall cutouts: hole world corners projected through world
   space so coincident wall pairs cut at the same physical position.
4. **§H** — `yRotOffset` from objathor pkl added to per-instance rotation.
5. **§L** — `doorSegment` / `windowSegment` world-coordinate ground truth
   used as the **primary** placement path; the polygon-corner code is
   kept as fallback only.

Verification: regenerated `floor_1.json` from `public_healthcare_3f_300_fp001#0`,
loaded the USD with pxr's `UsdGeom.XformCache`, and confirmed every
door/window world position matches mansion's segment midpoint to 5
decimal places (max Δ = 0.00000 m, n=20). Wall cutout count = 40/82,
identical to the MJCF output.

### Visual validation

The conversion produces a syntactically valid USDA (`pxr.Usd.Stage.Open`
succeeds, references resolve) and the geometry is numerically verified: every
room floor vertex lands at `M(unity_json_vertex)` (7/7 rooms). It has not been
opened in IsaacSim to confirm the rendered result. If something looks off:

1. ~~Concave floor polygons render as overlapping triangles~~ — **fixed**:
   `_add_polygon_mesh` ear-clips the polygon (`_triangulate_polygon`).
2. ~~Z-up / handedness wrong~~ — **fixed**: `/World` applies the reflection
   `M` (see "Coordinate convention" and `docs/mansion_to_mjcf.md` §M). If
   faces render inside-out, that is a winding/normals interaction with the
   reflecting root transform — set the mesh `orientation` or flip normals.
3. Object instances upside-down / flipped → check `yRotOffset` handling.

---

## 7. Reference: how the audit numbers were obtained

```bash
# from ~/Projects/molmospaces
python - <<'PY'
import json
from pathlib import Path
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

j = json.load(open("/home/jkim3662/Projects/mansion/mansionworld/"
                   "public_healthcare_3f_300_fp001#0/floor_1.json"))
all_ids = set(o["assetId"] for src in ("objects","doors","windows")
                            for o in (j.get(src) or []) if o.get("assetId"))

objathor = {p.name for p in (Path.home()/".objathor-assets/2023_09_23/assets").iterdir()}
thor_usd = {p.name for p in (Path("/home/jkim3662/Projects/molmospaces/assets/usd/objects/thor")).iterdir()}
print(f"{len(all_ids)} unique, {sum(a in objathor for a in all_ids)} in objathor, "
      f"{sum(a in thor_usd for a in all_ids)} in molmospaces USD thor")
PY
```

---

## 8. Why we don't use `ms-convert-houses`

`ms-convert-houses --mode convert-single` expects an MJCF house file built by
`molmo_spaces.housegen.builder.MlSpacesSceneBuilder.load_from_json()`. That
builder, in turn, reads MJCF assets from
`${MLSPACES_ASSETS_DIR}/objects/{thor,objaverse}/`. For mansion scenes, the
objaverse MJCF cache is empty (molmospaces never baked objathor → MJCF for
this UID set), so the builder would silently drop ~80 % of mansion objects.

Bypassing the MJCF intermediate and going straight from objathor `.pkl.gz` to
USD is both more accurate (we have the source meshes) and shorter (one less
hop).
