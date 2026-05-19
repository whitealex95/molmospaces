"""Audit a MansionWorld floor JSON against available asset libraries.

For every unique ``assetId`` referenced by ``objects``, ``doors``, and
``windows``, report whether it can be resolved from:

- ``~/.objathor-assets/2023_09_23/assets/`` (mansion-side cache; includes
  mansion_patch additions like ``small_stair``, ``toilet-suite`` after
  running ``setup_mansion.py``)
- the molmospaces USD library at ``assets/usd/objects/{thor,objaverse}/``

Run from the molmospaces repo root::

    PYTHONPATH=. python scripts/mansion/audit_assetids.py \\
        --scene-json /path/to/floor_1.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

OBJATHOR_UID_RE = re.compile(r"^[0-9a-f]{32}$")


def _index_dir(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {p.name for p in path.iterdir() if p.is_dir()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-json", required=True, type=Path)
    parser.add_argument(
        "--objathor-dir",
        type=Path,
        default=Path.home() / ".objathor-assets" / "2023_09_23" / "assets",
    )
    parser.add_argument(
        "--thor-usd-dir",
        type=Path,
        default=Path("/home/jkim3662/Projects/molmospaces/assets/usd/objects/thor"),
    )
    parser.add_argument(
        "--objaverse-usd-dir",
        type=Path,
        default=Path("/home/jkim3662/Projects/molmospaces/assets/usd/objects/objaverse"),
    )
    args = parser.parse_args()

    scene = json.loads(args.scene_json.read_text())
    ids: set[str] = set()
    for src in ("objects", "doors", "windows"):
        for o in scene.get(src) or []:
            aid = o.get("assetId")
            if aid:
                ids.add(aid)

    objathor = _index_dir(args.objathor_dir)
    thor_usd = _index_dir(args.thor_usd_dir)
    objaverse_usd = _index_dir(args.objaverse_usd_dir)

    in_objathor = {a for a in ids if a in objathor}
    in_thor_usd = {a for a in ids if a in thor_usd}
    in_objaverse_usd = {a for a in ids if a in objaverse_usd}
    resolvable = in_objathor | in_thor_usd | in_objaverse_usd
    missing = sorted(ids - resolvable)

    print(f"Scene: {args.scene_json}")
    print(f"  unique assetIds: {len(ids)}")
    print(f"  in objathor cache ({args.objathor_dir}): {len(in_objathor)}")
    print(f"  in molmospaces USD thor      ({args.thor_usd_dir}): {len(in_thor_usd)}")
    print(f"  in molmospaces USD objaverse ({args.objaverse_usd_dir}): {len(in_objaverse_usd)}")
    print(f"  fully resolvable (any source): {len(resolvable)} / {len(ids)}")
    if missing:
        print(f"\n  MISSING ({len(missing)}):")
        for a in missing:
            kind = "uid" if OBJATHOR_UID_RE.match(a) else "named"
            print(f"    [{kind}] {a}")
        return 1

    print("  ✅ all assetIds resolvable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
