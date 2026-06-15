#!/usr/bin/env python
"""Build a lightweight, viewer-only symlink tree for `python -m mujoco.viewer`.

The resource manager links scenes *per file* into ``$MLSPACES_ASSETS_DIR`` --
correct for version tracking, but ~1.3M "slow" symlinks (>5 GB on ext4) once
every dataset is linked. For the sole purpose of opening a scene in
``mujoco.viewer`` that is vastly overkill: a scene XML resolves its assets with
relative paths like ``../../objects/thor/...``, and MuJoCo normalizes those
``..`` segments *lexically* before opening files (verified) -- so a single
whole-directory symlink per dataset is enough.

This script builds exactly that -- one symlink per scene dataset and one per
object library:

    <out>/scenes/procthor-10k-val -> <cache>/scenes/procthor-10k-val/<version>
    <out>/objects/thor            -> <cache>/objects/thor/<version>

About 7 symlinks (~28 KB) expose every extracted scene, versus 1.3M / 5+ GB.

It is independent of the resource manager and never writes into the cache store,
so generated artifacts (``*_map.png`` THORMAP caches, navigation outputs) are
NOT supported in this tree -- use the resource-manager view
(``$MLSPACES_ASSETS_DIR``) for pipeline / datagen work. This tree is for
eyeballing scenes only.

Usage (pure standard library -- runs from any environment):

    python scripts/assets/build_viewer_view.py
    python scripts/assets/build_viewer_view.py --out ~/scenes_view --force

Then, from the `mlspaces-mujoco` env (the classic OpenGL renderer):

    python -m mujoco.viewer --mjcf <out>/scenes/procthor-10k-val/val_50.xml
"""

import argparse
import os
from pathlib import Path

_DEFAULT_CACHE = Path(
    os.environ.get("MLSPACES_CACHE_DIR", "~/.cache/molmo-spaces-resources")
).expanduser()
_DEFAULT_OUT = Path(
    os.environ.get("MLSPACES_VIEWER_DIR", "~/.cache/molmospaces/viewer")
).expanduser()


def _has_xml(directory):
    """True if ``directory`` directly contains at least one .xml file."""
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.endswith(".xml") and entry.is_file():
                return True
    return False


def _pick_version(source_dir, require_xml):
    """Return the newest version subdir of ``source_dir`` with usable content."""
    # Version directories are date-stamped (e.g. 20251217), so reverse-sorting
    # by name puts the newest first.
    for version_dir in sorted(source_dir.iterdir(), reverse=True):
        if not version_dir.is_dir():
            continue
        if not require_xml or _has_xml(version_dir):
            return version_dir
    return None


def _link(link_path, target, force):
    """Point ``link_path`` at ``target``; never deletes a real file/directory."""
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return "exists"
        if not force:
            return "skipped"
        link_path.unlink()
        link_path.symlink_to(target)
        return "replaced"
    if link_path.exists():
        return "blocked"  # a real file/dir is in the way -- left untouched
    link_path.symlink_to(target)
    return "created"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=_DEFAULT_CACHE,
        help=f"resource cache store (default: {_DEFAULT_CACHE})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"viewer view to build (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing symlinks that point elsewhere",
    )
    args = parser.parse_args()

    cache = args.cache.expanduser()
    out = args.out.expanduser()
    if not (cache / "scenes").is_dir():
        raise SystemExit(f"No scenes found in cache: {cache / 'scenes'}")

    (out / "scenes").mkdir(parents=True, exist_ok=True)
    (out / "objects").mkdir(parents=True, exist_ok=True)
    print(f"Cache store : {cache}")
    print(f"Viewer view : {out}")

    blocked = 0
    for kind, require_xml in (("scenes", True), ("objects", False)):
        src_root = cache / kind
        if not src_root.is_dir():
            continue
        for source_dir in sorted(src_root.iterdir()):
            if not source_dir.is_dir():
                continue
            version_dir = _pick_version(source_dir, require_xml)
            if version_dir is None:
                print(f"  {kind}/{source_dir.name}: no usable version -- skipped")
                continue
            status = _link(out / kind / source_dir.name, version_dir, args.force)
            if status == "blocked":
                blocked += 1
            hint = {
                "skipped": "  (points elsewhere; use --force)",
                "blocked": "  (real file/dir in the way; left untouched)",
            }.get(status, "")
            print(f"  {kind}/{source_dir.name} -> {version_dir.name}  [{status}]{hint}")

    print(
        "\nDone. View any scene from the `mlspaces-mujoco` env:\n"
        f"  python -m mujoco.viewer --mjcf {out}/scenes/<dataset>/<scene>.xml"
    )
    if blocked:
        raise SystemExit(f"{blocked} entr(y/ies) blocked by real files -- see above.")


if __name__ == "__main__":
    main()
