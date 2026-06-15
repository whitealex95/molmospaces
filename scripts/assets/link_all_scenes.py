#!/usr/bin/env python
"""Link every already-extracted scene into the symlink view (MLSPACES_ASSETS_DIR).

The resource manager links large scene datasets *lazily*: archives are extracted
into the cache store, but per-file symlinks under ``$MLSPACES_ASSETS_DIR/scenes``
are only created on demand (see ``post_setup`` in ``molmo_spaces_constants.py``
and ``install_scene_from_path`` in ``lazy_loading_utils.py``). A scene XML's
relative asset paths (e.g. ``../../objects/thor/...``) only resolve inside that
version-flattened symlink view -- never against the versioned cache store.

Raw ``python -m mujoco.viewer --mjcf ...`` bypasses the resource manager
entirely, so it can only open a scene that has already been linked. This script
materializes the symlinks for *every* scene currently extracted in the cache,
letting you point the viewer at any of them.

It is idempotent and resumable: scenes already linked are skipped, and it never
triggers a download (it only links what is physically present in the cache).

Usage (run from any conda env -- linking needs no renderer):

    python scripts/assets/link_all_scenes.py                    # every dataset
    python scripts/assets/link_all_scenes.py --dataset procthor-10k-val
    python scripts/assets/link_all_scenes.py --dry-run          # report only

Then view any scene from the `mlspaces-mujoco` env (classic OpenGL renderer):

    python -m mujoco.viewer --mjcf \\
        "$MLSPACES_ASSETS_DIR/scenes/<dataset>/<scene>.xml"
"""

import argparse
import time

from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, DATA_CACHE_DIR
from molmo_spaces.utils.lazy_loading_utils import install_scene_from_path

# Per scene the cache ships several XML *variants*. Linking the main scene's
# archive brings every variant along, so these are skipped as separate entries.
_VARIANT_SUFFIXES = ("_ceiling", "_non_settled", "_orig")


def _main_scene_stems(version_dir):
    """Return the stems of the *main* scene XMLs in one cache version directory."""
    stems = set()
    for entry in version_dir.iterdir():
        if entry.suffix != ".xml":
            continue
        stem = entry.stem
        if any(stem.endswith(suffix) for suffix in _VARIANT_SUFFIXES):
            continue
        stems.add(stem)
    return stems


def _scene_sources():
    """Map each cache scene source to its sorted list of main scene stems."""
    scenes_root = DATA_CACHE_DIR / "scenes"
    if not scenes_root.is_dir():
        raise SystemExit(f"No scenes found in cache: {scenes_root}")

    sources = {}
    for source_dir in sorted(scenes_root.iterdir()):
        if not source_dir.is_dir():
            continue
        stems = set()
        for version_dir in source_dir.iterdir():
            if version_dir.is_dir():
                stems |= _main_scene_stems(version_dir)
        if stems:
            sources[source_dir.name] = sorted(stems)
    return sources


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        action="append",
        metavar="SOURCE",
        help="restrict to this scene source, e.g. procthor-10k-val (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be linked without creating any symlinks",
    )
    args = parser.parse_args()

    sources = _scene_sources()
    if args.dataset:
        unknown = [d for d in args.dataset if d not in sources]
        if unknown:
            raise SystemExit(f"Unknown dataset(s) {unknown}; available: {sorted(sources)}")
        sources = {k: v for k, v in sources.items() if k in args.dataset}

    total = sum(len(stems) for stems in sources.values())
    print(f"Cache store : {DATA_CACHE_DIR}")
    print(f"Symlink view: {ASSETS_DIR}")
    print(
        f"{total} scene(s) across {len(sources)} dataset(s): "
        + ", ".join(f"{k}={len(v)}" for k, v in sources.items())
    )

    linked = skipped = failed = 0
    started = time.time()
    try:
        for source, stems in sources.items():
            for stem in stems:
                view_xml = ASSETS_DIR / "scenes" / source / f"{stem}.xml"
                if view_xml.is_symlink() or view_xml.exists():
                    skipped += 1
                    continue
                if args.dry_run:
                    linked += 1
                    continue
                try:
                    install_scene_from_path(view_xml)
                    linked += 1
                except Exception as exc:
                    failed += 1
                    print(f"  FAIL {source}/{stem}: {type(exc).__name__}: {exc}")
                if (linked + failed) % 500 == 0:
                    elapsed = time.time() - started
                    print(
                        f"  ...{linked} linked, {skipped} already, {failed} failed [{elapsed:.0f}s]"
                    )
    except KeyboardInterrupt:
        print("\nInterrupted -- partial progress is kept; re-run to resume.")

    verb = "would link" if args.dry_run else "linked"
    print(
        f"\nDone in {time.time() - started:.0f}s: {verb} {linked}, "
        f"already-linked {skipped}, failed {failed}."
    )


if __name__ == "__main__":
    main()
