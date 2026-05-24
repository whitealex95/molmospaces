#!/usr/bin/env python3
"""Launch the MuJoCo viewer with a scene (and optionally the G1 merged in).

Run from `mlspaces-mujoco` -- the Filament-built wheel in `mlspaces` lacks
the classic OpenGL UI symbols, so any `mujoco.viewer` entry point fails there
(see CLAUDE.md "Visualizing a scene").

    conda activate mlspaces-mujoco
    python scripts/navigation/open_mujoco_gui.py <scene.xml> [--g1 [X Y]]

Without `--g1` this is essentially `python -m mujoco.viewer --mjcf <scene>`.
With `--g1` it merges the Unitree G1 into the scene the same way
`run_mujoco.py` does (same camera + fill light), so the viewer can switch to
the `ego` camera and visually inspect what the agent sees. Pass `X Y` to set
the G1's xy spawn position (default 0 0).

Mansion MJCF:
  ~/Projects/mansion/mjcf_export/public_healthcare_3f_300_fp001_0/floor_1/scene.xml
Procthor (ceilinged):
  ~/.cache/molmospaces/assets/<base64>/scenes/procthor-10k-val/val_0_ceiling.xml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")  # interactive viewer needs glfw, not egl

import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_mujoco import G1_XML, PELVIS_Z, build_model, yaw_quat  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", type=Path, help="scene .xml")
    ap.add_argument(
        "--g1",
        nargs="*",
        type=float,
        default=None,
        help="Merge the G1 robot. Optionally pass `X Y` for the spawn xy (default 0 0).",
    )
    ap.add_argument("--g1-xml", type=Path, default=G1_XML, help="G1 MJCF path")
    ap.add_argument(
        "--renderer",
        choices=["opengl", "filament"],
        default="opengl",
        help="Only opengl is supported in this env; filament viewer needs the unavailable classic UI.",
    )
    args = ap.parse_args()

    if args.g1 is None:
        # No G1 -- load the scene as-is so the viewer behaves like `-m mujoco.viewer`.
        model = mujoco.MjModel.from_xml_path(str(args.scene))
    else:
        model = build_model(args.scene, args.g1_xml, args.renderer)

    data = mujoco.MjData(model)
    if args.g1 is not None:
        x, y = (args.g1 + [0.0, 0.0])[:2]
        base_adr = model.joint("g1_floating_base_joint").qposadr[0]
        data.qpos[base_adr : base_adr + 7] = [x, y, PELVIS_Z, *yaw_quat(0.0)]
    mujoco.mj_forward(model, data)

    # Use launch_passive so the viewer does NOT step physics. The scene MJCFs
    # (mansion, procthor) author floors/walls as visual-only geoms
    # (contype=0 conaffinity=0), so a stepped G1 just falls through the floor
    # under gravity. run_mujoco.py is also kinematic-only (writes qpos every
    # frame, calls mj_forward; no mj_step), so this matches the actual render
    # pipeline. We sync at ~30 Hz to keep the UI responsive.
    import time

    print(
        "MuJoCo viewer is running (kinematic; no physics stepping). "
        "Mouse to orbit, Ctrl+drag to manipulate, Tab for side panel. "
        "Close the window or Ctrl+C to exit."
    )
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 30.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
