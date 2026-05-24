#!/usr/bin/env python3
"""Launch IsaacSim with its GUI (non-headless) for interactive probing.

Run this YOURSELF in a terminal whose $DISPLAY points at the screen you are
looking at (e.g. :20 under Chrome Remote Desktop) -- the GUI window opens
there. Launches empty by default so the window comes up fast; pass a USD to
open it directly, or use File > Open inside the GUI.

    conda activate mlspaces-isaac
    python scripts/navigation/open_isaac_gui.py [scene.usda] [--g1 [X Y]]

Pass `--g1` to also reference the Unitree G1 USD at the scene origin (or at
X, Y if given) -- same geometry layer that run_isaac.py loads.

Mansion USD:
  ~/Projects/mansion/usd_export/public_healthcare_3f_300_fp001_0/floor_1/scene.usda

Heads-up: this is a second IsaacSim instance. If a GPU RL training is running,
the RTX viewport will render slowly or stall -- that contention is exactly
what we're trying to diagnose.
"""

import argparse
import sys
from pathlib import Path

G1_USD = Path.home() / "Projects/CAMDM/PyTorch/visualize/assets/g1_isaac/configuration/g1_base.usd"
G1_GROUND_Z = 0.315  # see run_isaac.py

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("scene", nargs="?", default=None, help="USD scene to open")
ap.add_argument("--g1", nargs="*", type=float, default=None,
                help="Reference the G1 USD into the scene. Optionally pass `X Y` for the base xy (default 0 0).")
ap.add_argument("--g1-usd", default=str(G1_USD), help="G1 USD path")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": False})

import omni.usd  # noqa: E402  (must import after the app launches)

if args.scene:
    usd = str(Path(args.scene).expanduser().resolve())
    print(f"opening stage: {usd}")
    omni.usd.get_context().open_stage(usd)
else:
    print("launched empty -- use File > Open to load a USD")

if args.g1 is not None:
    from pxr import Gf, UsdGeom  # noqa: E402

    x, y = (args.g1 + [0.0, 0.0])[:2]
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        # need a stage; create a new one so the G1 has somewhere to live
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
    # session-layer edit so the on-disk scene USD is never touched
    stage.SetEditTarget(stage.GetSessionLayer())
    g1 = stage.DefinePrim("/g1", "Xform")
    g1.GetReferences().AddReference(str(Path(args.g1_usd).expanduser().resolve()))
    g1x = UsdGeom.Xformable(g1)
    g1x.ClearXformOpOrder()
    g1x.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), G1_GROUND_Z))
    print(f"G1 referenced at /g1, base xyz = ({x:.3f}, {y:.3f}, {G1_GROUND_Z})")

print("IsaacSim GUI is running. Close the window (or Ctrl+C) to exit.")
while app.is_running():
    app.update()
app.close()
