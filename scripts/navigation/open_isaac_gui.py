#!/usr/bin/env python3
"""Launch IsaacSim with its GUI (non-headless) for interactive probing.

Run this YOURSELF in a terminal whose $DISPLAY points at the screen you are
looking at (e.g. :20 under Chrome Remote Desktop) -- the GUI window opens
there. Launches empty by default so the window comes up fast; pass a USD to
open it directly, or use File > Open inside the GUI.

    conda activate mlspaces-isaac
    python scripts/navigation/open_isaac_gui.py [scene.usda]

Mansion USD:
  ~/Projects/mansion/usd_export/public_healthcare_3f_300_fp001_0/floor_1/scene.usda

Heads-up: this is a second IsaacSim instance. If a GPU RL training is running,
the RTX viewport will render slowly or stall -- that contention is exactly
what we're trying to diagnose.
"""

import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": False})

import omni.usd  # noqa: E402  (must import after the app launches)

if len(sys.argv) > 1:
    usd = str(Path(sys.argv[1]).expanduser().resolve())
    print(f"opening stage: {usd}")
    omni.usd.get_context().open_stage(usd)
else:
    print("launched empty -- use File > Open to load a USD")

print("IsaacSim GUI is running. Close the window (or Ctrl+C) to exit.")
while app.is_running():
    app.update()
app.close()
