from pathlib import Path
import numpy as np

import math
import numpy as np

from brainrender import Scene
from brainrender.actors import Cylinder
from brainrender import VideoMaker
from brainrender import Animation
from brainrender import settings
from bg_atlasapi import show_atlases
from brainrender.actors import Points

scene = Scene(title="Silicon Probe Visualization")

# Visualise the probe target regions
MEC = scene.add_brain_region("MEC", alpha=0.15)

# render
scene.render()
print("")