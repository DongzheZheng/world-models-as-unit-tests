"""Public geometry and interface limits of the annex installation (published through api.task_card()).  Internal
module: it holds no hidden physical values (those live in _internal.py)."""
import math

CONTROL_HZ = 50.0
ZONE_LOW = (0.30, -0.20)        # tool zone (commands are clipped to it), m
ZONE_HIGH = (0.55, 0.20)
ZONE_CENTRE = (0.425, 0.0)
POST_RADIUS = 0.03
POST_HEIGHT = 0.08
N_POINTS = 20
LENGTH = 0.70                   # rest arc length between point 1 and point 20 (interface fact: it is what the points span)
THICK = 0.005                   # radius of the tracked body (internal; not published)
TOOL_HEIGHT = THICK             # height of the tool centre above the table
TOOL_VMAX = 4.0                 # m/s, speed limit of the tool's motion reference
TOOL_AMAX = 60.0                # m/s^2, acceleration limit of the tool's motion reference
TOOL_FMAX = 60.0                # N, force limit of each tool axis
TOOL_KP, TOOL_KV = 6000.0, 110.0
TOOL_MASS = 0.5
# post placement (relative to the zone centre): bearing from +x, distance
POST_BEARING = (-0.75, 0.75)    # rad (about +-43 deg)
POST_DIST = (0.40, 0.56)        # m
V_REST = 0.02                   # m/s, "at rest" threshold (finite difference of consecutive readings)
ZONE_TOL = 0.01                 # m, the tool counts as inside the zone within this margin
W_MIN = 1.25 * math.pi          # Wrap goal: required |winding| (rad)
POCKET_R = 0.10                 # Pocket goal: target disc radius, m
POCKET_BEHIND = 0.12            # Pocket goal: target centre this far behind the post centre (away from the zone)
POCKET_W = math.pi / 2           # Pocket goal: the points must also pass the post on the required side (side*W >= this)
