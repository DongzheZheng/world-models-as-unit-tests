"""Public geometry and limits of the cable rig (published through api.task_card()).  Internal module."""
import math

TABLE_LOW = (0.0, -0.4)
TABLE_HIGH = (1.2, 0.4)
ANCHOR = (0.05, 0.0)            # clamp point of the cable's first end (centre line), m
CLAMP_DIR = (1.0, 0.0)          # the clamp holds the first ~2.6 cm of cable pointing along +x
CABLE_LENGTH = 1.0
CABLE_RADIUS = 0.004            # 8 mm diameter
N_MARKERS = 20
PEG_RADIUS = 0.03
PEG_HEIGHT = 0.08
GRIP_HEIGHT = 0.012             # height of the gripper centre above the table
GRIP_RADIUS = 0.015             # the gripper is a ball of this radius; it collides with the pegs only
GRIP_VMAX = 0.30                # m/s, maximum speed of the gripper's motion target
GRIP_FMAX = 40.0                # N, maximum force of each gripper axis
CONTROL_HZ = 20.0
TIME_LIMIT = 600                # agent steps (30 s)
CLIP_TOL = 0.02                 # m
REST_SPEED = 0.02               # m/s, "rests" = gripper speed below this
TENSION_LIMIT = 10.0            # N, mean anchor force over one control period
SIDE_THRESHOLD = math.pi / 2    # |winding| below this -> side 0 (undefined)
