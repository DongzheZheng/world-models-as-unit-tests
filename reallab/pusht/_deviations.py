"""Hidden deviations of the real block / pusher from the published datasheet.  NOT part of the manual.

Pilots must not import or read this module.  `NOMINAL` is the datasheet object (used only by lab-side references).
"""
# The T's centre of gravity (body-frame coordinates, same frame as the datasheet) is moved off the axis of symmetry.
# nominal: (0, 45)
# Global velocity damping 0.01 instead of 0 (pymunk Space.damping, velocity multiplier per second): in the nominal rig
# the block stops the instant the pusher stops pushing (quasi-static); here it glides on after being pushed.
DEVIATIONS = dict(block_cog=(24.0, 30.0), damping=0.01)
NOMINAL = dict(block_cog=None, k_p=100.0, k_v=20.0)

# ---- the L block (second object) ------------------------------------------------------------------------------------
# nominal L centre of gravity: (45, 37.5) (mean of the two rectangle centroids).  Hidden: moved to (23, 55.5), i.e.
# towards the long leg and away from the foot (the T's shift was (+24, -15)); same gliding as the T rig; plus a
# pusher-block friction (nominal 0).
DEVIATIONS_L = dict(block_cog=(23.0, 55.5), damping=0.01, friction=0.6)  # set on pusher and block shapes: pair coefficient 0.36
NOMINAL_L = dict(block_cog=None, k_p=100.0, k_v=20.0)
