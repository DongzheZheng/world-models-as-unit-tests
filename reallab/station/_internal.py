"""Hidden physical properties of the real installation (a cable).  NOT part of the manual (which publishes no
physical data at all).

Pilots must not import or read this module.  NOMINAL is the lab's ordinary design cable (the "deviations off"
world used for validation); DEVIATIONS overrides some of its entries in the real rig.  Setting the environment variable CAWM_STATION_NODEV=1 switches the
deviations off (lab-side reference runs only).
"""
import os

# Ordinary cable (validation reference; not published).
NOMINAL = dict(
    bend_EI=1.5e-3,        # bending stiffness E*I, N m^2
    mu_table=0.40,         # cable-table friction coefficient
    mu_peg=0.30,           # cable-peg friction coefficient
    mass_per_m=0.060,      # kg/m
    drag=0.0,              # N s/m per metre of cable: viscous drag against the table
)

# The real cable ("fights back when you hurry").  Chosen in lab tuning runs (reallab/station/_tune.py, _grid*.sh):
# with these values the naive router drops from ~85% (nominal) to ~27% success, a slow load-aware reference ~79%.
DEVIATIONS = dict(
    bend_EI=4.5e-3,        # 3x nominal: springs out of wraps, needs tension to conform to a post
    mu_peg=1.0,            # 3.3x nominal: strong capstan effect around the posts
    drag=90.0,             # nominal 0: viscous drag against the table, N s/m per metre of cable (speed-dependent)
)


def physical(nodev: bool | None = None) -> dict:
    """The physical parameters of the real cable (nominal + deviations unless switched off)."""
    if nodev is None:
        nodev = os.environ.get("CAWM_STATION_NODEV", "") not in ("", "0")
    p = dict(NOMINAL)
    if not nodev:
        p.update(DEVIATIONS)
        ov = os.environ.get("CAWM_STATION_DEV", "")      # lab tuning only: "k=v,k=v" overrides the deviations
        for kv in filter(None, ov.split(",")):
            k, v = kv.split("=")
            p[k] = float(v)
    return p
