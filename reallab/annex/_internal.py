"""Hidden physical properties of the annex installation.  NOT part of the manual (which publishes no physical data).

Pilots must not import or read this module.  Lab tuning may override entries through the environment variable
CAWM_ANNEX_PHYS="k=v,k=v" (never used for the shipped installation) or by passing a dict to _scene.Scene.
"""
import os

REAL = dict(
    mass_per_m=0.080,      # kg/m of the uniform body (0.7 m -> 56 g)
    tip_mass=0.030,        # kg extra lumped at point 1 (the free end): the "weighted tip" a first guess gets wrong
    bend_EI=2.0e-3,        # bending stiffness E*I, N m^2 (springy, rod-like line; chosen over 4e-4..1e-2 in tuning)
    mu_table=0.10,         # Coulomb friction with the table (chosen over 0.1..0.3 in tuning)
    mu_post=0.25,          # Coulomb friction with the post
    drag=0.15,             # N s/m per metre: viscous drag of each vertex against the table (x, y)
    damping=0.002,         # internal (bending) damping of the elastic model
)

PERTURBABLE = ("mass_per_m", "tip_mass", "bend_EI", "mu_table", "mu_post", "drag")


def physical() -> dict:
    p = dict(REAL)
    for kv in filter(None, os.environ.get("CAWM_ANNEX_PHYS", "").split(",")):
        k, v = kv.split("=")
        p[k] = float(v)
    return p
