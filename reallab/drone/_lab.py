"""Internal (lab-only) constructor exposing switches that the public make() does not: flight_mode (reference
autopilots on the non-site tasks), anomalies=False (site, smoke only), _course (port, smoke only)."""
from .api import RealDrone

_LAB_KW = {"flight_mode", "anomalies", "_course"}


def make_lab(task, purpose, tag="", seed=None, **kw):
    lab = {k: kw.pop(k) for k in list(kw) if k in _LAB_KW}
    return RealDrone(task, purpose, tag, seed, _lab=lab, **kw)
