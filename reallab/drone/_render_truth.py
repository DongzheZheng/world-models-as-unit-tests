"""Internal (lab-only): reveal-the-truth overlays for the renderer."""
import numpy as np

from . import _site


def rotor_offset(scale=0.65):
    s = np.array([[0.028, -0.028], [-0.028, 0.028], [0.028, 0.028], [-0.028, -0.028]][_site.ROTOR]) / 0.028
    return np.array([s[0], s[1], 0.0]) * scale


def _jet_boxes():
    out = []
    for j, side in ((_site.JET1, +1), (_site.JET2, -1)):
        x0, x1 = j["xc"] - 1.8, j["xc"] + 1.8
        ys = (-2.0, 12.0) if side > 0 else (-12.0, 2.0)
        out.append((x0, x1, ys[0], ys[1], j))
    return out


def add_truth_bodies(pb, cid, body, vbox):
    for x0, x1, y0, y1, j in _jet_boxes():
        body(vbox([(x1 - x0) / 2, (y1 - y0) / 2, 0.02], (0.2, 0.55, 1.0, 1.0)), [(x0 + x1) / 2, (y0 + y1) / 2, 0.08])


def draw_truth_topdown(ax):
    from matplotlib.patches import Rectangle
    for x0, x1, y0, y1, j in _jet_boxes():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="tab:blue", alpha=0.25, ec="none", zorder=1))
        d = j["dir"]
        ax.annotate("", xy=(j["xc"] + 0 * d[0], (y0 + y1) / 2 + 3.5 * d[1]), xytext=(j["xc"], (y0 + y1) / 2 - 3.5 * d[1]),
                    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2), zorder=2)
        ax.text(j["xc"] + 2.2, 8.0 if d[1] < 0 else -8.0, f"jet {j['U']:.0f} m/s", ha="left", fontsize=7, color="tab:blue", zorder=9)
    ax.text(30, -11.4, f"damaged rotor #{_site.ROTOR}: {int(_site.ROTOR_EFF * 100)}% thrust+torque", ha="center",
            fontsize=8, color="m")


marker_offset = rotor_offset   # neutral alias used by the public renderer
