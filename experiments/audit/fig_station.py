"""Figure for the cable task: the code world as a judge of 20 programs it had never been fitted to.

Left and middle panels: a simple open-loop program (tight path around the posts, one constant speed, no load
feedback) at eight speeds and two clearances -- real success rate on 50 unseen episodes (seeds 6000-6049) against
the success rate the code world predicted from the reset readings of the same episodes, with zero real episodes.
Right panel: the four programs the two pilots wrote.
Reads runs/probe/station_probe.json.  Colours come from the page's CSS tokens.
usage: python experiments/audit/fig_station.py -> runs/probe/fig_station.html  (also prints the agreement numbers)
"""
import json
from pathlib import Path

P = Path("runs/probe")
d = json.loads((P / "station_probe.json").read_text())
R = {k.split("@")[0]: v for k, v in d["real"].items() if k.endswith("@6000")}
Wd = {k.split("@")[0]: v for k, v in d["world"].items() if k.endswith("@6000")}

SPEEDS = [4, 6, 8, 10, 12, 15, 20, 25]                      # cm/s
W, H, L, Rm, T, B = 320, 214, 38, 12, 40, 34
y = lambda s: T + (H - T - B) * (1 - s)


def axes(title, sub, aria):
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{aria}">',
         f'<text class="tt" x="{L}" y="15">{title}</text><text class="ts" x="{L}" y="29">{sub}</text>']
    for s in (0, 0.5, 1):
        o.append(f'<line class="grid" x1="{L}" y1="{y(s):.1f}" x2="{W - Rm}" y2="{y(s):.1f}"/>'
                 f'<text class="tk" x="{L - 6}" y="{y(s) + 3.5:.1f}" text-anchor="end">{int(s * 100)}%</text>')
    return o


def series(o, xs, labels, real, pred, names):
    o.append('<polyline class="cw dash" points="' + " ".join(f"{a:.1f},{y(s):.1f}" for a, s in zip(xs, pred)) + '"/>')
    o.append('<polyline class="cp" points="' + " ".join(f"{a:.1f},{y(s):.1f}" for a, s in zip(xs, real)) + '"/>')
    for a, lb, r, q, nm in zip(xs, labels, real, pred, names):
        o.append(f'<text class="tk" x="{a:.1f}" y="{H - B + 14}" text-anchor="middle">{lb}</text>')
        o.append(f'<rect class="cwd" x="{a - 3.2:.1f}" y="{y(q) - 3.2:.1f}" width="6.4" height="6.4"><title>{nm}: world model predicted {q * 100:.0f}%</title></rect>')
        o.append(f'<circle class="cpd" cx="{a:.1f}" cy="{y(r):.1f}" r="3"><title>{nm}: real {r * 100:.0f}%</title></circle>')


def speed_panel(clear_mm, title):
    keys = [f"open_loop_c{clear_mm:03d}_v{v:02d}" for v in SPEEDS]
    real, pred = [R[k]["success"] for k in keys], [Wd[k]["success"] for k in keys]
    n = len(keys)
    xs = [L + (W - L - Rm) * (i + 0.5) / n for i in range(n)]
    o = axes(title, "success rate of a constant-speed program", f"{title}: predicted against real success rate at eight speeds")
    series(o, xs, [str(v) for v in SPEEDS], real, pred, [f"{v} cm/s" for v in SPEEDS])
    o.append(f'<text class="tk" x="{(L + W - Rm) / 2:.1f}" y="{H - 5}" text-anchor="middle">tool speed (cm/s)</text></svg>')
    return "".join(o), real, pred


def pilots_panel():
    keys = ["cap_v1", "cap_v2", "cap_v3", "code_wm_ctrl"]
    lab = ["v1", "v2", "v3", "world's"]
    real, pred = [R[k]["success"] for k in keys], [Wd[k]["success"] for k in keys]
    n = len(keys)
    xs = [L + (W - L - Rm) * (i + 0.5) / n for i in range(n)]
    o = axes("The agents' own programs", "all four regulate speed on the live load reading", "The agents' programs: predicted against real success rate")
    series(o, xs, lab, real, pred, ["CaP v1", "CaP v2", "CaP v3", "code-as-world-model agent's controller"])
    o.append(f'<text class="tk" x="{(L + W - Rm) / 2:.1f}" y="{H - 5}" text-anchor="middle">CaP v1&ndash;v3, code-as-world-model agent</text></svg>')
    return "".join(o), real, pred


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va, vb = sum((u - ma) ** 2 for u in a), sum((u - mb) ** 2 for u in b)
    return sum((u - ma) * (w - mb) for u, w in zip(a, b)) / (va * vb) ** 0.5 if va and vb else float("nan")


html, allr, allp = ['<div class="curves">'], [], []
for svg, real, pred in (speed_panel(90, "Open loop, 9 cm from each post"), speed_panel(65, "Open loop, 6.5 cm from each post"), pilots_panel()):
    html.append(svg)
    allr += real
    allp += pred
html.append("</div>")
(P / "fig_station.html").write_text("\n".join(html) + "\n")
err = [abs(a - b) for a, b in zip(allr, allp)]
same = [sum(a == b for a, b in zip(d["real"][k]["per_seed"], d["world"][k]["per_seed"])) for k in d["real"] if k in d["world"] and k.endswith("@6000")]
print(f"n={len(allr)} mean|err|={sum(err) / len(err) * 100:.1f} pts  within 5: {sum(e <= 0.05 + 1e-9 for e in err)}  within 10: {sum(e <= 0.10 + 1e-9 for e in err)}  "
      f"max|err|={max(err) * 100:.1f}  r={pearson(allr, allp):.3f}  per-episode outcome agreement {sum(same)}/{50 * len(same)} = {sum(same) / (50 * len(same)) * 100:.1f}%")
