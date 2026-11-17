"""Build the "code world as judge" figure: for every program version the code-as-policy pilot wrote, the success
rate the code world predicted for it (blind, zero real episodes) next to its true success rate on unseen seeds.

Reads runs/probe/screen_{drone,pusht,doom}.json (predictions) and runs/probe/curves.json (truth).
Colours come from the page's CSS tokens.  usage: python experiments/audit/fig_judge.py -> runs/probe/fig_judge.html
Also prints the summary numbers (per-task and overall agreement).
"""
import json
from pathlib import Path

P = Path("runs/probe")
truth = json.loads((P / "curves.json").read_text())
sd, sp, sm = (json.loads((P / f"screen_{k}.json").read_text()) for k in ("drone", "pusht", "doom"))

PANELS = [
    dict(title="Drone, mission A", task="drone_A", vs=[1, 2, 3, 4, 5], pred=lambda v: sd["A"][f"v{v}"]["success"], shipped=5),
    dict(title="Drone, mission C", task="drone_C", vs=[6, 7, 8, 9, 10, 11, 12, 13, 14], pred=lambda v: sd["C"][f"v{v}"]["success"], shipped=11),
    dict(title="Push-T", task="push_T", vs=[1, 2, 3, 4, 5, 6, 7, 8], pred=lambda v: sp["T"][f"v{v}"]["success"], shipped=5),
    dict(title="Push-L", task="push_L", vs=[9, 10, 11, 12, 13, 14, 15], pred=lambda v: sp["L"][f"v{v}"]["success"], shipped=10),
    dict(title="Doom take_cover", task="doom", vs=[1, 2, 3, 4, 5], pred=lambda v: sm[f"v{v}"]["frac_ge_750"], shipped=5,
         sub="share of episodes surviving 750 game ticks"),
]
W, H, L, R, T, B = 320, 214, 38, 12, 40, 34


def panel(p):
    n = len(p["vs"])
    x = lambda i: L + (W - L - R) * (i + 0.5) / n
    y = lambda s: T + (H - T - B) * (1 - s)
    real = [truth[f'{p["task"]}/aspire_v{v}']["success"] for v in p["vs"]]
    pred = [p["pred"](v) for v in p["vs"]]
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{p["title"]}: predicted against real success rate for each program version">',
         f'<text class="tt" x="{L}" y="15">{p["title"]}</text><text class="ts" x="{L}" y="29">{p.get("sub", "success rate of each program version")}</text>']
    for s in (0, 0.5, 1):
        o.append(f'<line class="grid" x1="{L}" y1="{y(s):.1f}" x2="{W - R}" y2="{y(s):.1f}"/>'
                 f'<text class="tk" x="{L - 6}" y="{y(s) + 3.5:.1f}" text-anchor="end">{int(s * 100)}%</text>')
    for i, v in enumerate(p["vs"]):
        o.append(f'<text class="tk" x="{x(i):.1f}" y="{H - B + 14}" text-anchor="middle">v{v}</text>')
    o.append(f'<text class="tk" x="{(L + W - R) / 2:.1f}" y="{H - 5}" text-anchor="middle">program version, in the order written</text>')
    o.append('<polyline class="cw dash" points="' + " ".join(f"{x(i):.1f},{y(s):.1f}" for i, s in enumerate(pred)) + '"/>')
    o.append('<polyline class="cp" points="' + " ".join(f"{x(i):.1f},{y(s):.1f}" for i, s in enumerate(real)) + '"/>')
    for i, (v, r, q) in enumerate(zip(p["vs"], real, pred)):
        o.append(f'<rect class="cwd" x="{x(i) - 3.2:.1f}" y="{y(q) - 3.2:.1f}" width="6.4" height="6.4"><title>v{v}: world model predicted {q * 100:.0f}%</title></rect>')
        o.append(f'<circle class="cpd" cx="{x(i):.1f}" cy="{y(r):.1f}" r="3"><title>v{v}: real {r * 100:.0f}%</title></circle>')
        if v == p["shipped"]:
            o.append(f'<circle class="ship" cx="{x(i):.1f}" cy="{y(r):.1f}" r="7"/>')
    o.append("</svg>")
    return "".join(o), real, pred


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((u - ma) ** 2 for u in a)
    vb = sum((u - mb) ** 2 for u in b)
    return sum((u - ma) * (w - mb) for u, w in zip(a, b)) / (va * vb) ** 0.5 if va and vb else float("nan")


html, allr, allp = ['<div class="curves">'], [], []
for p in PANELS:
    svg, real, pred = panel(p)
    html.append(svg)
    allr += real
    allp += pred
    err = [abs(a - b) for a, b in zip(real, pred)]
    best_pred = p["vs"][max(range(len(pred)), key=lambda i: pred[i])]
    best_real = p["vs"][max(range(len(real)), key=lambda i: real[i])]
    print(f'{p["task"]:8s} n={len(real)} mean|err|={sum(err) / len(err) * 100:4.1f} pts  max|err|={max(err) * 100:4.1f}  r={pearson(real, pred):.3f}  '
          f'world picks v{best_pred} (real {real[p["vs"].index(best_pred)] * 100:.0f}%), truly best v{best_real} ({max(real) * 100:.0f}%)')
html.append("</div>")
(P / "fig_judge.html").write_text("\n".join(html) + "\n")
err = [abs(a - b) for a, b in zip(allr, allp)]
print(f"ALL      n={len(allr)} mean|err|={sum(err) / len(err) * 100:.1f} pts  within 5 pts: {sum(e <= 0.05 for e in err)}  within 10 pts: {sum(e <= 0.10 for e in err)}  "
      f"max|err|={max(err) * 100:.1f}  r={pearson(allr, allp):.3f}")
