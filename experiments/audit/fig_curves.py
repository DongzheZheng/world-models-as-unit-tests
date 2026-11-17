"""Build the blog's learning-curve figure as an HTML fragment of inline SVGs from runs/probe/curves.json.

Each panel: x = real lives consumed before a program version existed, y = that version's true success rate on
seeds no pilot ever saw.  Colours come from the page's CSS tokens (classes cp / cw / ax), so the figure follows
the light and dark themes.  usage: python experiments/audit/fig_curves.py  ->  runs/probe/fig_curves.html
"""
import json
from pathlib import Path

D = json.loads(Path("runs/probe/curves.json").read_text())
W, H, L, R, T, B = 320, 214, 38, 12, 40, 34

PANELS = [
    dict(task="drone_A", title="Drone, mission A", sub="every episode is nearly the same", xmax=5, xt=[0, 1, 2, 3, 4, 5],
         cp=[1, 2, 3, 4, 5], cw=["drone_A/code_wm_after1", "drone_A/code_wm_after2", "drone_A/code_wm_after3"], shipped=5),
    dict(task="drone_C", title="Drone, mission C", sub="other aircraft, different every episode", xmax=20, xt=[0, 5, 10, 15, 20],
         cp=[7, 8, 9, 10, 11, 12, 13, 14],
         cw=["drone_C/code_wm_missionA_ctrl", "drone_C/code_wm_after1", "drone_C/code_wm_after2", "drone_C/code_wm_after3",
             "drone_C/code_wm_after4"], shipped=11),
    dict(task="push_T", title="Push-T", sub="a different start pose every episode", xmax=13, xt=[0, 3, 6, 9, 12],
         cp=[1, 2, 3, 4, 5, 6, 7, 8], cw=["push_T/code_wm_after2"], shipped=5),
    dict(task="push_L", title="Push-L, a block never seen before", sub="a different start pose every episode", xmax=8,
         xt=[0, 2, 4, 6, 8], cp=[9, 10, 11, 12, 13, 14, 15], cw=["push_L/code_wm_after2"], shipped=10),
    dict(task="doom", title="Doom take_cover", sub="share of episodes surviving 750 game ticks", xmax=5, xt=[0, 1, 2, 3, 4, 5],
         cp=[1, 2, 3, 4, 5], cw=["doom_cw/code_wm_after3"], shipped=5),
]


def panel(p):
    x = lambda v: L + (W - L - R) * v / p["xmax"]
    y = lambda s: T + (H - T - B) * (1 - s)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{p["title"]}: true success rate against real trials used">']
    o.append(f'<text class="tt" x="{L}" y="15">{p["title"]}</text><text class="ts" x="{L}" y="29">{p["sub"]}</text>')
    for s in (0, 0.5, 1):
        o.append(f'<line class="grid" x1="{L}" y1="{y(s):.1f}" x2="{W - R}" y2="{y(s):.1f}"/>'
                 f'<text class="tk" x="{L - 6}" y="{y(s) + 3.5:.1f}" text-anchor="end">{int(s * 100)}%</text>')
    for v in p["xt"]:
        o.append(f'<text class="tk" x="{x(v):.1f}" y="{H - B + 14}" text-anchor="middle">{v}</text>')
    o.append(f'<text class="tk" x="{(L + W - R) / 2:.1f}" y="{H - 5}" text-anchor="middle">real trials used</text>')

    cp = [(D[f'{p["task"]}/aspire_v{v}']["lives_before"], D[f'{p["task"]}/aspire_v{v}']["success"], v) for v in p["cp"]]
    pts = " ".join(f"{x(a):.1f},{y(s):.1f}" for a, s, _ in cp)
    o.append(f'<polyline class="cp" points="{pts}"/>')
    if cp[-1][0] < p["xmax"]:
        o.append(f'<line class="cp dash" x1="{x(cp[-1][0]):.1f}" y1="{y(cp[-1][1]):.1f}" x2="{x(p["xmax"]):.1f}" y2="{y(cp[-1][1]):.1f}"/>')
    for a, s, v in cp:
        o.append(f'<circle class="cpd" cx="{x(a):.1f}" cy="{y(s):.1f}" r="3"><title>program v{v}, after {a} real trials: {s * 100:.0f}%</title></circle>')
        if v == p["shipped"]:
            o.append(f'<circle class="ship" cx="{x(a):.1f}" cy="{y(s):.1f}" r="7"/>')
    cw = [(D[k]["lives_before"], D[k]["success"]) for k in p["cw"] if k in D]
    if cw:
        if len(cw) > 1:
            o.append('<polyline class="cw" points="' + " ".join(f"{x(a):.1f},{y(s):.1f}" for a, s in cw) + '"/>')
        if cw[-1][0] < p["xmax"]:
            o.append(f'<line class="cw dash" x1="{x(cw[-1][0]):.1f}" y1="{y(cw[-1][1]):.1f}" x2="{x(p["xmax"]):.1f}" y2="{y(cw[-1][1]):.1f}"/>')
        for a, s in cw:
            o.append(f'<rect class="cwd" x="{x(a) - 3.2:.1f}" y="{y(s) - 3.2:.1f}" width="6.4" height="6.4"><title>code-as-world-model agent, after {a} real trials: {s * 100:.0f}%</title></rect>')
    o.append("</svg>")
    return "".join(o)


html = ['<div class="curves">'] + [panel(p) for p in PANELS] + ["</div>"]
Path("runs/probe/fig_curves.html").write_text("\n".join(html) + "\n")
print("panels", len(PANELS), "bytes", len("\n".join(html)))
for p in PANELS:
    print(p["task"], [(D[f'{p["task"]}/aspire_v{v}']["lives_before"], round(D[f'{p["task"]}/aspire_v{v}']["success"], 2)) for v in p["cp"]],
          "| cw", [(D[k]["lives_before"], round(D[k]["success"], 2)) for k in p["cw"] if k in D])
