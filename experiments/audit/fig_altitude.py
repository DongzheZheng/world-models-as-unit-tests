"""Side view of one real mission-C episode (evaluation seed 1000): altitude against distance along the canyon.

Shows what the videos only hint at: the code-world pilot's controller climbs above the altitudes the other
aircraft use and drops to each relay, while the code-as-policy programs stay at relay height.  Built from the
real evaluation step logs; other aircraft positions come from the lab's rendering-only key in those logs.
Colours come from the page's CSS tokens.  usage: python experiments/audit/fig_altitude.py -> runs/probe/fig_altitude.html
"""
import glob
import gzip
import json
from pathlib import Path

SEED = 1000
RUNS = [
    ("cp5", "runs/drone/aspire_style/eval_logs_aspire_style_C", "CaP agent, 5 real trials"),
    ("cp20", "runs/drone/aspire_style/eval_logs_aspire_style_extended_C", "CaP agent, 20 real trials"),
    ("cw", "runs/drone/code_wm/eval_logs/code_wm_C", "code-as-world-model agent, 5 real trials"),
]
W, H, L, R, T, B = 760, 260, 44, 14, 16, 36
X0, X1, Z0, Z1 = 0.0, 60.0, 0.0, 6.0
x = lambda v: L + (W - L - R) * (v - X0) / (X1 - X0)
y = lambda v: T + (H - T - B) * (1 - (v - Z0) / (Z1 - Z0))


def load(folder):
    path = sorted(glob.glob(f"{folder}/*_{SEED}.jsonl.gz"))[0]
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    obs = [rows[0]["obs"]] + [r["obs"] for r in rows if r["type"] == "step"]
    pos = [o["pos"] for o in obs]
    info = rows[-1]["info"] if rows[-1]["type"] == "end" else {}
    outcome = next((k for k in ("success", "collision", "out_of_bounds", "timeout") if info.get(k)), "unknown")
    movers = [r.get("render_movers") for r in rows if r["type"] in ("reset", "step")]
    return dict(pos=pos, relays=obs[0]["relays"], outcome=outcome, movers=movers)


runs = {k: load(f) for k, f, _ in RUNS}
o = [f'<svg class="alt" viewBox="0 0 {W} {H}" role="img" aria-label="Altitude against distance along the canyon for three controllers on one mission-C episode">']
for z in (0, 2, 4, 6):
    o.append(f'<line class="grid" x1="{L}" y1="{y(z):.1f}" x2="{W - R}" y2="{y(z):.1f}"/>'
             f'<text class="tk" x="{L - 6}" y="{y(z) + 3.5:.1f}" text-anchor="end">{z} m</text>')
for v in range(0, 61, 10):
    o.append(f'<text class="tk" x="{x(v):.1f}" y="{H - B + 14}" text-anchor="middle">{v}</text>')
o.append(f'<text class="tk" x="{(L + W - R) / 2:.1f}" y="{H - 6}" text-anchor="middle">distance along the canyon (m)</text>')
o.append(f'<text class="tk" x="{W - R}" y="{y(6) - 4:.1f}" text-anchor="end">flight ceiling</text>')

# where the other aircraft were during the code-world flight (every 6th step, inside the canyon)
zs = []
for step in runs["cw"]["movers"][::6]:
    for m in step or []:
        p = m[:3] if isinstance(m, (list, tuple)) else m.get("pos", m)[:3]
        if abs(p[1]) <= 2.0 and X0 <= p[0] <= X1:
            zs.append(p[2])
            o.append(f'<circle class="mv" cx="{x(p[0]):.1f}" cy="{y(p[2]):.1f}" r="1.6"/>')
for key, cls in (("cp5", "cp dash"), ("cp20", "cp"), ("cw", "cw")):
    pts = runs[key]["pos"]
    o.append(f'<polyline class="{cls}" points="' + " ".join(f"{x(p[0]):.1f},{y(p[2]):.1f}" for p in pts[::2]) + '"/>')
    if runs[key]["outcome"] != "success":
        px, pz = x(pts[-1][0]), y(pts[-1][2])
        o.append(f'<path class="crash" d="M{px - 5:.1f},{pz - 5:.1f} L{px + 5:.1f},{pz + 5:.1f} M{px - 5:.1f},{pz + 5:.1f} L{px + 5:.1f},{pz - 5:.1f}"/>')
for i, r in enumerate(runs["cw"]["relays"]):
    # label near the floor, clear of every flight path, joined to its marker by a faint guide
    o.append(f'<line class="grid" x1="{x(r[0]):.1f}" y1="{y(r[2]) + 6:.1f}" x2="{x(r[0]):.1f}" y2="{y(0.75):.1f}"/>'
             f'<circle class="relay" cx="{x(r[0]):.1f}" cy="{y(r[2]):.1f}" r="5"/>'
             f'<text class="tk" x="{x(r[0]):.1f}" y="{y(0.35):.1f}" text-anchor="middle">checkpoint {i + 1}</text>')
o.append("</svg>")
Path("runs/probe").mkdir(parents=True, exist_ok=True)
Path("runs/probe/fig_altitude.html").write_text("".join(o) + "\n")
cw = runs["cw"]["pos"]
print({k: (v["outcome"], len(v["pos"])) for k, v in runs.items()})
print("code world altitude: max", round(max(p[2] for p in cw), 2), "| other aircraft altitude range in canyon:",
      (round(min(zs), 2), round(max(zs), 2)) if zs else None, "| samples", len(zs))
print("cp20 altitude range", round(min(p[2] for p in runs["cp20"]["pos"]), 2), round(max(p[2] for p in runs["cp20"]["pos"]), 2))
