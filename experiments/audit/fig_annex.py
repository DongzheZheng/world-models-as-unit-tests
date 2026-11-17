"""Figure for the dynamic rope task (annex).  Orchestrator only.

Panel 1: success on the 50 evaluation episodes (seeds 1000-1049) against the number of real lives used, for the two
         pilots; the dashed line is the lab's reference that plans inside the real simulator itself.
Panel 2: the same submitted programs on 50 episodes nobody had used (seeds 5000-5049).
Panel 3: how far ahead the code-world pilot's world predicts the rope's far end, on real lives it was not fitted to:
         its first world (fitted to 4 lives) against its revised world (fitted to 8 lives).
Reads runs/annex/results/*.json, runs/annex/aspire_style/result_extended_after_6_lives.json,
runs/annex/_lab/baselines.json, runs/probe/annex_audit.json, runs/annex/code_wm/xfid_v{1,2}_on_ctrl_lives.json.
usage: python experiments/audit/fig_annex.py -> runs/probe/fig_annex.html (and prints every plotted number)
"""
import json
from pathlib import Path

P = Path("runs/probe")
W, H, L, Rm, T, B = 320, 214, 38, 12, 40, 34


def frame(title, sub, aria, ticks, ymax, fmt):
    y = lambda v: T + (H - T - B) * (1 - v / ymax)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{aria}">',
         f'<text class="tt" x="{L}" y="15">{title}</text><text class="ts" x="{L}" y="29">{sub}</text>']
    for s in ticks:
        o.append(f'<line class="grid" x1="{L}" y1="{y(s):.1f}" x2="{W - Rm}" y2="{y(s):.1f}"/>'
                 f'<text class="tk" x="{L - 6}" y="{y(s) + 3.5:.1f}" text-anchor="end">{fmt(s)}</text>')
    return o, y


def line(o, cls, pts):
    o.append(f'<polyline class="{cls}" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')


def main():
    res = lambda c: json.loads(Path(f"runs/annex/results/annex__{c}.json").read_text())["eval"]["mean"]
    cap5, cap19, cw4, cw20 = res("aspire_style"), res("aspire_style_extended"), res("code_wm"), res("code_wm_extended")
    cap6 = json.loads(Path("runs/annex/aspire_style/result_extended_after_6_lives.json").read_text())["eval"]["mean"]
    base = json.loads(Path("runs/annex/_lab/baselines.json").read_text())
    ref = float(base["priv"]["success"])
    aud = json.loads((P / "annex_audit.json").read_text())
    pct = lambda v: f"{int(round(v * 100))}%"

    # ---- panel 1: success against lives
    o, y = frame("Held-out episodes, by real trials used", "success on seeds 1000&ndash;1049", "Success rate against real trials used for the two agents",
                 (0, 0.5, 1), 1.0, pct)
    x = lambda n: L + (W - L - Rm) * n / 21.0
    o.append(f'<line class="cp dash" x1="{L}" y1="{y(ref):.1f}" x2="{W - Rm}" y2="{y(ref):.1f}"/>'
             f'<text class="tk" x="{W - Rm}" y="{y(ref) - 4:.1f}" text-anchor="end">planning in the real simulator: {pct(ref)}</text>')
    capp, cwp = [(5, cap5), (6, cap6), (19, cap19)], [(4, cw4), (20, cw20)]
    line(o, "cp", [(x(n), y(v)) for n, v in capp])
    line(o, "cw", [(x(n), y(v)) for n, v in cwp])
    for n, v in capp:
        o.append(f'<circle class="cpd" cx="{x(n):.1f}" cy="{y(v):.1f}" r="3"><title>CaP agent after {n} real trials: {pct(v)}</title></circle>')
    for n, v in cwp:
        o.append(f'<rect class="cwd" x="{x(n) - 3.2:.1f}" y="{y(v) - 3.2:.1f}" width="6.4" height="6.4"><title>code-as-world-model agent after {n} real trials: {pct(v)}</title></rect>'
                 f'<text class="tk" x="{x(n) + (7 if n < 10 else -7):.1f}" y="{y(v) - 6:.1f}" text-anchor="{"start" if n < 10 else "end"}">{pct(v)}</text>')
    o.append(f'<text class="tk" x="{x(19):.1f}" y="{y(cap19) - 7:.1f}" text-anchor="middle">{pct(cap19)}</text>')
    for n in (0, 5, 10, 15, 20):
        o.append(f'<text class="tk" x="{x(n):.1f}" y="{H - B + 14}" text-anchor="middle">{n}</text>')
    o.append(f'<text class="tk" x="{(L + W - Rm) / 2:.1f}" y="{H - 5}" text-anchor="middle">real trials used (one trial = one 4.7 s episode)</text></svg>')
    html = ['<div class="curves">', "".join(o)]

    # ---- panel 2: unseen episodes
    o, y = frame("Second held-out set", "success on seeds 5000&ndash;5049", "Success of the submitted programs on unseen episodes", (0, 0.5, 1), 1.0, pct)
    bars = [("5 trials", aud["cap_v6"], "cpd"), ("19 trials", aud["cap_v19"], "cpd"),
            ("4 trials", aud["code_wm_official"], "cwd"), ("20 trials", aud["code_wm_ext"], "cwd")]
    bw = (W - L - Rm) / len(bars)
    for i, (lb, v, cls) in enumerate(bars):
        cx = L + bw * (i + 0.5)
        o.append(f'<rect class="{cls}" x="{cx - bw * 0.28:.1f}" y="{y(v):.1f}" width="{bw * 0.56:.1f}" height="{max(y(0) - y(v), 1.2):.1f}"><title>{lb}: {pct(v)}</title></rect>'
                 f'<text class="tk" x="{cx:.1f}" y="{y(v) - 5:.1f}" text-anchor="middle">{pct(v)}</text>'
                 f'<text class="tk" x="{cx:.1f}" y="{H - B + 14}" text-anchor="middle">{lb}</text>')
    o.append(f'<text class="tk" x="{(L + W - Rm) / 2:.1f}" y="{H - 5}" text-anchor="middle">the same four programs, real trials each agent used</text></svg>')
    html.append("".join(o))

    # ---- panel 3: prediction horizon of the code world
    hz = ("5", "20", "60")
    f1 = json.loads(Path("runs/annex/code_wm/xfid_v1_on_ctrl_lives.json").read_text())["horizons"]
    f2 = json.loads(Path("runs/annex/code_wm/xfid_v2_on_ctrl_lives.json").read_text())["horizons"]
    e1, e2 = [f1[h]["p1_mean"] * 100 for h in hz], [f2[h]["p1_mean"] * 100 for h in hz]
    o, y = frame("How far ahead the world model sees", "error of the rope's far end, on real trials it was not fitted to",
                 "Prediction error of the far end against steps ahead, first and revised world model", (0, 20, 40), 45.0, lambda v: f"{int(v)} cm")
    xs = [L + (W - L - Rm) * (i + 0.5) / 3 for i in range(3)]
    line(o, "cw dash", [(a, y(v)) for a, v in zip(xs, e1)])
    line(o, "cw", [(a, y(v)) for a, v in zip(xs, e2)])
    for a, h, v1, v2 in zip(xs, hz, e1, e2):
        o.append(f'<circle class="cwd" cx="{a:.1f}" cy="{y(v1):.1f}" r="2.6"><title>first world model, {h} steps ahead: {v1:.1f} cm</title></circle>'
                 f'<text class="tk" x="{a:.1f}" y="{y(v1) - 6:.1f}" text-anchor="middle">{v1:.0f}</text>'
                 f'<rect class="cwd" x="{a - 3.2:.1f}" y="{y(v2) - 3.2:.1f}" width="6.4" height="6.4"><title>revised world model, {h} steps ahead: {v2:.1f} cm</title></rect>'
                 f'<text class="tk" x="{a:.1f}" y="{y(v2) + 14:.1f}" text-anchor="middle">{v2:.0f}</text>'
                 f'<text class="tk" x="{a:.1f}" y="{H - B + 14}" text-anchor="middle">{h} ({int(h) / 50:.1f} s)</text>')
    o.append(f'<text class="tk" x="{(L + W - Rm) / 2:.1f}" y="{H - 5}" text-anchor="middle">steps ahead; MPC re-plans every 8</text></svg>')
    html.append("".join(o))
    html.append("</div>")
    (P / "fig_annex.html").write_text("\n".join(html) + "\n")
    print("panel 1: cap", capp, "code world", cwp, "reference", ref)
    print("panel 2:", [(b[0], b[1]) for b in bars])
    print("panel 3: first world cm", [round(v, 1) for v in e1], "revised world cm", [round(v, 1) for v in e2])


if __name__ == "__main__":
    main()
