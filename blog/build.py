"""Assemble blog/index.html: stylesheet + the post body (written by Claude Opus 4.6) + the two charts + script.

The body text is inserted verbatim; only the two chart placeholders are replaced.
usage (from the project root): python blog/build.py
"""
from pathlib import Path

B = Path("blog")
P = Path("runs/probe")

LEG_CURVES = ('<div class="legend">'
              '<span><i class="k-cp"></i>CaP agent: each program version it wrote</span>'
              '<span><i class="k-ship"></i>the version it submitted for evaluation</span>'
              '<span><i class="k-cw"></i>code-as-world-model agent: each controller it produced</span></div>')
LEG_ALT = ('<div class="legend">'
           '<span><i class="k-line cw"></i>code-as-world-model agent, 5 real trials</span>'
           '<span><i class="k-line"></i>CaP agent, 20 real trials</span>'
           '<span><i class="k-line dash"></i>CaP agent, 5 real trials (crashed at the cross)</span>'
           '<span><i class="k-mv"></i>other aircraft during the flight</span>'
           '<span><i class="k-relay"></i>checkpoints</span></div>')

LEG_JUDGE = ('<div class="legend">'
             '<span><i class="k-cp"></i>real success rate of each CaP program (held-out seeds)</span>'
             '<span><i class="k-cw"></i>success rate the world model predicted for it, without a real trial</span>'
             '<span><i class="k-ship"></i>the version that was submitted</span></div>')

LEG_STATION = ('<div class="legend">'
               '<span><i class="k-cp"></i>real success rate of each program (50 held-out episodes)</span>'
               '<span><i class="k-cw"></i>success rate the world model predicted for it, without a real trial</span></div>')

LEG_ANNEX = ('<div class="legend">'
             '<span><i class="k-cp"></i>CaP agent</span>'
             '<span><i class="k-cw"></i>code-as-world-model agent</span>'
             '<span><i class="k-line dash"></i>left: oracle planner (true simulator)</span>'
             '<span><i class="k-line cw dash"></i>right: first world model (fitted on 4 real trials)</span>'
             '<span><i class="k-line cw"></i>right: revised world model (fitted on 8 real trials)</span></div>')

body = (B / "body.html").read_text()
for marker, legend, frag in (("<!--CURVES-->", LEG_CURVES, "fig_curves.html"), ("<!--ALTITUDE-->", LEG_ALT, "fig_altitude.html"),
                             ("<!--JUDGE-->", LEG_JUDGE, "fig_judge.html"), ("<!--STATION-->", LEG_STATION, "fig_station.html"),
                             ("<!--ANNEX-->", LEG_ANNEX, "fig_annex.html")):
    if marker in body:
        body = body.replace(marker, f'<div class="chart">{legend}{(P / frag).read_text()}</div>')
    else:
        print("placeholder missing:", marker)
if "<!--OVERVIEW-->" in body:      # the four-step overview is a hand-written fragment kept next to the body
    body = body.replace("<!--OVERVIEW-->", f'<div class="chart">{(B / "fig_overview.html").read_text()}</div>')
else:
    print("placeholder missing: <!--OVERVIEW-->")
out = (B / "_head.html").read_text() + body + (B / "_tail.html").read_text()
(B / "index.html").write_text(out)
media = sorted(set(m for m in __import__("re").findall(r'(?:src|poster)="(media/[^"]+)"', body)))
missing = [m for m in media if not (B / m).exists()]
print("index.html bytes", len(out.encode()), "| media referenced", len(media), "| missing", missing)
