"""Draw the share image of the Space (1200 x 630): the page title next to one recorded episode on each side.

    python space/make_thumbnail.py            (from the repository root; writes space/thumbnail.jpg)

The two episodes are run here with space/runner.py (program v5 on the T block: the first world-model episode of the
recorded batch, seed 50000, and the second environment episode, seed 6001) and their last frames are drawn the way
the panel draws them.  All strings come from space/text_static.json and the counts from space/data/index.json.
"""
from __future__ import annotations

import json
import math
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

W, H = 1200, 630
BG, SURFACE, INK, MUTED, RULE = "#f7f8fb", "#ffffff", "#141920", "#586172", "#dde2ea"
ACCENT, GOOD, BAD = "#1d57c9", "#177a49", "#c2362c"
TASK, VERSION = "T", 5
VIEW = 165.0          # half-width of the drawn square, in table units
EPISODES = {"world": (50000, 1), "real": (6001, 2)}       # side -> (seed, number of the episode in its batch)


def placed(poly, pose):
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    return [(x + c * u - s * v, y + s * u + c * v) for u, v in poly]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    from space import runner

    text = json.loads((HERE / "text_static.json").read_text(encoding="utf-8"))
    catalog = json.loads((HERE / "data" / "index.json").read_text(encoding="utf-8"))
    program = next(p for p in catalog["programs"] if p["task"] == TASK and p["version"] == VERSION)
    header = (HERE / "card_header.yaml").read_text(encoding="utf-8").splitlines()
    title = next(line.split(":", 1)[1].strip() for line in header if line.startswith("title:"))

    runs = {"world": runner.run_world(TASK, VERSION, EPISODES["world"][0], 1),
            "real": runner.run_real(TASK, VERSION, EPISODES["real"][0], 1, tag="space_thumbnail")}

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    px = lambda x: x / W                 # noqa: E731  pixels from the left
    py = lambda y: 1 - y / H             # noqa: E731  pixels from the top

    def put(x, y, s, size, color=INK, weight="normal", family="DejaVu Sans", ha="left"):
        fig.text(px(x), py(y), s, fontsize=size, color=color, fontweight=weight, family=family, ha=ha, va="baseline")

    # left: kicker, title, panel heading, program
    put(64, 96, text["kicker"].upper(), 12.5, MUTED, family="DejaVu Sans Mono")
    y = 152
    for line in textwrap.wrap(title, 13):
        put(62, y, line, 34, INK, "bold")
        y += 48
    y += 4
    for line in textwrap.wrap(text["heading"], 28):
        put(64, y, line, 17, MUTED)
        y += 28
    task_name = text[f"task_{TASK}"].split(" (")[0]
    put(64, 548, f"{task_name}  v{VERSION}", 14, INK, "bold", family="DejaVu Sans Mono")
    put(64, 574, text["tag_submitted"], 13, MUTED)

    # right: one stage per side
    size, top = 318, 150
    for i, side in enumerate(("world", "real")):
        left = 494 + i * (size + 30)
        res, cnt = runs[side], program[side]
        put(left, 92, text[f"{side}_title"], 17, INK, "bold")
        put(left, 124, text["result_template"].format(k=cnt["k"], n=cnt["n"]), 15.5, INK, "bold")
        put(left + size, 124, f"{100 * cnt['k'] / cnt['n']:.0f}%", 12.5, MUTED, family="DejaVu Sans Mono", ha="right")

        ax = fig.add_axes([px(left), py(top + size), size / W, size / H])
        geo = res["geometry"]
        gx, gy = geo["goal"][0], geo["goal"][1]
        ax.set_xlim(gx - VIEW, gx + VIEW)                   # the middle of the table, around the goal
        ax.set_ylim(gy + VIEW, gy - VIEW)                   # y downward, as the panel draws it
        ax.set_facecolor(SURFACE)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(RULE); sp.set_linewidth(1.2)
        traj = res["first"]["trajectory"]
        last = traj[-1]
        solved = bool(res["first"]["success"])
        for poly in geo["block"]:
            ax.add_patch(Polygon(placed(poly, geo["goal"]), closed=True, facecolor=GOOD, alpha=0.14, edgecolor="none"))
            ax.add_patch(Polygon(placed(poly, geo["goal"]), closed=True, fill=False, edgecolor=GOOD, linewidth=1.6,
                                 linestyle=(0, (4, 3))))
            ax.add_patch(Polygon(placed(poly, last[2:5]), closed=True, facecolor=GOOD if solved else MUTED,
                                 alpha=0.86, edgecolor="none"))
        tail = traj[-45:]
        ax.plot([r[0] for r in tail], [r[1] for r in tail], color=ACCENT, alpha=0.45, linewidth=1.4, solid_capstyle="round")
        ax.add_patch(Circle((last[0], last[1]), geo["pusher_radius"], facecolor=ACCENT, edgecolor="none"))

        n_ep = EPISODES[side][1]
        put(left, top + size + 30, text["episode_template"].format(i=n_ep, n=cnt["n"]), 12.5, INK, "bold")
        outcome = (text["outcome_success"].format(t=f"{res['first']['n_steps'] / 10:.1f}") if solved
                   else text["outcome_fail"])
        put(left, top + size + 52, outcome, 12.5, GOOD if solved else BAD)

    out = HERE / "thumbnail.jpg"
    fig.savefig(out, dpi=100, facecolor=BG, pil_kwargs={"quality": 92})
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
