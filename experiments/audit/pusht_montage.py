"""Push-T: independent audit of the code_wm controller on unseen seeds, then a side-by-side GIF of four
controllers pushing the SAME real episode (evaluation seed 1000).

Run by the orchestrator.  Frames come from the real rig's own renderer (recorded by the lab during real
purpose="eval" episodes); the code world renders nothing.  Labels carry 50-episode real evaluation numbers.

usage: python experiments/audit/pusht_montage.py [audit|montage|both]
  -> runs/pusht/audit.json, runs/pusht/gifs/pusht_showdown.{mp4,gif}
"""
import glob
import json
import statistics
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

from reallab.pusht.parallel import evaluate

SEED, FPS, SPEED = 1000, 10, 2
OUT, RAW, RES = Path("runs/pusht/gifs"), Path("runs/pusht/montage_raw"), Path("runs/pusht/results")
PLANNER = json.loads(Path("runs/pusht/code_wm/eval_code_wm.json").read_text())["cfg"]["planner"]
CODE_WM = "experiments.pusht.code_wm.controller:mpc_policy"


def stat(controller):
    r = json.loads((RES / f"pusht__{controller}.json").read_text())
    return f"{r['eval']['mean'] * 100:.0f}% of 50 · mean best coverage {r['other']['max_coverage']['mean']:.2f}"


def audit():
    path = Path("runs/pusht/audit.json")
    out = json.loads(path.read_text()) if path.exists() else {}
    eps = evaluate(CODE_WM, PLANNER, seeds=list(range(5000, 5050)), purpose="eval", pilot="audit",
                   tag="audit_code_wm", n_workers=8)
    out["code_wm"] = dict(n=len(eps), success=statistics.fmean(float(e["success"]) for e in eps),
                          max_coverage=statistics.fmean(e["max_coverage"] for e in eps),
                          n_steps=statistics.fmean(e["n_steps"] for e in eps))
    print("code_wm", out["code_wm"], flush=True)
    path.write_text(json.dumps(out, indent=1))


def font(size):
    for pat in ("/usr/share/fonts/**/NotoSansCJK*.ttc", "/usr/share/fonts/**/NotoSerifCJK*.ttc",
                "/usr/share/fonts/**/*CJK*.tt*", "/usr/share/fonts/**/DejaVuSans.ttf"):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return ImageFont.truetype(hits[0], size)
    return ImageFont.load_default()


def montage():
    base = json.loads(Path("runs/pusht/baselines.json").read_text())
    panels = [
        dict(key="random", factory="reallab.pusht.parallel:random_policy", kw={"seed": 0}, title="Random policy",
             lives="0 real trials", stat="0% of 50"),
        dict(key="aspire5", factory="pilots.aspire_pusht.policy_v5:make_policy", kw={},
             title="Code as Policies agent (ASPIRE-style)", lives="5 real trials", stat=stat("aspire_style")),
        dict(key="aspire13", factory="pilots.aspire_pusht.policy_v8:make_policy", kw={},
             title="Code as Policies agent, more trials", lives="13 real trials", stat=stat("aspire_style_extended")),
        dict(key="code_wm", factory=CODE_WM, kw=PLANNER, title="Code-as-world-model agent", lives="4 real trials",
             stat=stat("code_wm")),
    ]
    del base
    clips = []
    for p in panels:
        vdir = RAW / p["key"]
        ep = evaluate(p["factory"], p["kw"], seeds=[SEED], purpose="eval", pilot="montage", tag=f"montage_{p['key']}",
                      n_workers=1, record_first=1, video_dir=str(vdir))[0]
        gif = sorted(vdir.glob("*.gif"))[0]
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(gif))][::SPEED]
        clips.append(dict(p, frames=frames, ep=ep))
        print(p["key"], "success" if ep["success"] else "fail", ep["n_steps"], "steps", round(ep["max_coverage"], 3),
              len(frames), "frames", flush=True)

    fw, fh = clips[0]["frames"][0].size
    cols, pad, bar, head, foot = 4, 8, 66, 40, 30
    W, H = cols * fw + (cols + 1) * pad, head + bar + fh + 2 * pad + foot
    f_big, f_mid, f_small = font(17), font(14), font(12)
    n = max(len(c["frames"]) for c in clips)
    out = []
    for k in range(n + FPS * 3):
        kk = min(k, n - 1)
        canvas = Image.new("RGB", (W, H), (16, 18, 22))
        d = ImageDraw.Draw(canvas)
        d.text((pad, 9), f"Push-T · real physics (pymunk) · same episode (seed {SEED}) · {SPEED}x speed · success = coverage > 95%",
               font=f_big, fill=(235, 235, 235))
        for i, c in enumerate(clips):
            x, y = pad + i * (fw + pad), head + pad
            ended = kk >= len(c["frames"]) - 1
            frame = c["frames"][min(kk, len(c["frames"]) - 1)]
            if ended and not c["ep"]["success"]:
                frame = Image.blend(frame, Image.new("RGB", frame.size, (60, 0, 0)), 0.35)
            canvas.paste(frame, (x, y + bar))
            d.rectangle([x, y, x + fw, y + bar], fill=(34, 38, 46))
            d.text((x + 6, y + 4), c["title"], font=f_mid, fill=(255, 255, 255))
            d.text((x + 6, y + 25), f"{c['lives']} · {c['stat']}", font=f_small, fill=(170, 200, 255))
            if ended:
                if c["ep"]["success"]:
                    d.text((x + 6, y + 45), f"solved at step {c['ep']['n_steps']}", font=f_small, fill=(120, 235, 140))
                else:
                    d.text((x + 6, y + 45), f"not solved (300-step limit, best coverage {c['ep']['max_coverage']:.2f})", font=f_small,
                           fill=(255, 110, 110))
            else:
                d.text((x + 6, y + 45), f"step {kk * SPEED}", font=f_small, fill=(220, 220, 220))
        d.text((pad, H - foot + 6), "The block's centre of mass was secretly moved to one side and it keeps sliding after release; the agents were told only that it pushes differently from the drawing.",
               font=f_small, fill=(150, 150, 150))
        out.append(canvas)

    OUT.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(OUT / "pusht_showdown.mp4", fps=FPS, macro_block_size=1) as w:
        for im in out:
            w.append_data(np.asarray(im))
    scale, step = 0.75, 1
    while True:
        sel = [im.resize((int(W * scale), int(H * scale)), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=64)
               for im in out[::step]]
        sel[0].save(OUT / "pusht_showdown.gif", save_all=True, append_images=sel[1:],
                    duration=int(round(1000 / FPS * step / 10)) * 10, loop=0, optimize=True)
        size = (OUT / "pusht_showdown.gif").stat().st_size
        print("gif", size, "bytes scale", scale, "step", step, flush=True)
        if size <= 9_000_000 or step >= 4:
            break
        step += 1
    out[n + 5].save(OUT / "pusht_showdown_last.png")
    print("done")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("montage", "both"):
        montage()
    if what in ("audit", "both"):
        audit()
