"""Mission C (other aircraft in the yard): independent audit on unseen seeds, then a side-by-side GIF.

Run by the orchestrator.  Audit: the pilots' final programs, 100 real episodes on seeds 5000-5099.
GIF: four controllers on the same real episode (evaluation seed 1000), replayed from their step logs through
the lab's renderer with the hidden truth revealed for the viewer (other aircraft, wind zones, damaged rotor).

usage: python experiments/audit/drone_c.py [audit|montage|both]
"""
import glob
import json
import statistics
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

from reallab.drone.parallel import evaluate
from reallab.drone.render import load_log, render_episode_gif, render_topdown_gif

SEED, FPS = 1000, 15
OUT, RAW, RES = Path("runs/drone/gifs"), Path("runs/drone/montage_raw_C"), Path("runs/drone/results")
CODE_WM_KW = json.loads(Path("runs/drone/code_wm/cfg/eval_code_wm_C.json").read_text())["pilot_kwargs"]
CASES = [
    ("aspire_style_C_5lives", "pilots.aspire_drone.policy_v11:make_policy", {}),
    ("aspire_style_C_20lives", "pilots.aspire_drone.policy_v14:make_policy", {}),
    ("code_wm_C", "experiments.drone.code_wm.controller:make_pilot", CODE_WM_KW),
]
OUTCOME = {"success": "completed", "collision": "crashed", "timeout": "timed out", "out_of_bounds": "out of bounds", "unknown": "ended"}


def audit():
    path = Path("runs/drone/audit.json")
    out = json.loads(path.read_text()) if path.exists() else {}
    for name, factory, kw in CASES:
        rows = evaluate(factory, kw, "site", seeds=list(range(5000, 5100)), purpose="eval", tag=f"audit_{name}",
                        n_workers=16, log_dir=f"runs/drone/audit_logs/{name}", pilot="audit", mission="C",
                        command_mode="ctbr")
        ok = [r for r in rows if r["success"]]
        out[name] = dict(n=len(rows), success=len(ok) / len(rows),
                         time_success_s=statistics.fmean(r["time_s"] for r in ok) if ok else None,
                         collision=sum(bool(r["collision"]) for r in rows), timeout=sum(bool(r["timeout"]) for r in rows),
                         out_of_bounds=sum(bool(r["out_of_bounds"]) for r in rows))
        print(name, out[name], flush=True)
        path.write_text(json.dumps(out, indent=1))


def stat(controller):
    r = json.loads((RES / f"site_C__{controller}.json").read_text())
    t = r["other"].get("time_success_s", {}).get("mean")
    return f"{r['eval']['mean'] * 100:.0f}% of 100" + (f" · mean {t:.1f} s" if t else "")


def font(size):
    for pat in ("/usr/share/fonts/**/NotoSansCJK*.ttc", "/usr/share/fonts/**/NotoSerifCJK*.ttc",
                "/usr/share/fonts/**/*CJK*.tt*", "/usr/share/fonts/**/DejaVuSans.ttf"):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return ImageFont.truetype(hits[0], size)
    return ImageFont.load_default()


def montage():
    panels = [
        dict(key="stock", title="Stock autopilot (built-in primitive)", lives="no learning", stat="0% of 100",
             log="runs/drone/logs/site_nominal_autopilot_C"),
        dict(key="aspire5", title="Code as Policies agent (ASPIRE-style)", lives="5 real trials on mission C", stat=stat("aspire_style"),
             log="runs/drone/aspire_style/eval_logs_aspire_style_C"),
        dict(key="aspire20", title="Code as Policies agent, four times the trials", lives="20 real trials on mission C", stat=stat("aspire_style_extended"),
             log="runs/drone/aspire_style/eval_logs_aspire_style_extended_C"),
        dict(key="code_wm", title="Code-as-world-model agent", lives="5 real trials on mission C", stat=stat("code_wm"),
             log="runs/drone/code_wm/eval_logs/code_wm_C"),
    ]
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    clips = []
    for p in panels:
        path = sorted(glob.glob(f"{p['log']}/*_{SEED}.jsonl.gz"))[0]
        log = load_log(path)
        gif = RAW / f"{p['key']}.gif"
        render_episode_gif(path, str(gif), title="", fps=FPS, size=(400, 300), speed=1.0, reveal_truth=True, max_mb=40.0)
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(gif))]
        clips.append(dict(p, frames=frames, outcome=log["outcome"], dur=(len(log["pos"]) - 1) / 30.0, path=path))
        print(p["key"], log["outcome"], round(clips[-1]["dur"], 2), "s", len(frames), "frames", flush=True)

    fw, fh = clips[0]["frames"][0].size
    cols, pad, bar, head, foot = 2, 8, 66, 40, 30
    rows = -(-len(clips) // cols)
    W, H = cols * fw + (cols + 1) * pad, head + rows * (bar + fh) + (rows + 1) * pad + foot
    f_big, f_mid, f_small = font(16), font(14), font(12)
    n = max(len(c["frames"]) for c in clips)
    out = []
    for k in range(n + FPS * 3):
        kk = min(k, n - 1)
        canvas = Image.new("RGB", (W, H), (16, 18, 22))
        d = ImageDraw.Draw(canvas)
        d.text((pad, 10), f"Mission C: other aircraft with stochastic behaviour · real physics · same episode (seed {SEED})", font=f_big, fill=(235, 235, 235))
        for i, c in enumerate(clips):
            x = pad + (i % cols) * (fw + pad)
            y = head + pad + (i // cols) * (bar + fh + pad)
            ended = kk >= len(c["frames"]) - 1
            frame = c["frames"][min(kk, len(c["frames"]) - 1)]
            if ended and c["outcome"] != "success":
                frame = Image.blend(frame, Image.new("RGB", frame.size, (60, 0, 0)), 0.45)
            canvas.paste(frame, (x, y + bar))
            d.rectangle([x, y, x + fw, y + bar], fill=(34, 38, 46))
            d.text((x + 6, y + 4), c["title"], font=f_mid, fill=(255, 255, 255))
            d.text((x + 6, y + 25), f"{c['lives']} · {c['stat']}", font=f_small, fill=(170, 200, 255))
            if ended:
                ok = c["outcome"] == "success"
                d.text((x + 6, y + 45), f"{OUTCOME[c['outcome']]} @ {c['dur']:.1f} s", font=f_small,
                       fill=(120, 235, 140) if ok else (255, 110, 110))
            else:
                d.text((x + 6, y + 45), f"flying {kk / FPS:.1f} s", font=f_small, fill=(220, 220, 220))
        d.text((pad, H - foot + 6), "Magenta: other aircraft. Floor patches: cross-wind jets. Agents see only aircraft within 15 m, in line of sight.",
               font=f_small, fill=(150, 150, 150))
        out.append(canvas)

    with imageio.get_writer(OUT / "drone_C_showdown.mp4", fps=FPS, macro_block_size=1) as w:
        for im in out:
            w.append_data(np.asarray(im))
    scale, step = 0.9, 1
    while True:
        sel = [im.resize((int(W * scale), int(H * scale)), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=96)
               for im in out[::step]]
        sel[0].save(OUT / "drone_C_showdown.gif", save_all=True, append_images=sel[1:],
                    duration=int(round(1000 / FPS * step / 10)) * 10, loop=0, optimize=True)
        size = (OUT / "drone_C_showdown.gif").stat().st_size
        print("gif", size, "bytes scale", scale, "step", step, flush=True)
        if size <= 9_000_000 or step >= 4:
            break
        step += 1
    out[n + 5].save(OUT / "drone_C_showdown_last.png")
    out[min(n - 1, int(5.0 * FPS))].save(OUT / "drone_C_showdown_mid.png")
    render_topdown_gif([c["path"] for c in clips], str(OUT / "drone_topdown_C.gif"),
                       labels=["stock autopilot", "ASPIRE-style, 5 lives", "ASPIRE-style, 20 lives", "Code WM, 5 lives"],
                       title=f"mission C, seed {SEED} (other aircraft and wind revealed)", speed=1.5, reveal_truth=True)
    print("done")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("montage", "both"):
        montage()
    if what in ("audit", "both"):
        audit()
