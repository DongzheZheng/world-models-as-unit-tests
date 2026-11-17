"""English-labelled showdown GIFs for the blog.

Run by the orchestrator.  Re-uses material already recorded from the real simulators (Doom mp4s, drone step
logs, Push-T GIFs); only the L-block panels are recorded here, once, and cached.  The code worlds render
nothing: every frame comes from the simulator that plays the real world.

usage: python experiments/audit/montage_en.py [doom] [drone_a] [drone_c] [pusht] [pusht_l]
  -> runs/<demo>/gifs/<name>_en.{gif,mp4} and <name>_en_last.png
"""
import glob
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

GREEN, RED, GREY, BLUE, WHITE = (120, 235, 140), (255, 110, 110), (215, 215, 215), (170, 200, 255), (255, 255, 255)


def font(size, bold=False):
    names = (["DejaVuSans-Bold.ttf"] if bold else []) + ["DejaVuSans.ttf", "NotoSansCJK-Regular.ttc"]
    for n in names:
        hits = sorted(glob.glob(f"/usr/share/fonts/**/{n}", recursive=True))
        if hits:
            return ImageFont.truetype(hits[0], size)
    return ImageFont.load_default()


def result(demo, scenario, controller):
    return json.loads(Path(f"runs/{demo}/results/{scenario}__{controller}.json").read_text())


def compose(clips, header, footer, cols, stem, fps, hold_s=3.0, gif_scale=0.75, colors=64, max_bytes=9_000_000):
    """clips: dicts with title, sub, frames (PIL RGB, already on the output timeline), ok (bool),
    live(k) -> status while running, final -> status once the clip has ended."""
    fw, fh = clips[0]["frames"][0].size
    pad, bar, head, foot = 8, 66, 40, 30
    rows = -(-len(clips) // cols)
    W, H = cols * fw + (cols + 1) * pad, head + rows * (bar + fh) + (rows + 1) * pad + foot
    f_head, f_title, f_small = font(15, True), font(14, True), font(12)
    n = max(len(c["frames"]) for c in clips)
    out = []
    for k in range(n + int(fps * hold_s)):
        kk = min(k, n - 1)
        canvas = Image.new("RGB", (W, H), (16, 18, 22))
        d = ImageDraw.Draw(canvas)
        d.text((pad, 11), header, font=f_head, fill=(235, 235, 235))
        for i, c in enumerate(clips):
            x = pad + (i % cols) * (fw + pad)
            y = head + pad + (i // cols) * (bar + fh + pad)
            ended = kk >= len(c["frames"]) - 1
            frame = c["frames"][min(kk, len(c["frames"]) - 1)]
            if ended and not c["ok"]:
                frame = Image.blend(frame, Image.new("RGB", frame.size, (60, 0, 0)), 0.45)
            canvas.paste(frame, (x, y + bar))
            d.rectangle([x, y, x + fw, y + bar], fill=(34, 38, 46))
            d.text((x + 6, y + 5), c["title"], font=f_title, fill=WHITE)
            d.text((x + 6, y + 26), c["sub"], font=f_small, fill=BLUE)
            if ended:
                d.text((x + 6, y + 46), c["final"], font=f_small, fill=GREEN if c["ok"] else RED)
            else:
                d.text((x + 6, y + 46), c["live"](kk), font=f_small, fill=GREY)
        d.text((pad, H - foot + 7), footer, font=f_small, fill=(150, 150, 150))
        out.append(canvas)
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(f"{stem}.mp4", fps=fps, macro_block_size=1) as w:
        for im in out:
            w.append_data(np.asarray(im))
    step = 1
    while True:
        sel = [im.resize((int(W * gif_scale), int(H * gif_scale)), Image.LANCZOS)
               .convert("P", palette=Image.ADAPTIVE, colors=colors) for im in out[::step]]
        sel[0].save(f"{stem}.gif", save_all=True, append_images=sel[1:],
                    duration=int(round(1000 / fps * step / 10)) * 10, loop=0, optimize=True)
        size = Path(f"{stem}.gif").stat().st_size
        if size <= max_bytes or step >= 4:
            break
        step += 1
    out[n + 2].save(f"{stem}_last.png")
    out[n // 2].save(f"{stem}_mid.png")
    print(stem.name, "gif bytes", size, "frames", len(out), "step", step, flush=True)


# ------------------------------------------------------------------------------------------------ Doom
def doom():
    raw = Path("runs/doom/montage_raw")
    info = json.loads((raw / "info.json").read_text())
    rnd = json.loads(Path("runs/doom/baselines.json").read_text())["take_cover"]["random"]["ret"]["mean"]
    mean = lambda c: result("doom", "take_cover", c)["eval"]["mean"]
    ev = json.loads(Path("runs/doom/code_wm/eval.json").read_text())["episodes"]
    full = next(e["length_tics"] for e in ev if e["seed"] == 1000)
    spec = [("random", "Random policy", f"0 real trials · mean {rnd:,.0f} ticks"),
            ("ppo5", "PPO on real trials", f"5 real trials · mean {mean('ppo_real_5lives'):,.0f} ticks"),
            ("aspire", "Code as Policies agent (ASPIRE-style)", f"5 real trials · mean {mean('aspire_style'):,.0f} ticks"),
            ("code_wm", "Code-as-world-model agent", f"4 real trials · mean {mean('code_wm'):,.0f} ticks")]
    cap, tpf = 3000, 8
    n = cap // tpf
    clips = []
    for key, title, sub in spec:
        src = [np.asarray(f) for f in imageio.get_reader(info[key]["path"])]     # one source frame per 2 tics
        dead = info[key]["dead"]
        m = min(n, math.ceil(len(src) / 4)) if dead else n
        frames = [Image.fromarray(src[min(i * 4, len(src) - 1)]) for i in range(m)]
        final = (f"dead at tick {info[key]['tics']:,}" if dead
                 else f"alive at tick {cap:,} (episode ran {full:,} ticks)")
        clips.append(dict(title=title, sub=sub, frames=frames, ok=not dead, final=final,
                          live=lambda k: f"alive · tic {k * tpf:,}"))
    compose(clips, "Doom take_cover · real VizDoom · same episode (seed 1000) · time-lapse",
            "One real trial = one episode to learn from. Mean over 100 held-out episodes; 35 ticks per second.",
            2, "runs/doom/gifs/doom_showdown_en", fps=15)


# ------------------------------------------------------------------------------------------------ drone
def _drone_clips(panels, raw, seed=1000, fps=15):
    from reallab.drone.render import load_log, render_episode_gif
    raw.mkdir(parents=True, exist_ok=True)
    word = {"success": "completed in", "collision": "crashed at", "timeout": "timed out at",
            "out_of_bounds": "left the yard at", "unknown": "ended at"}
    clips = []
    for key, title, sub, folder in panels:
        path = sorted(glob.glob(f"{folder}/*_{seed}.jsonl.gz"))[0]
        log = load_log(path)
        gif = raw / f"{key}.gif"
        render_episode_gif(path, str(gif), title="", fps=fps, size=(400, 300), speed=1.0, reveal_truth=True, max_mb=40.0)
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(gif))]
        dur = (len(log["pos"]) - 1) / 30.0
        clips.append(dict(title=title, sub=sub, frames=frames, ok=log["outcome"] == "success",
                          final=f"{word[log['outcome']]} {dur:.1f} s", live=lambda k: f"flying · {k / fps:.1f} s"))
        print(key, log["outcome"], round(dur, 2), flush=True)
    return clips


def _drone_stat(scenario, controller, n):
    r = result("drone", scenario, controller)
    t = r["other"].get("time_success_s", {}).get("mean")
    return f"{r['eval']['mean'] * 100:.0f}% of {n}" + (f" · mean {t:.1f} s" if t else "")


def drone_a():
    s = lambda c: _drone_stat("site_A", c, 50)
    panels = [
        ("stock", "Stock autopilot (built-in primitive)", "no learning · 0% of 50", "runs/drone/logs/site_nominal_autopilot_A"),
        ("ppo5", "PPO on real trials", f"5 real trials · {s('ppo_real_5lives')}", "runs/drone/baselines_rl/evlogs/ppo_real_5lives"),
        ("nwm", "Neural dynamics model + planner", f"5 real trials · {s('neural_wm_5lives')}", "runs/drone/baselines_rl/evlogs/neural_wm_5lives"),
        ("aspire", "Code as Policies agent (ASPIRE-style)", f"5 real trials · {s('aspire_style')}", "runs/drone/aspire_style/eval_logs_aspire_style"),
        ("code_wm", "Code-as-world-model agent", f"4 real trials · {s('code_wm')}", "runs/drone/code_wm/eval_logs/code_wm"),
        ("ppo_big", "PPO on real trials, no cap", f"14,811 real trials · {s('ppo_real_14811lives')}", "runs/drone/baselines_rl/evlogs/ppo_real_14811lives"),
    ]
    compose(_drone_clips(panels, Path("runs/drone/montage_raw_en")),
            "Container-yard canyon · pass three checkpoints within 17 s · real physics (PyBullet) · same episode (seed 1000) · real time",
            "Floor patches: hidden cross-wind jets. Magenta marker: damaged rotor. Revealed for the viewer only; the agents had to infer both from their own flight logs.",
            3, "runs/drone/gifs/drone_showdown_en", fps=15, colors=96)


def drone_c():
    s = lambda c: _drone_stat("site_C", c, 100)
    panels = [
        ("stock", "Stock autopilot (built-in primitive)", "no learning · 0% of 100", "runs/drone/logs/site_nominal_autopilot_C"),
        ("aspire5", "Code as Policies agent (ASPIRE-style)", f"5 real trials on mission C · {s('aspire_style')}", "runs/drone/aspire_style/eval_logs_aspire_style_C"),
        ("aspire20", "Code as Policies agent, four times the trials", f"20 real trials on mission C · {s('aspire_style_extended')}", "runs/drone/aspire_style/eval_logs_aspire_style_extended_C"),
        ("code_wm", "Code-as-world-model agent", f"5 real trials on mission C · {s('code_wm')}", "runs/drone/code_wm/eval_logs/code_wm_C"),
    ]
    compose(_drone_clips(panels, Path("runs/drone/montage_raw_C_en")),
            "Mission C: other aircraft with stochastic behaviour · real physics · same episode (seed 1000)",
            "Magenta: other aircraft. Floor patches: cross-wind jets. Agents see only aircraft within 15 m, in line of sight.",
            2, "runs/drone/gifs/drone_C_showdown_en", fps=15, gif_scale=0.9, colors=96)


# ------------------------------------------------------------------------------------------------ Push-T / Push-L
def _push_clips(spec, raw, info, speed=2):
    clips = []
    for key, title, sub in spec:
        gif = sorted((raw / key).glob("*.gif"))[0]
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(gif))][::speed]
        e = info[key]
        final = (f"solved at step {e['n_steps']}" if e["success"]
                 else f"not solved in 300 steps (best coverage {e['max_coverage']:.2f})")
        clips.append(dict(title=title, sub=sub, frames=frames, ok=bool(e["success"]), final=final,
                          live=lambda k: f"step {k * speed}"))
    return clips


def pusht():
    raw = Path("runs/pusht/montage_raw")
    cache = raw / "info.json"
    if not cache.exists():      # outcomes printed by the recording run (experiments/audit/pusht_montage.py, seed 1000)
        cache.write_text(json.dumps({
            "random": dict(success=False, n_steps=300, max_coverage=0.007),
            "aspire5": dict(success=False, n_steps=300, max_coverage=0.937),
            "aspire13": dict(success=True, n_steps=116, max_coverage=0.955),
            "code_wm": dict(success=True, n_steps=136, max_coverage=0.957)}))
    info = json.loads(cache.read_text())
    aud = json.loads(Path("runs/pusht/audit.json").read_text())
    pct = lambda c: result("pusht", "pusht", c)["eval"]["mean"] * 100
    spec = [("random", "Random policy", "0 real trials · 0% of 50"),
            ("aspire5", "Code as Policies agent (ASPIRE-style)", f"5 real trials · {pct('aspire_style'):.0f}% (re-test {aud['aspire_v5']['success'] * 100:.0f}%)"),
            ("aspire13", "Code as Policies agent, more trials", f"13 real trials · {pct('aspire_style_extended'):.0f}% (re-test {aud['aspire_v8']['success'] * 100:.0f}%)"),
            ("code_wm", "Code-as-world-model agent", f"4 real trials · {pct('code_wm'):.0f}% (re-test {aud['code_wm']['success'] * 100:.0f}%)")]
    compose(_push_clips(spec, raw, info),
            "Push-T · real physics (pymunk) · same episode (seed 1000) · 2x speed · success = coverage > 0.95",
            "The block's centre of mass was secretly moved off-centre and it keeps gliding after release. Pilots were told only that it \"does not push the way the drawing suggests\".",
            4, "runs/pusht/gifs/pusht_showdown_en", fps=10)


def pusht_l():
    """Records the four L-block panels once (real eval episodes, pilot='montage'), then composes."""
    from reallab.pusht.parallel import evaluate
    raw = Path("runs/pusht/montage_raw_L")
    cfg = json.loads(Path("experiments/audit/pusht_l_panels.json").read_text())      # written once the pilots have reported
    cache = raw / "info.json"
    info = json.loads(cache.read_text()) if cache.exists() else {}
    for p in cfg["panels"]:
        if p["key"] in info:
            continue
        ep = evaluate(p["factory"], p.get("kw", {}), seeds=[1000], purpose="eval", pilot="montage",
                      tag=f"montage_L_{p['key']}", n_workers=1, record_first=1, video_dir=str(raw / p["key"]), object="L")[0]
        info[p["key"]] = dict(success=bool(ep["success"]), n_steps=int(ep["n_steps"]), max_coverage=float(ep["max_coverage"]))
        raw.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(info))
        print(p["key"], info[p["key"]], flush=True)
    spec = [(p["key"], p["title"], p["sub"]) for p in cfg["panels"]]
    compose(_push_clips(spec, raw, info),
            "A block never seen before (L) · two real trials to adapt · real physics · same episode (seed 1000) · 2x speed",
            cfg["footer"], 4, "runs/pusht/gifs/pusht_L_showdown_en", fps=10)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["doom", "drone_a", "drone_c", "pusht"]
    for name in todo:
        {"doom": doom, "drone_a": drone_a, "drone_c": drone_c, "pusht": pusht, "pusht_l": pusht_l}[name]()
