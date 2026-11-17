"""Side-by-side GIF of the four Doom controllers playing the SAME real episode (same seed), time-lapsed.

Run by the orchestrator.  Each panel is a fresh real episode (purpose="eval", pilot="montage") recorded from
VizDoom's own screen buffer; nothing is rendered by the code world.  The scores in the labels are the
100-episode real evaluation means from runs/doom/results, not the single episode shown.

usage: python experiments/audit/doom_montage.py   ->  runs/doom/gifs/doom_showdown.{mp4,gif}
"""
import glob
import importlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reallab.doom import api

SEED, CAP_TICS, TICS_PER_FRAME, FPS = 1000, 3000, 8, 15
OUT = Path("runs/doom/gifs")
RAW = Path("runs/doom/montage_raw")
RES = Path("runs/doom/results")


def _mean(controller):
    return json.loads((RES / f"take_cover__{controller}.json").read_text())["eval"]["mean"]


def _full_length(path, seed):
    d = json.loads(Path(path).read_text())
    eps = d["episodes"] if isinstance(d, dict) else d
    return next(e["length_tics"] for e in eps if e["seed"] == seed)


PANELS = [
    dict(key="random", factory="reallab.doom.parallel:random_policy", kw={"seed": 0},
         title="Random policy", lives="0 real trials",
         mean=json.loads(Path("runs/doom/baselines.json").read_text())["take_cover"]["random"]["ret"]["mean"]),
    dict(key="ppo5", factory="experiments.doom.rl_baseline.ppo_real:make_policy",
         kw={"ckpt": "runs/doom/rl_baseline/ckpt/ppo_real_5lives.pt"},
         title="PPO on real trials", lives="5 real trials", mean=_mean("ppo_real_5lives")),
    dict(key="aspire", factory="pilots.aspire_doom.policy_v5:make_policy", kw={},
         title="Code as Policies agent (ASPIRE-style)", lives="5 real trials", mean=_mean("aspire_style")),
    dict(key="code_wm", factory="experiments.doom.code_wm.planner:planner",
         kw=dict(horizon=24, margin=2.0, w_center=0.3),
         title="Code-as-world-model agent", lives="4 real trials", mean=_mean("code_wm")),
]


def capture(p):
    """Play one real episode (stopped at CAP_TICS) and record VizDoom's screen to mp4."""
    path = RAW / f"{p['key']}.mp4"
    env = api.make("take_cover", "eval", tag=f"montage_{p['key']}", frame_skip=4, record_path=str(path),
                   record_format="mp4", record_title=p["key"], pilot="montage")
    mod, fn = p["factory"].split(":")
    policy = getattr(importlib.import_module(mod), fn)(**p["kw"])
    if hasattr(policy, "bind"):
        policy.bind("take_cover", env.action_names)
    if hasattr(policy, "set_seed"):
        policy.set_seed(SEED)
    policy.reset()
    obs = env.reset(SEED)
    done = False
    while not done and obs["tic"] < CAP_TICS + 16:
        obs, _, term, trunc, _ = env.step(int(policy.act(obs)))
        done = term or trunc
    env.stop_recording()
    env.close()
    return dict(dead=bool(done), tics=int(obs["tic"]), path=str(path))


def font(size):
    for pat in ("/usr/share/fonts/**/NotoSansCJK*.ttc", "/usr/share/fonts/**/NotoSerifCJK*.ttc",
                "/usr/share/fonts/**/*CJK*.tt*", "/usr/share/fonts/**/DejaVuSans.ttf"):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return ImageFont.truetype(hits[0], size)
    return ImageFont.load_default()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    full = {"code_wm": _full_length("runs/doom/code_wm/eval.json", SEED),
            "aspire": _full_length("runs/doom/aspire_style/eval_aspire_style.json", SEED)}
    clips = []
    cache = RAW / "info.json"
    infos = json.loads(cache.read_text()) if cache.exists() else {}
    for p in PANELS:
        info = infos.get(p["key"]) or capture(p)
        infos[p["key"]] = info
        cache.write_text(json.dumps(infos))
        frames = [np.asarray(f) for f in imageio.get_reader(info["path"])]
        clips.append(dict(p, **info, frames=frames))
        print(p["key"], "dead" if info["dead"] else "alive at cap", info["tics"], "frames", len(frames), flush=True)

    fh, fw = clips[0]["frames"][0].shape[:2]
    pad, bar, head, foot = 8, 66, 40, 30
    W, H = 2 * fw + 3 * pad, head + 2 * (bar + fh) + 3 * pad + foot
    f_big, f_mid, f_small = font(17), font(14), font(12)
    n_out = CAP_TICS // TICS_PER_FRAME
    src_step = TICS_PER_FRAME // 2                      # the recorder stores one frame per 2 tics
    out = []
    for k in range(n_out + FPS * 3):                    # hold the last picture for 3 s
        kk = min(k, n_out - 1)
        tic = kk * TICS_PER_FRAME
        canvas = Image.new("RGB", (W, H), (16, 18, 22))
        d = ImageDraw.Draw(canvas)
        d.text((pad, 9), f"Doom take_cover · real VizDoom · same episode (seed {SEED}) · time-lapse", font=f_big, fill=(235, 235, 235))
        for i, c in enumerate(clips):
            x = pad + (i % 2) * (fw + pad)
            y = head + (i // 2) * (bar + fh + pad)
            idx = kk * src_step
            dead = c["dead"] and idx >= len(c["frames"]) - 1
            frame = Image.fromarray(c["frames"][min(idx, len(c["frames"]) - 1)])
            if dead:
                frame = Image.blend(frame, Image.new("RGB", frame.size, (60, 0, 0)), 0.55)
            canvas.paste(frame, (x, y + bar))
            d.rectangle([x, y, x + fw, y + bar], fill=(34, 38, 46))
            d.text((x + 6, y + 4), c["title"], font=f_mid, fill=(255, 255, 255))
            d.text((x + 6, y + 25), f"{c['lives']} · mean {c['mean']:,.0f} ticks over 100 episodes", font=f_small, fill=(170, 200, 255))
            if dead:
                status, col = f"dead at tick {c['tics']}", (255, 110, 110)
            elif k >= n_out and c["key"] in full:
                status, col = f"still alive (episode ran {full[c['key']]:,} ticks)", (120, 235, 140)
            else:
                status, col = f"alive at tick {tic}", (120, 235, 140)
            d.text((x + 6, y + 45), status, font=f_small, fill=col)
        d.text((pad, H - foot + 6), "Each agent could use only the stated number of real trials; scores are mean survival over 100 held-out episodes.",
               font=f_small, fill=(150, 150, 150))
        out.append(canvas)

    with imageio.get_writer(OUT / "doom_showdown.mp4", fps=FPS, macro_block_size=1) as w:
        for im in out:
            w.append_data(np.asarray(im))
    scale, step = 0.75, 1
    while True:                                          # shrink until the GIF is shareable
        sel = [im.resize((int(W * scale), int(H * scale)), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=64)
               for im in out[::step]]
        sel[0].save(OUT / "doom_showdown.gif", save_all=True, append_images=sel[1:],
                    duration=int(round(1000 / FPS * step / 10)) * 10, loop=0, optimize=True)
        size = (OUT / "doom_showdown.gif").stat().st_size
        print("gif", size, "bytes scale", scale, "step", step, flush=True)
        if size <= 9_000_000 or step >= 4:
            break
        step += 1
    out[n_out + 5].save(OUT / "doom_showdown_last.png")
    out[n_out // 4].save(OUT / "doom_showdown_mid.png")
    print({c["key"]: (c["dead"], c["tics"]) for c in clips})


if __name__ == "__main__":
    main()
