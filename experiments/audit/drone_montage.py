"""Side-by-side GIFs of the drone pilots flying the SAME real episode (mission A, evaluation seed 1000).

Run by the orchestrator.  Every panel replays a real evaluation flight from its step log through the lab's
renderer (PyBullet TinyRenderer); the hidden truth (wind jets on the floor, damaged rotor marker) is revealed
here for the viewer -- no pilot ever saw it.  Labels carry the 50-episode real evaluation numbers.

usage: python experiments/audit/drone_montage.py
  -> runs/drone/gifs/drone_showdown.{mp4,gif}, drone_topdown_A.gif, drone_topdown_B.gif
"""
import glob
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

from reallab.drone.render import load_log, render_episode_gif, render_topdown_gif

SEED, FPS = 1000, 15
OUT, RAW, RES = Path("runs/drone/gifs"), Path("runs/drone/montage_raw"), Path("runs/drone/results")


def rec(scenario, controller):
    r = json.loads((RES / f"{scenario}__{controller}.json").read_text())
    t = r["other"].get("time_success_s", {}).get("mean")
    return r["eval"]["mean"], t


def stat(scenario, controller):
    s, t = rec(scenario, controller)
    return f"50 局成功率 {s * 100:.0f}%" + (f" · 平均 {t:.1f} s" if t else "")


def log_of(folder):
    return sorted(glob.glob(f"{folder}/*_{SEED}.jsonl.gz"))[0]


PANELS = [
    dict(key="stock", title="出厂自动驾驶（预置原语）", lives="不学习", stat="50 局成功率 0%",
         log="runs/drone/logs/site_nominal_autopilot_A"),
    dict(key="ppo5", title="直接 PPO（无世界模型）", lives="5 条命", stat=stat("site_A", "ppo_real_5lives"),
         log="runs/drone/baselines_rl/evlogs/ppo_real_5lives"),
    dict(key="nwm", title="神经网络世界模型 + 规划", lives="5 条命", stat=stat("site_A", "neural_wm_5lives"),
         log="runs/drone/baselines_rl/evlogs/neural_wm_5lives"),
    dict(key="aspire", title="ASPIRE 式 · 代码即策略", lives="5 条命", stat=stat("site_A", "aspire_style"),
         log="runs/drone/aspire_style/eval_logs_aspire_style"),
    dict(key="code_wm", title="Code WM · 代码即世界模型", lives="4 条命", stat=stat("site_A", "code_wm"),
         log="runs/drone/code_wm/eval_logs/code_wm"),
    dict(key="ppo_big", title="直接 PPO（不限命数）", lives="14,811 条命", stat=stat("site_A", "ppo_real_14811lives"),
         log="runs/drone/baselines_rl/evlogs/ppo_real_14811lives"),
]
OUTCOME = {"success": "完成", "collision": "坠毁", "timeout": "超时", "out_of_bounds": "出界", "unknown": "结束"}


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
    clips = []
    for p in PANELS:
        path = log_of(p["log"])
        log = load_log(path)
        gif = RAW / f"{p['key']}.gif"
        render_episode_gif(log, str(gif), title="", fps=FPS, size=(400, 300), speed=1.0, reveal_truth=True, max_mb=40.0)
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(gif))]
        clips.append(dict(p, frames=frames, outcome=log["outcome"], dur=(len(log["pos"]) - 1) / 30.0, path=path))
        print(p["key"], log["outcome"], round(clips[-1]["dur"], 2), "s", len(frames), "frames", flush=True)

    fw, fh = clips[0]["frames"][0].size
    cols, pad, bar, head, foot = 3, 8, 66, 40, 30
    rows = -(-len(clips) // cols)
    W, H = cols * fw + (cols + 1) * pad, head + rows * (bar + fh) + (rows + 1) * pad + foot
    f_big, f_mid, f_small = font(17), font(14), font(12)
    n = max(len(c["frames"]) for c in clips)
    out = []
    for k in range(n + FPS * 3):
        kk = min(k, n - 1)
        canvas = Image.new("RGB", (W, H), (16, 18, 22))
        d = ImageDraw.Draw(canvas)
        d.text((pad, 9), f"港区峡谷 · 17 秒内切断三处中继 · 真实物理（PyBullet）· 同一局（种子 {SEED}）· 实时播放",
               font=f_big, fill=(235, 235, 235))
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
                d.text((x + 6, y + 45), f"飞行中 {kk / FPS:.1f} s", font=f_small, fill=(220, 220, 220))
        d.text((pad, H - foot + 6), "地面色块 = 隐藏的横风区，洋红标记 = 受损旋翼：只为观众揭示，飞手从未被告知，只能从自己的黑匣子日志里推断。",
               font=f_small, fill=(150, 150, 150))
        out.append(canvas)

    with imageio.get_writer(OUT / "drone_showdown.mp4", fps=FPS, macro_block_size=1) as w:
        for im in out:
            w.append_data(np.asarray(im))
    scale, step = 0.75, 1
    while True:
        sel = [im.resize((int(W * scale), int(H * scale)), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=96)
               for im in out[::step]]
        sel[0].save(OUT / "drone_showdown.gif", save_all=True, append_images=sel[1:],
                    duration=int(round(1000 / FPS * step / 10)) * 10, loop=0, optimize=True)
        size = (OUT / "drone_showdown.gif").stat().st_size
        print("gif", size, "bytes scale", scale, "step", step, flush=True)
        if size <= 9_000_000 or step >= 4:
            break
        step += 1
    out[n + 5].save(OUT / "drone_showdown_last.png")
    out[min(n - 1, int(4.5 * FPS))].save(OUT / "drone_showdown_mid.png")

    # top-down comparisons with the truth revealed
    order = ["stock", "ppo5", "nwm", "aspire", "code_wm", "ppo_big"]
    by = {c["key"]: c for c in clips}
    render_topdown_gif([by[k]["path"] for k in order], str(OUT / "drone_topdown_A.gif"),
                       labels=["stock autopilot", "PPO, 5 lives", "neural WM, 5 lives", "ASPIRE-style, 5 lives",
                               "Code WM, 4 lives", "PPO, 14,811 lives"],
                       title=f"mission A, seed {SEED} (hidden wind jets revealed)", speed=1.5, reveal_truth=True)
    b_logs = [log_of("runs/drone/logs/site_nominal_autopilot_B"),
              log_of("runs/drone/aspire_style/eval_logs_aspire_style_B"),
              log_of("runs/drone/code_wm/eval_logs/code_wm_B")]
    render_topdown_gif(b_logs, str(OUT / "drone_topdown_B.gif"),
                       labels=["stock autopilot", "ASPIRE-style, 0 extra lives", "Code WM, 0 extra lives"],
                       title=f"mission B (new route, never flown before), seed {SEED}", speed=1.5, reveal_truth=True)
    print("done")


if __name__ == "__main__":
    main()
