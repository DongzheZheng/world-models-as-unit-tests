"""Write runs/doom/code_wm/diagnosis.json and summary.json from the world constants, logs and result files."""
import gzip
import json
from pathlib import Path

from worlds.fps import take_cover_world as W

OUT = Path("runs/doom/code_wm")
D = [
    ("room / player position", dict(player_x=W.PX, player_y0=W.PY0, y_limits=[W.Y_LO, W.Y_HI]),
     "player x=-416 and angle 0 at every step of lives 1-4 (except hit knock-back); stops dead at y=16/752; floor_plan room x[-448,128] y[0,768]", "log", "high"),
    ("strafe dynamics per tic: v+=A*dir; y+=v; clamp; v*=F", dict(A=W.ACCEL, F=W.FRICTION, control_from_tic=W.CONTROL_TIC),
     "least squares over 600 step transitions (lives 1-2): 1-step y RMSE 0.012 units; held-out 25-step error < 1.6 units (lives 2,3)", "log", "high"),
    ("imps are static at x=64", dict(x=W.IMP_X), "every imp in lives 1-4 has x=64, v=0", "log", "high"),
    ("initial imps", dict(y=list(W.IMP0_Y)), "identical in all lives", "log", "high"),
    ("imp spawning", dict(first=W.SPAWN_FIRST, period=W.SPAWN_PERIOD, success="(13-n)/13", max_imps=W.IMP_SPOTS, y="uniform [20,748]"),
     "all spawn tics = 1+60k (lives 1-3, 20 spawns); life 3 saturated at 13 imps, last spawn tic 6841; the 'random spot, fails if taken' form and uniform y are my model", "log", "medium"),
    ("imp wake delay + early burst", dict(delay=W.WAKE_DELAY, window=W.WAKE_WINDOW, rate_range=list(W.WAKE_RATE)),
     "first throws 58.5-88 tics after appearing for 8 of 13 imps (lives 1-2); rate randomised over the binomial CI; lab noop baseline suggests early throws are rarer than this", "log", "medium"),
    ("throw rate per imp", dict(rate=W.FIRE_RATE, scale_range=list(W.FIRE_RATE_SCALE), refractory=W.FIRE_REFRACTORY),
     "life 3: 1492 throws / 218k imp-tics (0.0058-0.0080 per 2000-tic period); lives 1-2: 0.0048; shortest same-imp interval 22 tics", "log", "medium"),
    ("throw is a memoryless Bernoulli process per tic", {}, "modelling assumption (PRIOR: Doom monsters attack with a random check in their chase loop); not tested beyond rates", "prior", "low"),
    ("fireball flight", dict(speed=W.BALL_SPEED, straight_line=True, spawn_lead=W.SPAWN_LEAD),
     "|v|=10.000 for all balls; dead-reckoned position error < 0.001 units after 40 tics (lives 2,3); spawn lead is prior", "log", "high"),
    ("aim at player's current position", dict(aim_noise_std=W.AIM_NOISE),
     "back-projected aim residual mean 11, std 20 units (life 1), dominated by my timing error; no lead/prediction", "log", "medium"),
    ("hit box", dict(half_width=W.HIT_HALF), "PRIOR radii 6+16=22; life 1 no hit at (19,25), life 2 hit at |dx|=22.3", "log+prior", "medium"),
    ("damage and health", dict(health=W.HEALTH0, damage="21*U{1..8}"),
     "hit damages 105,168,21,63,84 (lives 1-3) all multiples of 21 up to 8x; the uniform 1..8 law is PRIOR (Doom 1d8)", "log+prior", "medium"),
    ("hit knock-back", dict(modelled=False), "life 3: player x displaced up to -432 after hits; ignored", "log", "high"),
    ("ball explodes at walls", dict(x_min=W.BALL_X_MIN, y=[W.BALL_Y_LO, W.BALL_Y_HI]), "balls stop at x=-440..-442 / y=760 (life 1)", "log", "high"),
    ("visibility = 90 deg cone with object radius margin", dict(rule="|dy| < dx + r", r_imp=W.IMP_R, r_ball=W.BALL_R),
     "matches the engine's 'visible' flag for 100% (life 1), 99.7% (life 2), 99.9% (life 3) of gt objects; occlusion ignored", "log", "high"),
    ("reward", dict(per_tic=1, death_step="tics until death"), "returns = tics alive; life 2's death step paid 1", "log", "high"),
]
diag = [dict(name=n, value=v, evidence=e, source=s, confidence=c) for n, v, e, s, c in D]
(OUT / "diagnosis.json").write_text(json.dumps(diag, indent=1))

lives = {}
for p in sorted(Path("runs/doom/blackbox/take_cover/code_wm").glob("life_*.jsonl.gz")):
    recs = [json.loads(l) for l in gzip.open(p, "rt")]
    lives[p.name] = dict(seed=recs[0]["meta"]["seed"], steps=len(recs) - 2, ret=sum(r["reward"] for r in recs[2:]))
sel = json.loads((OUT / "selection_v3world.json").read_text())
tr = {t: json.loads((OUT / f"train_{t}.json").read_text()) for t in ("v1", "v2")}
ev = json.loads((OUT / "eval.json").read_text()) if (OUT / "eval.json").exists() else None
summary = dict(
    lives_used=len(lives), lives=lives, real_steps_total=sum(v["steps"] for v in lives.values()),
    world_file_bytes=Path(W.__file__).stat().st_size,
    final_controller=dict(kind="hand-written feature-space planner (cawm-style lookahead in the code world's player dynamics "
                               "over remembered fireballs)", params=dict(horizon=24, margin=2.0, w_center=0.3),
                          selected_by="code-world mean over 512 episodes (selection_v3world.json)"),
    code_world_score_final=sel["candidates"][sel["best"]],
    code_world_scores_all=sel["candidates"],
    ppo=dict({t: dict(wm_steps=r["wm_steps"], wall_s=r["train_wall_s"], sps=r["sps"]) for t, r in tr.items()}),
    wm_steps_note="PPO steps (v1+v2) plus planner tuning/selection rollouts (see wm_steps_total)",
    real_eval=None if ev is None else dict(mean=ev["record"]["eval"]["mean"], sem=ev["record"]["eval"]["sem"],
                                           median=ev["record"]["eval"]["median"], n=ev["record"]["eval"]["n"],
                                           min=ev["record"]["eval"]["min"], max=ev["record"]["eval"]["max"]),
    gifs=sorted(str(p) for p in (OUT / "gifs").glob("*.gif")),
)
extra = json.loads((OUT / "cost.json").read_text()) if (OUT / "cost.json").exists() else {}
summary.update(extra)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1)[:3000])
