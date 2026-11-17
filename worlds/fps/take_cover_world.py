"""Code world of the `take_cover` scenario (written from the code_wm pilot's black-box lives; pure torch).

Picture (all numbers from the logs unless marked PRIOR):
  A 576 x 768 room.  The player stands at x = -416 facing +x and can only strafe along y (MOVE_LEFT = +y).
  Imps stand still at x = 64 on the far side; more of them appear over time.  Each imp now and then throws a
  fireball that flies in a straight line at 10 units/tic towards where the player was at the moment of the throw.
  A hit takes 21 x U{1..8} of the player's 100 health (so about half of first hits kill).  At most 13 imps.
  Reward = +1 per tic alive.

Functional API (state = dict of tensors with leading batch dim B; nothing is mutated in place):
    s = init_state(B, device, generator=None)        # also samples the per-episode randomised parameters
    s2, reward, done = step(s, action)               # action (B,) long in {0 noop, 1 MOVE_LEFT, 2 MOVE_RIGHT}
    o = observe(s)                                   # (B, OBS_DIM) features of what is *visible* now
Real-world bridge: encode_obs_dict(obs, mem) -> (np.float32 (OBS_DIM,), mem) builds the same features from a
real observation dict (which contains only visible objects) through the same `encode` function.
TakeCoverEnv wraps all of this as an auto-resetting batched env for cawm.ppo.
"""
from __future__ import annotations

import math

import numpy as np
import torch

# ------------------------------------------------------------------ geometry
PX = -416.0          # player x, constant; life 1: x = -416.0 at every step (only strafing, angle 0)
PY0 = 384.0          # player start y; life 1 reset obs (= room centre)
Y_LO, Y_HI = 16.0, 752.0   # player y limits; life 1: stops dead at 16.0 / 752.0 (walls y=0/768, radius 16)
IMP_X = 64.0         # imp x; life 1: all 7 imps at x = 64.0, velocity 0 throughout (they never move)
IMP_Y_LO, IMP_Y_HI = 20.0, 748.0   # imp y range; life 1 spawns y in {20,61,224,278,366,513,652}; 20 = wall+radius
BALL_X_MIN = -442.0  # fireball explodes at the back wall; life 1: balls stop at x = -440..-442 (wall -448, r 6)
BALL_Y_LO, BALL_Y_HI = 6.0, 762.0  # side walls; life 1 ball #5 stopped at y=760

# ------------------------------------------------------------------ player dynamics (per tic)
TICS_PER_STEP = 4    # frame_skip of the lab
ACCEL = 0.75         # thrust per tic while a strafe button is held; fit.py least squares on life 1 (95 steps)
FRICTION = 0.9065    # velocity multiplier per tic, applied after the move; fit.py (1-step y RMSE 0.012 units)
#                      -> terminal speed ACCEL*F/(1-F) = 7.27 u/tic, life 1 observed 7.25
CONTROL_TIC = 8      # first tic at which input acts; life 1: step tics 5..8 gave v=0.68 = one tic of thrust

# ------------------------------------------------------------------ fireballs
BALL_SPEED = 10.0    # units/tic; life 1: |v| = 10.000 for all 8 balls (std 3e-6)
HIT_HALF = 23.0      # hit if |dx|<23 and |dy|<23: PRIOR ball radius 6 + player radius 16 = 22; life 1: the killing
#                      ball was at (19,25) one tic before the hit (no hit); life 2: hit registered at |dx|=22.3
HEALTH0 = 100.0      # player health at reset (manual; every life's reset obs)
DAMAGE_UNIT, DAMAGE_MAX_K = 21.0, 8  # damage per hit = 21 * U{1..8}: hits seen did 105 (life 1), 168 (life 2),
#                      21, 63, 84 (life 3): all multiples of 21, max 8x; PRIOR: Doom damage = base * 1d8
#                      (the knock-back of a hit, up to 15 units in x in life 3, is not modelled: x stays -416)
AIM_NOISE = 10.0     # std (units at the player's x) of aim error; life 1 aim residual std 20 is mostly my timing
#                      error (player moves 7 u/tic); kept small but nonzero for robustness
SPAWN_LEAD = 0.5     # ball starts half a step ahead of the imp (PRIOR: Doom P_CheckMissileSpawn)

# ------------------------------------------------------------------ imps (behaviour)
IMP0_Y = (652.0, 278.0, 20.0)  # imps present at tic 1: identical x=64, y in lives 1 and 2 (fixed, not random)
N_IMP0 = len(IMP0_Y)
SPAWN_FIRST, SPAWN_PERIOD = 61, 60   # every new imp appeared at a tic 1 + 60k, k >= 1 (lives 1-2: 10 spawns)
IMP_SPOTS = 13       # spawn attempt succeeds with prob (13 - n_imps)/13 (a random spot, fails if taken): life 3
#                      reached 13 imps and never more (18657 tics; last spawn tic 6841, spawns ever rarer);
#                      n=3 gives 0.77, lives 1-2: 10 of 15 early attempts succeeded
WAKE_DELAY = 57      # an imp cannot throw for this many tics after it appears; lives 1-2: earliest first throw 58.5
WAKE_WINDOW = 33     # then a burst: 8 of 13 imps threw within tics 57..90 of appearing: 1-exp(-33h) = 0.62
WAKE_RATE = (0.012, 0.059)  # -> h = 0.029/tic; sampled log-uniformly per episode over the binomial 95% CI of 8/13
#                             (the lab's noop baseline suggests early throws are rarer than in my 2 lives)
FIRE_RATE = 0.0068   # afterwards fireballs per imp per tic; life 3: 1492 throws / 218k imp-tics (2000-tic periods
#                      0.0058..0.0080); lives 1-2 gave 0.0048 (17 throws / 3554 imp-tics)
FIRE_RATE_SCALE = (0.65, 1.3)  # log-uniform per-episode multiplier covering both estimates
FIRE_REFRACTORY = 20  # min tics between two throws of one imp; lives 1-2 shortest same-imp interval 22 tics
#                       (PRIOR: the imp attack animation lasts ~20 tics)

N_IMP_MAX = IMP_SPOTS
BALLS_PER_IMP = 5    # flight time <= 90 tics and refractory 20 -> at most 5 balls of one imp in the air

# ------------------------------------------------------------------ observation
FOV_HALF_TAN = 1.0   # 90 deg horizontal FOV (manual)
IMP_R, BALL_R = 20.0, 6.0  # visibility margin = object radius: rule |dy| < dx + r matched the 'visible' flag of
#                            every gt object in life 1 (accuracy 1.000, fit.py)
K_BALLS = 5
N_BINS = 8
OBS_DIM = 3 + 5 * K_BALLS + N_BINS


def _u(shape, device, generator):
    return torch.rand(shape, device=device, generator=generator)


def init_state(B: int, device="cpu", generator=None) -> dict:
    NI, NB = N_IMP_MAX, N_IMP_MAX * BALLS_PER_IMP
    imp_on = torch.zeros(B, NI, dtype=torch.bool, device=device)
    imp_on[:, :N_IMP0] = True
    lo, hi = math.log(FIRE_RATE_SCALE[0]), math.log(FIRE_RATE_SCALE[1])
    imp_y0 = torch.tensor(IMP0_Y, device=device)
    d = dict(
        tic=torch.ones(B, dtype=torch.long, device=device),
        py=torch.full((B,), PY0, device=device), vy=torch.zeros(B, device=device),
        imp_on=imp_on, imp_y=IMP_Y_LO + (IMP_Y_HI - IMP_Y_LO) * _u((B, NI), device, generator),
        imp_born=torch.ones(B, NI, dtype=torch.long, device=device),
        imp_ready=torch.full((B, NI), 1 + WAKE_DELAY, dtype=torch.long, device=device),
        imp_seen=torch.zeros(B, NI, dtype=torch.bool, device=device),
        ball_seen=torch.zeros(B, NB, dtype=torch.bool, device=device),
        imp_shots=torch.zeros(B, NI, dtype=torch.long, device=device),
        n_imp=torch.full((B,), N_IMP0, dtype=torch.long, device=device),
        ball_on=torch.zeros(B, NB, dtype=torch.bool, device=device),
        bx=torch.zeros(B, NB, device=device), by=torch.zeros(B, NB, device=device),
        bvx=torch.zeros(B, NB, device=device), bvy=torch.zeros(B, NB, device=device),
        rate=FIRE_RATE * torch.exp(lo + (hi - lo) * _u((B,), device, generator)),
        wake_rate=WAKE_RATE[0] * (WAKE_RATE[1] / WAKE_RATE[0]) ** _u((B,), device, generator),
        health=torch.full((B,), HEALTH0, device=device),
        alive=torch.ones(B, dtype=torch.bool, device=device),
    )
    d["imp_y"][:, :N_IMP0] = imp_y0
    return d


def _tic(s: dict, d: torch.Tensor, generator=None) -> dict:
    """Advance one engine tic. d: (B,) float in {-1,0,1} (+1 = MOVE_LEFT = +y)."""
    s = dict(s)
    B, dev = d.shape[0], d.device
    t = s["tic"]
    # player: thrust, move, clamp at the walls (velocity zeroed), friction
    vy = s["vy"] + ACCEL * d * (t >= CONTROL_TIC).float()
    py = s["py"] + vy
    wall = (py < Y_LO) | (py > Y_HI)
    py = py.clamp(Y_LO, Y_HI)
    vy = torch.where(wall, torch.zeros_like(vy), vy) * FRICTION
    # balls fly, explode on walls, hit the player
    on = s["ball_on"]
    bx, by = s["bx"] + s["bvx"], s["by"] + s["bvy"]
    hit = on & ((bx - PX).abs() < HIT_HALF) & ((by - py[:, None]).abs() < HIT_HALF)
    on = on & (bx > BALL_X_MIN) & (by > BALL_Y_LO) & (by < BALL_Y_HI) & ~hit
    k = torch.randint(1, DAMAGE_MAX_K + 1, hit.shape, device=dev, generator=generator).float()
    health = s["health"] - DAMAGE_UNIT * (k * hit.float()).sum(1)
    s["health"] = health
    alive = s["alive"] & (health > 0)
    # imps throw: Bernoulli(rate) per ready imp, aimed at the player's current position (+ noise)
    ready = s["imp_on"] & (t[:, None] >= s["imp_ready"])
    in_win = (t[:, None] - s["imp_born"]) < WAKE_DELAY + WAKE_WINDOW
    haz = torch.where(in_win, s["wake_rate"][:, None], s["rate"][:, None])
    fire = ready & (_u(ready.shape, dev, generator) < haz)
    aim_y = py[:, None] + AIM_NOISE * torch.randn(ready.shape, device=dev, generator=generator)
    dx, dy = PX - IMP_X, aim_y - s["imp_y"]
    n = torch.sqrt(dx * dx + dy * dy)
    vx_new, vy_new = BALL_SPEED * dx / n, BALL_SPEED * dy / n
    slot = torch.arange(N_IMP_MAX, device=dev)[None] * BALLS_PER_IMP + s["imp_shots"] % BALLS_PER_IMP  # (B,NI)
    sel = torch.zeros_like(on).scatter(1, slot, fire)  # balls that are (re)launched now
    src = lambda v: torch.zeros_like(bx).scatter(1, slot, v)
    bx = torch.where(sel, src(IMP_X + SPAWN_LEAD * vx_new), bx)
    by = torch.where(sel, src(s["imp_y"] + SPAWN_LEAD * vy_new), by)
    s["bvx"] = torch.where(sel, src(vx_new), s["bvx"])
    s["bvy"] = torch.where(sel, src(vy_new), s["bvy"])
    on = on | sel
    s["ball_seen"] = s["ball_seen"] & ~sel
    s["imp_ready"] = torch.where(fire, t[:, None] + FIRE_REFRACTORY, s["imp_ready"])
    s["imp_shots"] = s["imp_shots"] + fire.long()
    # imp spawning: an attempt every SPAWN_PERIOD tics from SPAWN_FIRST
    attempt = (t >= SPAWN_FIRST) & ((t - SPAWN_FIRST) % SPAWN_PERIOD == 0) & (s["n_imp"] < N_IMP_MAX)
    ok = attempt & (_u((B,), dev, generator) < (IMP_SPOTS - s["n_imp"]).float() / IMP_SPOTS)
    new = ok[:, None] & (torch.arange(N_IMP_MAX, device=dev)[None] == s["n_imp"][:, None])
    s["imp_on"] = s["imp_on"] | new
    s["imp_ready"] = torch.where(new, t[:, None] + WAKE_DELAY, s["imp_ready"])
    s["imp_born"] = torch.where(new, t[:, None], s["imp_born"])
    s["n_imp"] = s["n_imp"] + ok.long()
    s.update(tic=t + 1, py=py, vy=vy, ball_on=on, bx=bx, by=by, alive=alive)
    return s


def step(state: dict, action: torch.Tensor, generator=None):
    """One agent step = TICS_PER_STEP tics with the action held. Returns (state, reward (B,), done (B,)).
    reward = tics alive in this step incl. the tic of death (life 2's last step paid 1, life 1's paid 4)."""
    d = torch.tensor([0.0, 1.0, -1.0], device=action.device)[action]
    s = state
    alive0 = state["alive"]
    reward = torch.zeros_like(state["py"])
    for _ in range(TICS_PER_STEP):
        reward = reward + s["alive"].float()
        s = _tic(s, d, generator)
    done = alive0 & ~s["alive"]
    # the observer's memory: anything that was visible at an observation is remembered (dead-reckoned) afterwards
    py = s["py"][:, None]
    s["ball_seen"] = s["ball_seen"] | (s["ball_on"] & visible(s["bx"] - PX, s["by"] - py, BALL_R))
    s["imp_seen"] = s["imp_seen"] | (s["imp_on"] & visible(torch.full_like(s["imp_y"], IMP_X - PX), s["imp_y"] - py, IMP_R))
    return s, reward, done


# ------------------------------------------------------------------ observation (shared with reality)
def encode(py, vy, tic, bx, by, bvx, bvy, bmask, imp_y, imask):
    """Features from VISIBLE objects only. Shapes: py,vy,tic (B,); balls (B,N) + mask; imps (B,M) + mask.
    Balls: the K soonest to reach the player's x, each (present, time-to-arrival, crossing offset, offset now, vy)."""
    B = py.shape[0]
    tau = (bx - PX) / (-bvx).clamp(min=1e-3)
    valid = bmask & (bvx < -1.0) & (tau > -2.0)
    key = torch.where(valid, tau, torch.full_like(tau, 1e9))
    k = min(K_BALLS, key.shape[1])
    order = key.topk(k, dim=1, largest=False).indices
    g = lambda v: v.gather(1, order)
    vt, tt = g(valid).float(), g(tau)
    cross = g(by) + g(bvy) * tt - py[:, None]
    feats = torch.stack([vt, vt * tt / 40.0, vt * cross / 64.0, vt * (g(by) - py[:, None]) / 256.0,
                         vt * g(bvy) / 10.0], -1).reshape(B, -1)
    if k < K_BALLS:
        feats = torch.cat([feats, torch.zeros(B, 5 * (K_BALLS - k), device=py.device)], 1)
    bins = ((imp_y / 768.0) * N_BINS).long().clamp(0, N_BINS - 1)
    hist = torch.zeros(B, N_BINS, device=py.device).scatter_add(1, bins, imask.float()) / 2.0
    base = torch.stack([(py - PY0) / 368.0, vy / 7.27, (tic.float() / 2000.0).clamp(max=2.0)], -1)  # no health:
    # (kept out of the features so that the PPO candidates trained on the earlier one-hit world still apply)
    return torch.cat([base, feats, hist], 1)


def visible(dx, dy, r):
    """In the 90-degree view cone (life 1: exact match with the engine's 'visible' flag)."""
    return (dx > 0) & (dy.abs() < dx * FOV_HALF_TAN + r)


def observe(s: dict) -> torch.Tensor:
    """Features of what the observer KNOWS: objects visible now or seen earlier (fireballs fly exactly straight,
    lives 1-2: dead-reckoning error < 0.001 units after 40 tics, so remembered positions are exact while in flight)."""
    py = s["py"][:, None]
    bk = s["ball_on"] & (s["ball_seen"] | visible(s["bx"] - PX, s["by"] - py, BALL_R))
    ik = s["imp_on"] & (s["imp_seen"] | visible(torch.full_like(s["imp_y"], IMP_X - PX), s["imp_y"] - py, IMP_R))
    return encode(s["py"], s["vy"], s["tic"], s["bx"], s["by"], s["bvx"], s["bvy"], bk, s["imp_y"], ik)


def encode_obs_dict(obs: dict, mem=None):
    """Real observation dict -> features (same encode).  obs['objects'] holds only visible objects; `mem` keeps
    what was seen before: fireballs by id (dead-reckoned, dropped once past the back/side walls) and imps by id."""
    mem = mem or dict(balls={}, imps={})
    tic, p = obs["tic"], obs["player"]
    for o in obs["objects"]:
        if "Ball" in o["name"]:
            if math.hypot(o["vx"], o["vy"]) > 1.0:          # exploded balls report v = 0
                mem["balls"][o["id"]] = (o["x"], o["y"], o["vx"], o["vy"], tic)
            else:
                mem["balls"].pop(o["id"], None)
        elif o["name"] == "DoomImp":
            mem["imps"][o["id"]] = o["y"]
    balls = []
    for bid, (x, y, vx, vy, t0) in list(mem["balls"].items()):
        x, y = x + vx * (tic - t0), y + vy * (tic - t0)
        if x <= BALL_X_MIN or y <= BALL_Y_LO or y >= BALL_Y_HI:
            del mem["balls"][bid]
        else:
            balls.append((x, y, vx, vy))
    imps = list(mem["imps"].values())
    col = lambda i: torch.tensor([[b[i] for b in balls]] if balls else [[0.0]], dtype=torch.float32)
    o = encode(torch.tensor([p["y"]], dtype=torch.float32), torch.tensor([p["vy"]], dtype=torch.float32),
               torch.tensor([tic]), col(0), col(1), col(2), col(3), torch.tensor([[bool(balls)] * max(1, len(balls))]),
               torch.tensor([imps or [0.0]], dtype=torch.float32), torch.tensor([[bool(imps)] * max(1, len(imps))]))
    return o[0].numpy().astype(np.float32), mem


# ------------------------------------------------------------------ batched env for cawm.ppo
class TakeCoverEnv:
    n_actions, obs_dim = 3, OBS_DIM

    def __init__(self, num_envs: int, device="cuda", max_steps: int = 1500, seed: int = 0):
        self.num_envs, self.device, self.max_steps = num_envs, torch.device(device), max_steps
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

    def reset(self):
        self.s = init_state(self.num_envs, self.device, self.gen)
        self.t = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return observe(self.s)

    def step(self, action):
        s, r, done = step(self.s, action, self.gen)
        self.t = self.t + 1
        tout = (self.t >= self.max_steps) & ~done
        end = done | tout
        term_obs = observe(s)
        if end.any():
            fresh = init_state(self.num_envs, self.device, self.gen)
            s = {k: torch.where(end.view(-1, *([1] * (v.dim() - 1))), fresh[k], v) for k, v in s.items()}
            self.t = torch.where(end, torch.zeros_like(self.t), self.t)
        self.s = s
        return observe(s), r / TICS_PER_STEP, end, dict(time_out=tout, terminal_obs=term_obs)
