import time, collections, numpy as np
from reallab.doom import api
env = api.make("take_cover", purpose="smoke", pilot="ppo_real", tag="probe", seed=0)
print(env.action_names)
names = collections.Counter(); t0=time.time(); n=0
rng = np.random.default_rng(0)
for ep in range(6):
    o = env.reset(ep); first=True; L=0
    while True:
        if first or L in (20,):
            p=o["player"]; print("player", {k:round(float(p[k]),2) for k in ("x","y","angle","vx","vy","health")})
            for ob in o["objects"]: print("  ", ob["name"], round(ob["x"]), round(ob["y"]), round(ob["vx"],1), round(ob["vy"],1), ob["sx"],ob["sw"])
            first=False
        o,r,te,tr,_ = env.step(int(rng.integers(3))); n+=1; L+=1
        for ob in o["objects"]: names[ob["name"]]+=1
        if te or tr: break
    print("ep",ep,"steps",L,"hp",o["player"]["health"])
print(names, "sps", n/(time.time()-t0))
