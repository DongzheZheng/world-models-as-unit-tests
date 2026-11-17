import numpy as np
from reallab.drone.api import make
env = make("site", purpose="oracle_train", tag="probe", pilot="ppo_real")
for T in (0.30, 0.34, 0.38, 0.42):
    obs,_=env.reset(seed=1_000_100); n=0
    z=[]; 
    while True:
        obs,r,te,tru,info=env.step(np.array([0,0,0,T])); n+=1
        if n%10==0: z.append((obs["pos"].round(1).tolist(), obs["lin_vel"].round(1).tolist(), (obs["quat"][:3]).round(2).tolist()))
        if te or tru or n>150: break
    print(T, n, info.get("collision"))
    for s in z[:8]: print("   ", s)
