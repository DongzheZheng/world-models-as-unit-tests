"""Run one program in the code world for given seeds and save lab-format traces (imagined; for failure analysis).
python pilots/aspire_pusht/cw_trace.py <version> '<obj_over json>' <out_dir> <seed>..."""
import json, math, sys, importlib
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    import torch
    from experiments.pusht.code_wm import screen as S
    from worlds.pusht import pusht_world as W
    from reallab.pusht import api
    ver, out = sys.argv[1], Path(sys.argv[3]); over = json.load(open(sys.argv[2][1:])) if sys.argv[2].startswith("@") else json.loads(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    W.set_object("L"); card = api.task_card(object="L")
    m = importlib.import_module(f"pilots.aspire_pusht.policy_v{ver}")
    f = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
    for s in map(int, sys.argv[4:]):
        p, q = S._init_state(np.random.default_rng(s), W, *S.RANGES["L"])
        st = W.make_state(f([p]), torch.zeros(1, 2), f([q]))
        pol = m.make_policy(obj_over=over); pol.bind(card); pol.reset()
        AP, AV, BP, C, A = [], [], [], [], []
        for t in range(301):
            pose = st["q"].double().numpy()[0]; bp = np.array([pose[0], pose[1], pose[2] % (2 * math.pi)])
            cov = api.coverage_of(tuple(pose), object="L")
            AP.append(st["p"].double().numpy()[0]); AV.append(st["pv"].double().numpy()[0]); BP.append(bp); C.append(cov)
            if cov > 0.95 or t == 300: break
            a = np.clip(pol.act(dict(agent_pos=AP[-1].copy(), agent_vel=AV[-1].copy(), block_pose=bp, goal_pose=np.array(card["goal_pose"]), coverage=cov, t=t)), 0, 512)
            A.append(a); st = W.physics(st, f([a]))
        np.savez(out / f"cw_{s}.npz", agent_pos=np.array(AP), agent_vel=np.array(AV), block_pose=np.array(BP), coverage=np.array(C),
                 actions=np.array(A).reshape(-1, 2), goal_pose=np.tile(card["goal_pose"], (len(C), 1)), success=C[-1] > 0.95,
                 max_coverage=max(C), n_steps=len(A))
        print(s, "success", C[-1] > 0.95, "max %.3f" % max(C), "steps", len(A), "init eth %.0f" % math.degrees(abs((card['goal_pose'][2] - q[2] + math.pi) % (2*math.pi) - math.pi)))
