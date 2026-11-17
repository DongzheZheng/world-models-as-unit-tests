"""Evaluate the saved oracle PPO checkpoints (50 real eval episodes each) and write ppo_curve.json."""
import json, re, sys
from experiments.drone.baselines.eval_ctrl import run

def main():
    man = json.load(open("runs/drone/baselines_rl/oracle/manifest.json"))
    seen, sel = set(), []
    for m in man:
        if m["real_episodes"] in seen:
            continue
        seen.add(m["real_episodes"]); sel.append(m)
    curve = []
    for m in sel:
        n = m["real_episodes"]
        out = run("experiments.drone.baselines.common:rl_policy", dict(ckpt=m["ckpt"]), f"ppo_real_{n}lives", "model_free_rl",
                  m["real_steps"], m["train_wall_s"], f"PPO trained directly in the real site (oracle_train), 16 parallel real workers; checkpoint after {n} real episodes / {m['real_steps']} agent steps; deterministic mean action at eval",
                  n_workers=20)
        curve.append(dict(real_episodes=n, real_steps=m["real_steps"], success_rate=out["success"], n_success=out["n_success"],
                          mean_time_success_s=out["mean_time"], mean_relays=out["relays"], train_wall_s=m["train_wall_s"], ckpt=m["ckpt"]))
        print(curve[-1], flush=True)
    # training-time (stochastic policy) success from the training log, as a finer supplementary curve
    tc = []
    try:
        ep = None
        for l in open("runs/logs/oracle_main.log"):
            m_ = re.search(r"eps (\d+) succ_train_recent (\d+)/(\d+) relays ([\d.]+)", l)
            if m_:
                tc.append(dict(real_episodes=int(m_.group(1)), train_success_last300=int(m_.group(2)) / int(m_.group(3)), mean_relays=float(m_.group(4))))
    except Exception as e:
        print("no train log", e)
    ok = [c for c in curve if c["success_rate"] >= 0.5]
    first = min(ok, key=lambda c: c["real_episodes"])["real_episodes"] if ok else None
    tfirst = next((c["real_episodes"] for c in tc if c["train_success_last300"] >= 0.5), None)
    json.dump(dict(curve=curve, smallest_evaluated_checkpoint_with_eval_success_ge_50pct=first,
                   first_log_line_with_stochastic_train_success_ge_50pct=tfirst, train_curve=tc), open("runs/drone/baselines_rl/ppo_curve.json", "w"), indent=1)
    print("lives to 50%:", first, "| train-time:", tfirst)


if __name__ == '__main__':
    main()
