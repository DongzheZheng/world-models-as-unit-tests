"""List failing imagined episodes of a candidate with their initial features.
python pilots/aspire_pusht/cw_fails.py <version> @<kwargs.json> <n> <seed0>"""
import json, sys
if __name__ == "__main__":
    from experiments.pusht.code_wm.screen import run_in_world
    from pilots.aspire_pusht.cw_screen import init_features
    kw = json.load(open(sys.argv[2][1:])) if sys.argv[2].startswith("@") else json.loads(sys.argv[2])
    n, s0 = int(sys.argv[3]), int(sys.argv[4])
    eps = run_in_world(f"pilots.aspire_pusht.policy_v{sys.argv[1]}:make_policy", kw, "L", n, s0, n_workers=12, batch=max(1, n // 12))
    F = init_features(range(s0, s0 + n))
    for e in eps:
        if not e["success"]:
            f = F[e["seed"]]; print(e["seed"], "eth %.0f wall %.0f dist %.0f maxcov %.3f" % (f["eth_deg"], f["wall"], f["dist"], e["max_coverage"]))
