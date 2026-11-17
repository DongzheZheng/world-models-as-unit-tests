"""Forensics on the cable task, from real step logs of the evaluation episodes (seeds 1000-1049).  Orchestrator only.

Questions:
 1. Did the code-as-policy program really route the cable around the posts, or scrape past the geometric success rule?
    -> final winding angle about every post against the pi/2 threshold; whether the cable touches each post at the end.
 2. Why is its peak load ~5 N when the constant-speed program overloads?  -> speeds, time spent above 4 N, clearance.
 3. Why does the lab's "informed" controller only reach 78%?  -> its failure split and speed.
 4. How would the pilots' programs fare under a hard time limit?  -> share of episodes finished within 170 / 240 steps.
 5. How different are the episodes from one another?  -> required-side patterns, weaving, wrap of the taut route.
usage: python experiments/audit/station_forensics.py  -> runs/probe/station_forensics.json (and a printed table)
"""
import glob
import json
from pathlib import Path

import numpy as np

SETS = {
    "constant_speed_15cm_s": "runs/station/audit_logs/gif_naive",
    "code_as_policy_v3": "runs/station/aspire_style/eval_logs",
    "code_world_controller": "runs/station/code_wm/eval_logs",
    "lab_informed": "runs/station/_lab/logs/informed",
}
POST_R, CABLE_R = 0.03, 0.004


def winding(pts, posts):
    out = []
    for c in posts:
        a = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        out.append(((np.diff(a) + np.pi) % (2 * np.pi) - np.pi).sum())
    return np.array(out)


def seg_dist(pts, c):
    a, b = pts[:-1, :2], pts[1:, :2]
    ab = b - a
    t = np.clip(((c - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return np.linalg.norm(a + t[:, None] * ab - c, axis=1).min()


def episode(path):
    d = np.load(path, allow_pickle=False)
    pts, posts, sides = d["points"], d["posts"][0], d["post_sides"][0]
    load, tool = np.asarray(d["load"], float), d["tool_pos"]
    T = len(d["actions"])
    W = winding(pts[-1][:, :2], posts)
    margin = sides * W - np.pi / 2                       # >= 0 means the side rule is met for that post
    touch_end = [seg_dist(pts[-1], c) < POST_R + CABLE_R + 0.006 for c in posts]
    touch_any = np.zeros(3, bool)
    max_touch = 0
    for i in range(0, T + 1, 4):
        tt = np.array([seg_dist(pts[i], c) < POST_R + CABLE_R + 0.006 for c in posts])
        touch_any |= tt
        max_touch = max(max_touch, int(tt.sum()))
    step = np.linalg.norm(np.diff(tool, axis=0), axis=1)
    overload = bool(load.max() > 10.0)
    dist = float(np.linalg.norm(tool[-1] - d["target"][0]))
    ok = (not overload) and T < 600 and bool((margin >= 0).all()) and dist <= 0.02
    order = np.argsort(posts[:, 0])
    s = sides[order]
    return dict(seed=int(d["seed"]), T=T, ok=ok, overload=overload, timeout=bool(T >= 600 and not overload),
                peak=float(load.max()), frac_above4=float((load > 4).mean()), frac_above7=float((load > 7).mean()),
                mean_speed=float(step.sum() / (T * 0.05)), max_speed=float(step.max() / 0.05),
                path_len=float(step.sum()), min_margin=float(margin.min()), absW=[float(abs(w)) for w in W],
                touch_end=int(sum(touch_end)), touch_any=int(touch_any.sum()), max_touch=max_touch,
                min_tool_post=float(min(np.linalg.norm(tool - c, axis=1).min() for c in posts)),
                pattern="".join("+" if v > 0 else "-" for v in s), weave=int((np.diff(s) != 0).sum()),
                final_sides_ok=bool((margin >= 0).all()), final_dist=dist)


def main():
    out = {}
    for name, folder in SETS.items():
        files = sorted(glob.glob(f"{folder}/ep_1*.npz"))
        files = [f for f in files if 1000 <= int(Path(f).stem.split("_")[1]) < 1050]
        if not files:
            print(name, "no logs")
            continue
        E = [episode(f) for f in files]
        ok = [e for e in E if e["ok"]]
        q = lambda k, src=E: [e[k] for e in src]
        out[name] = dict(
            n=len(E), success=len(ok) / len(E), overload=sum(q("overload")) / len(E), timeout=sum(q("timeout")) / len(E),
            wrong_side_at_end=sum((not e["final_sides_ok"]) and not e["overload"] for e in E) / len(E),
            steps_mean=float(np.mean(q("T"))), steps_max=int(np.max(q("T"))),
            done_within_170=sum(e["ok"] and e["T"] <= 170 for e in E) / len(E),
            done_within_240=sum(e["ok"] and e["T"] <= 240 for e in E) / len(E),
            peak_mean=float(np.mean(q("peak"))), peak_max=float(np.max(q("peak"))),
            frac_time_above_4N=float(np.mean(q("frac_above4"))), frac_time_above_7N=float(np.mean(q("frac_above7"))),
            mean_speed=float(np.mean(q("mean_speed"))), max_speed=float(np.max(q("max_speed"))),
            path_len_mean=float(np.mean(q("path_len"))), min_tool_post_mean=float(np.mean(q("min_tool_post"))),
            # among successes: how far past the pi/2 threshold is the least-wound post, and does the cable touch posts
            succ_min_margin_min=float(np.min(q("min_margin", ok))) if ok else None,
            succ_min_margin_median=float(np.median(q("min_margin", ok))) if ok else None,
            succ_absW_mean=float(np.mean([w for e in ok for w in e["absW"]])) if ok else None,
            succ_absW_min=float(np.min([w for e in ok for w in e["absW"]])) if ok else None,
            succ_posts_touched_at_end_mean=float(np.mean(q("touch_end", ok))) if ok else None,
            succ_posts_touched_ever_mean=float(np.mean(q("touch_any", ok))) if ok else None,
            succ_max_simultaneous_touch_mean=float(np.mean(q("max_touch", ok))) if ok else None,
            per_seed={e["seed"]: [int(e["ok"]), e["T"], round(e["peak"], 2), e["pattern"], e["weave"]] for e in E})
    # variety of the 50 evaluation situations (from any complete set) and how the constant-speed program fares by type
    ref = out.get("constant_speed_15cm_s") or next(iter(out.values()))
    pat = {}
    for seed, (ok, T, peak, p, w) in ref["per_seed"].items():
        pat.setdefault(p, []).append(ok)
    cap = out.get("code_as_policy_v3", {}).get("per_seed", {})
    variety = dict(patterns={p: dict(n=len(v), constant_speed_success=float(np.mean(v))) for p, v in sorted(pat.items())},
                   weave_counts={str(k): sum(1 for v in ref["per_seed"].values() if v[4] == k) for k in (0, 1, 2)},
                   constant_speed_success_by_weave={str(k): float(np.mean([v[0] for v in ref["per_seed"].values() if v[4] == k] or [np.nan])) for k in (0, 1, 2)},
                   cap_steps_by_weave={str(k): float(np.mean([v[1] for v in cap.values() if v[4] == k] or [np.nan])) for k in (0, 1, 2)},
                   cap_peak_by_weave={str(k): float(np.mean([v[2] for v in cap.values() if v[4] == k] or [np.nan])) for k in (0, 1, 2)})
    L = np.load("runs/station/observation/layouts.npz")
    posts, tgt = L["posts"], L["target"]
    variety["snapshots_200"] = dict(post_x=[float(posts[..., 0].min()), float(posts[..., 0].max())],
                                    post_y=[float(posts[..., 1].min()), float(posts[..., 1].max())],
                                    target_x=[float(tgt[:, 0].min()), float(tgt[:, 0].max())],
                                    target_y=[float(tgt[:, 1].min()), float(tgt[:, 1].max())],
                                    post_y_std=float(posts[..., 1].std()), post_x_std=float(posts[..., 0].std()))
    out["variety"] = variety
    Path("runs/probe/station_forensics.json").write_text(json.dumps(out, indent=1))
    for name in SETS:
        if name in out:
            r = out[name]
            print(f"\n== {name}: success {r['success']:.2f} overload {r['overload']:.2f} timeout {r['timeout']:.2f} wrong-side-at-end {r['wrong_side_at_end']:.2f}")
            print(f"   steps mean {r['steps_mean']:.0f} max {r['steps_max']} | done within 170: {r['done_within_170']:.2f}  within 240: {r['done_within_240']:.2f}")
            print(f"   peak load mean {r['peak_mean']:.2f} max {r['peak_max']:.2f} | time above 4 N {r['frac_time_above_4N']:.2f}  above 7 N {r['frac_time_above_7N']:.3f}")
            print(f"   tool speed mean {r['mean_speed']:.3f} m/s (max step speed {r['max_speed']:.2f}) | path {r['path_len_mean']:.2f} m | closest tool-post distance {r['min_tool_post_mean']:.3f} m")
            if r["succ_absW_mean"] is not None:
                print(f"   successes: |W| mean {r['succ_absW_mean']:.2f} rad, min {r['succ_absW_min']:.2f} (threshold 1.57); least margin over threshold: min {r['succ_min_margin_min']:.2f}, median {r['succ_min_margin_median']:.2f}")
                print(f"   successes: posts touched by the cable at the end {r['succ_posts_touched_at_end_mean']:.2f}/3, ever {r['succ_posts_touched_ever_mean']:.2f}/3, most at once {r['succ_max_simultaneous_touch_mean']:.2f}")
    print("\n== variety", json.dumps(variety, indent=1))


if __name__ == "__main__":
    main()
