# Writes runs/station/code_wm/hypothesis.json on the server (Step 0, before any life).
import json, os, time
H = {
 "written": time.strftime("%Y-%m-%d %H:%M:%S"),
 "what_it_is": (
  "A flexible, practically inextensible rope (cable/chain) about 0.99 m long lying flat on the table. The 20 tracked points "
  "are equally spaced markers along it (spacing ~0.052 m). Point 1 is the rope end clamped to the table at (0.05, 0); point 20 "
  "is the other end, held by the tool, which drags it. The posts are fixed pegs the rope can be routed around (and that block "
  "the tool). The task is a cable-routing task: drag the rope through the peg field so it passes each peg on the required side "
  "and park the end at the target. The 'load' reading is the pulling force the rope exerts on the tool (rope tension at the tool "
  "end), which comes from friction of the rope sliding on the table (grows with how fast and how much rope is dragged) and is "
  "amplified where the rope wraps around pegs (capstan friction)."),
 "evidence": [
  "layouts.npz: point 1 is at exactly (0.05, 0.00) in all 200 snapshots (std 0); in all 10 episode logs point 1 never moves (max displacement 0.0000 m) -> clamped end.",
  "layouts.npz: |points[19] - tool_pos| <= 1.4e-6 m -> point 20 is attached to the tool.",
  "layouts.npz: consecutive point spacing 0.0514..0.0526 m, polyline length 0.9939 +- 0.0009 m across 200 snapshots; in episodes the polyline length stays 0.988..0.994 and spacing 0.0487..0.0527 even under ~10 N -> inextensible links (strain < ~5%).",
  "z of all points = 0.0039..0.0042 m always -> lying on the table (a ~8 mm thick rope resting on z=0); never lifts.",
  "Initial winding of every snapshot |W| < 1.41 rad < ... , sides_now = 0 at reset; snapshots show the rope coiled near the clamp (tool x in [0.05, 0.32]) while posts are at x 0.38..0.72 and targets at x 0.61..0.90 -> rope must be pulled out through the pegs.",
  "Episode logs: load vs. rope kinematics (1287 steps): corr(load, sum_i |v_i|) = 0.92 (rmse 0.77 N) vs corr(load, |sum_i v_i|) = 0.78 and corr(load, tool speed) = 0.25 -> load is set by how much rope is sliding and how fast (tension accumulates along a rope dragged along its own path), not by tool motion alone.",
  "Episode logs: load / sum|v_i| = 3.9 when rope turning near pegs < 1 rad, 5.2 for 1..2 rad, 6.1 for 2..3 rad; fit load = c*sum|v_i|*exp(mu*wrap) gives c=3.87 N s/m per node, mu=0.18, rmse 0.60 N -> capstan amplification at pegs.",
  "Episode logs: node speeds decay from the tool (0.15 m/s) toward the clamp (0); nodes near the clamp stay still -> friction-dominated, overdamped rope (only the pulled part slides).",
  "episode_903 t=60..75: when the program reverses/slows, load drops from 7.6 N to 1.0 N within ~5 steps -> load follows sliding speed with little memory/inertia.",
  "episode_906: tool gets stuck on post 2 (tool-post distance 0.044-0.045 = 0.03+0.015 contact) with servo lag growing to 0.094 m, then slips past and the servo catches up at 0.74 m/s tool speed -> load 21 N, overload. episode_900 similarly ends with tool speed 1.9 m/s and load 10.8 N. 'Fights back when you hurry'.",
  "Tool lag behind command 0.0115..0.0137 m at 0.15 m/s commanded path speed; logged tool_vel magnitude ~0.12 while position increments give ~0.149 m/s (tool_vel reading under-reads steady speed by ~20% at this pace)."
 ],
 "alternatives_rejected": [
  {"alt": "an elastic band / spring between clamp and tool (load = stiffness * stretch)", "ruled_out_by": "polyline length constant within 0.6% while load varies 0.2..10 N; corr(load, end-to-end distance) is weak; load falls to ~1 N when the tool slows although the geometry is unchanged."},
  {"alt": "a chain of rigid unconnected beads / granular objects pushed by the tool", "ruled_out_by": "spacing stays 0.049..0.053 m in every log -> points are linked."},
  {"alt": "load = tool servo force / inertial force of the tool", "ruled_out_by": "corr(load, tool speed) 0.25, corr(load, acceleration) -0.04; corr(lag, load) = -0.09."},
  {"alt": "a rope hanging in the air / pendulum", "ruled_out_by": "all z = 0.004 constant; no oscillation after stops (load drops monotonically)."},
  {"alt": "pure Coulomb friction (speed independent) for the rope on the table", "ruled_out_by": "partially only: Coulomb-style direction sum explains load poorly (corr 0.18) and catch-up events at 0.7-1.9 m/s give 10-21 N vs 5-7 N at 0.15 m/s, suggesting speed dependence; but the program always moves at one speed, so viscous vs Coulomb is NOT pinned down (life 1 tests it)."}
 ],
 "mechanisms": [
  {"name": "inextensible rope, clamped at point 1, end held by the tool", "pinned": "link length 0.0523 m, clamp (0.05, 0)", "not_pinned": "bending stiffness (looks small: rope keeps curved shapes, coils)."},
  {"name": "overdamped sliding friction rope-table (only pulled nodes move; nodes stay where left)", "pinned": "node speed profiles along the rope at 0.15 m/s pulling", "not_pinned": "dependence on speed (viscous v^1 vs Coulomb v^0 vs mixed) and on direction (along vs across the rope)."},
  {"name": "load = rope tension at tool ~ c * sum_i |v_i| * exp(mu * wrap angle at pegs)", "pinned": "c ~ 3.9 N s/m per node at 0.15 m/s, mu ~ 0.18 (proxy wrap angle)", "not_pinned": "speed exponent; tension when the rope becomes taut against the clamp (never reached: end-to-end <= 0.81 m); reading at rest 0.23..0.85 N (offset or residual)."},
  {"name": "pegs: rigid cylinders r=0.03 that block rope and tool; rope wraps around them", "pinned": "contact distance 0.045 for the tool (seen in 906, 908)", "not_pinned": "rope-peg friction coefficient separately from the drag."},
  {"name": "tool servo: reference moves toward command at <=0.30 m/s, tool tracks with a force-limited PD; lags under obstruction and then catches up fast", "pinned": "lag ~0.012 m at 0.15 m/s", "not_pinned": "servo gain, force limit, behaviour after big lag (only 2 events)."}
 ],
 "why_the_existing_program_fails": (
  "It drags the rope at a fixed 0.15 m/s along a fixed-clearance waypoint path. As more rope gets set sliding and the rope wraps around "
  "pegs, the friction tension at the tool grows (~0.6 N per sliding node, amplified by wraps) and crosses 10 N mid-route (901, 902, 905, "
  "907, 909: overload at tool speed 0.14-0.15 m/s with ~15 nodes sliding). In 906 and 908 the tool snags on a peg (its fixed clearance "
  "is too small for the tool radius + peg radius on some corners), the servo lag builds up and the tool then lurches at 0.3-0.7 m/s -> load "
  "spike. 900 ends in a 1.9 m/s lurch. 903 and 904 succeed because their routes need less wrapping/less dragged rope (max load 8.4 and 9.4 N), "
  "i.e. the program is marginal everywhere: it never slows down when the load rises."),
 "unknowns_and_experiments": [
  "Never varied in observation: drag speed (always 0.15 m/s), so viscous vs Coulomb friction is unknown. Life 1: pull the rope out in a straight line with speed steps 0.02/0.05/0.10/0.15/0.20 m/s separated by slow baselines, with a load guard; also stop-and-hold to see load relaxation.",
  "Never reached: taut rope (end-to-end near 0.99 m). Life 1 end: pull slowly toward full extension with a guard, to see where tension takes off.",
  "Tool servo dynamics and peg contact: life 1 middle: press the tool gently into a peg and release; a fast slack move to see tool response and tool_vel readout.",
  "Direction dependence (dragging rope sideways vs along itself): life 1/2 lateral sweep.",
  "Lives 2-3: run the code-world controller end-to-end on snapshot seeds (refit after each). Lives 4-5: reserved for end-to-end tests / fixing what lives 2-3 reveal."
 ]
}
os.makedirs("runs/station/code_wm", exist_ok=True)
p = "runs/station/code_wm/hypothesis.json"
assert not os.path.exists(p), "hypothesis already written"
json.dump(H, open(p, "w"), indent=1)
print("wrote", p)
