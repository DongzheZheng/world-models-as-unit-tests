import json, os
H = {
 "written_before_first_life": True,
 "what_it_is": ("A flexible rope/cable (or chain-like strand) about 0.70 m long and ~1 cm thick lying flat on a table, "
   "one end gripped by the tool (point 20), the other end free (point 1). The 20 tracked points are markers along it. "
   "The post is a short vertical peg (r 0.03 m, h 0.08 m) standing on the table out of the tool's reach, and the target "
   "is a disc on the table directly behind the post as seen from the middle of the tool zone. The task is to throw/cast "
   "the rope (a whip / fly-cast style flick) so that its free end slides out past the post and comes to rest inside the "
   "target disc, with the tool parked in its zone."),
 "evidence": [
   "Consecutive tracked points are 0.03666-0.03685 m apart in 3-d (0.70/19 = 0.03684): inextensible or very stiff in tension; total arc 0.699-0.700 m.",
   "Turning angle between consecutive segments up to 0.39 rad, total turning between -3.65 and +3.41 rad, end-to-end chord 0.20-0.70 m: the strand bends freely (curled, S-shapes), so it is flexible in bending.",
   "All points at z = 0.0049-0.0051 m in every snapshot: it lies on the table; 0.005 m is consistent with the centre-line of a ~1 cm diameter rope resting on z=0.",
   "Point 20 coincides with the tool to 2e-6 m: that end is clamped to the tool.",
   "Rope centroid lies on average 0.16 m on the -x side of the tool (range -0.33 .. +0.24): it mostly lies behind the tool, away from the post and target.",
   "Post centre is 0.400-0.559 m from the zone centre (0.425, 0) and 0.20-0.43 m beyond the nearest zone point; target centre is exactly 0.12 m from the post, and the direction post->target is exactly radial from the zone centre (angle error < 3e-6 deg): the target sits straight behind the post as seen from the zone centre.",
   "Target is 0.32-0.55 m from the nearest point of the tool zone and 0.40-0.83 m from the initial tool position; the rope is 0.70 m long, so the target is reachable only with the rope well stretched out ahead of the tool, while the tool is confined to a 0.25 x 0.40 m zone.",
   "Initial point-1-to-target distance 0.26-1.24 m (mean 0.88): the free end never starts in the disc.",
   "A straight line from the zone centre to the target runs through the post (lateral offset 0, post radius 0.03): a rope laid straight towards the target along that line would hit the post; it has to pass beside it (allowed on either side) or be thrown so it bends around/over it."
 ],
 "what_the_field_note_means": {
   "you get one go at it": ("Once the rope has been thrown out and is lying on the table, the tool can only pull it back towards the zone (it is a rope: it cannot be pushed); "
       "any second attempt drags the free end away from the target and the rope may snag on the post. So the program must make one well-aimed throw and then hold still."),
   "doing it slowly gets you nowhere": ("Table friction: when the tool moves slowly the rope just trails behind the tool along its path and the free end never gets beyond the tool zone "
       "(the tool cannot reach the target). Only a fast stroke that gives the rope momentum, then stops, lets the rope slide/unroll out ahead of the tool (whip/fly-cast loop) "
       "before friction stops it."),
   "a small change in how you do it changes where it ends up": ("The outcome is sensitive to the stroke (speed, stop point, direction, timing and the rope's initial curl): "
       "the dynamics of a flung flexible rope with friction and possible contact with the post are close to chaotic, so the landing spot scatters; "
       "the program should rely on robust mechanisms (e.g. the tether: a fully straightened rope ends exactly 0.70 m from the tool) rather than fine-tuned speeds.")
 },
 "unknowns": [
   "Friction coefficient between rope and table (sliding deceleration), and whether the rope rolls/lifts off the table in a fast flick (z). Life 1 trace: deceleration of points after the tool stops; z excursions.",
   "Whether a single fast forward stroke actually throws the free end out ahead of the tool (rope unrolling past the tool) and how far: ratio of tip travel to stroke speed. Life 1-2.",
   "Whether the rope rebounds after it straightens against the tether, and how long it takes to come to rest (the rest threshold 0.02 m/s is strict; time limit 4.72 s). Life 1 trace.",
   "How the rope behaves at the post: snag/wrap, slide around, or go over the 0.08 m post. Lives 2-4, by aiming the throw beside the post.",
   "How much the initial curl of the rope matters, and whether a preparatory slow drag (to straighten the rope behind the tool) makes throws repeatable. Compared across lives on different seeds.",
   "Mass of the rope / whether the 60 N servo force limit makes the tool lag the reference in a hard stroke (tool_pos vs command in the trace)."
 ]
}
os.makedirs("runs/annex/aspire_style", exist_ok=True)
p = "runs/annex/aspire_style/hypothesis.json"
assert not os.path.exists(p), "hypothesis already exists"
json.dump(H, open(p, "w"), indent=1)
print("wrote", p)
