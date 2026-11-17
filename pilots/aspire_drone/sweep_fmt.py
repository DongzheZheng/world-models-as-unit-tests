"""Format world_runner output lines compactly (stdin -> stdout)."""
import json
import sys

for line in sys.stdin:
    if line.startswith("{"):
        print(line.strip())
        continue
    if " {" not in line:
        continue
    k, _, j = line.partition(" {")
    d = json.loads("{" + j)
    print(f"{d['success']:.4f} wall {d['coll_wall']:.4f} mover {d['coll_mover']:.4f} {d['coll_mover_by_kind']} oob {d['out_of_bounds']:.4f}"
          f" t {d['time_success_s']:.2f}  <- {k}")
