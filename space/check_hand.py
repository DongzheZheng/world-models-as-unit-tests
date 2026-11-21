#!/usr/bin/env python
"""Check space/hand, the data and the videos of the screwdriver panel (space/panel_hand.html).

    python space/check_hand.py [--hand space/hand]

  totals       the successes per stage, program and seed set, the world-model totals and the success rates the
               agent predicted equal the figures of the lab's evaluation (EXPECTED below), and the success window
               and the strip limit are the task's
  consistency  every seed set has one outcome and one hold per seed, and k is its number of "ok"; the hold of an
               "ok" episode lies in the window, of a "hold_low" one below it, of a "hold_high" one above it and not
               above the strip limit; a "damaged" episode has no hold left; the world-model strings have the stated
               lengths and counts
  videos       every stage and held-out seed has its video and the folder holds nothing else; each file is an MP4
               whose index precedes the media data (it can start before it has loaded completely), with one H.264
               video track of the stated size and no audio track; all together at most 80 MB
Reads the files only; the environment is not needed.  Standard library only.  Exit status 1 if anything differs.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

WINDOW, STRIP = [0.05, 0.0862], 0.125               # N m
SETS = [("held_out", 1000, 50, True), ("retest", 1050, 50, False)]      # id, first seed, episodes, has videos
OUTCOMES = ["ok", "hold_low", "hold_high", "damaged", "tool_lost"]
# trials -> successes per seed set (held-out, re-test), world model (successes, episodes), world model from the
# held-out resets, and the success rate the world-model agent predicted for its program
EXPECTED = {
    5: dict(cap=(16, 7), wm=(35, 30), world=(174, 200), paired=(43, 50), predicted=0.60),
    15: dict(cap=(29, 27), wm=(28, 34), world=(198, 200), paired=(50, 50), predicted=0.75),
}
MAX_TOTAL_BYTES = 80_000_000


def boxes(data: bytes, start: int = 0, end: int | None = None):
    """(type, start of the payload, end of the box) for each box of an MP4 between start and end."""
    end = len(data) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack(">I4s", data[pos:pos + 8])
        head = 8
        if size == 1:
            size, head = struct.unpack(">Q", data[pos + 8:pos + 16])[0], 16
        elif size == 0:
            size = end - pos
        if size < head or pos + size > end:
            raise ValueError("broken box structure")
        yield kind.decode("latin-1"), pos + head, pos + size
        pos += size


def child(data: bytes, start: int, end: int, *path: str):
    """Payload range of the first box at the given path below start..end, or None."""
    for name in path:
        found = next(((a, b) for kind, a, b in boxes(data, start, end) if kind == name), None)
        if found is None:
            return None
        start, end = found
    return start, end


def inspect(path: Path) -> dict:
    """Layout of an MP4: whether the index comes first, its duration, and (handler, size, codec) per track."""
    data = path.read_bytes()
    top = list(boxes(data))
    kinds = [kind for kind, _, _ in top]
    if kinds[:1] != ["ftyp"] or "moov" not in kinds or "mdat" not in kinds:
        raise ValueError("not an MP4 with ftyp, moov and mdat")
    _, a, b = top[kinds.index("moov")]
    mvhd = child(data, a, b, "mvhd")
    if mvhd is None:
        raise ValueError("no mvhd")
    if data[mvhd[0]] == 1:
        timescale, duration = struct.unpack(">IQ", data[mvhd[0] + 20:mvhd[0] + 32])
    else:
        timescale, duration = struct.unpack(">II", data[mvhd[0] + 12:mvhd[0] + 20])
    tracks = []
    for kind, ta, tb in boxes(data, a, b):
        if kind != "trak":
            continue
        tkhd, hdlr = child(data, ta, tb, "tkhd"), child(data, ta, tb, "mdia", "hdlr")
        stsd = child(data, ta, tb, "mdia", "minf", "stbl", "stsd")
        width, height = struct.unpack(">II", data[tkhd[1] - 8:tkhd[1]])       # 16.16 fixed point
        tracks.append((data[hdlr[0] + 8:hdlr[0] + 12].decode("latin-1"), (width >> 16, height >> 16),
                       data[stsd[0] + 12:stsd[0] + 16].decode("latin-1")))
    return dict(index_first=kinds.index("moov") < kinds.index("mdat"), seconds=duration / timescale, tracks=tracks)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hand", default=str(HERE / "hand"))
    a = ap.parse_args()
    hand = Path(a.hand)
    index = json.loads((hand / "index.json").read_text(encoding="utf-8"))
    ok = True

    def report(good: bool, line: str):
        nonlocal ok
        ok &= bool(good)
        print(f"{line}  [{'PASS' if good else 'FAIL'}]", flush=True)

    lo, hi = index["window"]
    report(index["window"] == WINDOW and index["strip_torque"] == STRIP and index["outcomes"] == OUTCOMES
           and [(s["id"], s["seed0"], s["n"], s["video"]) for s in index["sets"]] == SETS
           and [st["trials"] for st in index["stages"]] == list(EXPECTED) and index["programs"] == ["cap", "wm"],
           f"constants: window {lo} to {hi} N m, strip limit {index['strip_torque']} N m, "
           f"{len(index['sets'])} seed sets, stages {[st['trials'] for st in index['stages']]}")

    for st in index["stages"]:
        want = EXPECTED.get(st["trials"], {})
        for program in index["programs"]:
            e = st["programs"][program]
            ks, sound, n_all = [], True, 0
            for name, _, n, _ in SETS:
                s = e["sets"][name]
                marks = [OUTCOMES[int(c)] for c in s["outcome"]]
                sound &= len(marks) == len(s["hold"]) == n and s["k"] == marks.count("ok")
                for o, h in zip(marks, s["hold"]):
                    sound &= (lo <= h <= hi if o == "ok" else h <= lo if o == "hold_low" else      # holds are stored
                              hi <= h <= STRIP if o == "hold_high" else                             # to 4 decimals
                              h == 0 if o == "damaged" else 0 <= h <= STRIP)
                ks.append(s["k"])
                n_all += n
            sound &= e["k"] == sum(ks) and e["n"] == n_all
            line = (f"{st['trials']:2d} trials  {program:3s}  environment {e['k']}/{e['n']} "
                    f"(held-out {ks[0]}/{SETS[0][2]}, re-test {ks[1]}/{SETS[1][2]})")
            good = sound and tuple(ks) == want.get(program)
            if program == "wm":
                w, p = e["world"], e["paired"]
                sound_w = (len(w["success"]) == w["n"] and w["success"].count("1") == w["k"]
                           and len(p["success"]) == p["n"] == SETS[0][2] and p["success"].count("1") == p["k"])
                good &= sound_w and (w["k"], w["n"]) == want.get("world") and (p["k"], p["n"]) == want.get("paired") \
                    and abs(e["predicted"] - want.get("predicted", -1)) < 1e-9
                line += (f"; world model {w['k']}/{w['n']}, from the held-out resets {p['k']}/{p['n']}, "
                         f"predicted by the agent {e['predicted']:.2f}")
            report(good, line)

    media = hand / "media"
    seeds = [seed0 + i for _, seed0, n, video in SETS if video for i in range(n)]
    names = [st["video"].replace("{seed}", str(seed)) for st in index["stages"] for seed in seeds]
    size = (index["video"]["width"], index["video"]["height"])
    present, sizes, wrong = 0, [], []
    for name in names:
        path = media / name
        if not path.exists():
            wrong.append(f"{name} missing")
            continue
        present += 1
        sizes.append(path.stat().st_size)
        try:
            info = inspect(path)
        except (ValueError, TypeError, struct.error) as ex:
            wrong.append(f"{name}: {ex}")
            continue
        if not (info["index_first"] and info["tracks"] == [("vide", size, "avc1")] and info["seconds"] > 1):
            wrong.append(f"{name}: index first {info['index_first']}, tracks {info['tracks']}, {info['seconds']:.1f} s")
    extra = sorted(p.name for p in media.iterdir() if p.name not in names) if media.is_dir() else []
    total = sum(sizes)
    report(present == len(names) and not wrong and not extra and total <= MAX_TOTAL_BYTES,
           f"videos: {present}/{len(names)} present ({len(index['stages'])} stages x {len(seeds)} held-out seeds), "
           f"H.264 {size[0]}x{size[1]} without audio and with the index first, {total / 1e6:.1f} MB in total"
           + (f", {min(sizes) / 1e6:.2f} to {max(sizes) / 1e6:.2f} MB each" if sizes else "")
           + (f"; problems: {'; '.join(wrong[:5])}" + (" ..." if len(wrong) > 5 else "") if wrong else "")
           + (f"; unexpected files: {', '.join(extra[:5])}" if extra else ""))
    print(f"check_hand: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
