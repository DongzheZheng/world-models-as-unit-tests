#!/usr/bin/env python
"""Assemble the folder that is uploaded as the Space repository.

    python space/assemble.py <out_dir>

Writes into <out_dir> (new, empty, or a fresh clone of the Space repository: .git, .gitattributes, README.md):
    README.md    space/card_header.yaml as YAML front matter, with short_description from the key
                 card_short_description of space/text.json, followed by the key card_body as the body
    Dockerfile   space/Dockerfile
    .gitattributes   the videos and posters of the page are stored with Git LFS
    and the parts of the repository the Space needs (FILES below): the Space's own code, the page sources and
    media, the pushing environment, its world model, the programs, the evaluation code and the recorded results
    they are compared with.
Prints the number of files and their total size.  Standard library only.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from space.runner import N_EPISODES  # noqa: E402

SHORT_DESCRIPTION_MAX = 60      # limit of the Space card

# Patterns relative to the repository root; every pattern must match at least one file.
FILES = [
    "LICENSE",
    # the Space itself
    "space/__init__.py", "space/app.py", "space/runner.py", "space/build_page.py", "space/check_runner.py",
    "space/panel.html", "space/panel.css", "space/panel_shared.js", "space/requirements.txt",
    # the article: sources, chart fragments, media
    "blog/_head.html", "blog/_tail.html", "blog/body.html", "blog/build.py", "blog/fig_*.html",
    "site/build.py", "site/media/*",
    "runs/probe/fig_*.html",
    # recorded results shown in the panel, and the two evaluations space/check_runner.py compares with
    "runs/probe/screen_pusht.json", "runs/probe/curves.json",
    "runs/pusht/aspire_style/eval_aspire_style/episodes.json",
    "runs/pusht/aspire_style/eval_L_cap_in_code_world/episodes.json",
    # the environment, the world model, the programs, the evaluation code
    "cawm/__init__.py", "cawm/ledger.py",
    "reallab/__init__.py", "reallab/pusht/__init__.py", "reallab/pusht/api.py", "reallab/pusht/parallel.py",
    "reallab/pusht/_sim.py", "reallab/pusht/_deviations.py", "reallab/pusht/_geometry.py",
    "worlds/__init__.py", "worlds/pusht/pusht_world.py",
    "pilots/__init__.py", "pilots/aspire_pusht/__init__.py", "pilots/aspire_pusht/policy_v*.py",
    "experiments/__init__.py", "experiments/pusht/code_wm/__init__.py", "experiments/pusht/code_wm/screen.py",
]
OPTIONAL = ["space/text.json"]
KEEP = {".git", ".gitattributes", "README.md"}      # what a newly created Space repository contains
GITATTRIBUTES = "*.mp4 filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n"


def readme() -> str:
    header = (HERE / "card_header.yaml").read_text(encoding="utf-8").strip()
    text = {}
    if (HERE / "text.json").exists():
        text = {k: v.replace("{n}", str(N_EPISODES))
                for k, v in json.loads((HERE / "text.json").read_text(encoding="utf-8")).items()}
    short, body = text.get("card_short_description", "").strip(), text.get("card_body", "").strip()
    if short:
        if len(short) > SHORT_DESCRIPTION_MAX:
            raise SystemExit(f"card_short_description has {len(short)} characters; the limit is {SHORT_DESCRIPTION_MAX}")
        header += "\nshort_description: " + json.dumps(short, ensure_ascii=False)
    return "---\n" + header + "\n---\n" + ("\n" + body + "\n" if body else "")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1]).resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit("choose an output directory outside the repository")
    if out.exists() and any(p.name not in KEEP for p in out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    files = []
    for pattern in FILES + OPTIONAL:
        hits = sorted(p for p in ROOT.glob(pattern) if p.is_file())
        if not hits and pattern not in OPTIONAL:
            raise SystemExit(f"nothing matches {pattern}")
        files += hits
    out.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = out / src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    (out / "README.md").write_text(readme(), encoding="utf-8")
    (out / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    shutil.copyfile(HERE / "Dockerfile", out / "Dockerfile")
    written = [p for p in out.rglob("*") if p.is_file() and ".git" not in p.relative_to(out).parts]
    size = sum(p.stat().st_size for p in written)
    print(f"{out}: {len(written)} files, {size / 1e6:.2f} MB "
          f"(largest: {max(written, key=lambda p: p.stat().st_size).relative_to(out)}, "
          f"{max(p.stat().st_size for p in written) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
