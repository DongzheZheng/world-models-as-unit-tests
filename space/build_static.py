#!/usr/bin/env python
"""Build the static variant of the Space: the article with a panel that replays recorded runs.  No server is needed.

    python space/build_static.py <out_dir>

Writes into <out_dir> (new, empty, or a fresh clone of the Space repository: .git, .gitattributes, README.md)
everything the static Space consists of:
    index.html       the article (site/build.py) with space/panel_static.html inserted after its header; the strings
                     come from space/text_static.json (placeholders below if that file is missing) and the catalog
                     space/data/index.json is inlined
    media/           the article's videos and posters
    data/            the recorded runs, space/data (written by space/make_data.py): one file per program, fetched
                     by the page when the program is selected
    README.md        space/card_header.yaml as YAML front matter for a static Space (sdk: static, app_file:
                     index.html), with short_description from the key card_short_description of text_static.json,
                     followed by the key card_body as the body
    .gitattributes   the videos and posters are stored with Git LFS
All paths in the page are relative.  Prints the number of files and their total size.  Standard library only.
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

from space import build_page  # noqa: E402

SHORT_DESCRIPTION_MAX = 60      # limit of the Space card
KEEP = {".git", ".gitattributes", "README.md"}      # what a newly created Space repository contains
GITATTRIBUTES = "*.mp4 filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n"

# Placeholders, replaced by space/text_static.json (same keys).
FALLBACK = dict(
    kicker="Records", heading="Recorded runs", intro="Select a program, then an episode.",
    label_task="Task", task_T="Push-T", task_L="Push-L", label_program="Program",
    tag_submitted="submitted", tag_extended="extended", tag_with_tests="with tests",
    world_title="World model", world_sub="worlds/pusht/pusht_world.py", real_title="Environment",
    real_sub="reallab.pusht", result_template="{k}/{n}", hint_marks="Select a mark.",
    episode_template="Episode {i}/{n}", outcome_success="success after {t} s", outcome_fail="no success",
    label_replay="Replay", anim_caption="Goal pose and one episode.", chart_title="World model vs. environment",
    chart_x="World model", chart_y="Environment", chart_legend_all="Programs", chart_legend_selected="Selected",
    note_recorded="Recorded runs.", note_scope="Pushing tasks only.", link_repo="Repository", loading="Loading",
    load_error="Not loaded. Retry.", card_short_description="Interactive article", card_body="Interactive article.",
)


def readme(text: dict) -> str:
    """The Space card: the header of the live Space with the SDK changed to static."""
    header = []
    for line in (HERE / "card_header.yaml").read_text(encoding="utf-8").strip().splitlines():
        if line.startswith("sdk:"):
            header += ["sdk: static", "app_file: index.html"]
        elif not line.startswith("app_port:"):
            header.append(line)
    if "sdk: static" not in header:
        raise SystemExit("space/card_header.yaml has no sdk line")
    short, body = text["card_short_description"].strip(), text["card_body"].strip()
    if short:
        if len(short) > SHORT_DESCRIPTION_MAX:
            raise SystemExit(f"card_short_description has {len(short)} characters; the limit is {SHORT_DESCRIPTION_MAX}")
        header.append("short_description: " + json.dumps(short, ensure_ascii=False))
    return "---\n" + "\n".join(header) + "\n---\n" + ("\n" + body + "\n" if body else "")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1]).resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit("choose an output directory outside the repository")
    if out.exists() and any(p.name not in KEEP for p in out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    data = HERE / "data"
    if not (data / "index.json").exists():
        raise SystemExit("space/data/index.json not found: run space/make_data.py first")
    catalog = json.loads((data / "index.json").read_text(encoding="utf-8"))
    files = [data / "index.json"] + [data / p["file"] for p in catalog["programs"]]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"space/data lacks {missing}: run space/make_data.py")

    text = build_page.load_text("text_static.json", FALLBACK, n=None)
    index = build_page.build_article(out, lambda repo_url: build_page.render(
        "panel_static.html", text, repo_url=repo_url, data_json=build_page.script_json(catalog)))
    (out / ".nojekyll").unlink(missing_ok=True)         # written by site/build.py for another host
    (out / "data").mkdir()
    for src in files:
        shutil.copyfile(src, out / "data" / src.name)
    (out / "README.md").write_text(readme(text), encoding="utf-8")
    (out / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    if (HERE / "thumbnail.jpg").exists():       # the share image named by `thumbnail:` in the card header
        shutil.copyfile(HERE / "thumbnail.jpg", out / "thumbnail.jpg")      # (space/make_thumbnail.py)
    written =[p for p in out.rglob("*") if p.is_file() and ".git" not in p.relative_to(out).parts]
    size = lambda paths: sum(p.stat().st_size for p in paths) / 1e6  # noqa: E731
    print(f"wrote {index} ({index.stat().st_size} bytes) with the recorded panel")
    print(f"{out}: {len(written)} files, {size(written):.2f} MB (index.html {size([index]):.2f}, "
          f"media {size(p for p in written if p.parent.name == 'media'):.2f}, "
          f"data {size(p for p in written if p.parent.name == 'data'):.2f} MB)")


if __name__ == "__main__":
    main()
