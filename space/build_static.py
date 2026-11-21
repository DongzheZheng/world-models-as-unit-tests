#!/usr/bin/env python
"""Build the static variant of the Space: the article with two panels that replay recorded runs.  No server is needed.

    python space/build_static.py <out_dir>

Writes into <out_dir> (new, empty, or a fresh clone of the Space repository: .git, .gitattributes, README.md)
everything the static Space consists of:
    index.html       the article (site/build.py) with two panels inserted after its header: space/panel_static.html
                     (the pushing tasks, with the catalog space/data/index.json inlined) and after it
                     space/panel_hand.html (the screw-turning task, with space/hand/index.json inlined); the strings
                     come from space/text_static.json (placeholders below if that file is missing), those of the
                     second panel under keys that start with hand_
    media/           the article's videos and posters
    data/            the recorded pushing runs, space/data (written by space/make_data.py): one file per program,
                     fetched by the page when the program is selected
    hand/            the screwdriver panel's data space/hand/index.json and its videos space/hand/media/*.mp4, one
                     per stage and held-out seed, requested by the page when a seed is selected.  The videos are
                     not kept on GitHub (space/.gitignore): copy hand/media/ from the Space repository before
                     building.  space/check_hand.py checks both
    README.md        space/card_header.yaml as YAML front matter for a static Space (sdk: static, app_file:
                     index.html), with short_description from the key card_short_description of text_static.json,
                     followed by the key card_body as the body
    .gitattributes   the videos and posters are stored with Git LFS (hand/media/*.mp4 included)
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
    # the screwdriver panel (space/panel_hand.html)
    hand_kicker="Records", hand_heading="Screw task", hand_intro="Select a stage, then a mark.",
    hand_label_stage="Stage", hand_stage_template="{t} trials", hand_video_label="Video",
    hand_seed_template="Seed {seed}", hand_caption_template="{name}: {outcome}, {hold}", hand_hold_template="{h} N·m",
    hand_hint_marks="Select a mark.", hand_video_caption="Left: program A. Right: program B.",
    hand_video_error="Video not loaded.", hand_name_cap="Program A", hand_name_wm="Program B",
    hand_cap_title="Program A", hand_cap_sub="Subtitle A", hand_wm_title="Program B", hand_wm_sub="Subtitle B",
    hand_result_template="{k} of {n}", hand_outcome_ok="tightened", hand_outcome_hold_low="too loose",
    hand_outcome_hold_high="too tight", hand_outcome_damaged="thread stripped", hand_outcome_tool_lost="tool dropped",
    hand_set_held_out_template="Set 1, seeds {a}-{b}", hand_set_retest_template="Set 2, seeds {a}-{b}",
    hand_set_result_template="{k}/{n}", hand_cap_note="No world model.", hand_world_title="World model",
    hand_world_result_template="{k} of {n}", hand_world_success="model: success", hand_world_fail="model: failure",
    hand_paired_template="Same starts: {k}/{n}", hand_predicted_template="Predicted: {p}", hand_chart_title="Hold",
    hand_chart_x="Hold (N·m)", hand_chart_window="window", hand_chart_strip="limit", hand_chart_note="Note.",
    hand_foot="Recorded runs.",
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


def hand_files() -> tuple:
    """(the screwdriver panel's index, its videos): one video per stage and seed of the set that has videos."""
    hand = HERE / "hand"
    if not (hand / "index.json").exists():
        raise SystemExit("space/hand/index.json not found: the data of the screwdriver panel (see space/panel_hand.html)")
    index = json.loads((hand / "index.json").read_text(encoding="utf-8"))
    seeds = [s["seed0"] + i for s in index["sets"] if s["video"] for i in range(s["n"])]
    videos = [hand / "media" / st["video"].replace("{seed}", str(seed)) for st in index["stages"] for seed in seeds]
    missing = [p.name for p in videos if not p.exists()]
    if missing:
        raise SystemExit(f"space/hand/media lacks {len(missing)} of the {len(videos)} videos of the screwdriver panel "
                         f"(the first is {missing[0]}).  They are not kept on GitHub: copy hand/media/ from the Space "
                         "repository into space/hand/media/")
    return index, videos


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
    hand_index, videos = hand_files()

    text = build_page.load_text("text_static.json", FALLBACK, n=None)
    push = {k: v for k, v in text.items() if not k.startswith("hand_")}       # each panel carries its own strings
    hand = {k: v for k, v in text.items() if k.startswith("hand_")}
    index = build_page.build_article(out, lambda repo_url: (
        build_page.render("panel_static.html", push, repo_url=repo_url, data_json=build_page.script_json(catalog))
        + "\n\n" + build_page.render("panel_hand.html", hand, data_json=build_page.script_json(hand_index))))
    (out / ".nojekyll").unlink(missing_ok=True)         # written by site/build.py for another host
    (out / "data").mkdir()
    for src in files:
        shutil.copyfile(src, out / "data" / src.name)
    (out / "hand" / "media").mkdir(parents=True)
    shutil.copyfile(HERE / "hand" / "index.json", out / "hand" / "index.json")
    for src in videos:
        shutil.copyfile(src, out / "hand" / "media" / src.name)
    (out / "README.md").write_text(readme(text), encoding="utf-8")
    (out / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")     # *.mp4 covers hand/media as well
    if (HERE / "thumbnail.jpg").exists():       # the share image named by `thumbnail:` in the card header
        shutil.copyfile(HERE / "thumbnail.jpg", out / "thumbnail.jpg")      # (space/make_thumbnail.py)
    written =[p for p in out.rglob("*") if p.is_file() and ".git" not in p.relative_to(out).parts]
    size = lambda paths: sum(p.stat().st_size for p in paths) / 1e6  # noqa: E731
    print(f"wrote {index} ({index.stat().st_size} bytes) with the two recorded panels")
    print(f"{out}: {len(written)} files, {size(written):.2f} MB (index.html {size([index]):.2f}, "
          f"media {size(p for p in written if p.parent == out / 'media'):.2f}, "
          f"data {size(p for p in written if p.parent == out / 'data'):.2f}, "
          f"hand {size(p for p in written if out / 'hand' in p.parents):.2f} MB)")


if __name__ == "__main__":
    main()
