#!/usr/bin/env python
"""Build the article page with the live interactive panel inserted after the article header.

    python space/build_page.py [out_dir]          (default: space/_site)

Runs site/build.py (the page of the repository, with its media), then inserts space/panel.html directly after the
</header> of the article's <header class="post-head"> block.  The panel's strings are read from space/text.json
(a flat JSON object of strings) and inlined; "{n}" in any string becomes the number of episodes per batch.  Without
that file the placeholder strings below are used.  Standard library only.

A panel template may contain {{include <file>}} (that file of this folder, verbatim) and {{<name>}} (a string of
the text file, HTML-escaped, or a value given by the build script).  space/build_static.py uses the same functions
for the recorded variant of the panel.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from space.runner import N_EPISODES  # noqa: E402

DEFAULT_OUT = HERE / "_site"
HEADER = re.compile(r'<header class="post-head">.*?</header>', re.S)
INCLUDE = re.compile(r"\{\{include ([A-Za-z_.]+)\}\}")
PLACEHOLDER = re.compile(r"\{\{([a-z_A-Z]+)\}\}")

# Placeholders, replaced by space/text.json (same keys).
FALLBACK = dict(
    kicker="Demo", heading="Run a program", intro="Pick a program and press Run ({n} episodes per side).",
    label_task="Task", task_T="Push-T", task_L="Push-L", label_program="Program",
    tag_submitted="submitted", tag_extended="extended", tag_with_tests="with tests",
    button_run="Run", button_busy="Running", world_title="World model", world_sub="worlds/pusht/pusht_world.py",
    real_title="Environment", real_sub="reallab.pusht", result_template="{k}/{n}",
    status_world="Running (world model)", status_real="Running (environment)", status_queued="Queued",
    status_error="Error", time_template="{s} s", anim_caption="First episode of each batch.",
    chart_title="World model vs. environment", chart_x="World model", chart_y="Environment",
    chart_legend_recorded="Recorded", chart_legend_live="This page", note_sample="{n} random episodes per side.",
    note_scope="Pushing tasks only.", link_repo="Repository", card_short_description="Interactive article",
    card_body="Interactive article.",
)


def load_text(name: str = "text.json", fallback: dict = FALLBACK, n=N_EPISODES) -> dict:
    """The strings of a panel: space/<name> over the placeholders, with {n} filled in (unless n is None)."""
    text, path = dict(fallback), HERE / name
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
            raise SystemExit(f"space/{name} must be a flat JSON object of strings")
        missing, unknown = sorted(set(fallback) - set(data)), sorted(set(data) - set(fallback))
        if missing:
            print(f"space/{name} has no {missing}: using the placeholders for them")
        if unknown:
            print(f"space/{name} has unknown keys {unknown}: ignored")
        text.update({k: v for k, v in data.items() if k in fallback})
    else:
        print(f"space/{name} not found: using the placeholder strings")
    return text if n is None else {k: v.replace("{n}", str(n)) for k, v in text.items()}


def script_json(value) -> str:
    """JSON that is safe inside a <script> element."""
    return json.dumps(value, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def render(template: str, text: dict, **values) -> str:
    """space/<template> with its includes inserted and its placeholders filled in."""
    source = INCLUDE.sub(lambda m: (HERE / m.group(1)).read_text(encoding="utf-8").rstrip("\n"),
                         (HERE / template).read_text(encoding="utf-8"))
    values = {**{k: html.escape(v) for k, v in text.items()}, **values,
              "text_json": script_json({k: v for k, v in text.items() if not k.startswith("card_")})}

    def fill(m):
        if m.group(1) not in values:
            raise SystemExit(f"space/{template}: unknown placeholder {m.group(0)}")
        return values[m.group(1)]

    return PLACEHOLDER.sub(fill, source)


def build_article(out_dir, panel) -> Path:
    """site/build.py into out_dir, then panel(repository url) inserted after the article header."""
    out = Path(out_dir).resolve()
    subprocess.run([sys.executable, str(ROOT / "site" / "build.py"), str(out)], cwd=ROOT, check=True)
    index = out / "index.html"
    try:
        doc = index.read_text(encoding="utf-8")
        headers = HEADER.findall(doc)
        if len(headers) != 1:
            raise SystemExit('expected one <header class="post-head"> ... </header> block in the page, '
                             f"found {len(headers)}")
        repo = re.search(r'href="(https://github\.com/[^"]+)"', headers[0])
        if not repo:
            raise SystemExit("no repository link found in the article header")
        doc = doc.replace(headers[0], headers[0] + "\n\n" + panel(html.escape(repo.group(1))).strip() + "\n", 1)
    except BaseException:
        index.unlink()          # never leave the page without its panel
        raise
    index.write_text(doc, encoding="utf-8")
    return index


def build(out_dir=DEFAULT_OUT) -> Path:
    index = build_article(out_dir, lambda repo_url: render("panel.html", load_text(), repo_url=repo_url))
    print(f"wrote {index} ({index.stat().st_size} bytes) with the live panel, {N_EPISODES} episodes per batch")
    return index


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
