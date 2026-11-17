"""Build the standalone web page (used by the GitHub Pages workflow).

    python site/build.py [out_dir]        (default: _site)

Runs blog/build.py, which assembles the page fragment from blog/_head.html, blog/body.html, the chart fragments in
runs/probe/ and blog/_tail.html, then wraps the fragment into a complete HTML document and copies site/media/ next
to it.  Standard library only.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = ("World Models as Unit Tests. Agents observe their task environment, write its world model as code, "
               "and use it to help write and verify policies, including Code-as-Policies programs.")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "_site")
    subprocess.run([sys.executable, "blog/build.py"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    fragment = (ROOT / "blog" / "index.html").read_text(encoding="utf-8")
    head, sep, rest = fragment.partition("</style>")
    if not sep:
        raise SystemExit("unexpected page fragment: no </style> found")
    head = re.sub(r'<meta name="claude:[^>]*>\n?', "", head)       # hint for a hosting service, not needed here
    title = re.search(r"<title>(.*?)</title>", head).group(1)
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
           f'<meta name="description" content="{DESCRIPTION}">\n'
           f'<meta property="og:title" content="{title}">\n'
           f'<meta property="og:description" content="{DESCRIPTION}">\n'
           '<meta property="og:type" content="article">\n'
           + head + "</style>\n</head>\n<body>\n" + rest.lstrip("\n") + "\n</body>\n</html>\n")
    missing = sorted(m for m in set(re.findall(r'(?:src|poster)="(media/[^"]+)"', doc))
                     if not (ROOT / "site" / m).exists())
    if missing:
        raise SystemExit(f"media referenced by the page but missing from site/media: {missing}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(doc, encoding="utf-8")
    shutil.copytree(ROOT / "site" / "media", out / "media", dirs_exist_ok=True)
    (out / ".nojekyll").write_text("")
    print(f"wrote {out / 'index.html'} ({len(doc.encode())} bytes) and {len(list((out / 'media').iterdir()))} media files")


if __name__ == "__main__":
    main()
