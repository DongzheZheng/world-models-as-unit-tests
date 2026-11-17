"""Plain outputs: Markdown tables and an SVG scatter plot (numbers and file names, no prose)."""
from __future__ import annotations

import math
from html import escape


def rate(summary: dict | None) -> str:
    """Success rate of a summary (harness/evaluate.py: summarise) as text."""
    if not summary or summary.get("success_rate") is None:
        return "-"
    return f"{summary['success_rate']:.3f}"


def count(summary: dict | None) -> str:
    if not summary or not summary.get("episodes"):
        return "-"
    return f"{summary['successes']}/{summary['episodes']}"


def flagged(summary: dict | None) -> str:
    if not summary or not summary.get("episodes"):
        return "-"
    if not summary["flagged"]:
        return "0"
    return f"{summary['flagged']} (" + ", ".join(f"{k} {v}" for k, v in sorted(summary["flags"].items())) + ")"


def cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def mean_abs_error(xs: list[float], ys: list[float]) -> float | None:
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs) if xs else None


def scatter_svg(points: list[tuple[str, float, float]], xlabel: str, ylabel: str, note: str = "") -> str:
    """Predicted against real success rate, both on [0, 1], one labelled point per program."""
    size, left, top, plot = 520, 70, 30, 400

    def px(x):
        return left + plot * x

    def py(y):
        return top + plot * (1.0 - y)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
           f'font-family="sans-serif" font-size="12">',
           f'<rect width="{size}" height="{size}" fill="white"/>',
           f'<rect x="{left}" y="{top}" width="{plot}" height="{plot}" fill="none" stroke="#444"/>']
    for i in range(6):
        v = i / 5
        out.append(f'<line x1="{px(v):.1f}" y1="{top}" x2="{px(v):.1f}" y2="{top + plot}" stroke="#ddd"/>')
        out.append(f'<line x1="{left}" y1="{py(v):.1f}" x2="{left + plot}" y2="{py(v):.1f}" stroke="#ddd"/>')
        out.append(f'<text x="{px(v):.1f}" y="{top + plot + 18}" text-anchor="middle">{v:.1f}</text>')
        out.append(f'<text x="{left - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>')
    out.append(f'<line x1="{px(0):.1f}" y1="{py(0):.1f}" x2="{px(1):.1f}" y2="{py(1):.1f}" stroke="#999" stroke-dasharray="5 4"/>')
    placed: list[tuple[float, float, float]] = []       # label boxes so far: left, right, baseline
    for name, x, y in points:
        out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4.5" fill="#1f6fb2" fill-opacity="0.85"/>')
        anchor, dx = ("end", -8) if x > 0.8 else ("start", 8)
        width = 7.0 * len(name)
        x0 = px(x) + dx - (width if anchor == "end" else 0.0)
        base = py(y) - 6
        while any(abs(base - b) < 13 and x0 < hi and lo < x0 + width for lo, hi, b in placed):
            base -= 13                                   # move a label up until it is clear of the others
        placed.append((x0, x0 + width, base))
        out.append(f'<text x="{px(x) + dx:.1f}" y="{base:.1f}" text-anchor="{anchor}">{escape(name)}</text>')
    out.append(f'<text x="{left + plot / 2}" y="{top + plot + 42}" text-anchor="middle" font-size="13">{escape(xlabel)}</text>')
    out.append(f'<text transform="translate(20 {top + plot / 2}) rotate(-90)" text-anchor="middle" font-size="13">{escape(ylabel)}</text>')
    if note:
        out.append(f'<text x="{left}" y="{top - 10}" font-size="12">{escape(note)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
