#!/usr/bin/env python3
"""Render a research topic's Markdown to a self-contained HTML site.

Usage:
    python3 convert.py [TOPIC_DIR]

Defaults to the current working directory. Reads every ``*.md`` under the topic
(excluding the output directory ``html/``, hidden directories, and any paths
matched by an optional ``.htmlignore`` file at the topic root) and writes
``<topic>/html/<same relative path>.html`` plus ``<topic>/html/index.html``.
Relative links between Markdown files are rewritten to point at the generated
HTML. All CSS is embedded, so the output works offline.

The Markdown files remain canonical; everything under ``html/`` is a preview and
is safe to delete and regenerate.

Dependencies: Python 3.8+ and the ``markdown`` package (``pip install markdown``).
"""

from __future__ import annotations

import fnmatch
import html as html_mod
import os
import re
import shutil
import sys
from pathlib import Path

try:
    import markdown
except ImportError:  # pragma: no cover
    sys.exit("error: the 'markdown' package is required (pip install markdown)")

MD_EXTENSIONS = ["extra", "toc", "sane_lists", "smarty"]
MD_EXT_CONFIGS = {"toc": {"permalink": False}}

SKILL_DIR = Path(__file__).resolve().parent.parent
MATHJAX_ASSET = SKILL_DIR / "assets" / "mathjax" / "tex-svg.js"

CSS = """
:root {
  --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --border: #d1d9e0;
  --accent: #0969da; --code-bg: #f6f8fa; --card: #f6f8fa; --table-stripe: #f6f8fa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --border: #30363d;
    --accent: #4493f8; --code-bg: #161b22; --card: #161b22; --table-stripe: #161b22;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
.wrap { max-width: 860px; margin: 0 auto; padding: 0 24px 96px; }
nav.top {
  position: sticky; top: 0; z-index: 5; background: var(--bg);
  border-bottom: 1px solid var(--border); margin-bottom: 32px;
}
nav.top .wrap { display: flex; flex-wrap: wrap; gap: 4px 18px; padding: 12px 24px; align-items: center; }
nav.top a { color: var(--muted); text-decoration: none; font-size: 14px; font-weight: 600; }
nav.top a:hover { color: var(--accent); }
nav.top a.home { color: var(--fg); }
main { padding-top: 8px; }
h1, h2, h3, h4 { line-height: 1.25; margin: 1.6em 0 0.6em; }
h1 { font-size: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3em; }
h2 { font-size: 1.45rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3em; }
h3 { font-size: 1.15rem; }
a { color: var(--accent); }
hr { border: 0; border-top: 1px solid var(--border); margin: 2.5em 0; }
code {
  background: var(--code-bg); border-radius: 6px; padding: 0.15em 0.4em;
  font: 0.85em/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
pre {
  background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px;
  padding: 14px 16px; overflow: auto;
}
pre code { background: none; padding: 0; font-size: 0.85em; }
blockquote {
  margin: 1em 0; padding: 0.4em 1em; color: var(--muted);
  border-left: 4px solid var(--border);
}
table { border-collapse: collapse; width: 100%; margin: 1.2em 0; display: block; overflow: auto; }
th, td { border: 1px solid var(--border); padding: 7px 12px; text-align: left; vertical-align: top; }
th { background: var(--card); }
tbody tr:nth-child(even) { background: var(--table-stripe); }
img { max-width: 100%; }
.footer {
  margin-top: 56px; padding-top: 16px; border-top: 1px solid var(--border);
  color: var(--muted); font-size: 13px;
}
.index-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; margin: 1.4em 0; }
.card { border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; background: var(--card); }
.card a { text-decoration: none; font-weight: 600; }
.card p { margin: 6px 0 0; color: var(--muted); font-size: 13px; }
.card .path { display: block; margin-top: 8px; color: var(--muted); font-size: 12px; font-family: ui-monospace, Menlo, monospace; }
.group-title { margin-top: 2em; }
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<nav class="top"><div class="wrap">
<a class="home" href="{home}">{topic}</a>
{navlinks}
</div></nav>
<div class="wrap">
<main>
{body}
</main>
<div class="footer">
Rendered from <code>{source}</code> &middot; preview only, the Markdown file is canonical.
</div>
</div>
{mathjax}
</body>
</html>
"""

ROOT: Path = Path.cwd().resolve()
OUT: Path = ROOT / "html"
TOPIC: str = ROOT.name
IGNORE: list[str] = []


def load_ignore() -> list[str]:
    """Read optional ``.htmlignore`` patterns (relative to ROOT) from the root.

    Blank lines and lines starting with ``#`` are ignored. A pattern containing
    ``/`` is matched with :func:`fnmatch.fnmatch` against a Markdown file's path
    relative to ROOT (POSIX separators); a pattern without ``/`` is matched
    against each individual path component, so a bare ``data`` excludes any file
    under a directory named ``data`` at any depth.
    """
    path = ROOT / ".htmlignore"
    if not path.is_file():
        return []
    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def is_ignored(rel: Path) -> bool:
    """True if ``rel`` (relative to ROOT) matches any ``.htmlignore`` pattern.

    A pattern containing ``/`` is matched against the full relative path; a pattern
    without ``/`` is matched against every individual path component, so a bare
    ``data`` excludes anything under a directory named ``data`` at any depth.
    """
    rel_posix = rel.as_posix()
    for pattern in IGNORE:
        if "/" in pattern:
            if fnmatch.fnmatch(rel_posix, pattern):
                return True
        elif any(fnmatch.fnmatch(part, pattern) for part in rel.parts):
            return True
    return False


def collect_markdown() -> list[Path]:
    files = []
    for p in sorted(ROOT.rglob("*.md")):
        rel = p.relative_to(ROOT)
        if OUT in p.parents or OUT == p.parent:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        if is_ignored(rel):
            continue
        files.append(p)
    return files


def out_path(md_path: Path) -> Path:
    return (OUT / md_path.relative_to(ROOT)).with_suffix(".html")


def rel_href(from_html: Path, to_html: Path) -> str:
    return os.path.relpath(to_html, from_html.parent).replace(os.sep, "/")


def render_body(text: str) -> str:
    md = markdown.Markdown(extensions=MD_EXTENSIONS, extension_configs=MD_EXT_CONFIGS)
    return md.convert(text)


# --- math -------------------------------------------------------------------
# Protect LaTeX math from Markdown before conversion, then restore it wrapped in
# \(...\) / \[...\] so the bundled MathJax renders it client-side.
MATH_MASTER = re.compile(
    r"(`{3,})([^\n]*)\n(.*?)(?:\n\1|$)"       # 1 fence, 2 info, 3 body
    r"|(`[^`\n]*`)"                           # 4 inline code
    r"|(\$\$(.+?)\$\$)"                       # 5 whole, 6 body
    r"|(\\\[(.+?)\\\])"                       # 7 whole, 8 body
    r"|(\\\((.+?)\\\))"                       # 9 whole, 10 body
    r"|(\$([^\$\n]*[\\_^{}=+\-<>|][^\$\n]*|[A-Za-z])\$)",  # 11 whole, 12 body
    re.S,
)
MATH_TOKEN_RE = re.compile(r"MATHTOKEN(\d+)ENDMATHTOKEN")
MATHJAX_SNIPPET = r"""<script>
window.MathJax = {
  tex: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']] },
  svg: { fontCache: 'global' },
  options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }
};
</script>
<script src="__SRC__"></script>"""


def _extract_math(text: str) -> tuple[str, list[tuple[str, str]]]:
    store: list[tuple[str, str]] = []

    def repl(match: re.Match) -> str:
        if match.group(4) is not None:  # inline code span
            return match.group(0)
        if match.group(1) is not None:  # fenced block
            info = (match.group(2) or "").strip().lower()
            if info in ("math", "latex", "tex"):
                store.append(("display", match.group(3)))
                return f"\n\nMATHTOKEN{len(store) - 1}ENDMATHTOKEN\n\n"
            return match.group(0)
        for body_index, kind in ((6, "display"), (8, "display"), (10, "inline"), (12, "inline")):
            value = match.group(body_index)
            if value is not None:
                if kind == "inline" and (value == "" or value[:1].isspace() or value[-1:].isspace()):
                    return match.group(0)
                store.append((kind, value))
                return f"MATHTOKEN{len(store) - 1}ENDMATHTOKEN"
        return match.group(0)

    return MATH_MASTER.sub(repl, text), store


def _restore_math(body: str, store: list[tuple[str, str]]) -> str:
    def repl(match: re.Match) -> str:
        kind, tex = store[int(match.group(1))]
        escaped = tex.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"\\[{escaped}\\]" if kind == "display" else f"\\({escaped}\\)"

    return MATH_TOKEN_RE.sub(repl, body)


def render_body_with_math(text: str) -> tuple[str, bool]:
    protected, store = _extract_math(text)
    body = render_body(protected)
    if store:
        body = _restore_math(body, store)
    return body, bool(store)


def rewrite_links(body: str, md_path: Path, converted: dict[Path, Path]) -> str:
    def repl(match: re.Match) -> str:
        href = match.group(1)
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", href) or href.startswith("#") or href.startswith("/"):
            return match.group(0)
        path, _, frag = href.partition("#")
        if not path:
            return match.group(0)
        target = (md_path.parent / path).resolve()
        cand = target if target.suffix == ".md" else target.with_suffix(".md")
        if cand in converted:
            new = rel_href(out_path(md_path), converted[cand])
            if frag:
                new += "#" + frag
            return f'href="{new}"'
        return match.group(0)

    return re.sub(r'href="([^"]+)"', repl, body)


def group_name(md_path: Path) -> str:
    rel = md_path.relative_to(ROOT)
    return "Topic root" if len(rel.parts) == 1 else rel.parts[0]


def ordered_groups(groups: dict[str, list[Path]]) -> list[str]:
    preferred = ["Topic root", "notes", "sources", "outputs", "experiments", "papers", "data", "assets"]
    return [g for g in preferred if g in groups] + [g for g in sorted(groups) if g not in preferred]


def nav_links(current: Path, groups: dict[str, list[Path]]) -> str:
    parts = []
    for group in ordered_groups(groups):
        target = groups[group][0]
        label = "README" if group == "Topic root" else group
        parts.append(f'<a href="{rel_href(out_path(current), out_path(target))}">{html_mod.escape(label)}</a>')
    return "".join(parts)


def first_paragraph(body: str) -> str:
    text = re.sub(r"<[^>]+>", " ", body)
    text = html_mod.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return (text[:160] + "…") if len(text) > 160 else text


def first_heading(body: str, fallback: str) -> str:
    match = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    if match:
        title = html_mod.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()
        if title:
            return title
    return fallback


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_index(converted: dict[Path, Path]) -> str:
    groups: dict[str, list[Path]] = {}
    for md_path in sorted(converted):
        groups.setdefault(group_name(md_path), []).append(md_path)

    parts = [f"<h1>{html_mod.escape(TOPIC)}</h1>",
             "<p>HTML preview of the topic's Markdown. The Markdown files are canonical.</p>"]
    for group in ordered_groups(groups):
        parts.append(f'<h2 class="group-title" id="group-{slug(group)}">{html_mod.escape(group)}</h2>')
        parts.append('<div class="index-grid">')
        for md_path in groups[group]:
            rel = md_path.relative_to(ROOT).as_posix()
            body = render_body(md_path.read_text(encoding="utf-8"))
            fallback = md_path.stem.replace("-", " ").replace("_", " ")
            title = first_heading(body, fallback)
            parts.append(
                f'<div class="card"><a href="{rel_href(OUT / "index.html", converted[md_path])}">'
                f'{html_mod.escape(title)}</a><p>{html_mod.escape(first_paragraph(body))}</p>'
                f'<span class="path">{html_mod.escape(rel)}</span></div>'
            )
        parts.append("</div>")
    return "\n".join(parts)


def main() -> None:
    global ROOT, OUT, TOPIC, IGNORE
    if len(sys.argv) > 1:
        ROOT = Path(sys.argv[1]).expanduser().resolve()
        OUT = ROOT / "html"
        TOPIC = ROOT.name
    if not ROOT.is_dir():
        sys.exit(f"error: not a directory: {ROOT}")

    IGNORE = load_ignore()
    OUT.mkdir(parents=True, exist_ok=True)
    md_files = collect_markdown()
    if not md_files:
        sys.exit(f"error: no Markdown files found under {ROOT}")
    converted = {p: out_path(p) for p in md_files}
    groups: dict[str, list[Path]] = {}
    for p in sorted(converted):
        groups.setdefault(group_name(p), []).append(p)

    mathjax_dest = OUT / "assets" / "mathjax" / "tex-svg.js"
    uses_math = False

    for md_path in md_files:
        dest = converted[md_path]
        dest.parent.mkdir(parents=True, exist_ok=True)
        body, has_math = render_body_with_math(md_path.read_text(encoding="utf-8"))
        body = rewrite_links(body, md_path, converted)
        title = md_path.stem.replace("-", " ").replace("_", " ")
        mathjax = ""
        if has_math:
            uses_math = True
            mathjax = MATHJAX_SNIPPET.replace("__SRC__", rel_href(dest, mathjax_dest))
        dest.write_text(
            TEMPLATE.format(
                title=html_mod.escape(title), css=CSS,
                home=rel_href(dest, OUT / "index.html"),
                topic=html_mod.escape(TOPIC),
                navlinks=nav_links(md_path, groups),
                body=body,
                source=html_mod.escape(md_path.relative_to(ROOT).as_posix()),
                mathjax=mathjax,
            ),
            encoding="utf-8",
        )

    if uses_math:
        mathjax_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MATHJAX_ASSET, mathjax_dest)

    (OUT / "index.html").write_text(
        TEMPLATE.format(
            title=html_mod.escape(TOPIC), css=CSS, home="index.html",
            topic=html_mod.escape(TOPIC), navlinks="",
            body=build_index(converted), source="(index)", mathjax="",
        ),
        encoding="utf-8",
    )

    print(f"Wrote {len(md_files) + 1} HTML files to {OUT}/")
    for md_path in md_files:
        print(f"  {md_path.relative_to(ROOT)} -> {converted[md_path].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
