---
name: md-to-html
description: Render a research topic's Markdown into a self-contained HTML site under <topic>/html/ for browser viewing. Use whenever you create or edit Markdown files in a topic, or when the user asks to view/render/preview docs in a browser. Produces html/index.html plus one HTML page per Markdown file, with relative links rewritten and embedded CSS (works offline).
---

# Markdown to HTML

Render the Markdown files of one research topic into a browsable, self-contained
HTML site. The Markdown stays canonical; `html/` is a disposable preview.

## When to use

- After creating or editing **any** Markdown file inside a topic (this repo's
  `AGENTS.md` requires the preview to stay in sync — see `§8`).
- When the user asks to view, render, preview, or export docs to a browser.

## Run it

```bash
# from the repository root, pass the topic directory:
python3 .agents/skills/md-to-html/scripts/convert.py <topic-dir>

# or from inside the topic (defaults to the current directory):
python3 ../.agents/skills/md-to-html/scripts/convert.py
```

The script is at `scripts/convert.py` relative to this skill directory. Resolve it
against the skill's own path if you are not in the repo root.

What it does:

- Reads every `*.md` under the topic, excluding `html/`, hidden directories
  (`.git`, `.agents`, …), and any path matched by an optional `.htmlignore` at the
  topic root (see *Excluding files* below).
- Writes `html/<same relative path>.html` and `html/index.html`.
- Rewrites relative `.md` links to `.html` (fragments preserved); leaves external
  URLs (`https:`, `doi.org`, arXiv, GitHub) untouched.
- Renders LaTeX math in the browser. `$...$`, `$$...$$`, `\(...\)`, `\[...\]`, and
  ```math fences are protected from Markdown and restored for MathJax. A bundled
  MathJax (`assets/mathjax/tex-svg.js`) is copied into
  `<topic>/html/assets/mathjax/` and loaded **only** on pages that contain math, so
  the output stays fully offline.
- Inlines all CSS (system fonts, tables, code blocks, responsive layout, automatic
  dark mode). No network access and no CDN, so the output works offline.

## Output

```
<topic>/html/
├── index.html              # landing page: one card per document
├── README.html
├── notes/<doc>.html
├── sources/<doc>.html
└── assets/mathjax/         # only present if some page uses math
```

Point the user at the absolute path, e.g.:

```
file://<abs-path-to-topic>/html/index.html
```

## Steps

1. Identify the topic directory (the active topic). Do not render the whole repo —
   the one exception is the GitHub Pages build, which renders the repo root so all
   topics appear together (see the root `.htmlignore` for files to keep out).
2. Run `scripts/convert.py <topic-dir>`.
3. Verify the internal links resolve:

   ```bash
   python3 - <<'PY'
   import re
   from pathlib import Path
   out = Path("<topic-dir>/html").resolve(); bad = n = 0
   for h in out.rglob("*.html"):
       for href in re.findall(r'href="([^"]+)"', h.read_text(encoding="utf-8")):
           if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', href) or href.startswith("#"):
               continue
           n += 1
           if not (h.parent / href.split("#")[0]).resolve().exists():
               bad += 1; print("BROKEN", h.name, href)
   print(f"{n} internal links, {bad} broken")
   PY
   ```

4. Report the `file://` path to `html/index.html`.

## Notes and troubleshooting

- **Dependency:** Python 3.8+ and the `markdown` package (`pip install markdown`).
  It is usually already present; if not, install it before running.
- **Pandoc:** if `pandoc` is installed, it is a fine alternative for a single file
  (`pandoc in.md -s -o out.html`), but this script is preferred because it builds
  the index, mirrors the directory tree, and rewrites links. Do not require pandoc.
- **Regenerate after every Markdown edit**, including `README.md` and `notes/`.
  A stale `html/` tree is a documentation defect (see `AGENTS.md` §8).
- **Never hand-edit files under `html/`** — they are overwritten on the next run.
- `html/` is generated output. Deleting it is safe; re-run the script to rebuild.
- **Excluding files:** create a `.htmlignore` at the build root with one `fnmatch`
  pattern per line (`#` starts a comment). Each pattern is matched against a file's
  path relative to the root *and* against its first path component, so a bare
  directory name excludes that whole subtree. Example: a multi-topic root can list
  `AGENTS.md` in `.htmlignore` to keep the agent contract off the generated site.
