# Project extensions

## `math.ts` — terminal math rendering

Renders LaTeX math as readable Unicode in the Pi interactive transcript using the
`registerMarkdownTransformer` hook. It converts inline `$...$` / `\(...\)`, display
`$$...$$` / `\[...\]`, and fenced ```math / ```latex / ```tex blocks. Code spans and
non-math fenced blocks are left untouched, and prose dollars (e.g. `$PATH`) are not
mistaken for math.

Example: `\sum_{t=1}^{T} \frac{\alpha_t}{\beta^2}` → `Σₜ₌₁ᵀ (αₜ)/(β²)`.

The converter is intentionally lossy (it is a terminal preview, not a typesetter):
matrices, alignment, and unknown macros degrade to readable plain text. For fully
typeset math, view the generated HTML (which uses MathJax) in a browser.

**Loading:** this is a *project* extension, so it requires project trust. Run
`/reload` after editing it (or restart Pi). If project trust is not granted, copy it
to `~/.pi/agent/extensions/` for user-level loading instead.
