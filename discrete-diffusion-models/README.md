# Discrete Diffusion Models

**Scope.** Generative modeling with diffusion processes over *discrete / categorical*
state spaces — tokens, VQ codebook indices, segmentation labels, graphs — as opposed
to continuous Gaussian diffusion. The topic spans the underlying theory (discrete-time
Markov corruption, continuous-time Markov chains, concrete-score matching) and its
applications (diffusion language models, masked/VQ generative image models).

**Status:** initialized 2026-09-27. Foundational reading pinned; initial overview
written. No experiments yet.

## Core questions

1. What is the right *corruption kernel* for a given discrete domain, and how does
   the choice (uniform vs. absorbing/mask vs. structured) drive quality, likelihood,
   and sampler efficiency?
2. How does discrete diffusion relate to autoregressive and masked language modeling
   (any-order generation, infilling, parallel decoding)?
3. Can discrete diffusion match or beat autoregressive likelihood/perplexity at scale?
4. What is the correct continuous-time formulation, and how do the discrete-time
   ELBO and the score-entropy/concrete-score objectives relate?
5. Where are the compute/quality trade-offs in sampling (step budget, semi-AR decoding,
   guidance, controllable infilling)?

## Start here

- `sources/foundational-papers.md` — annotated bibliography: the **Core 5** plus an
  **Extended set** of 5 runner-ups (VQ-Diffusion, MaskGIT, CDCD, Plaid, MDLM), with
  arXiv/S2 IDs, DOIs, citation snapshots, and provenance. Raw API responses in
  `sources/raw/`.
- `notes/2026-09-27-overview.md` — high-level map of the field: the two formalisms
  (discrete-time Markov vs. continuous-time CTMC), the corruption-kernel design axis,
  and a short history.

## Layout

| Path | Purpose |
| --- | --- |
| `notes/` | Findings, summaries, decision logs, reading notes. |
| `data/` | Datasets and processed artifacts. |
| `sources/` | Paper references, citations, extracted text. |
| `experiments/` | Code, configs, run outputs, metrics. |
| `papers/` | Draft write-ups, LaTeX, figures. |
| `assets/` | Images, diagrams, non-data media. |
| `html/` | Generated HTML preview of the Markdown; open `html/index.html`. |

## Viewing rendered docs

Open `discrete-diffusion-models/html/index.html` in a browser (self-contained, no
network needed). Regenerate after editing any Markdown:

```bash
# from the repository root
python3 .agents/skills/md-to-html/scripts/convert.py discrete-diffusion-models
# or from inside this topic (defaults to the current directory)
python3 ../.agents/skills/md-to-html/scripts/convert.py
```

This is handled by the `md-to-html` skill. The Markdown files are canonical;
`html/` is a disposable preview.

## Changelog

- **2026-09-27** — Topic created. Scaffolded directories. Pinned 5 foundational papers
  (Sohl-Dickstein 2015; Hoogeboom 2021; Austin 2021 D3PM; Meng 2022 Concrete Score;
  Lou 2024 SEDD) with runner-ups; wrote initial field overview in `notes/`.
- **2026-09-27** — Promoted the 5 runner-ups to fully documented entries in
  `sources/foundational-papers.md` (VQ-Diffusion, MaskGIT, CDCD, Plaid, MDLM) with
  verified Semantic Scholar metadata and citation snapshots; saved raw responses to
  `sources/raw/`; added an applied-track map to `notes/2026-09-27-overview.md`.
- **2026-09-27** — Confirmed `SEMANTIC_SCHOLAR_API_KEY` is active in the agent
  environment. Completed the metadata set by retrying Plaid (S2
  `d9ffb44ee3c8ec0b6692df8a90451384c1edd89b`, 171 citations); all 10 raw records now
  valid.
- **2026-09-27** — Added `build-html.py`; renders all topic Markdown to a
  self-contained HTML site under `html/` (with `html/index.html`) and rewrites
  `.md` links to `.html`. All 33 internal links verified.
- **2026-09-27** — Moved the renderer into the repo-level `md-to-html` skill
  (`.agents/skills/md-to-html/`); removed the topic-local `build-html.py`. Rebuilt
  `html/` (64 internal links, 0 broken).
- **2026-09-27** — Started a chronological paper walkthrough in
  `notes/paper-walkthrough.md`; entry 1 covers Sohl-Dickstein et al. 2015 (3 key
  points + main innovation). `html/` regenerated.
- **2026-09-27** — Added math rendering: repo-level terminal extension
  (`.pi/extensions/math.ts`, LaTeX→Unicode) and bundled MathJax in the `md-to-html`
  skill. Converted the ELBO equations in `notes/paper-walkthrough.md` to LaTeX.
  Rebuilt `html/` (73 internal links, 0 broken).
- **2026-09-27** — Extended the bound deep dive with the Bayes/telescoping derivation
  of Eq. (15) and the paper's per-step entropy-production bounds (Appendix B).
- **2026-09-27** — Deep dive on the 2015 variational bound (ELBO `K`, its exact form
  and the DDPM rearrangement) added to `notes/paper-walkthrough.md`.
- **2026-09-27** — Added a 12-move *intuition* walkthrough of the bound (why each
  equation exists, and the design space it yields) ahead of the formal deep dive in
  `notes/paper-walkthrough.md`. `html/` regenerated.
- **2026-09-27** — Added a *binomial (discrete) case* deep dive to the Sohl-Dickstein
  entry: exact Table App.1 kernels, the symmetric-channel mechanics, the derived
  closed-form posterior, and its identification with D3PM's uniform kernel at $K=2$.
- **2026-09-27** — Paper 2 (Hoogeboom et al. 2021, *Argmax Flows and Multinomial
  Diffusion*) written up in `notes/paper-walkthrough.md` (3 key points + main
  innovation; progress 2/10). `html/` regenerated.
- **2026-09-27** — Added a cross-cutting *$x_0$- vs. noise-parameterization* subsection to
  paper 2 in `notes/paper-walkthrough.md` (continuous affine-twin argument; why $x_0$ is the
  natural discrete coordinate; when a noise formulation returns). `html/` regenerated.
- **2026-09-27** — Appended a companion *which discrete state spaces admit a noise
  formulation* subsection (native score/ratios, lattice, group, $\pm1$ bits, continuous
  embedding, counts; table; verified Analog Bits 2208.04202 and Dirichlet Diffusion
  2305.10699). `html/` regenerated.
- **2026-09-27** — Added the two new references (Analog Bits 2208.04202, Dirichlet
  Diffusion 2305.10699) to `sources/foundational-papers.md` under “Additional references”.
  `html/` regenerated.
- **2026-09-27** — Published the workspace to GitHub Pages. Repo
  [`anvuongb/ai-research`](https://github.com/anvuongb/ai-research) (public); a repo-level
  Actions workflow (`.github/workflows/pages.yml`) renders all Markdown at the root and
  deploys it. **Live site:** <https://me.anvuong.dev/ai-research/> (the account's custom
  domain; `anvuongb.github.io/ai-research/` redirects there). `html/` is gitignored and
  rebuilt in CI.
