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
- `notes/group-theory-primer.md` — group theory from zero, at the depth this topic needs:
  groups and the four axioms, the recurring examples ($\mathbb{Z}_N$, $(\mathbb{Z}_2)^n$,
  $\mathbb{Z}$, $S_n$, $SO(3)$, $SE(3)$), Cayley graphs, abelian vs non-abelian (the line
  between a scalar dispersion $\psi$ and matrix-valued representations), characters and the
  dual group, convolution as translation-invariance, random walks as the forward diffusion
  process, why absorbing/mask is *not* a group, Lie groups, and the group-as-state-space
  vs group-as-symmetry trap.

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
- **2026-09-27** — Added a *group diffusion* subsection to paper 2 in
  `notes/paper-walkthrough.md`: the translation-invariant family (convolution forward, group
  Fourier diagonalization, group-element noise), the result that Hoogeboom's uniform kernel
  is its degenerate maximally-mixing member (with the spectral derivation), an
  instances-by-group table, `S_n` and Lie-group case studies, the non-abelian difficulty, and
  the equivariance contrast. Backed by a verified literature pass (SymmetricDiffusers ICLR
  2025; Soft-Rank ICML 2026; Blackout Diffusion ICML 2023; finite-group Fourier/convolution
  semigroups 2026; cycle-graph Markov processes; FrameDiff/SO(3)/Riemannian/Lie-group
  representations; plus equivariance contrasts). `html/` regenerated.
- **2026-09-27** — Added references A3–A16 (group diffusion) to
  `sources/foundational-papers.md` and their verified records to
  `sources/raw/group-diffusion-refs.json`. `html/` regenerated.
- **2026-09-27** — Added **paper 3**: *Harmonic flows and Markov dynamics on finite groups via
  pseudo-differential operators* (Torresblanca-Badillo, Barrios-Garizao & Quiñonez-Martínez;
  J. Pseudo-Differ. Oper. Appl. 17(1), Art. 18, 2026; DOI `10.1007/s11868-025-00759-7`).
  Written **from first principles** (no operator theory assumed): a from-zero primer on the
  toolkit — functions on Z_N, convolution, the Fourier transform as the diagonalizer,
  convolution semigroups, negative definite functions / Lévy–Khinchin, Feller semigroups,
  generators, pseudo-differential operators, dissipativity and self-adjointness — followed by
  the paper's construction step by step, the bridge to discrete diffusion models, and caveats.
  The walkthrough was renumbered from 10 to 11 papers: the reading order is now a **study
  order**, and D3PM is paper 4. `html/` regenerated.
- **2026-09-27** — Added `experiments/check-dispersion-example.py`, a reproducible check of the
  new paper's dispersion-function conditions. It shows the paper's worked example
  (psi(n)=n^2 on Z_5) is **not admissible** — psi is not even, so Z_t is complex and cannot be
  a probability measure — and that even the symmetrized min(m,N-m)^2 fails the Lévy–Khinchin
  cone (c_2 < 0). The admissible quadratic dispersion is the cycle-Laplacian symbol
  2(1-cos(2*pi*m/N)). Written up as a correction in the paper 3 entry; A7 in
  `sources/foundational-papers.md` now points to it.
- **2026-09-27** — Added `notes/group-theory-primer.md`: group theory built from zero at the
  depth this topic needs — the four axioms; the recurring examples ($\mathbb{Z}_N$,
  $(\mathbb{Z}_2)^n$, $\mathbb{Z}$, $S_n$, $SO(3)$, $SE(3)$) and the labelling caveat; the
  Cayley graph as the walk's geometry; abelian vs **non-abelian** (the line between a scalar
  dispersion $\psi$ and matrix-valued representations, i.e. why $S_n$/$SO(3)$ have no
  spectral closed form); homomorphisms, characters and the dual group ("frequency" =
  character); convolution as translation-invariance; the random walk as the forward diffusion
  process, with a three-line derivation of Hoogeboom's symbol $1-\beta$; why absorbing/mask is
  *not* a group translation and how the sub-probability slack in paper 3's definition connects
  to it; Lie groups; and the group-as-state-space vs group-as-symmetry trap. Cross-linked from
  paper 2's group-diffusion section and paper 3's operator primer. `html/` regenerated.
