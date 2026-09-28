# Foundational & Important Papers — Discrete Diffusion Models

Retrieval date: **2026-09-27**. Metadata retrieved from the arXiv API, Semantic Scholar
Graph API, and OpenAlex. Citation counts are Semantic Scholar snapshots at retrieval
time, not current values. Raw per-paper API responses are saved under `raw/`.

Two tiers below: the **Core 5** (the conceptual spine of the field) and the
**Extended set** (runner-ups that carry the theory into language and vision practice,
or supply important variants — read these next).

---

# Core 5 (foundational)

## 1. Deep Unsupervised Learning using Nonequilibrium Thermodynamics
- **Authors:** Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli
- **Venue / year:** ICML 2015
- **arXiv:** [1503.03585](https://arxiv.org/abs/1503.03585) (v8, cs.LG)
- **S2:** `2dcef55a07f8607a819c21fe84131ea269cc2e3c` · citations **10,749** (influential 538)
- **Why it matters:** The origin of diffusion models. Introduces the forward
  structure-destroying diffusion and the learned reverse process. Notably already
  formulates a *discrete* (binomial) diffusion case, so the discrete track is
  present from the beginning rather than a later retrofit.
- **Read for:** the original ELBO/variational derivation and the diffusion template.

## 2. Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions
- **Authors:** Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, Max Welling
- **Venue / year:** NeurIPS 2021
- **arXiv:** [2102.05379](https://arxiv.org/abs/2102.05379) (v3, stat.ML)
- **S2:** `1913d3edcc00d0aba097a9df190dd16f6fdfbf0c` · citations **759** (influential 57)
- **Why it matters:** First strong *categorical* diffusion model. A uniform-transition
  Markov corruption over `K` classes with a closed-form ELBO; strong log-likelihood
  results on text and segmentation. Also introduces Argmax Flows. This is the
  discrete-state baseline that D3PM generalizes.
- **Read for:** uniform categorical corruption and its ELBO.

## 3. Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)
- **Authors:** Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg
- **Venue / year:** NeurIPS 2021
- **arXiv:** [2107.03006](https://arxiv.org/abs/2107.03006) (v3, cs.LG)
- **S2:** `91b32fc0a23f0af53229fceaae9cce43a0406d2e` · citations **2,168** (influential 299)
- **Why it matters:** The canonical framework. Generalizes multinomial diffusion to
  arbitrary transition matrices `Q_t` (Gaussian-like kernels, embedding nearest-neighbor,
  and absorbing/`[MASK]` states). Connects diffusion to autoregressive and mask-based
  generative models, and introduces the ELBO + auxiliary cross-entropy loss. Approaches
  DDPM sample quality and exceeds its CIFAR-10 log-likelihood.
- **Read first.** The reference formulation for discrete-state diffusion.

## 4. Concrete Score Matching: Generalized Score Matching for Discrete Data
- **Authors:** Chenlin Meng, Kristy Choi, Jiaming Song, Stefano Ermon
- **Venue / year:** NeurIPS 2022
- **arXiv:** [2211.00802](https://arxiv.org/abs/2211.00802)
- **DOI:** [10.48550/arXiv.2211.00802](https://doi.org/10.48550/arXiv.2211.00802)
- **S2:** `3d8a2753649f3c493e2c237b0f4049858e958ae6` · citations **161** (influential 14)
- **Why it matters:** Theoretical backbone of continuous-time discrete diffusion.
  Defines the **concrete score** — the ratio of probabilities of neighboring states
  under a predefined neighborhood structure — as the discrete analog of the (Stein)
  score, and derives the corresponding matching objective. Enables CTMC-based
  discrete diffusion and samplers.
- **Read for:** the discrete score definition and continuous-time rate-matrix view.

## 5. Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD)
- **Authors:** Aaron Lou, Chenlin Meng, Stefano Ermon
- **Venue / year:** ICML 2024 (Oral); arXiv 2023
- **arXiv:** [2310.16834](https://arxiv.org/abs/2310.16834) (v3, stat.ML)
- **S2:** `ce806f8d32f6fb1eaa821248a1bc4fa2cd949fbb` · citations **747** (influential 133)
- **Why it matters:** Introduces **score entropy**, a loss that makes concrete-score
  learning tractable and integrates into discrete diffusion. SEDD beats prior diffusion
  LMs by 25–75% perplexity, outperforms GPT-2, generates faithful text without
  temperature annealing, trades compute for quality (comparable quality with ~32× fewer
  network evaluations), and enables controllable infilling.
- **Code:** https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

---

# Extended set (runner-ups)

These five are not merely adjacent — each fills a distinct gap: VQ-Diffusion and MaskGIT
push discrete diffusion into vision; CDCD keeps the input space continuous; Plaid and MDLM
push the language-model likelihood gap shut. Together with SEDD they define the current
applied frontier.

## E1. Vector Quantized Diffusion Model for Text-to-Image Synthesis (VQ-Diffusion)
- **Authors:** Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, Baining Guo
- **Venue / year:** CVPR 2022
- **arXiv:** [2111.14822](https://arxiv.org/abs/2111.14822) (v3, cs.CV)
- **DOI:** [10.1109/CVPR52688.2022.01043](https://doi.org/10.1109/CVPR52688.2022.01043)
- **S2:** `414e554d281d529401c873cb9c97186365ec5dd8` · citations **1,108** (influential 99)
- **OpenAlex:** [W4312388283](https://openalex.org/W4312388283)
- **Contribution:** Models a VQ-VAE latent space with a *conditional* discrete diffusion
  process using a **mask-and-replace** corruption. Removes the unidirectional bias of
  autoregressive text-to-image and avoids error accumulation; a reparameterization makes
  generation ~15× faster than comparable AR models at better image quality.
- **Why interesting:** The clearest demonstration that discrete diffusion scales to
  real generative modeling of images via a two-stage (VQ + diffusion) pipeline; the
  mask-and-replace kernel is an early appearance of the absorbing-state idea in vision.

## E2. MaskGIT: Masked Generative Image Transformer
- **Authors:** Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, William T. Freeman
- **Venue / year:** CVPR 2022
- **arXiv:** [2202.04200](https://arxiv.org/abs/2202.04200)
- **DOI:** [10.1109/CVPR52688.2022.01103](https://doi.org/10.1109/CVPR52688.2022.01103)
- **S2:** `7c597874535c1537d7ddff3b3723015b4dc79d30` · citations **1,294** (influential 233)
- **Contribution:** A bidirectional transformer trained to predict randomly masked
  tokens, but *sampled* by iterative parallel refinement: predict all tokens at once,
  then re-mask and re-predict the least-confident ones. Up to 64× faster decoding than
  raster-scan AR, with strong ImageNet results and easy extension to editing tasks.
- **Why interesting:** The mask/absorbing branch in its purest algorithmic form. It is
  technically a masked-token model rather than a full diffusion process, but it is the
  direct ancestor of iterative masked-decoding schedulers used throughout modern
  discrete diffusion — worth reading to understand the sampling schedule independently
  of the diffusion formalism.

## E3. CDCD — Continuous diffusion for categorical data
- **Authors:** Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H. Richemond, Arnaud Doucet, Robin Strudel
- **Venue / year:** arXiv 2022 (DeepMind)
- **arXiv:** [2211.15089](https://arxiv.org/abs/2211.15089) (v3, cs.CL)
- **DOI:** [10.48550/arXiv.2211.15089](https://doi.org/10.48550/arXiv.2211.15089)
- **S2:** `22775e58932cdfbd273a2a835a22c5d86800a458` · citations **225** (influential 21)
- **Contribution:** Keeps diffusion continuous in *both* time and input space for
  categorical data: a continuous flow over probability simplices with a categorical
  noise process, plus a self-conditioning scheme. Demonstrates competitive language
  modeling while retaining the benefits of continuous-time diffusion.
- **Why interesting:** The main alternative to native discrete-state diffusion. Useful
  as a foil — it shows you can get a continuous-time score-based model for language
  without a CTMC, which sharpens the question of why the discrete formulation is worth
  it (any-order/parallel sampling, clean infilling, simple likelihood bounds).

## E4. Plaid — Likelihood-Based Diffusion Language Models
- **Authors:** Ishaan Gulrajani, Tatsunori B. Hashimoto
- **Venue / year:** NeurIPS 2023
- **arXiv:** [2305.18619](https://arxiv.org/abs/2305.18619)
- **DOI:** [10.48550/arXiv.2305.18619](https://doi.org/10.48550/arXiv.2305.18619)
- **S2:** `d9ffb44ee3c8ec0b6692df8a90451384c1edd89b` · citations **171** (influential 24)
- **Contribution:** A maximum-likelihood training recipe for diffusion LMs plus scaling
  laws for diffusion models. Trains and releases **Plaid 1B**, which outperforms
  GPT-2 124M in likelihood on standard benchmarks and generates fluent unconditional /
  zero-shot-controlled samples.
- **Why interesting:** The strongest early evidence that discrete diffusion is a
  *likelihood* method that can be scaled like AR models, not just a sampler heuristic.
  Its compute-optimal regimes differ substantially from AR scaling laws — a concrete,
  testable claim for anyone planning to train diffusion LMs.

## E5. MDLM — Simple and Effective Masked Diffusion Language Models
- **Authors:** Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, Volodymyr Kuleshov
- **Venue / year:** NeurIPS 2024
- **arXiv:** [2406.07524](https://arxiv.org/abs/2406.07524) (v2, cs.CL)
- **DOI:** [10.48550/arXiv.2406.07524](https://doi.org/10.48550/arXiv.2406.07524)
- **S2:** `f8d357d38bbcdd93889fe71762eb57842b2ab063` · citations **851** (influential 215)
- **Contribution:** Shows simple masked discrete diffusion is far better than previously
  reported. Derives a simplified, **Rao-Blackwellized** objective that reduces to a
  mixture of classical masked-language-modeling losses, enabling encoder-only models
  with efficient (including semi-autoregressive, arbitrary-length) samplers. Sets a new
  SOTA among diffusion LMs and approaches AR perplexity.
- **Why interesting:** The practical counterpart to SEDD. Where SEDD attacks the loss
  (score entropy over concrete scores), MDLM attacks the estimator (Rao-Blackwellization
  of the masking objective) and lands on a strikingly simple loss. Reading SEDD and MDLM
  together is the fastest way to understand the current training landscape for diffusion LMs.
- **Code:** https://github.com/kuleshov-group/mdlm

---

# Additional references (noise-formulation taxonomy)

Two further papers, retrieved **2026-09-27** while mapping *which discrete state spaces
admit a noise (rather than clean-state) parameterization*. They are not part of the Core 5
or Extended set; they are cited in `notes/paper-walkthrough.md` → "which discrete state
spaces admit a noise formulation".

## A1. Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning (Bit Diffusion)
- **Authors:** Ting Chen, Ruixiang Zhang, Geoffrey E. Hinton
- **Venue / year:** ICLR 2022
- **arXiv:** [2208.04202](https://arxiv.org/abs/2208.04202) (cs.LG)
- **DOI:** [10.48550/arXiv.2208.04202](https://doi.org/10.48550/arXiv.2208.04202)
- **S2:** `b64537bdf7a103aa01972ba06ea24a9c08f7cd74` · citations **505**
- **Open access:** <http://arxiv.org/pdf/2208.04202>
- **Why it matters:** Represents discrete data as real-valued "analog bits" and runs a
  standard *continuous* diffusion model on them, thresholding to bits at the end. A
  noise/`epsilon`-prediction formulation for binary data obtained by changing the
  **embedding** (bits as reals) rather than the corruption semantics. Also introduces
  self-conditioning and asymmetric time intervals.
- **Read for:** the embedding route to `epsilon`-prediction on discrete data (taxonomy row 4).

## A2. Dirichlet Diffusion Score Model for Biological Sequence Generation
- **Authors:** Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, Jian Zhou
- **Venue / year:** ICML 2023
- **arXiv:** [2305.10699](https://arxiv.org/abs/2305.10699) (q-bio.QM / cs.LG)
- **DOI:** [10.48550/arXiv.2305.10699](https://doi.org/10.48550/arXiv.2305.10699)
- **S2:** `4319a5faceb0f94fc791e49dc0b94dd4d142f90e` · citations **114**
- **PMID:** [37292476](https://pubmed.ncbi.nlm.nih.gov/37292476/)
- **Why it matters:** Runs a score-based SDE on the probability **simplex** (a continuous
  Dirichlet space) for discrete sequence generation — the "manufacture a continuous
  space" route, so standard score/`epsilon` prediction applies directly. The main
  alternative to native CTMC diffusion for discrete data.
- **Read for:** the simplex/Dirichlet route to a noise formulation (taxonomy row 5).

---

# Additional references (group diffusion)

References gathered **2026-09-27** for the walkthrough subsection "group diffusion". These
cover the *translation-invariant* family: state spaces that are groups, forward corruption
that is a random group translation, and the Fourier/convolution structure that follows. The
first group is the discrete instances and the general theory; the second is the continuous
(Lie-group) instances; the last two use groups for *equivariance* rather than as the state
space, and are included as contrasts. Records: `raw/group-diffusion-refs.json`.

## A3. SymmetricDiffusers: Learning Discrete Diffusion on Finite Symmetric Groups
- **Authors:** Yongxing Zhang, Donglin Yang, Renjie Liao
- **Venue / year:** ICLR 2025 (Oral); arXiv 2024
- **arXiv:** [2410.02942](https://arxiv.org/abs/2410.02942) (v2, cs.LG)
- **DOI:** [10.48550/arXiv.2410.02942](https://doi.org/10.48550/arXiv.2410.02942)
- **S2:** `a541288ed3336f561350407b2cb4b2f248373658` · citations **7**
- **Why it matters:** Discrete diffusion directly on the finite symmetric group `S_n`. The
  forward process is a **riffle shuffle** — a random walk on the finite group — and the
  diffusion length is chosen from random-walk-on-finite-groups mixing theory; the reverse is a
  generalized Plackett–Luce distribution. Tasks: sorting 4-digit MNIST, jigsaw puzzles, TSP.
- **Code:** https://github.com/DSL-Lab/SymmetricDiffusers

## A4. Learning Permutation Distributions via Reflected Diffusion on Ranks (Soft-Rank Diffusion)
- **Authors:** Sizhuang He, Yangtian Zhang, Shiyang Zhang, David van Dijk
- **Venue / year:** ICML 2026
- **arXiv:** [2603.17353](https://arxiv.org/abs/2603.17353) (v2, cs.LG)
- **Why it matters:** Follow-up to A3. Keeps the permutation state space but replaces
  shuffle-based corruption with a *soft-rank* continuous lift and uses contextualized
  generalized Plackett–Luce denoisers; better on long sequences.

## A5. Blackout Diffusion: Generative Diffusion Models in Discrete-State Spaces
- **Authors:** Javier E. Santos, Zachary R. Fox, Nicholas Lubbers, Yen Ting Lin
- **Venue / year:** ICML 2023 (PMLR v202, pp. 9034–9059)
- **arXiv:** [2305.11089](https://arxiv.org/abs/2305.11089) (cs.LG)
- **DOI:** [10.48550/arXiv.2305.11089](https://doi.org/10.48550/arXiv.2305.11089)
- **S2:** `fb03154bbf1348e796f9a82b2372a7d5e7e4b45a` · citations **30**
- **Why it matters:** Exact (non-variational) reverse-time analysis for *arbitrary*
  discrete-state Markov forward processes — the general framework in which group diffusion is
  the translation-invariant special case.
- **Note:** PMLR lists the paper as "Generative Diffusion Models in Discrete-State Spaces"; the
  arXiv title carries the "Blackout Diffusion" prefix. Same paper.

## A6. Convergence Analysis of Discrete Diffusion Model: Exact Implementation through Uniformization
- **Authors:** Hongrui Chen, Lexing Ying
- **Venue / year:** arXiv 2024
- **arXiv:** [2402.08095](https://arxiv.org/abs/2402.08095) (v2, stat.ML)
- **Why it matters:** CTMC formulation with a uniformization algorithm; Total-Variation and
  KL guarantees for sampling any distribution on a **hypercube** `(Z/2)^d`.

## A7. Harmonic flows and Markov dynamics on finite groups via pseudo-differential operators
- **Authors:** Anselmo Torresblanca-Badillo, Ronald Barrios-Garizao, Ronny Quiñonez-Martínez
- **Venue / year:** Journal of Pseudo-Differential Operators and Applications, **17**(1), Article 18 (2026), open access
- **DOI:** [10.1007/s11868-025-00759-7](https://doi.org/10.1007/s11868-025-00759-7)
- **Why it matters:** The closest thing found to the *mathematics of group diffusion*: builds
  diffusion on the finite abelian group `Z_N` via discrete Fourier analysis, convolution
  semigroups, negative definite functions and pseudo-differential operators; the generators
  are proved Feller, `m`-dissipative and self-adjoint. Explicitly flags extension to
  **non-abelian** finite groups as future work (spectral diagonalization becomes harder).

## A8. Markov processes on a circular lattice
- **Authors:** Sourav Majumdar
- **Venue / year:** arXiv 2026
- **arXiv:** [2603.02890](https://arxiv.org/abs/2603.02890) (math.ST)
- **Why it matters:** The cyclic-group instance — diffusion-generated families on the `m`-point
  discrete circle (cycle graph), with an explicit transition kernel, exact trigonometric
  moments, convergence to uniformity, and discrete von Mises / wrapped Cauchy stationary laws.

## A9. SE(3) diffusion model with application to protein backbone generation (FrameDiff)
- **Authors:** Jason Yim, Brian L. Trippe, Valentin De Bortoli, Emile Mathieu, Arnaud Doucet, Regina Barzilay, Tommi Jaakkola
- **Venue / year:** ICML 2023
- **arXiv:** [2302.02277](https://arxiv.org/abs/2302.02277) (v3, cs.LG)
- **Why it matters:** Invariant diffusion over rigid-body *frames* on `SE(3)`, with an
  `SE(3)`-equivariant score — the continuous-group analogue of "noise is a group element".

## A10. Unified framework for diffusion generative models in SO(3)
- **Authors:** Yesukhei Jagvaral, Francois Lanusse, Rachel Mandelbaum
- **Venue / year:** AAAI 2024
- **arXiv:** [2312.11707](https://arxiv.org/abs/2312.11707) (cs.LG)
- **Why it matters:** Extends both score-based and DDPM formulations to the Lie group `SO(3)`,
  exploiting its tractable heat kernel.

## A11. Denoising Diffusion Probabilistic Models on SO(3) for Rotational Alignment
- **Authors:** Adam Leach, Sebastian M. Schmon, Matteo T. Degiacomi, Chris G. Willcocks
- **Venue / year:** ICLR 2022 Workshop on Geometrical and Topological Representation Learning
- **URL:** <https://iclr.cc/virtual/2022/8698> · ML Anthology: `iclrw/2022`
- **Why it matters:** An early DDPM defined on the rotation group `SO(3)`.
- **Note:** arXiv ID not confirmed this session.

## A12. Riemannian Score-Based Generative Modelling
- **Authors:** Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, Arnaud Doucet
- **Venue / year:** NeurIPS 2022
- **arXiv:** [2202.02763](https://arxiv.org/abs/2202.02763) (v3, cs.LG)
- **Why it matters:** The general Riemannian-manifold theory of which the `SO(3)`/`SE(3)` group
  models are special cases.

## A13. Diffusion Generative Modeling on Lie Group Representations
- **Authors:** Marco Bertolini, Tuan Le, Djork-Arné Clevert
- **Venue / year:** NeurIPS 2025 (Spotlight)
- **arXiv:** [2502.02513](https://arxiv.org/abs/2502.02513) (v2, cs.LG)
- **Why it matters:** Score-based diffusion in the *representation space* of any (non-abelian)
  Lie group; states that ordinary Euclidean score matching is recovered as the special case of
  the **translation group**. Applications: `SO(3)` conformers, `SE(3)` docking.

## A14. Permutation-Symmetrized Diffusion for Unconditional Molecular Generation
- **Authors:** Gyeonghoon Ko, Juho Lee
- **Venue / year:** arXiv 2026 (ICLR 2026)
- **arXiv:** [2603.23255](https://arxiv.org/abs/2603.23255) (cs.LG)
- **Why it matters (as a contrast):** Diffuses on the *quotient manifold* `R^{d x N}/S_N`
  instead of on the group itself — the group action is removed rather than used as the state
  space.

## A15. SymDiff: Equivariant Diffusion via Stochastic Symmetrisation
- **Authors:** Leo Zhang, Kianoosh Ashouritaklimi, Yee Whye Teh, Rob Cornish
- **Venue / year:** ICLR 2025
- **arXiv:** [2410.06262](https://arxiv.org/abs/2410.06262) (v2, cs.LG)
- **Why it matters (as a contrast):** Uses groups for *equivariance* of the model, with a
  Euclidean state space — the common meaning of "group" in generative modeling, distinct from
  group-valued state spaces.

## A16. Structure Preserving Diffusion Models
- **Authors:** Haoye Lu, Spencer Szabados, Yaoliang Yu
- **Venue / year:** arXiv 2024
- **arXiv:** [2402.19369](https://arxiv.org/abs/2402.19369) (v2, cs.LG)
- **Why it matters (as a contrast):** Diffusion processes that preserve group-invariant
  properties of the data distribution; again a network/process symmetry, not a group state
  space.

---

## Provenance

| Source | Endpoint | Used for |
| --- | --- | --- |
| arXiv API | `export.arxiv.org/api/query?id_list=...` | Titles, authors, abstracts, versions, dates |
| Semantic Scholar Graph API | `api.semanticscholar.org/graph/v1/paper/arXiv:<id>` (header `x-api-key`) | S2 paper IDs, citation/influential snapshots, venues, DOIs, TLDRs |
| Semantic Scholar Graph API (bulk search) | `.../paper/search/bulk?query=...` (header `x-api-key`) | Additional refs A1–A2 (Analog Bits 2208.04202, Dirichlet Diffusion 2305.10699) |
| OpenAlex | `api.openalex.org/works` | DOI/venue cross-checks (VQ-Diffusion) |
| arXiv API (batch `id_list`) | `export.arxiv.org/api/query?id_list=...` | Group-diffusion refs A3–A6, A8–A16 titles/authors/dates/versions |
| Semantic Scholar Graph API (bulk search) | `.../paper/search/bulk?query=...` | S2 IDs and citation snapshots for A3 (SymmetricDiffusers) and A5 (Blackout Diffusion) |
| Web search + Springer article page | `link.springer.com/article/10.1007/s11868-025-00759-7` | A7 bibliographic details (authors, journal, volume, open access) and A11 venue |

- Raw Semantic Scholar responses: `raw/s2-<arxiv-id>.json` (all 10 papers).
- Group-diffusion records (A3–A16): `raw/group-diffusion-refs.json` (arXiv API + S2 bulk
  search + web/Springer, retrieved 2026-09-27).
- Semantic Scholar and OpenAlex were intermittently rate-limited during retrieval
  (HTTP 429); retries with the API key completed the set. All identifiers and counts
  above were returned directly by the listed endpoints.
