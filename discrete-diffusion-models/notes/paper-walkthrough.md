# Paper Walkthrough — Study Order

Durable notes from a paper-by-paper read of the topic. Each entry
records **3 key points** and the **main innovation**. This is an evolving document;
entries are refined as follow-up questions are answered.

**Progress:** 3 / 11.

## Reading order

Numbered in **study order** (venue in parentheses). Entries 1, 2 and 4–11 follow first arXiv
posting; entry 3 is a **foundations interlude** inserted deliberately — it is a 2026 journal
article, and its operator-theoretic view of translation-invariant (group-convolution) kernels
is the right preparation for D3PM's structured kernels. See the note under entry 3.

1. 2015-03-12 — Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* (ICML 2015) — `1503.03585`
2. 2021-02-10 — Hoogeboom et al., *Argmax Flows and Multinomial Diffusion* (NeurIPS 2021) — `2102.05379`
3. 2026 — Torresblanca-Badillo, Barrios-Garizao & Quiñonez-Martínez, *Harmonic flows and Markov dynamics on finite groups via pseudo-differential operators* (J. Pseudo-Differ. Oper. Appl. 17(1), Art. 18) — `10.1007/s11868-025-00759-7`
4. 2021-07-07 — Austin et al., *Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)* (NeurIPS 2021) — `2107.03006`
5. 2021-11-29 — Gu et al., *Vector Quantized Diffusion (VQ-Diffusion)* (CVPR 2022) — `2111.14822`
6. 2022-02-08 — Chang et al., *MaskGIT* (CVPR 2022) — `2202.04200`
7. 2022-11-01 — Meng et al., *Concrete Score Matching* (NeurIPS 2022) — `2211.00802`
8. 2022-11-28 — Dieleman et al., *CDCD: Continuous diffusion for categorical data* (arXiv 2022) — `2211.15089`
9. 2023-05-30 — Gulrajani & Hashimoto, *Plaid: Likelihood-Based Diffusion Language Models* (NeurIPS 2023) — `2305.18619`
10. 2023-10-25 — Lou, Meng, Ermon, *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD)* (ICML 2024 Oral) — `2310.16834`
11. 2024-06-11 — Sahoo et al., *Simple and Effective Masked Diffusion Language Models (MDLM)* (NeurIPS 2024) — `2406.07524`

---

## 1. Deep Unsupervised Learning using Nonequilibrium Thermodynamics
*Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli — ICML 2015 · arXiv:1503.03585 · S2 `2dcef55a…`*

### 3 key points

1. **Generation as iterative denoising.** The paper defines a fixed *forward*
   Markov chain that gradually destroys structure in the data
   (`q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)`), and learns the *reverse*
   chain (`p_θ(x_{t-1} | x_t)`) to reconstruct it. A hard generative problem is thus
   replaced by a long sequence of small, easy denoising steps.
2. **Tractable training *and* evaluation from one bound.** The reverse chain is fit
   by a variational lower bound on the data log-likelihood. Under a Gaussian forward
   the bound simplifies into a sum of per-step denoising terms (a weighted
   reconstruction/denoising loss — the direct ancestor of DDPM's simplified
   objective, later recognized as a form of denoising score matching). The same bound
   yields an analytically computable entropy / bits-per-dim, so sampling, likelihood
   evaluation, and conditional inference (e.g. inpainting) all come from one model.
3. **The discrete case is already here.** Besides Gaussian diffusion, the paper
   derives a **binomial diffusion** over binary state spaces (bit-flip corruption with
   a learned Bernoulli reverse). Experiments were still mostly continuous/binary-image
   (CIFAR-10, MNIST), but this is the conceptual seed of discrete diffusion: the
   corruption-and-denoise recipe is not tied to continuous states.

### Main innovation

The **diffusion probabilistic model** itself: a fixed forward noising process plus a
learned reverse denoising process trained by a variational bound, motivated by
nonequilibrium statistical mechanics. It is the template that DDPM (2020) popularized
and that every discrete diffusion model in this topic reuses — swapping the Gaussian
corruption for a discrete one. Notably, the paper predates DDPM by roughly five years
and went largely unnoticed until DDPM demonstrated strong sample quality.

### Why it matters for this topic

Everything downstream is a variation on this template. The two levers the rest of the
field explores are already visible here: (a) *what corruption process to use* (Gaussian
vs. binomial, later uniform/absorbing/structured), and (b) *what objective/bound to
optimize* (variational ELBO vs. simplified denoising loss, later concrete-score /
score-entropy objectives).

### Intuition: the thinking behind the bound

A narrative of *why* each equation exists, in the order you would invent them.

1. **The intractable object.** The model is defined through $T$ latents, so
   $p(x_0) = \int p(x_{0:T})\,dx_{1:T}$ is a hopeless integral. The first move is to
   stop trying to compute it and look for a sample-estimable surrogate.

2. **Reframe: don't integrate, traverse.** Declare that generation *is* a path — a
   fixed forward process that destroys data into noise, plus a learned reverse
   process. The goal becomes "learn reverse transitions," not "compute a density."

3. **An identity lets sampling stand in for integration.** For any $q$,
   $$\log p(x_0) = \log \mathbb{E}_{q(x_{1:T}\mid x_0)}\!\left[\frac{p(x_{0:T})}{q(x_{1:T}\mid x_0)}\right].$$
   The left side is an integral; the right side is an expectation we can Monte-Carlo
   by sampling one trajectory and evaluating the ratio.

4. **Jensen, because $\log$ is concave.** Pushing the log inside the expectation can
   only lower it, giving the bound $K$.

5. **The slack is a promise.**
   $\log p(x_0)-K = D_{\mathrm{KL}}(q(x_{1:T}\mid x_0)\,\|\,p_\theta(x_{1:T}\mid x_0))$,
   exactly the mismatch between the true reverse path and the learned one. So raising
   $K$ is a principled surrogate for maximum likelihood.

6. **Choose $q$ = the fixed forward chain.** Sampling a trajectory then needs no
   learning, the forward transition is known, and the only unknown left in $K$ is the
   reverse chain.

7. **Expanding shows an asymmetry.** Forward terms are $x_{t-1}\to x_t$ (known);
   reverse terms are $x_t\to x_{t-1}$ (learned). To train step $t$ we need a
   computable target pointing the *same way as the model*.

8. **The Bayes move (the heart).**
   $q(x_t\mid x_{t-1}) = \dfrac{q(x_{t-1}\mid x_t,x_0)\,q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}$.
   The target $q(x_{t-1}\mid x_t,x_0)$ now points the model's direction, the marginals
   telescope, and $K$ becomes a sum of per-step KLs. This rewrite is *why* diffusion
   training is a sequence of denoising problems.

9. **Why that target is computable — and what it means.** The forward is chosen so its
   posterior stays in a simple family. Conceptually $q(x_{t-1}\mid x_t,x_0)$ is the
   **Bayes-optimal reverse step** for the fixed forward process; training is
   *supervised distillation of the exact posterior*, one noise level at a time. At
   sampling $x_0$ is unavailable, so the network learns the $x_0$-averaged version —
   which is what "denoising" means.

10. **Entropy constants are the ends of the telescope.** $H_q(X_1\mid X_0)$,
    $H_q(X_T\mid X_0)$, $H_p(X_T)$ do not involve $\theta$; they are dropped for
    optimization but kept for a valid likelihood bound.

11. **DDPM's regrouping is bookkeeping.** Prior-matching, denoising, reconstruction —
    the same terms, merely ordered. Parameterization and weighting choices act on
    these identical terms.

12. **Why the paper says "thermodynamics."** Each reverse step dissipates a little
    free energy; the KL *is* the dissipation; the sum is total dissipation; the bound
    is the second law; equality is the quasi-static/reversible limit (infinitely slow,
    small $\beta$). Hence the schedule controls how much is lost, and Appendix B
    quantifies the unavoidable per-step entropy production.

**The design space this yields:** (1) choose the forward kernel so the posterior is
fixed/tractable and the reverse expressive; (2) parameterize the reverse
(mean/$\varepsilon$/$x_0$, or the concrete score); (3) weight the per-step KLs.
Discrete diffusion changes (1) only — replace the Gaussian with a categorical $Q_t$
and every step above is unchanged.

### Deep dive: the variational bound (K)

This is the paper's training objective and the ancestor of every discrete-diffusion
loss. It is the standard **evidence lower bound (ELBO)**, obtained by Jensen's
inequality on the latent diffusion chain:

$$
\begin{aligned}
\log p_\theta(x_0)
&= \log \int p_\theta(x_{0:T})\,dx_{1:T} \\
&= \log \mathbb{E}_{q(x_{1:T}\mid x_0)}\!\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right] \\
&\ge \mathbb{E}_q\!\left[\log p_\theta(x_{0:T}) - \log q(x_{1:T}\mid x_0)\right] \;=\; K
\end{aligned}
$$

The **gap is exact and interpretable**:

$$
\log p_\theta(x_0) - K = D_{\mathrm{KL}}\!\left(q(x_{1:T}\mid x_0)\,\|\,p_\theta(x_{1:T}\mid x_0)\right)
$$

So $K = \log p_\theta(x_0)$ iff the learned reverse trajectory reproduces the true
posterior — the paper's *quasi-static* (thermodynamically reversible) case.

Expanding the chain gives the starting form:

$$
K = \mathbb{E}_q\!\left[\log p(x_T) + \sum_{t=1}^{T}\log p_\theta(x_{t-1}\mid x_t) - \sum_{t=1}^{T}\log q(x_t\mid x_{t-1})\right]
$$

The paper rearranges this (Eq. 15), using the fact that the *forward posterior*
$q(x_{t-1}\mid x_t, x_0)$ is tractable, into:

$$
K = -\sum_{t=2}^{T}\mathbb{E}_{q(x_0,x_t)}\!\left[D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\right)\right]
    + H_q(X_T\mid X_0) - H_q(X_1\mid X_0) - H_p(X_T)
$$

- The **sum of KLs** is the learnable part: match the learned reverse step to the
  closed-form forward posterior. Each KL is between two distributions of the *same
  family* (Gaussians for Gaussian diffusion, Bernoullis for binomial diffusion), so
  it is analytic and minimizing it is just regression — on the mean/variance, or on
  the bit-flip probability.
- $H_q(X_T\mid X_0)$, $H_q(X_1\mid X_0)$, $H_p(X_T)$ are **entropy constants** for a
  fixed forward $\beta_t$ schedule (in this paper they also *learn* $\beta_{2:T}$ by
  gradient ascent on $K$, with $\beta_1$ fixed small).
- The $t=0$ (reconstruction) edge is handled by construction: they set the final
  reverse step equal to the corresponding forward kernel to avoid an edge effect.

**The modern (DDPM) rearrangement** of the same bound is the clearest way to read it:

$$
\begin{aligned}
L ={} & D_{\mathrm{KL}}\!\left(q(x_T\mid x_0)\,\|\,p(x_T)\right) && \text{prior matching} \\
& + \sum_{t=2}^{T}\mathbb{E}_q\!\left[D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\right)\right] && \text{denoising} \\
& - \mathbb{E}_q\!\left[\log p_\theta(x_0\mid x_1)\right] && \text{reconstruction}
\end{aligned}
$$

with $\log p_\theta(x_0) \ge -L$.

**Why this matters for discrete diffusion.** The bound is *kernel-agnostic*: swap the
Gaussian forward for uniform/absorbing/structured corruption and $q(x_{t-1}\mid x_t,x_0)$
becomes a tractable product of categoricals, so each KL becomes a sum of categorical
cross-entropies. That is exactly D3PM's ELBO (plus its auxiliary cross-entropy term).
Later objectives are departures from this bound: SEDD's score-entropy loss is a
*different estimator* (ratio matching), and MDLM's objective is a Rao-Blackwellized
rewrite of the masking-specific form of this same bound.

### Where Eq. (15) comes from (the step usually skipped)

Start from the trajectory form and use Bayes on a single forward step:

$$
q(x_t\mid x_{t-1}) = \frac{q(x_{t-1}\mid x_t, x_0)\, q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}
$$

Substituting this into $K$ and telescoping the ratio $q(x_t\mid x_0)/q(x_{t-1}\mid x_0)$
across $t=1..T$ collapses the two trajectory sums into per-step KLs against the
*tractable forward posterior* $q(x_{t-1}\mid x_t,x_0)$ — which is exactly Eq. (15).
The $t=1$ and $t=T$ edges leave the entropy constants. So the bound is not an
approximation layered on top: it is the chain rule plus Jensen, nothing more.

### Thermodynamic reading

The paper frames $K$ as a free-energy / entropy-production quantity:

- The gap $\log p_\theta(x_0) - K$ is the KL between the forward and reverse
  trajectories — the "work dissipated" by an irreversible process.
- Equality (zero gap) is the *quasi-static* / reversible limit where forward and
  reverse coincide.
- Appendix B bounds the per-step entropy production using only the forward chain:

$$
H_q(X_{t-1}\mid X_t) \;\ge\; H_q(X_t\mid X_{t-1}) + H_q(X_{t-1}\mid X_0) - H_q(X_t\mid X_0)
$$

$$
H_q(X_{t-1}\mid X_t) \;\le\; H_q(X_t\mid X_{t-1})
$$

Both sides are analytically computable, which is what makes the bound usable as a
diagnostic (and why the paper reports bits-per-dim rather than a sampling proxy).

### Deep dive: the binomial (discrete) case

The paper's second kernel — the seed of all discrete diffusion. Table App.1 defines it;
§2.2 and the appendix explain the reversal. Note carefully: the paper's $\mathcal{B}(u;r)$
is a **single Bernoulli trial**, not a binomial. The state is a vector of *independent*
binary variables corrupted bit-by-bit, so the modern reading is **uniform-state
categorical diffusion with $K=2$** and a factorized forward; the aggregate *count* of ones
is what is binomial.

Table App.1 (binomial column):

$$
\pi(x^{(T)}) = \mathcal{B}(x^{(T)}; 1/2), \qquad
q(x^{(t)}\mid x^{(t-1)}) = \mathcal{B}\!\left(x^{(t)};\; x^{(t-1)}(1-\beta_t) + \tfrac{\beta_t}{2}\right),
$$

$$
p_\theta(x^{(t-1)}\mid x^{(t)}) = \mathcal{B}\!\left(x^{(t-1)};\; f_b(x^{(t)},t)\right),
\qquad
\pi(x^{(T)})=\mathcal{B}(x^{(T)};1/2).
$$

**Mechanics.** With $\lambda_t = 1-\beta_t$ the forward is a *symmetric binary channel*:
stay with probability $(1+\lambda_t)/2$, flip with probability $(1-\lambda_t)/2 = \beta_t/2$
— the actual flip probability is $\beta_t/2$, not $\beta_t$. The correlation to the clean
bit multiplies across steps; with $\rho_t=\prod_{s\le t}\lambda_s$,

$$
q(x_t=1\mid x_0=1)=\frac{1+\rho_t}{2}, \qquad
q(x_t=1\mid x_0=0)=\frac{1-\rho_t}{2}.
$$

The number of ones is a sum of two binomials (hence "binomial diffusion"):
$X_t \sim \mathrm{Bin}\!\big(n,\tfrac{1+\rho_t}{2}\big) + \mathrm{Bin}\!\big(N-n,\tfrac{1-\rho_t}{2}\big)$,
where $n$ is the number of ones in $x_0$.

**The posterior the bound needs** (analytic per the paper, but never printed). Bayes on the
two-state chain, with $A=q(x^{(t-1)}=1\mid x_0)=\tfrac{1+\rho_{t-1}s_0}{2}$ and
$s_0=2x_0-1\in\{-1,+1\}$:

$$
q(x^{(t-1)}=1\mid x_t=1,x_0) = \frac{(1+\lambda_t)\,A}{(1+\lambda_t)\,A+(1-\lambda_t)(1-A)},
$$

$$
q(x^{(t-1)}=1\mid x_t=0,x_0) = \frac{(1-\lambda_t)\,A}{(1-\lambda_t)\,A+(1+\lambda_t)(1-A)}.
$$

Each is a Bernoulli probability, so the per-step KL is a sum of binary cross-entropies —
**verbatim the D3PM loss**.

**This *is* D3PM's uniform kernel.** D3PM's $Q_t=(1-\beta_t)I+\frac{\beta_t}{K}\mathbf{1}\mathbf{1}^\top$
at $K=2$ has diagonal $1-\beta_t/2$ and off-diagonal $\beta_t/2$ — identical to the
binomial kernel. So "binomial diffusion" is uniform-state discrete diffusion, restricted
to binary states.

**What is trained.** Gaussian: $f_\mu, f_\Sigma, \beta_{1..T}$. Binomial: **only** $f_b$,
with a fixed forward schedule — because, as the paper states, "the discrete state space
makes gradient ascent with frozen noise impossible" (no reparameterization trick for
discrete states).

**Closed-form guidance.** The perturbed reverse for the binomial case is *again* Bernoulli,
with a closed-form Bayes rate — no Taylor linearization, unlike the Gaussian case. An early
instance of classifier guidance in discrete space.

**Shortcomings (→ the rest of the reading list).** Binary only; uniform stationary only
(no absorbing/mask state, so no natural infilling story); mean-field reverse; no
simplified/reweighted loss, no score objective, no CTMC view.

---

## 2. Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions
*Hoogeboom, Nielsen, Jaini, Forré, Welling — NeurIPS 2021 · arXiv:2102.05379 ·
S2 metadata in `sources/foundational-papers.md`*

### 3 key points

1. **Two routes to categorical likelihoods.** The paper contributes both **Argmax Flows**
   (a continuous normalizing flow followed by an argmax layer, trained with a learned
   probabilistic inverse of argmax) and **Multinomial Diffusion** (diffusion defined
   directly on categorical variables). Both are likelihood-based, non-autoregressive
   alternatives to ARMs for language and segmentation maps.
2. **Multinomial Diffusion = uniform categorical corruption with a closed-form
   posterior.** States are one-hot $x_t \in \{0,1\}^K$; each step resamples a category
   uniformly with probability $\beta_t$:

$$
q(x_t\mid x_{t-1}) = \mathcal{C}\!\left(x_t \,\middle|\, (1-\beta_t)\,x_{t-1} + \beta_t/K\right)
$$

   Generalizing Sohl-Dickstein's binary binomial to $K$ categories, both the marginal and
   the posterior stay closed-form:

$$
q(x_t\mid x_0) = \mathcal{C}\!\left(x_t \,\middle|\, \bar\alpha_t\,x_0 + (1-\bar\alpha_t)/K\right)
$$

$$
\theta_{\mathrm{post}}(x_t,x_0) \;\propto\; \big[\alpha_t\,x_t + (1-\alpha_t)/K\big] \odot \big[\bar\alpha_{t-1}\,x_0 + (1-\bar\alpha_{t-1})/K\big]
$$

   with $\alpha_t = 1-\beta_t$, $\bar\alpha_t = \prod_{\tau\le t}\alpha_\tau$, and $\odot$ the
   elementwise (Hadamard) product, normalized over categories.
3. **The $x_0$-parameterization (the key practical move).** Rather than predict noise
   (hard for discrete data), the network outputs a **clean-category probability vector**
   $\hat x_0 = \mu(x_t,t)$ (softmax), which is plugged straight back into the posterior:

$$
p(x_{t-1}\mid x_t) = \mathcal{C}\!\left(x_{t-1} \,\middle|\, \theta_{\mathrm{post}}(x_t,\hat x_0)\right),
\qquad
\log p(x_0\mid x_1) = \sum_k x_{0,k}\log \hat x_{0,k}.
$$

   The per-step ELBO term is a categorical KL between
   $\theta_{\mathrm{post}}(x_t,x_0)$ and $\theta_{\mathrm{post}}(x_t,\hat x_0)$. So the whole
   discrete-diffusion objective collapses to **cross-entropy on a predicted clean
   distribution** — the direct ancestor of the modern masked-diffusion loss.

### Main innovation

**Multinomial (categorical) diffusion**: taking the 2015 binomial kernel to $K$ classes
and, crucially, introducing the **$x_0$-prediction parameterization** of the reverse
process. Instead of regressing noise, the model predicts the clean categorical
distribution and reuses the closed-form posterior, which turns the ELBO into a simple,
low-variance cross-entropy and makes discrete diffusion trainable at scale. (Argmax Flows
are the paper's other, independent contribution: extending continuous flows to
categorical data through an argmax layer.)

### Why it matters for this topic

This is where discrete diffusion becomes a *general categorical* method rather than a
binary curiosity. Two ideas propagate through the rest of the list:

- **Uniform corruption + closed-form posterior** — the "uniform" corner of the kernel
  design space, later contrasted with D3PM's absorbing/mask and structured kernels.
- **Predict $\hat x_0$, not noise** — inherited by D3PM's auxiliary $x_0$-loss and by the
  mask-prediction objective of MaskGIT/MDLM. Almost every discrete-diffusion loss later in
  this list is a variation on `cross_entropy(x_0, model(x_t, t))`.

Experiments: text8 (1.72 bpc) and enwik8 (1.75 bpb) with a 12-layer Transformer diffusion
model, plus unconditional image-segmentation maps. It loses to 64-layer Transformer ARMs
on text, but is a competitive non-AR likelihood model; the paper does not chase sample
quality (that arrives with VQ-Diffusion and MaskGIT).

### Cross-cutting: $x_0$- vs. noise-parameterization

Why discrete diffusion predicts the **clean state** rather than the noise. The distinction
flips between continuous and discrete state spaces.

**Continuous: the two targets are affine twins.** With
$q(x_t\mid x_0)=\mathcal{N}(x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$,

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,
$$

so the noise and clean targets are related by an invertible affine map,

$$
\varepsilon=\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}},
\qquad
x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon}{\sqrt{\bar\alpha_t}},
$$

and a model trained on one is the other up to loss weighting (a factor
$\bar\alpha_t/(1-\bar\alpha_t)$ per step). The reverse mean is identical either way:

$$
\mu_\theta=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta\Big)
=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,c_\theta
+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t,
$$

where $c_\theta$ is the predicted clean value (the paper's $\hat x_0$). The differences are
only (i) *target scaling / noise-level weighting*, (ii) *numerical conditioning*
($x_0$-prediction divides by $\sqrt{\bar\alpha_t}\to0$ at high noise, $\varepsilon$-prediction
amplifies at low noise — which is exactly why $v$-prediction and EDM preconditioning exist),
and (iii) whether the reverse variance is learned. The real degree of freedom is the **loss
weighting across $t$**: DDPM's "simple loss" $\|\varepsilon-\varepsilon_\theta\|^2$ simply drops
the exact ELBO weights.

**Discrete: $x_0$ is the natural coordinate, not a free choice.** There is no additive
noise — corruption is *replacement* (to uniform or to a mask token) — so there is no
$\varepsilon=x_t-\ldots$ to regress. The closed-form posterior is indexed by the clean state,

$$
\theta_{\mathrm{post}}(x_t,x_0)\propto[\alpha_t x_t+(1-\alpha_t)/K]\odot[\bar\alpha_{t-1}x_0+(1-\bar\alpha_{t-1})/K],
$$

so the network supplies a clean-token distribution $c_\theta=\mu_\phi(x_t,t)$ and the
*known* posterior does the rest: $\pi_\theta=\theta_{\mathrm{post}}(x_t,c_\theta)$,
$x_{t-1}\sim\mathcal{C}(\pi_\theta)$. Two consequences: softmax + posterior automatically
yield a valid reverse distribution (no projection step), and the **schedule, not the
network, sets the reverse entropy** — near $t{=}T$ the posterior is almost uniform and barely
depends on $c_\theta$, near $t{=}0$ it is sharp. So the schedule is the curriculum and the
cross-entropy carries an implicit $t$-dependent weight.

**The deep contrast.** Both routes let the network supply only the part that needs data and
let the fixed forward process supply the rest. Noise prediction learns a **residual** and
inverts the forward process through **algebra**; $x_0$-prediction learns a **distribution over
clean states** and applies the forward process through **Bayes' rule** — i.e. amortized
posterior inference. For absorbing/mask corruption the posterior says "unmasked positions
are known, masked positions must be guessed," so the bound reduces to weighted masked-LM
cross-entropy (MDLM, paper 11).

**When a noise formulation returns.** $x_0$-prediction is forced only by the *combination*
(replacement corruption) + (native categorical state space) + (variational-bound route).
Change any one: the **concrete-score** objective predicts probability *ratios* (the discrete
analogue of the continuous score) instead of the clean token (Meng 2022, SEDD 2023);
**lattice/ordinal** states admit integer-shift "noise" (D3PM's discretized-Gaussian kernel);
and **embedding** the categorical in a continuous space restores standard $\varepsilon$/score
prediction (CDCD, paper 8).

### Cross-cutting: which discrete state spaces admit a noise formulation

**Is $x_0$-prediction Bernoulli-specific?** No. Bernoulli is just $K{=}2$; the argument is
identical for any $K$-category categorical. The constraint is the *pair* (replacement
corruption) + (no algebraic structure on the state space) — not binary-ness. Formally, a
noise coordinate exists iff the corruption can be written $x_t=T(x_0,g)$ with $g$ independent
of $x_0$ and invertible enough that the reverse step can be built from $g$. Categorical
replacement fails this: the only "noise" is "$x_t$ is a different category," which is
*observed*, so there is no hidden residual to invert. That reframes the question as: *which
discrete state spaces carry such a structure?*

**1. Native discrete score (ratios) — no $x_0$.** Even in pure categorical space one avoids
$x_0$ by predicting the *concrete score* over one-token neighbours $y$,
$s_\theta(x_t)_y=p_\theta(y)/p_\theta(x_t)$ — the discrete analogue of $\nabla\log p$ — with the
reverse CTMC rate built directly from ratios. So $x_0$-prediction is one of *two* native
parameterizations, not the only one (Meng 2022, paper 7; SEDD 2023, paper 10).

**2. Lattice / ordinal states.** If $x\in\{0,\dots,K-1\}$ denotes a quantized ordinal value,
the displacement $\delta=x_t-x_0\in\mathbb{Z}$ is meaningful and the shift itself is the noise
target. This is D3PM's discretized-Gaussian kernel (paper 4); uniform and absorbing are
degenerate limits.

**3. Group-structured states.** For $x\in\mathbb{Z}/N\mathbb{Z}$ or permutations $S_n$, corruption
is a group translation $x_t=x_0\oplus g$; the noise is the group element $g$, the reverse
"de-translates," and the posterior is a group convolution — the discrete sibling of additive
Gaussian noise.

**4. Binary in $\pm1$ coordinates (a vector-space trick).** Encode a bit as $s\in\{-1,+1\}$ and
a flip becomes *multiplicative* noise $s_t=\xi_t s_0$, $\xi_t\in\{\pm1\}$; relaxing to reals
gives ordinary continuous diffusion with $\varepsilon$-prediction, thresholded at the end.
→ Analog Bits / "Bit Diffusion" (Chen, Zhang & Hinton, arXiv:2208.04202, ICLR 2022,
S2 `b64537bdf7a103aa01972ba06ea24a9c08f7cd74`).

**5. Continuous embedding (manufacture a vector space).** Put the categorical on the
probability simplex and run a continuous score-SDE (→ *Dirichlet Diffusion Score Model*,
Avdeyev et al., arXiv:2305.10699, ICML 2023, S2 `4319a5faceb0f94fc791e49dc0b94dd4d142f90e`,
PMID 37292476), or diffuse in a learned logit embedding (→ score interpolation / **CDCD**,
paper 8). Standard $\varepsilon$/score prediction then applies directly.

**6. Count / nonnegative-integer states.** If corruption is *additive* Poisson/binomial noise
on a count, the increment is a genuine additive noise target. (Principle only; no canonical
reference pinned in this session.)

| State space | Corruption | "Noise" variable | Example |
| --- | --- | --- | --- |
| Categorical $\{1..K\}$ | replacement → uniform/mask | none additive; use ratios | Meng 2022; SEDD |
| Ordinal $\{0..K-1\}$ | discretized Gaussian | integer shift $\delta$ | D3PM (paper 4) |
| Group $G$ ($\mathbb{Z}/N$, $S_n$) | group translation | group element $g$ | group diffusion |
| Binary $\{-1,+1\}$ | sign flip | $\xi\in\{\pm1\}$ | Analog Bits (2208.04202) |
| Simplex / logits | Gaussian | $\varepsilon$ / score | Dirichlet Diff. (2305.10699); CDCD (paper 8) |
| Counts $\mathbb{N}$ | Poisson/binomial add | count increment | count diffusion |

**The precise nuance.** "$x_0$ is forced" is shorthand for *the variational-bound posterior is
naturally indexed by $x_0$.* One can mimic noise prediction by predicting the replacement
indicator $g$ (determined by $(x_0,x_t)$), which given $x_t$ carries the same information as
$x_0$ — but the reverse step still needs a full *distribution* over the previous state, and
the closed-form posterior is cleanest in terms of the clean state. So that is a relabelling,
not a simplification. The one route that genuinely changes the **estimator** (rather than
relabelling it) is the ratio/score objective, which bypasses the ELBO entirely. Finally, the
neat $\odot$ "product of two noisy estimates" form is specific to the **uniform** kernel;
absorbing/mask corruption has a different posterior (deterministic on unmasked positions),
which is exactly why masked diffusion collapses to a plain cross-entropy.

---

### Cross-cutting: group diffusion (the translation-invariant family)

> **New to groups?** → [`group-theory-primer.md`](group-theory-primer.md) builds the algebra
> from zero (what a group is, abelian vs non-abelian, characters and the dual group,
> convolution as translation-invariance, random walks as the forward process, why
> absorbing/mask is *not* a group, Lie groups, and the group-as-state-space
> vs group-as-symmetry trap).

The taxonomy above listed "group $G$" as a single row. This section expands it, because it is
the one place where discrete diffusion keeps a genuine **additive-noise** picture — and
because paper 2's own kernel turns out to be the *degenerate* member of the family.

"Group diffusion" is a descriptive name rather than a single canonical paper: the state space
is a (finite or compact Lie) group $G$, and corruption is **translation by a random group
element** instead of replacement. The closest formal treatments found in this pass are a
finite-group Fourier construction [A7] (studied as **paper 3** below) and the exact discrete-state analysis of [A5]; the
applied instances are the permutation-group model SymmetricDiffusers [A3] and its soft-rank
follow-up [A4], the cycle-graph construction [A8], and the Lie-group models [A9]–[A13].

#### The mechanism

Write the group operation additively for abelian $G$ (composition for non-abelian). The
forward step is

$$x_t = x_{t-1} + g_t, \qquad g_t \sim \kappa_t \;\text{a distribution on } G.$$

Because translations compose, the cumulative corruption is the group convolution

$$\gamma_t = \kappa_1 \ast \kappa_2 \ast \cdots \ast \kappa_t,
  \qquad q(x_t \mid x_0) = \gamma_t(x_t - x_0),$$

the exact analogue of $q(x_t\mid x_0)=\mathcal{N}(\sqrt{\rho_t}x_0,\,(1-\rho_t)I)$. The reverse
posterior is a ratio of convolutions,

$$q(x_{t-1}\mid x_t,x_0)=
  \frac{\kappa_t(x_t-x_{t-1})\,\gamma_{t-1}(x_{t-1}-x_0)}{\gamma_t(x_t-x_0)},$$

and the stationary law is the Haar (uniform) measure. The payoff is the **group Fourier
transform**: convolution becomes pointwise multiplication in the dual group, so for every
character $\chi$ (abelian case)

$$\mathcal{F}(\gamma_t)(\chi)=\prod_{s\le t}\mathcal{F}(\kappa_s)(\chi).$$

The whole forward process — marginals, entropy terms, schedule — is therefore a family of
eigenvalues $\mathcal{F}(\kappa_t)(\chi)$ rather than a $K\times K$ matrix. This is the discrete
form of "the Gaussian is diagonal in the Fourier basis".

Because the corruption is invertible and additive, the group element $g$ is *observable* from
$(x_0,x_t)$ and can be predicted. Group diffusion is exactly the family in which the
$\varepsilon$-style parameterization returns (taxonomy row 3).

#### Paper 2 is the degenerate case: uniform replacement is a group diffusion

Hoogeboom's kernel (paper 2, Eq. 11), rewritten for a general finite group $G$ of order $K$
with identity $e$, *is* a translation kernel:

$$Q_t = (1-\beta_t)\,\delta_e + \beta_t\,\mathrm{Unif}(G)
  = \Big(1-\beta_t+\tfrac{\beta_t}{K}\Big)\delta_e + \sum_{g\neq e}\tfrac{\beta_t}{K}\,\delta_g .$$

The "noise" is a group element: $g_t=e$ with probability $1-\beta_t+\beta_t/K$, and uniform on
$G\setminus\{e\}$ otherwise. Multinomial diffusion is thus a group diffusion whose noise
distribution is the **most spread-out one possible**.

It pays to carry this through the Fourier picture once. For the cyclic group $\mathbb{Z}/K$ the
characters are $\chi_k(n)=e^{2\pi i kn/K}$, and the per-step symbol is

$$\mathcal{F}(\kappa_t)(\chi_k)=
  \begin{cases} 1 & k=0 \quad(\text{trivial / stationary}),\\
                 1-\beta_t & k\neq 0,\end{cases}$$

so the cumulative kernel has eigenvalues $1$ and $\alpha_{1:t}=\prod_{s\le t}(1-\beta_s)$
(Hoogeboom's $\bar\alpha_t$) — this is his closed-form marginal (Eq. 12) seen in the Fourier
basis. Note what the symbol does *not* depend on: the frequency $k$. Every non-constant mode
decays at the same rate, which is precisely what "replacement by uniform" means in spectral
language. Contrast D3PM's discretized-Gaussian kernel (paper 4), whose symbol is Gaussian in
$k$: low-frequency (smooth) modes survive longer than high-frequency ones, which is a genuine
notion of locality and scale. **That single difference is the entire reason to prefer the
structured kernel.** Paper 2 already lives inside the group-diffusion family; it just works
with the member that discards the geometry it secretly has.

#### The family, by group

| Group $G$ | Typical data | Forward $\kappa_t$ | Representative work |
| --- | --- | --- | --- |
| $\mathbb{Z}/K$ (cyclic) | circular / periodic values | local walk on the cycle | [A8] (discrete circle); theory [A7] |
| $(\mathbb{Z}/2)^n$ (hypercube) | bits | bit flips (XOR) | Sohl-Dickstein 2015 (paper 1); theory [A6] |
| $\mathbb{Z}$ (lattice) | ordered categories | discretized Gaussian | D3PM (paper 4) |
| $S_n$ (symmetric group) | permutations / rankings | riffle shuffle, random-transposition walk | SymmetricDiffusers [A3]; soft-rank [A4] |
| finite abelian $G$ (general) | arbitrary finite labels | convolution semigroup | [A7] |
| $SO(3)$, $SE(3)$, Lie groups (continuous) | rotations, poses, frames | heat / Brownian motion on the group | [A11], [A9], [A10], [A12], [A13] |

#### Case study: $S_n$ (permutations)

Learning a distribution over $S_n$ is the hardest common instance: $|S_n|=n!$ and the group is
non-abelian. SymmetricDiffusers [A3] takes the direct route: the forward process is a **riffle shuffle**, a random
walk on the finite group $S_n$, and the diffusion length is chosen from the
random-walk-on-finite-groups mixing theory for that walk; the reverse is a generalized
Plackett–Luce distribution, provably more expressive than plain PL. The forward is therefore
group-structured (a convolution), while the reverse is a *learned, ordered* distribution —
exactly the same "group-structured forward, general reverse" pattern as paper 2. [A4] keeps the
group but changes the corruption: it lifts the permutation to a continuous soft-rank
representation and denoises there, reporting better behaviour on long sequences. A related but
distinct idea is to quotient the ambient space by the group action rather than to put the
group in the state space [A14].

#### Case study: continuous Lie groups

Replace the finite group by a compact Lie group and the convolution walk by Brownian motion:
the heat kernel on the group is the noise distribution, and the "noise" is again a group
element (a small rotation, a small rigid motion). [A11] runs a DDPM on $SO(3)$ for rotational
alignment; [A10] gives a unified $SO(3)$ SGM/DDPM framework exploiting the tractable heat
kernel there; [A9] builds an $SE(3)$-invariant diffusion over rigid-body frames (FrameDiff)
with an $SE(3)$-equivariant score; [A12] develops the general Riemannian-manifold theory of
which these are special cases; and [A13] formulates score-based diffusion in the
*representation space* of an arbitrary (non-abelian) Lie group, noting explicitly that ordinary
Euclidean score matching is recovered as the special case of the translation group. This is the
continuous counterpart of the finite-group picture, and it is where the "predict the group
element" parameterization is standard practice.

#### Why non-abelian groups are hard

For abelian $G$ the dual group is again a group of scalars, so every irreducible representation
is one-dimensional and the forward kernel has scalar eigenvalues — hence [A7]'s (paper 3's) clean
Fourier/convolution-semigroup treatment on $\mathbb{Z}_N$. For non-abelian $G$ (notably $S_n$ and
$SO(3)$) the irreducible representations are matrix-valued, so $\mathcal{F}(\kappa_t)$ is a matrix
and the spectrum is not a list of numbers; [A7] (paper 3) names extension to non-abelian finite groups as
future work. This is why the $S_n$ literature leans on random-walk mixing theory and structured
reverse parameterizations [A3, A4] rather than on a spectral closed form.

#### Do not confuse this with equivariance

Group diffusion puts the group in the **state space and the corruption**. A different, more
common use of groups keeps the state space Euclidean and makes the *network* commute with a
group action — equivariant diffusion [A15], structure-preserving diffusion [A16], and the
$E(3)$-equivariant networks inside molecular diffusion models. Both are called "group
diffusion" in loose speech; only the first is what the taxonomy row means and what turns the
noise variable into a group element.

#### Cross-links

- Paper 1's binomial diffusion is the $(\mathbb{Z}/2)^n$ instance (forward = XOR by an i.i.d.
  Bernoulli vector); see the binomial deep dive above.
- Paper 2's uniform kernel is the degenerate, non-local member; paper 4's discretized-Gaussian
  kernel is the local member on $\mathbb{Z}$.
- The escapes to a *continuous* state space — Analog Bits / Bit Diffusion [A1] and Dirichlet
  Diffusion [A2] — are what one does when the discrete state space has no useful group
  structure to exploit.

**One-line summary.** Group diffusion = discrete diffusion whose corruption is a random group
translation $x_t=x_{t-1}+g_t$; the forward chain is a convolution diagonalized by the group
Fourier transform, the noise variable is an observable group element, uniform replacement
(paper 2) is its maximally-mixing degenerate case, and the continuous Lie-group versions
($SO(3)$, $SE(3)$) are where this parameterization is used in practice.

**Sources.** Verified records (arXiv IDs, DOIs, S2 IDs, venues, dates) are stored in
`sources/raw/group-diffusion-refs.json`; the A-labels above match that file.

---

## 3. Harmonic flows and Markov dynamics on finite groups via pseudo-differential operators
*Torresblanca-Badillo, Barrios-Garizao & Quiñonez-Martínez — J. Pseudo-Differ. Oper. Appl. **17**(1), Art. 18, 2026 · DOI [`10.1007/s11868-025-00759-7`](https://doi.org/10.1007/s11868-025-00759-7) · open access · no arXiv preprint*

> **Why this sits at 3.** It is the newest paper in this list (2026) yet is placed *before*
> D3PM on purpose. It is the mathematics underneath the *group diffusion* family: it answers
> *what makes a translation-invariant forward kernel a legitimate Markov kernel, and what is
> its spectrum?* D3PM (paper 4) then reads as the engineering instantiation of one particular
> choice of that spectrum. It is a pure analysis paper — **no learning, no reverse process, no
> generative model, no experiments beyond an illustration.** Read it as foundations.

### 3 key points

1. **Diffusion on a finite group is a convolution semigroup, and the group Fourier transform
   diagonalizes it.** The state space is $\mathbb{Z}_N$ (integers mod $N$: the $N$-cycle), and
   "corruption" is *translation by a random group element*, so the forward kernel is a
   convolution. Under $\mathcal{F}$, convolution becomes pointwise multiplication, so the whole
   time evolution collapses from an $N\times N$ matrix to $N$ scalar decay rates.

2. **One non-negative "dispersion function" $\psi$ on the dual group specifies the entire
   process.** The transition kernel at time $t$ is
   $$Z_t=\mathcal{F}^{-1}\!\left(e^{-t\psi}\right),$$
   equivalently: the eigenvalues of the kernel are exactly $e^{-t\psi(m)}$. So $\psi$ *is* the
   noise spectrum — the discrete analogue of "the Gaussian's Fourier transform is a Gaussian".

3. **The generator is a Fourier multiplier — a pseudo-differential operator
   $A_\psi=-\mathcal{F}^{-1}(\psi\,\mathcal{F})$ — and it is bounded, self-adjoint and
   $m$-dissipative.** Together with the Feller-semigroup property of the operator family
   $T_t z = Z_t\ast z$, this is what guarantees the associated diffusion equation is
   well-posed, stable and probability-preserving. That is the analytic licence for using such
   kernels as a forward process.

### Main innovation

The paper builds a **self-contained harmonic-analysis/operator-theory construction of
discrete diffusion on a finite abelian group**, and in doing so isolates the exact
condition that makes a translation-invariant kernel a valid Markov kernel:

1. define probability measures as inverse Fourier transforms of $e^{-t\psi}$ (Thm 1);
2. show they compose in time, $Z_t\ast Z_s=Z_{t+s}$, i.e. form a convolution semigroup (Prop 1);
3. identify the **negative-definiteness** condition on $\psi$ that legitimizes step 1
   (Thm 2 — see the caveat below), and
4. prove the associated operator is a **Feller semigroup** (Thm 3) whose generator is
   **$m$-dissipative and self-adjoint** (Thm 4).

The framing is what is new: the same objects are classical in potential theory and in
Lévy-process theory on locally compact abelian groups, but here they are assembled
explicitly and concretely for the finite cyclic group, with the "dispersion function"
$\psi$ exposed as the single design knob. The paper also names two extensions it does *not*
do: **non-abelian finite groups** ("spectral diagonalization presents new challenges") and
**ultrametric / $p$-adic groups**. It claims applications in signal processing on finite
graphs, simulation of discrete stochastic systems, and numerical analysis — but runs no ML
experiments and never constructs a generative model.

### Why it matters for this topic

- It is the rigorous version of the "group diffusion" story already sketched under paper 2:
  a translation-invariant forward kernel is a **group convolution**, and its spectrum is
  $e^{-t\psi}$. The earlier spectral remark — that Hoogeboom's uniform kernel has symbol
  $1-\beta$ on *every* non-trivial character — is the special case $\psi\equiv\text{const}$ on
  $m\neq 0$.
- It tells us **when a structured kernel is admissible**: $\psi$ must be negative definite
  (equivalently, even and inside the Lévy–Khinchin cone). "Pick a nice-looking $\psi$" is not
  enough — see the correction below.
- The generator $A_\psi$ is exactly the **continuous-time Markov chain rate matrix** of a
  translation-invariant chain. That is the object papers 7 (concrete score / CTMC) and 10
  (SEDD) learn a *discrete* analogue of. This paper is the forward half of that picture.
- It is **forward-only**. There is no reverse process, no ELBO, no score, no learning. The
  generative half is papers 4, 7 and 10.

---

### Primer: the operator-theory toolkit, from zero

Nothing below assumes prior operator theory. Each notion gets the one-line intuition first,
then the formal statement, then why we care.

> **Algebra background:** group theory itself — the thing $\psi$ is a function *of* — is
> built from zero in [`group-theory-primer.md`](group-theory-primer.md).

#### 1. The state space is a group, so "noise" means "translation"

A group is a set with an operation you can undo (addition mod $N$ here). Writing
$\mathbb{Z}_N=\{0,1,\dots,N-1\}$ with addition mod $N$, a *translation* by $g$ is the map
$n\mapsto n+g$. The payoff of this structure is that translating, then translating again, is
translating by the sum:
$$(n+g_1)+g_2 = n+(g_1+g_2).$$
So if corruption is "add a random $g_t$", the effect of many steps is "add the sum of the
$g_t$" — which is why everything below factorizes. This is the group analogue of the
Gaussian case, where corruption is "add a random vector" and many steps add up to one
Gaussian.

#### 2. Functions on $\mathbb{Z}_N$ are vectors; convolution is the group's "blur"

A complex function $z:\mathbb{Z}_N\to\mathbb{C}$ is just a vector
$z=(z(0),\dots,z(N-1))$, and the space of them is written $L^2(\mathbb{Z}_N)=\mathbb{C}^N$.
The **convolution** of two functions is
$$(z\ast w)(m)=\sum_{n=0}^{N-1} z(m-n)\,w(n).$$
Convolution is the "smear $w$ according to $z$" operation. A probability distribution $\mu$
acting on a function is exactly a convolution: $\mu\ast z$. Convolution is commutative and
associative, and $z\ast\delta=z$ where $\delta=(1,0,\dots,0)$ is the point mass at $0$.

#### 3. The Fourier transform diagonalizes convolution

Define the (discrete) Fourier transform by
$$\mathcal{F}(z)(m)=\sum_{n=0}^{N-1} z(n)\,e^{-2\pi i mn/N}, \qquad
  \mathcal{F}^{-1}(z)(n)=\frac{1}{N}\sum_{m=0}^{N-1} z(m)\,e^{2\pi i mn/N}.$$
The **convolution theorem** is the whole game:
$$\mathcal{F}(z\ast w)=\mathcal{F}(z)\cdot\mathcal{F}(w) \quad\text{(pointwise)}.$$
In matrix language: a kernel that acts by convolution is a **circulant matrix**, and the
Fourier basis is exactly its eigenbasis. Equivalently, the characters
$\chi_k(n)=e^{2\pi i kn/N}$ are the eigenvectors:
$$(\mu\ast \chi_k)=\mathcal{F}(\mu)(k)\,\chi_k.$$
So "convolution operator" and "diagonal in the Fourier basis" are the same statement, and the
**Fourier transform of the kernel is its eigenvalue list.** Everything the paper does follows
from this.

#### 4. The dispersion function $\psi$ is the noise spectrum

The paper fixes a *dispersion function* $\psi:\mathbb{Z}_N\to\mathbb{R}_+$ with

- (i) $\psi(m)>0$ for $m\neq 0$, and
- (ii) $\psi(m)=0$ if and only if $m=0$.

Intuition: $\psi(m)$ is the **decay rate of frequency $m$**. Think of $m$ as a frequency
index and $\psi(m)$ as how fast that frequency is damped. $\psi(0)=0$ says the constant mode
(the total mass) never decays — that is mass conservation. Two familiar choices:

- $\psi(m)=2(1-\cos(2\pi m/N))$ — small for low frequencies, large for high ones: a
  **diffusion** that smooths (the cycle-graph Laplacian).
- $\psi(m)=\text{const}$ for $m\neq 0$ — all frequencies damp at the same rate: **maximally
  mixing**, no geometry.

#### 5. The heat kernel $Z_t$: how the process spreads in time

Given $\psi$, define
$$Z_t(n)=\mathcal{F}^{-1}\!\left(e^{-t\psi}\right)(n)
  =\frac{1}{N}\sum_{m=0}^{N-1} e^{-t\psi(m)}\,e^{2\pi i mn/N},\qquad t\ge 0 .$$
Read it as: **start from a point mass and let each frequency decay at its own rate.**
Because $\psi(m)\ge 0$, high-$\psi$ (rough) frequencies die first and $Z_t$ smooths out;
$\psi(0)=0$ keeps the total at $1$. In the Fourier domain the evolution is the trivial
multiplication $e^{-t\psi(m)}$; in the spatial domain it is the (generally complicated)
inverse transform.

#### 6. Semigroup = "time adds up"

A family $(\mu_t)_{t>0}$ of positive bounded measures is a **convolution semigroup** if
$$\mu_t\ast\mu_s=\mu_{t+s}\ \ \text{for all } t,s>0,\qquad \lim_{t\to 0}\mu_t=\delta .$$
Intuition: running the process for $t$ then $s$ is the same as running it for $t+s$; and at
time $0$ nothing has happened yet. This is the Markov/Chapman–Kolmogorov property. It is
trivial to check here, because in the Fourier domain it reads
$e^{-t\psi}\cdot e^{-s\psi}=e^{-(t+s)\psi}$.

#### 7. Probability measures: positivity + normalization

For $Z_t$ to be a *probability distribution* it needs two things:

- **Normalization:** $\sum_n Z_t(n)=1$. This is automatic: $\sum_n Z_t(n)=\mathcal{F}(Z_t)(0)=e^{-t\psi(0)}=1$.
- **Positivity:** $Z_t(n)\ge 0$ for all $n$. This is **not** automatic, and it is where all
  the subtlety lives.

Positivity is exactly a statement of **Bochner's theorem**: a function is the Fourier
transform of a positive measure iff it is *positive definite*. So we need $e^{-t\psi}$ to be
positive definite.

#### 8. Negative definite functions (the real hypothesis)

A function $\varphi:\mathbb{Z}_N\to\mathbb{C}$ is **negative definite** if
$$\sum_{i,j=1}^{r}\Big(\varphi(m_i)+\varphi(m_j)^{*}-\varphi(m_i-m_j)\Big)\lambda_i\lambda_j^{*}\ge 0$$
for all $m_1,\dots,m_r\in\mathbb{Z}_N$, $\lambda_1,\dots,\lambda_r\in\mathbb{C}$
(the $*$ denotes complex conjugate). Intuition: it is the abstract shape of a *squared
distance* — the prototype is $\varphi(x)=|x|^2$, and the quadratic form above is the
"energy" version of that. Its role here is the **Schoenberg correspondence**:
$$\psi \text{ is negative definite} \iff e^{-t\psi} \text{ is positive definite for every } t>0.$$
Chaining with Bochner: *$\psi$ negative definite $\Rightarrow$ $e^{-t\psi}$ positive definite
$\Rightarrow$ (Bochner) $Z_t=\mathcal{F}^{-1}(e^{-t\psi})$ is a positive measure.* On the
finite abelian group this condition has a concrete form, the **Lévy–Khinchin cone**:
$$\psi(m)=\sum_{k=1}^{\lfloor N/2\rfloor} c_k\big(1-\cos(2\pi km/N)\big),\qquad c_k\ge 0 .$$
Two immediate consequences, both used below: $\psi$ must be **even**, $\psi(m)=\psi(N-m)$, and
its "Fourier coefficients" $c_k$ must be non-negative. Conditions (i)–(ii) on $\psi$ alone are
**not** sufficient for any of this.

#### 9. Feller semigroup: the operator that applies the process

Instead of tracking distributions, track how the process acts on functions. Define
$$T_t z(n):=(Z_t\ast z)(n)\qquad(t>0),\qquad T_0:=I .$$
The family $(T_t)_{t\ge 0}$ is a **Feller semigroup** (adapted to the finite setting) if

- (i) $T_tT_s=T_{t+s}$ and $T_0=I$ (semigroup);
- (ii) $\|T_t z-z\|\to 0$ as $t\to 0$ (strong continuity — nothing jumps at $t=0$);
- (iii) $0\le z\le 1\ \Rightarrow\ 0\le T_tz\le 1$ (positivity / no new mass).

Intuition: $T_t$ is "expectation after time $t$". Property (i) is the Markov property,
(ii) says the process starts where you put it, and (iii) says probabilities stay
probabilities. The paper proves $\|T_t\|\le 1$ (contractivity) and all three properties
(Lemmas 4–6, Thm 3).

#### 10. Generator and pseudo-differential operator

For a semigroup, the **generator** is the time derivative at $t=0$:
$A=\frac{d}{dt}T_t\big|_{t=0}$. It is the "instantaneous rate of change" — for a Markov
chain it is the rate matrix (the $Q$ of a CTMC). Here, differentiating
$\mathcal{F}(T_tz)(m)=e^{-t\psi(m)}\mathcal{F}(z)(m)$ gives
$$A_\psi z=-\mathcal{F}^{-1}\!\big(\psi\cdot\mathcal{F}(z)\big),
  \qquad\text{i.e. } A_\psi=-\mathcal{F}^{-1}\psi\,\mathcal{F}.$$
A **pseudo-differential operator** is exactly this: an operator defined not by derivatives in
space but by multiplying by a *symbol* ($\psi$) in the Fourier domain. "Pseudo-differential"
here just means "Fourier multiplier". The evolution equation it generates,
$\partial_t u = A_\psi u$, is the discrete heat/diffusion equation; its solution is
$u(t)=T_t u(0)$.

#### 11. Dissipative, $m$-dissipative, self-adjoint

These three words are the guarantees that the equation is *physically sensible and numerically
stable*:

- **Dissipative:** $\|w-\lambda Aw\|\ge\|w\|$ for all $\lambda>0$. Equivalently (Lumer–Phillips)
  $\operatorname{Re}(Az,z)\le 0$: the operator never *injects* energy. For $A_\psi$ this is one
  line of Parseval:
  $$(A_\psi z,z)=-\frac{1}{N}\sum_{m}\psi(m)\,|\mathcal{F}(z)(m)|^2\le 0,$$
  because $\psi\ge 0$. Each frequency's energy decays; none grows.
- **$m$-dissipative:** dissipative *and* the equation $w-\lambda Aw=z$ is solvable for every
  $z$ and $\lambda>0$. The "$m$" (maximal) part rules out pathologies: it is the hypothesis of
  the Lumer–Phillips theorem guaranteeing that $A$ really is the generator of a contraction
  semigroup, so the flow exists for all time and is unique.
- **Self-adjoint:** $(A_\psi z,w)=(z,A_\psi w)$. In the Fourier basis this is just "multiply by
  the *real* number $-\psi(m)$" on each mode. Self-adjointness is why the kernel matrices are
  symmetric and the spectrum is real, and it is what makes the numerics well-behaved.

#### 12. The dictionary

| Machine-learning / diffusion object | This paper's object |
| --- | --- |
| forward kernel $Q_t$ (one step) | heat kernel $Z_t=\mathcal{F}^{-1}(e^{-t\psi})$ |
| cumulative marginal $q(x_t\mid x_0)$ | $Z_t(x_t-x_0)$ (a convolution kernel) |
| forward schedule $\beta_t$, $\bar\alpha_t$ | time $t$ plus the dispersion $\psi$; spectrum $e^{-t\psi}$ |
| eigenvalues of the forward kernel | $e^{-t\psi(m)}$ |
| CTMC rate matrix / generator | $A_\psi=-\mathcal{F}^{-1}\psi\,\mathcal{F}$ |
| noise type (Gaussian, uniform, jumps) | the choice of $\psi$ |
| maximally mixing (uniform) kernel | $\psi=\text{const}$ on $m\neq 0$ |
| D3PM's discretized-Gaussian kernel | $\psi(m)=2(1-\cos(2\pi m/N))$ |

### The paper's construction, step by step

1. **Setup (§2).** $L^2(\mathbb{Z}_N)=\mathbb{C}^N$; inner product; Fourier transform and its
   inverse; $\delta$; convolution. Establishes that $\mathcal{F}^{-1}\mathcal{F}=I$.
2. **Measures from $e^{-t\psi}$ (§3).** Fix $\psi$ with (i)–(ii). Define
   $Z_t=\mathcal{F}^{-1}(e^{-t\psi})$. Show $Z_t\ge 0$ (Lemma 1, via Bochner/positive
   definiteness), $\sum_n Z_t(n)=1$ (Lemma 2, via $\mathcal{F}(Z_t)(0)=e^{-t\psi(0)}=1$), hence
   $Z_t$ is a probability measure for every $t\ge 0$ (Thm 1). Illustration: $N=5$,
   $\psi(m)=m^2$, mass spreading from $n=0$ as $t$ grows (Figs. 1–2).
3. **Convolution semigroup (§4).** $Z_t\ast Z_s=Z_{t+s}$ (Lemma 3) by the convolution theorem;
   therefore $(Z_t)$ is a convolution semigroup (Prop 1). Then the negative-definiteness
   statement (Thm 2) — the step that ties the construction to classical Lévy theory. Also
   records that $\mathbb{Z}_N$ is a compact locally compact abelian group, which is why the
   classical machinery applies.
4. **Feller semigroup (§5).** Define $T_tz=Z_t\ast z$. Prove $T_t$ is well-defined, linear,
   bounded with $\|T_t\|\le 1$ (Lemma 4); $T_tT_s=T_{t+s}$ (Lemma 5); positivity
   $0\le z\le 1\Rightarrow 0\le T_tz\le 1$ (Lemma 6); strong continuity (Remark 6). Conclude
   $(T_t)$ is a Feller semigroup (Thm 3).
5. **The generator (§6).** Define $A_\psi=-\mathcal{F}^{-1}(\psi\,\mathcal{F})$. Prove
   dissipativity by Parseval (Lemma 7), self-adjointness (Lemma 8), hence $m$-dissipativity
   (Thm 4). Illustration: $N=5$, $\psi(m)=m^2$, $z=\delta$, giving
   $A_\psi z(0)=-6$ and conjugate-symmetric complex values elsewhere (Ex. 3, Table 1, Fig. 4).

### Worked example, and a correction

The paper's running illustration is $\mathbb{Z}_5$ with $\psi(m)=m^2$ (Examples 1–3). **That
$\psi$ is not admissible**, and the failure is easy to check by hand:

- $\psi$ must be even ($\psi(m)=\psi(N-m)$, from the requirement that $e^{-t\psi}$ be positive
  definite). But $\psi(1)=1$ while $\psi(4)=16$. So $Z_t$ comes out **complex** — direct
  computation at $t=1$ gives $\max_n|\operatorname{Im} Z_t(n)|\approx 0.072$ — and a complex
  "probability" contradicts the paper's own Lemma 1. Example 2's
  $\psi(m)=1+\sin(2\pi m/5)$ has the same defect ($\psi(1)=1.951$ vs $\psi(4)=0.049$).
- Even symmetrizing to $\psi(m)=\min(m,N-m)^2$ is not enough: fitting it into the
  Lévy–Khinchin cone gives $c_2<0$, and indeed $Z_t$ then dips to $-0.011$ near $t\approx0.14$.
- The **correct** "quadratic" dispersion on $\mathbb{Z}_N$ is the cycle-Laplacian symbol
  $$\psi(m)=2\big(1-\cos(2\pi m/N)\big),$$
  which is negative definite ($c_1=2$, all other $c_k=0$) and reduces to $(2\pi m/N)^2$ for
  small $m$ — the discrete analogue of the Gaussian's $k^2$. Its kernel is real and
  non-negative for all $t$, and its eigenvalues are exactly $e^{-\psi(m)}$.
- As a sanity check on the other end, $\psi=\text{const}$ on $m\neq 0$ gives the maximally
  mixing kernel $Z_t=e^{-t}\delta+(1-e^{-t})\,\text{uniform}$ — i.e. Hoogeboom's uniform
  replacement, in continuous time.

All of this is verified numerically in `experiments/check-dispersion-example.py`
(`python3 experiments/check-dispersion-example.py`), which reports the failure of the paper's
examples and the admissibility of the Laplacian and uniform choices.

The lesson is the paper's own implicit one: **conditions (i)–(ii) on $\psi$ are not enough —
negative definiteness is the real hypothesis**, and it is not just a technicality (it is
what forces evenness and the non-negative $c_k$). The framework (Thms 1, 3, 4) is sound;
the illustration picked $\psi$ outside its domain.

### Bridge to discrete diffusion models

- **This is the forward process, abstracted.** A translation-invariant forward chain with
  kernels $Q_1,Q_2,\dots$ is the discrete-time shadow of a convolution semigroup: the
  cumulative kernel $\bar Q_t$ is $Z_t$ for the right $\psi$, and the "schedule" is the map
  $t\mapsto\psi$ together with how $t$ advances. In D3PM's terms, the transition matrix is a
  **circulant** matrix and $e^{-t\psi}$ is its spectrum.
- **The kernel is chosen by choosing a spectrum.** This reframes the design question. Instead
  of inventing a transition matrix, pick the decay profile $\psi$: constant $\Rightarrow$
  uniform mixing (no structure); $2(1-\cos)$ $\Rightarrow$ local Gaussian-like smoothing;
  larger powers $\Rightarrow$ more local; non-smooth $\psi$ $\Rightarrow$ Lévy-type jumps. This
  is precisely the "which discrete state spaces admit a noise formulation" taxonomy from
  paper 2, now with an explicit spectral parameterization.
- **The generator is the rate matrix.** $A_\psi$ is the CTMC generator of the forward chain.
  Papers 7 (concrete score matching) and 10 (SEDD) work with the *continuous-time* CTMC view
  and learn the reverse rates; this paper supplies the forward half rigorously, for the
  translation-invariant (abelian) case.
- **Why non-abelian groups are hard, made precise.** For abelian $G$ the dual is again a group
  of scalars, so $\mathcal{F}$ turns the kernel into $N$ independent scalars $e^{-t\psi(m)}$.
  For non-abelian $G$ the Fourier transform is **matrix-valued** (irreducible representations
  are matrices), so there is no single real-valued dispersion function $\psi$ and the
  generator is not diagonalized by scalars. That is exactly the obstruction the paper names as
  future work, and the reason the $S_n$/Lie-group literature (see the group-diffusion
  subsection under paper 2) resorts to random-walk mixing theory and structured reverse
  parameterizations instead of a spectral closed form.

### Sharp edges and open questions

- **Theorem 2's wording looks inverted.** As printed, it says $e^{-t\psi}$ is *negative*
  definite. The standard statement — Schoenberg's correspondence, the same one behind
  Lévy–Khinchin on groups, where a convolution semigroup of probability measures has
  $\mathcal{F}(\mu_t)=e^{-t\eta}$ with $\eta$ *negative definite* — is: **$\psi$ is negative
  definite iff $e^{-t\psi}$ is positive definite for all $t>0**. With that reading the proof
  chain is exactly right (positive definiteness is what Bochner needs); as written it is
  likely a slip. Flagged rather than asserted — it does not affect Theorems 1, 3, 4.
- **The admissibility hypothesis is understated.** The paper's standing assumptions on $\psi$
  are only (i)–(ii), but Theorems 1/3 need $\psi$ negative definite. The worked examples
  violate it (above).
- **Forward-only.** No reverse process, no ELBO, no score, no learning, no ML evaluation. The
  generative machinery is entirely in papers 4, 7, 10.
- **Abelian and cyclic only.** $\mathbb{Z}_N$ (and it gestures at general finite abelian $G$).
  Non-abelian and $p$-adic extensions are stated as future work, not delivered.
- **"Harmonic flow" is a framing word.** The title's phrase is not given a separate formal
  definition in the body; it names these Fourier-diagonalized evolution families.
- **Sub-probabilities allowed.** Definition 3 permits $\mu_t(\mathbb{Z}_N)\le 1$ (possibly
  killing mass); the specific $Z_t$ built in §3 is normalized to $1$, so this is a harmless
  generality here but a difference from the diffusion setting, where the forward process
  conserves total mass.
- **Continuous time vs discrete steps.** The paper's $t$ is a continuous parameter and $Z_t$ is
  the exact kernel at time $t$. Diffusion models use a finite step schedule; the correspondence
  is that each step is $Z_{\Delta t}$ for some increment, or equivalently the continuous-time
  chain whose generator is $A_\psi$. The paper does not discuss discretization error.

**If you remember one thing:** a translation-invariant diffusion on a finite group is a
convolution, convolution is diagonal in the group Fourier basis, so the whole forward process
is *one non-negative even function $\psi$* — its spectrum $e^{-t\psi}$ — and the generator is
the Fourier multiplier $-\psi$.


