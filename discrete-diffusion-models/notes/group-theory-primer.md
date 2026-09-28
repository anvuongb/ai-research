# Group Theory Primer (for discrete diffusion)

**Why this file exists.** From paper 2 onward the topic uses group structure constantly —
paper 3's dispersion function $\psi$, the "group diffusion" family, the $S_n$ and $SO(3)$
papers, and the equivariance contrast all assume it. This primer builds group theory from
zero, in the order the topic needs it, and stops as soon as the payoff has been reached.

It assumes no prior algebra. Every rung is *concept → example → why this topic needs it*.
The companion primer for the analysis half (Fourier, semigroups, operators) lives inside the
paper 3 entry of [`paper-walkthrough.md`](paper-walkthrough.md); this file covers only the
algebra.

---

## 0. Why groups appear at all

To get "diffusion" cheaply on a set of labels you need exactly three things:

1. a **canonical noise operation** — what does "add noise" mean on labels?
2. an **associative law** — so that many noise steps collapse into one distribution;
3. a **Fourier theory** — so that the resulting kernels can be diagonalized.

A **group** supplies all three at once. That is the whole reason this literature is
organized around groups, and the reason it stops being clean when you lose them.

---

## 1. What a group is

A **group** is a set $G$ together with one binary operation $\cdot$ such that:

| Axiom | Statement | Plain meaning |
| --- | --- | --- |
| Closure | $a\cdot b \in G$ | combining two moves is a move |
| Associativity | $(a\cdot b)\cdot c = a\cdot(b\cdot c)$ | "do $a$, then $b$, then $c$" is unambiguous |
| Identity | $\exists e:\ e\cdot a = a\cdot e = a$ | there is a "do nothing" move |
| Inverses | $\forall a\ \exists a^{-1}:\ a\cdot a^{-1} = e$ | every move can be undone |

Read $G$ as a set of **moves**, $\cdot$ as "do one then the other", $e$ as "do nothing", and
$a^{-1}$ as "undo $a$".

Note what is **not** required: the operation need not be commutative, i.e. $a\cdot b$ need
not equal $b\cdot a$. That missing axiom is the most important fact in this primer — see §4.

The **order** of the group, written $|G|$, is the number of elements.

---

## 2. The groups you will actually meet

| Group | Elements & operation | Typical data here | Order $|G|$ |
| --- | --- | --- | --- |
| $\mathbb{Z}_N = \{0,\dots,N-1\}$ | add mod $N$ | a categorical label with $N$ values; a point on an $N$-cycle | $N$ |
| $(\mathbb{Z}_2)^n$ | bit strings under XOR | binary features / bits (Sohl-Dickstein's binomial diffusion) | $2^n$ |
| $\mathbb{Z}$ | integers, addition | ordinal / lattice values (D3PM's discretized Gaussian) | $\infty$ |
| $S_n$ | permutations of $n$ items, composition | rankings, orderings, matchings | $n!$ |
| $SO(3)$ | 3-D rotations, composition | orientations, poses | $\infty$ (continuous) |
| $SE(3)$ | rigid motions (rotation + translation) | full pose / frame | $\infty$ (continuous) |

Work the axioms once on $\mathbb{Z}_5$: closure $2+3=0$; associativity
$(2+3)+4 = 2+(3+4) = 4$; identity $0$; inverse of $2$ is $3$ since $2+3=0$. In $(\mathbb{Z}_2)^n$
the operation is XOR, the identity is the all-zero string, and **every element is its own
inverse** (flipping a bit twice restores it).

### Two caveats that matter in practice

- **The number of states must match $|G|$.** You cannot put a nice group structure on an
  arbitrary set of labels; for $N$ labels the natural choice is $\mathbb{Z}_N$.
- **$\mathbb{Z}_N$ imposes a cyclic order on your labels.** Token 0 and token 4 are
  "neighbours" in $\mathbb{Z}_5$ purely because of how you numbered the vocabulary. The
  kernel inherits that arbitrary choice. Consequently the *uniform* kernel
  ($\psi=\text{const}$) is the only genuinely **labelling-free** group kernel, and D3PM's
  ordinal kernel is legitimate only when the categories are *really* ordered (pixel
  intensities) rather than merely numbered.

---

## 3. The group as a shape: Cayley graphs

Choose a small set of **generators** $S \subseteq G$ — moves that, chained together, reach
every element. Draw one vertex per group element and one edge per generator. The result is the
**Cayley graph**:

- $\mathbb{Z}_N$ with generator $\{1\}$ → a **cycle** $C_N$;
- $(\mathbb{Z}_2)^n$ with the single-bit flips $\{e_1,\dots,e_n\}$ → the **hypercube**;
- $S_n$ with adjacent transpositions → the **permutohedron**.

**Why it matters here:** the Cayley graph *is* the graph your random walk lives on. The
generator set says which single-step moves are allowed, and the geometry of that graph is
exactly what a dispersion function $\psi$ measures — "rough structure" means rough *on this
graph*. Choosing a kernel is choosing a geometry on the Cayley graph.

---

## 4. Abelian vs non-abelian — the crux

A group is **abelian** (or commutative) if $a\cdot b = b\cdot a$ for all $a,b$ — the order of
operations does not matter.

- **Abelian:** $\mathbb{Z}_N$, $(\mathbb{Z}_2)^n$, $\mathbb{Z}$, and any direct product of
  cyclic groups.
- **Non-abelian:** $S_n$ for $n\ge 3$; $SO(3)$; $SE(3)$. Concretely, rotating 90° about $x$
  then 90° about $y$ gives a different orientation than doing them in the other order.

**Why this decides everything in this topic.** In the abelian case the Fourier transform is
built from **characters**, which are plain complex *numbers*. Convolution therefore becomes
pointwise multiplication, and the whole forward process collapses to one scalar decay rate per
frequency — that is paper 3's $\psi$. In the non-abelian case the Fourier transform is built
from **matrix-valued representations**, so the "spectrum" is a collection of matrices and no
single real-valued $\psi$ exists. That is precisely why:

- paper 3 names extension to non-abelian finite groups as future work, and
- the $S_n$ / $SO(3)$ works (SymmetricDiffusers, FrameDiff, Riemannian SGM) have no spectral
  closed form and instead rely on random-walk mixing theory and structured reverse processes.

"Abelian" is not a technicality; it is the line between *we can write the answer down* and
*we cannot*.

---

## 5. Homomorphisms, characters, and the dual group

A **homomorphism** is a map that respects the operation:
$$\varphi(a\cdot b) = \varphi(a)\cdot\varphi(b).$$
It represents an abstract group by concrete objects — here, by numbers.

A **character** is a homomorphism into the nonzero complex numbers (for finite or compact
groups, into the unit circle):
$$\chi: G \to \mathbb{C}^{*}, \qquad \chi(a\cdot b) = \chi(a)\,\chi(b).$$

For $\mathbb{Z}_N$ the characters are exactly
$$\chi_k(n) = e^{\,2\pi i k n / N}, \qquad k = 0,1,\dots,N-1,$$
which is precisely the **Fourier basis** from the paper 3 primer. "Frequency" and "character"
are the same word. A character is *how the group element $n$ is seen by frequency $k$*.

The characters themselves form a group under pointwise multiplication, called the **dual
group** $\widehat{G}$. For finite abelian $G$ one has $\widehat{G} \cong G$: same size, same
structure.

**Why it matters here:** paper 3's dispersion function is a function **on the dual group**,
$\psi:\widehat{G}\to\mathbb{R}_+$. Saying "$\psi$ lives on $\widehat{G}$" is exactly saying
"$\psi$ is a function of frequency" — i.e. that it is a *spectrum*. Meanwhile $Z_t$ and the
forward kernels live on $G$. The whole construction is a statement relating a function on $G$
to its spectrum on $\widehat{G}$.

---

## 6. Convolution and translation invariance

**Group convolution** of two functions on $G$ is
$$(f*g)(x) = \sum_{y\in G} f(y)\,g(y^{-1}x),$$
which for abelian $G$ is the familiar $(f*g)(x) = \sum_y f(y)g(x-y)$. It is the operation
"smear $g$ according to $f$", and it is automatically commutative and associative.

Now the identification that makes everything work. A Markov kernel $K(x,y)$ (probability of
moving $x\to y$) is **translation-invariant** if it depends only on the group difference:
$$K(x,y) = k(y-x) \quad\Longleftrightarrow\quad K \text{ acts by convolution with } k.$$

So the following four phrases are *the same statement*:

> translation-invariant kernel $\;=\;$ convolution kernel $\;=\;$ random walk on the group
> $\;=\;$ operator diagonal in the Fourier basis.

The "group diffusion" literature is the study of that single equivalence class.

---

## 7. A random walk *is* the forward diffusion process

The forward process is one line:
$$X_t = X_{t-1} + g_t, \qquad g_t \sim \mu_t \ \ (\text{a distribution on } G).$$
After $t$ steps, the law of $X_t$ given $X_0 = x$ is the **convolution of all the steps**,
$$\mu_1 * \mu_2 * \cdots * \mu_t \ \text{ shifted to } x,$$
which is exactly the forward marginal $q(x_t\mid x_0)$. The associativity axiom is what makes
this collapse to a single object, and the same law gives the Markov/Chapman–Kolmogorov
property. In short: **groups are the reason multi-step corruption is one convolution.**

Sanity check against a kernel already met in paper 2 — Hoogeboom's uniform categorical kernel
on $\mathbb{Z}_K$,
$$q(y\mid x) = (1-\beta)\,\delta_{y,x} + \frac{\beta}{K},$$
is a random walk whose single step is the distribution
$\mu = (1-\beta)\delta_0 + \beta\cdot\text{Uniform}(G)$. Push the nontrivial characters
$\chi_k$ ($k\neq 0$) through it:
$$\text{eigenvalue} = (1-\beta)\cdot 1 + \frac{\beta}{K}\sum_{m}\chi_k(m) = 1-\beta,$$
because the characters sum to zero over the group. That is exactly the "symbol equals
$1-\beta$ on every non-trivial character" fact recorded in paper 2's spectral remark —
now derived in three lines from group theory alone.

---

## 8. What a group does **not** give you: the absorbing / mask state

Consider absorbing (mask) corruption:
$$x_t = x_{t-1} \ \text{ with prob. } 1-\beta, \qquad x_t = [\text{MASK}] \ \text{ with prob. } \beta.$$

`[MASK]` is a **distinguished sink**: every state maps *into* it and nothing ever leaves. A
group translation moves *every* state by the same offset and therefore cannot single one
element out and trap mass there. So the absorbing kernel is **not** a group diffusion — it
sits at the opposite extreme from the translation-invariant family:

| | structure | kernel |
| --- | --- | --- |
| structured (group) corruption | translation-invariant | convolution $K(x,y)=k(y-x)$ |
| absorbing (mask) corruption | a sink | $K(x,\text{MASK})=\beta$, $K(\text{MASK},\text{MASK})=1$ |

One can still fit absorption into the framework by adjoining an absorbing element to the
group, giving a **killed random walk** in which total mass leaks away, $\mu_t(G) < 1$. This
is the natural reading of the fact that paper 3's definition of a convolution semigroup
allows sub-probabilities, $\mu_t(G)\le 1$, even though the kernels it actually constructs are
normalized to $1$: that slack is exactly the room needed for absorbed mass. *(The paper does
not discuss this; it is an observation, not a claim of theirs.)*

---

## 9. Lie groups: the continuous case

A **Lie group** is a group that is also a smooth manifold, so it can be differentiated.
$SO(3)$ (3-D rotations) and $SE(3)$ (rigid motions, $SO(3)\ltimes\mathbb{R}^3$) are the ones
that recur here.

- The tangent space at the identity is a vector space, the **Lie algebra**; the **exponential
  map** $\exp$ turns an infinitesimal move into a finite one.
- "A random group element" becomes "a probability distribution on a manifold", and the
  **heat kernel** is Brownian motion on the group.
- The generator is a differential operator (a Laplacian on the group) rather than a matrix.

This is the continuous mirror of paper 3: the same story — translation-invariant, convolution,
Fourier/representation diagonalization — with characters replaced by representations and sums
replaced by integrals. FrameDiff, the $SO(3)$ unified framework, and Riemannian SGM live here.

---

## 10. The trap: "group" has two meanings in this literature

| | Where the group lives | What is structured | Examples |
| --- | --- | --- | --- |
| **Group diffusion** | the group **is the state space** ($G=\mathbb{Z}_N$, $S_n$, $SO(3)$) | the kernel is translation-invariant → a convolution | paper 3; SymmetricDiffusers on $S_n$; FrameDiff on $SE(3)$ |
| **Equivariant networks** | the state space is Euclidean; $G$ **acts on it** | the network **commutes** with the group action | E(3)-equivariant molecular nets; SymDiff; SPDM |

An **action** assigns each $g\in G$ a transformation of the space, compatibly with
composition. **Equivariance** is $f(g\cdot x) = g\cdot f(x)$: transforming the input then
running the network equals running the network then transforming the output.

Both are called "group something", but in the first the group is *what the noise is made of*,
and in the second it is *a symmetry the model respects*. Keeping these apart prevents most of
the confusion in reading this area.

---

## 11. Cheat sheet

| Group | Data | Forward step | Kernel class | Where in this topic |
| --- | --- | --- | --- | --- |
| $(\mathbb{Z}_2)^n$ | bits | flip each bit w.p. $\beta/2$ (XOR a random string) | uniform on the hypercube | Sohl-Dickstein 2015 (paper 1) |
| $\mathbb{Z}_K$ | categorical label | replace with a uniform category w.p. $\beta$ | uniform convolution, symbol $1-\beta$ | Hoogeboom 2021 (paper 2), paper 3 |
| $\mathbb{Z}$ (lattice) | ordinal value | add a discretized Gaussian | discretized Gaussian / cycle Laplacian symbol $2(1-\cos(2\pi m/N))$ | D3PM (paper 4) |
| general finite abelian $G$ | any | convolution by a semigroup | spectral; one scalar per character | paper 3 |
| $S_n$ | ranking / ordering | riffle shuffle, random transposition | non-abelian: no scalar spectrum | SymmetricDiffusers, Soft-Rank Diffusion |
| $SO(3)$, $SE(3)$ | rotation / pose | Brownian motion on the group | continuous heat kernel | FrameDiff, Riemannian SGM |

### Common confusions

- **"Abelian" is about the *operation*, not the size.** $(\mathbb{Z}_2)^n$ is huge and
  abelian; $S_3$ has only 6 elements and is not.
- **A group is not the same as a group action.** See §10.
- **$\mathbb{Z}_K$ is a choice of labelling**, not a fact about your data. See §2.
- **Absorbing/mask is not a group translation.** See §8.
- **"Frequency" = character**, and $\psi$ is a function of frequency — not of state. See §5.

---

## 12. Cross-links

- **[`paper-walkthrough.md`](paper-walkthrough.md) → paper 3** — the operator-theory primer
  (Fourier on $\mathbb{Z}_N$, convolution semigroups, negative definite functions, Feller
  semigroups, generators, $m$-dissipativity) and the harmonic-flows construction this primer
  feeds into.
- **[`paper-walkthrough.md`](paper-walkthrough.md) → paper 2 → "Group diffusion"** — the
  translation-invariant family, the $\psi$ dictionary, the $S_n$ and Lie-group case studies,
  and the non-abelian obstruction.
- **`sources/foundational-papers.md` refs A3–A16** — the verified group-diffusion
  bibliography; raw records in `sources/raw/group-diffusion-refs.json`.
