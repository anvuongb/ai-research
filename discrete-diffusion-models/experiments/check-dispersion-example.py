#!/usr/bin/env python3
"""Reproducible check of the dispersion-function conditions in

    Torresblanca-Badillo, Barrios-Garizao & Quinonez-Martinez (2026),
    "Harmonic flows and Markov dynamics on finite groups via
     pseudo-differential operators",
    J. Pseudo-Differ. Oper. Appl. 17(1), Article 18.
    DOI 10.1007/s11868-025-00759-7

The paper defines, for a "dispersion function" psi: Z_N -> R_+ with psi(0) = 0
and psi(m) > 0 for m != 0,

    Z_t(n) = F^{-1}(e^{-t psi})(n)
           = (1/N) * sum_{m=0}^{N-1} e^{-t psi(m)} e^{2 pi i m n / N}     (Eq. 3.1)

and claims (Lemma 1 + Lemma 2 + Theorem 1) that Z_t is a *probability measure*
on Z_N for every t >= 0, i.e. Z_t(n) is real and non-negative for all n, and
sum_n Z_t(n) = 1.

WHY THIS IS NOT AUTOMATIC
-------------------------
Z_t >= 0 requires e^{-t psi} to be POSITIVE DEFINITE for every t > 0, which by
Bochner's theorem makes it the Fourier transform of a positive measure.  By
Schoenberg's correspondence this is equivalent to psi being NEGATIVE DEFINITE.
On a finite abelian group, negative definiteness is exactly the Levy-Khinchin
cone

    psi(m) = sum_{k=1}^{(N-1)//2} c_k (1 - cos(2 pi k m / N)),   c_k >= 0,

which forces psi to be EVEN: psi(m) = psi(N - m).

Conditions (i)-(ii) on psi alone (non-negative, zero only at 0) are NOT
sufficient, and the paper's worked examples (psi(n) = n^2 on Z_5) violate
evenness.  This script demonstrates the failure and identifies a psi that works.

Run:  python3 check-dispersion-example.py
"""

import cmath
import math

import numpy as np

N = 5


def Zt(psi, t):
    """Inverse DFT of e^{-t psi} (the paper's Eq. 3.1), as complex numbers."""
    return [
        sum(cmath.exp(-t * psi[m]) * cmath.exp(2j * math.pi * m * n / N)
            for m in range(N)) / N
        for n in range(N)
    ]


def classify(name, psi):
    """Report whether Z_t is a genuine probability measure for all t."""
    max_imag = 0.0
    worst = (float("inf"), None)
    for i in range(1, 4001):
        t = i * 0.005
        vals = Zt(psi, t)
        max_imag = max(max_imag, max(abs(v.imag) for v in vals))
        m = min(v.real for v in vals)
        if m < worst[0]:
            worst = (m, t)
    total = sum(v.real for v in Zt(psi, 1.0))
    ok = max_imag < 1e-12 and worst[0] >= -1e-9
    print(f"{name:28s} psi = {psi}")
    print(f"{'':28s}   sum_t Z_t        = {total:.6f}  (normalisation, Lemma 2)")
    print(f"{'':28s}   max_t |Im Z_t|   = {max_imag:.6f}  (must be 0: Z_t is a measure)")
    print(f"{'':28s}   min_t min_n Re Z_t = {worst[0]:+.6f} at t = {worst[1]:.3f}")
    print(f"{'':28s}   -> probability measure? {'YES' if ok else 'NO'}")
    print()
    return ok


def levy_khinchin(name, psi):
    """Least-squares fit of psi into the Levy-Khinchin cone; c_k must be >= 0."""
    K = (N - 1) // 2
    A = np.array([[1 - math.cos(2 * math.pi * k * m / N) for k in range(1, K + 1)]
                  for m in range(1, N)], dtype=float)
    b = np.array([psi[m] for m in range(1, N)], dtype=float)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = np.linalg.norm(A @ c - b)
    print(f"{name:28s} c_k = {np.round(c, 4)}   residual = {resid:.2e}   "
          f"all c_k >= 0? {bool((c >= -1e-9).all())}")
    return c


def main():
    print(__doc__)
    print("=" * 78)
    print("Candidate dispersion functions on Z_5\n")

    literal = [m * m for m in range(N)]
    sym_min = [min(m, N - m) ** 2 for m in range(N)]
    laplacian = [2 * (1 - math.cos(2 * math.pi * m / N)) for m in range(N)]
    uniform = [0] + [1] * (N - 1)

    classify("literal m^2 (paper Ex.1/3)", literal)
    classify("symmetric min(m,N-m)^2", sym_min)
    classify("Laplacian 2(1-cos(2pi m/N))", laplacian)
    classify("uniform  psi = 1 on m!=0", uniform)

    print("=" * 78)
    print("Levy-Khinchin feasibility (c_k >= 0 is the negative-definiteness test)\n")
    levy_khinchin("literal m^2", literal)
    levy_khinchin("symmetric min(m,N-m)^2", sym_min)
    levy_khinchin("Laplacian 2(1-cos)", laplacian)
    print()

    print("=" * 78)
    print("Spectral view: for a translation-invariant (circulant) kernel, the")
    print("eigenvalues of the kernel matrix are exactly e^{-t psi}, so the")
    print("dispersion function IS the negative log of the spectrum.\n")
    psi = laplacian
    c = np.array(Zt(psi, 1.0))
    C = np.array([[c[(i - j) % N] for j in range(N)] for i in range(N)])
    ev = np.linalg.eigvals(C)
    print("  kernel eigenvalues (t=1) :", np.round(np.sort(ev.real), 6))
    print("  e^{-psi}                 :", np.round([math.exp(-x) for x in psi], 6))
    print(f"  max |Im eigenvalue|      = {max(abs(x.imag) for x in ev):.2e}"
          "   (real, as expected for a symmetric circulant)")

    print()
    print("=" * 78)
    print("CONCLUSION")
    print("  * psi(n) = n^2 on Z_5 is not even (psi(1)=1 vs psi(4)=16), so Z_t is")
    print("    COMPLEX and cannot be a probability measure -> the paper's Example 1")
    print("    and Example 3 are inconsistent with its own Lemma 1.")
    print("  * psi = min(m, N-m)^2 is even but NOT in the Levy-Khinchin cone")
    print("    (c_2 < 0), so Z_t goes negative at t ~ 0.14 -> also invalid.")
    print("  * The correct 'quadratic' dispersion on Z_N is the cycle-Laplacian")
    print("    symbol psi(m) = 2(1 - cos(2 pi m / N)), which is negative definite")
    print("    (c_1 = 2, all other c_k = 0) and reduces to (2 pi m / N)^2 for small m.")
    print("  * psi = const on m != 0 reproduces the uniform/'lazy' kernel:")
    print("    Z_t = e^{-t} delta + (1 - e^{-t}) * uniform  -- the maximally")
    print("    mixing member of the family (cf. Hoogeboom's kernel).")


if __name__ == "__main__":
    main()
