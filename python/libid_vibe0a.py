#!/usr/bin/env python3
"""
Randomized linear-algebra routines.

This module implements
    * orth_sketch - build an orthonormal basis for the column space of A
    * rrqr_randomized - rank-revealing QR
    * rrsvd_randomized - truncated SVD
    * rrid_randomized - interpolative decomposition
along with a helper that constructs a Hilbert matrix.

Author: Your Name
SPDX-License-Identifier: TBD
"""

import numpy as np
import time                           # <-- timing helper
from scipy import linalg
from numpy.linalg import norm

# --------------------------------------------------------------------- #
# a) Draw a Gaussian test matrix Ω  (size n × block_size)
# --------------------------------------------------------------------- #

def _gaussian_omega(A, n, block_size, *, rng=None):
    """
    Return a Gaussian test matrix Ω with the same dtype as ``A``.

    Parameters
    ----------
    A : np.ndarray
        Reference matrix – its dtype (real or complex) determines the dtype
        of the returned Ω.
    n, block_size : int
        Desired shape of Ω (n rows, block_size columns).
    rng : np.random.Generator, optional
        Random number generator.  If omitted a fresh Generator is created
        using NumPy’s entropy source.

    Returns
    -------
    np.ndarray
        Gaussian matrix Ω.  For a complex ``A`` the real and imaginary parts
        are independent N(0, 1) draws; for a real ``A`` a plain N(0, 1) draw.
    """
    if rng is None:
        rng = np.random.default_rng()

    # -----------------------------------------------------------------
    # Real‑valued case – one call to `standard_normal` is enough.
    # -----------------------------------------------------------------
    if not np.iscomplexobj(A):
        # `dtype` argument does the cast in‑place – no extra .astype() needed.
        return rng.standard_normal((n, block_size), dtype=A.dtype)

    # -----------------------------------------------------------------
    # Complex‑valued case – generate real and imag parts separately
    # (both N(0,1)) and combine them.
    # -----------------------------------------------------------------
    real = rng.standard_normal((n, block_size), dtype=A.real.dtype)
    imag = rng.standard_normal((n, block_size), dtype=A.real.dtype)
    return (real + 1j * imag).astype(A.dtype, copy=False)


# ------------------------------------------------------------------------
# b) Draw a uniform test matrix Ω  (size n × block_size, values ∈ [-1, 1))
# ------------------------------------------------------------------------

def _uniform_omega(A, n, block_size, *, rng=None):
    """
    Return a uniform test matrix Ω with the same dtype as ``A``.

    The entries are drawn independently from the continuous uniform
    distribution on the half‑open interval [-1, 1).

    Parameters
    ----------
    A : np.ndarray
        Reference array – its dtype (real or complex) determines the dtype
        of the returned Ω.
    n, block_size : int
        Desired shape of Ω (n rows, block_size columns).
    rng : np.random.Generator, optional
        Random‑number generator.  If omitted a fresh Generator is created
        from NumPy’s entropy source.  Supplying a seeded ``rng`` makes the
        draw reproducible.

    Returns
    -------
    np.ndarray
        Uniform matrix Ω.  If ``A`` is complex, the real and imaginary
        parts are drawn independently from Uniform[-1, 1) and then
        combined; otherwise a real‑valued matrix is returned.
    """
    if rng is None:
        rng = np.random.default_rng()

    # -----------------------------------------------------------------
    # Real‑valued case – a single call to `uniform` does everything.
    # -----------------------------------------------------------------
    if not np.iscomplexobj(A):
        # `dtype` argument does the casting inside the RNG call.
        return rng.uniform(-1.0, 1.0, size=(n, block_size)).astype(A.dtype)

    # -----------------------------------------------------------------
    # Complex‑valued case – generate real and imag parts separately.
    # -----------------------------------------------------------------
    # Use the *real* component’s dtype for the two halves; then promote
    # to the full complex dtype of A in a single, cheap view‑cast.
    real = rng.uniform(-1.0, 1.0,
                       size=(n, block_size)).astype(A.real.dtype)
    imag = rng.uniform(-1.0, 1.0,
                       size=(n, block_size)).astype(A.real.dtype)
    return (real + 1j * imag).astype(A.dtype, copy=False)


# ----------------------------------------------------------------------
# Helper: Power iteration
# ----------------------------------------------------------------------
def _power_iteration(A: np.ndarray, x: np.ndarray,
                     flag_power: int = 0) -> np.ndarray:
    """Apply (A^H @ A) repeatedly to the random vector x."""
    for _ in range(flag_power):
        x = A.conj().T @ (A @ x)
        x, _R, _p = linalg.qr(x, mode='economic', pivoting=True)
    return x

# ----------------------------------------------------------------------
# 1. Orthogonal sketch
# ----------------------------------------------------------------------
def orth_sketch(A: np.ndarray, rtol: float,
                block_size: int = 42, flag_power: int = 0) -> tuple[int, np.ndarray]:
    """
    Build an orthonormal basis for the column space of A.

    Parameters
    ----------
    A : (m x n) array
    rtol : relative tolerance - when the smallest retained singular value
           divided by the largest column norm is below this, stop.
    block_size : initial number of random columns.
    flag_power : number of power iterations (default 0).

    Returns
    -------
    k : the number of basis vectors actually retained.
    Q : (m x k) orthonormal matrix.
    """
    m, n = A.shape

    # If the block is large enough, just return an empty basis
    if block_size >= min(m, n):
        return min(m, n), np.empty_like(A, shape=(0, 0))

    while True:
        # Random starting matrix - cast to the dtype of A
        x = _uniform_omega(A, n, block_size)
        x = _power_iteration(A, x, flag_power)

        y = A @ x
        Q, R, p = linalg.qr(y, mode='economic', pivoting=True)

        # Ratio of smallest diagonal of R to largest column norm of y
        r = R.diagonal()
        d = max(abs(r[-1:])) / max(norm(y, axis=0))

        if d <= rtol:
            return block_size, Q

        # Increase block size if we have not yet reached the tolerance
        block_size = min(block_size * 4, min(m, n))
        if block_size >= min(m, n):
            return min(m, n), np.empty_like(A, shape=(m, 0))


# ----------------------------------------------------------------------
# 2. Rank-revealing QR
# ----------------------------------------------------------------------
def rrqr_randomized(A: np.ndarray, rtol: float,
                    block_size: int = 42, flag_power: int = 0) -> tuple[np.ndarray,
                                                                     np.ndarray,
                                                                     np.ndarray]:
    """
    Rank-revealing QR factorization using a randomized sketch.

    Parameters
    ----------
    A : (m x n) array
    rtol : relative tolerance used to determine the rank.
    block_size : initial sketch size.
    flag_power : number of power iterations.

    Returns
    -------
    Q : (m x k) orthonormal matrix
    R : (k x n) upper-triangular matrix
    p : pivot vector (indices of the columns in the sketch)
    """
    m, n = A.shape
    k, q = orth_sketch(A, rtol, block_size, flag_power)

    if k >= min(m, n):
        Q, R, p = linalg.qr(A, mode='economic', pivoting=True)
        k = sum(norm(R, axis=1) >= rtol * norm(A))
        return Q[:, :k], R[:k, :], p

    Aproj = q.conj().T @ A
    Qproj, R, p = linalg.qr(Aproj, mode='economic', pivoting=True)
    Q = q @ Qproj
    k = sum(norm(R, axis=1) >= rtol * norm(Aproj))
    return Q[:, :k], R[:k, :], p


# ----------------------------------------------------------------------
# 3. Truncated SVD
# ----------------------------------------------------------------------
def rrsvd_randomized(A: np.ndarray, rtol: float,
                     block_size: int = 42, flag_power: int = 0) -> tuple[np.ndarray,
                                                                      np.ndarray,
                                                                      np.ndarray]:
    """
    Truncated SVD using a randomized sketch.

    Parameters
    ----------
    A : (m x n) array
    rtol : relative tolerance used to determine the rank.
    block_size : initial sketch size.
    flag_power : number of power iterations.

    Returns
    -------
    U : (m x k) left singular vectors
    s : (k,) singular values
    V : (k x n) right singular vectors (not transposed)
    """
    m, n = A.shape
    k, q = orth_sketch(A, rtol, block_size, flag_power)

    if k >= min(m, n):
        U, s, V = linalg.svd(A, full_matrices=False)
        k = sum(abs(s) >= rtol * norm(A))
        return U[:, :k], s[:k], V[:k, :]

    Aproj = q.conj().T @ A
    Uproj, s, V = linalg.svd(Aproj, full_matrices=False)
    U = q @ Uproj
    k = sum(abs(s) >= rtol * norm(Aproj))
    return U[:, :k], s[:k], V[:k, :]


# ----------------------------------------------------------------------
# 4. Interpolative decomposition
# ----------------------------------------------------------------------
def rrid_randomized(
    A: np.ndarray,
    rtol: float,
    block_size: int = 42,
    flag_power: int = 0,
    *,
    use_svd: bool = False,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Interpolative decomposition using a randomized QR.

    Parameters
    ----------
    A : (m x n) ndarray
        Input matrix.
    rtol : float
        Relative tolerance used to determine the rank **and** the
        singular‑value cutoff when ``use_svd=True``.
    block_size : int, optional
        Initial sketch size (default 42).
    flag_power : int, optional
        Number of power‑iteration steps (default 0).
    use_svd : bool, optional
        If True, compute the coefficient matrix ``T`` via a thin‑SVD of the
        leading triangular block ``R11``.  Singular values smaller than
        ``rtol * max(s)`` are discarded (treated as zero).  If False
        (default) solve the triangular system ``R11 * T = R12`` with
        ``scipy.linalg.solve``.

    Returns
    -------
    k : int
        Estimated rank (number of selected columns).
    piv : ndarray (length n)
        Pivot indices returned by the underlying randomized QR.
    T : ndarray (k x (n‑k))
        Coefficient matrix such that ``A[:, piv[:k]] @ T ≈ A[:, piv[k:]]``.
    """
    # -----------------------------------------------------------------
    # 1. Randomized rank‑revealing QR (same as before)
    # -----------------------------------------------------------------
    Q, R, piv = rrqr_randomized(A, rtol, block_size, flag_power)

    k = R.shape[0]                     # number of selected columns

    # -----------------------------------------------------------------
    # 2. Partition the upper‑triangular factor:
    #        R = [R11  R12]
    #            [ 0    0 ]
    # -----------------------------------------------------------------
    R11 = R[:k, :k]                    # (k x k) upper‑triangular
    R12 = R[:k, k:]                    # (k x (n‑k))

    # -----------------------------------------------------------------
    # 3. Solve for the interpolation matrix T.
    # -----------------------------------------------------------------
    if use_svd:
        # ---- SVD of the leading block --------------------------------
        # R11 = U * Σ * Vᴴ
        U, s, Vh = linalg.svd(R11, full_matrices=False)

        # ---- Determine which singular values to keep -----------------
        # Keep those >= rtol * max(s).  This mirrors the way the
        # other routines (orth_sketch, rrqr_randomized, rrsvd_randomized)
        # decide rank.
        thresh = rtol * np.max(s)
        keep = s >= thresh
        q = sum(keep)

        if not np.any(keep):
            # All singular values are below the threshold – the block is
            # effectively rank‑deficient.  Return a zero matrix for T.
            T = np.zeros_like(R12)
        else:
            # Invert only the kept singular values.
            inv_s = np.zeros_like(s)
            inv_s[keep] = 1.0 / s[keep]
            # T = R11 \ R12
            T = (Vh[:q,:].conj().T @ (np.diag(inv_s[:q]) @ (U[:,:q].conj().T @ R12)))
    else:
        # ---- Classical triangular solve (original behaviour) ----------
        T = linalg.solve(np.triu(R11), R12)

    return k, piv, T


# ----------------------------------------------------------------------
# 5. Hilbert matrix generator
# ----------------------------------------------------------------------
def _hilb(n: int, m: int) -> np.ndarray:
    """
    Construct an n x m Hilbert matrix.

    Parameters
    ----------
    n : number of rows
    m : number of columns

    Returns
    -------
    H : (n x m) Hilbert matrix
    """
    c = np.zeros(n)
    r = np.zeros(m)

    for i in range(n):
        c[i] = 1.0 / (i + 1)
    for i in range(m):
        r[i] = 1.0 / (i + n)

    return linalg.hankel(c, r)


# ----------------------------------------------------------------------
#  Helper for safe maximum‑absolute‑value extraction (used only in __main__)
# ----------------------------------------------------------------------
def _safe_max_abs(mat: np.ndarray) -> float:
    """
    Return ``max(|mat|)`` safely.

    If *mat* is empty (size == 0) the function returns ``0.0`` instead of
    raising a ``ValueError``.  This mirrors the behaviour one would expect
    for a “missing” matrix – the largest entry is effectively zero.
    """
    return np.max(np.abs(mat)) if mat.size else 0.0


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored test harness for the randomized low‑rank algorithms.

In addition to the four test matrices (real/complex Hilbert &
real/complex Gaussian) the script now also runs the *SciPy* reference
implementations:

    • QR (scipy.linalg.qr, with column pivoting)
    • SVD (scipy.linalg.svd)
    • Interpolative ID (scipy.linalg.interpolative.interp_decomp)

Both the custom routines and the SciPy baselines are timed and
validated, making it trivial to spot regressions or numerical
differences.

Replace the dummy placeholders (orth_sketch, rrqr_randomized, …) with
your actual implementations and run the file – it will print a tidy,
colour‑friendly report for every matrix / algorithm combination.
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import time
import warnings
from typing import Tuple

import numpy as np
from numpy.linalg import norm
from scipy.linalg import hilbert, qr as scipy_qr, svd as scipy_svd
from scipy.linalg import interpolative as sli

# ----------------------------------------------------------------------
# Helper utilities (keep them tiny – replace with the real implementations)
# ----------------------------------------------------------------------
def _safe_max_abs(X: np.ndarray) -> float:
    """Return max|X| safely – 0.0 for an empty array."""
    return float(np.max(np.abs(X))) if X.size else 0.0

# ----------------------------------------------------------------------
# Core test runner – one matrix, one label
# ----------------------------------------------------------------------
def run_one_test(A: np.ndarray, label: str, *, rtol: float = 1e-12) -> None:
    """
    Run the full suite (custom + SciPy reference) on a single matrix.

    Parameters
    ----------
    A : np.ndarray
        Input matrix (real or complex).
    label : str
        Human‑readable name printed in the header.
    rtol : float, optional
        Relative tolerance used for rank decisions.
    """
    print(f"\n=== {label} ===")
    print(f"Matrix shape: {A.shape}\n")

    # ------------------------------------------------------------------
    # 1️⃣  Orthogonal sketch (custom)
    # ------------------------------------------------------------------
    print("[orth_sketch] running...")
    t0 = time.perf_counter()
    k_range, Q_range = orth_sketch(A, rtol=rtol)
    t1 = time.perf_counter()

    if Q_range.size == 0:
        print("[orth_sketch]   ⚠️  Sketch matrix is empty.")
        print(f"[orth_sketch]   k = {k_range}, basis shape = {Q_range.shape}")
    else:
        ortho_err = norm(Q_range.conj().T @ Q_range - np.eye(k_range))
        print(f"[orth_sketch]   k = {k_range}, basis shape = {Q_range.shape}")
        print(f"[orth_sketch]   orthonormality error = {ortho_err:e}")
    print(f"[orth_sketch]   elapsed time = {t1 - t0:.3f} s\n")

    # ------------------------------------------------------------------
    # 2️⃣  Rank‑revealing QR (custom)
    # ------------------------------------------------------------------
    print("[rrqr_randomized] running...")
    t0 = time.perf_counter()
    Q_rrqr, R_rrqr, piv = rrqr_randomized(A, rtol=rtol)
    t1 = time.perf_counter()
    rank_rrqr = R_rrqr.shape[0]
    A_perm = A[:, piv]
    recon_err = norm(Q_rrqr @ R_rrqr - A_perm) / norm(A_perm)
    print(f"[rrqr_randomized] rank = {rank_rrqr}")
    print(f"[rrqr_randomized] reconstruction relative error = {recon_err:e}")
    print(f"[rrqr_randomized] elapsed time = {t1 - t0:.3f} s\n")

    # ------------------------------------------------------------------
    # 3️⃣  Truncated SVD (custom)
    # ------------------------------------------------------------------
    print("[rrsvd_randomized] running...")
    t0 = time.perf_counter()
    U_rrsvd, s_rrsvd, Vt_rrsvd = rrsvd_randomized(A, rtol=rtol)
    t1 = time.perf_counter()
    rank_svd = s_rrsvd.size
    A_svd = U_rrsvd @ np.diag(s_rrsvd) @ Vt_rrsvd
    svd_err = norm(A_svd - A) / norm(A)
    print(f"[rrsvd_randomized] rank = {rank_svd}")
    print(f"[rrsvd_randomized] reconstruction relative error = {svd_err:e}")
    print(f"[rrsvd_randomized] elapsed time = {t1 - t0:.3f} s\n")

    # ------------------------------------------------------------------
    # 4️⃣  Interpolative ID (custom – both variants)
    # ------------------------------------------------------------------
    for use_svd, variant in [(False, "triangular‑solve"), (True, "SVD‑based")]:
        print(f"[rrid_randomized – {variant}] running...")
        t0 = time.perf_counter()
        k_id, piv_id, T_id = rrid_randomized(A, rtol=rtol, use_svd=use_svd)
        t1 = time.perf_counter()

        A_id_approx = A[:, piv_id[:k_id]] @ T_id
        id_err = norm(A[:, piv_id[k_id:]] - A_id_approx) / norm(A)

        print(f"[rrid_randomized – {variant}] rank = {k_id}")
        print(f"[rrid_randomized – {variant}] interp. relative error = {id_err:e}")
        print(f"[rrid_randomized – {variant}] max(|T|) = {_safe_max_abs(T_id):.3e}")
        print(f"[rrid_randomized – {variant}] elapsed time = {t1 - t0:.3f} s\n")

    # ------------------------------------------------------------------
    # 5️⃣  **SciPy reference implementations**
    # ------------------------------------------------------------------
    print("=== SciPy reference algorithms ===")

    # ------------------------------------------------------------------
    # 5a) QR with column pivoting (SciPy)
    # ------------------------------------------------------------------
    print("[scipy.linalg.qr] running...")
    t0 = time.perf_counter()
    Q_sci, R_sci, piv_sci = scipy_qr(A, pivoting=True, mode="economic")
    t1 = time.perf_counter()
    # Rank by same relative‑tolerance rule as above
    diag = np.abs(np.diag(R_sci))
    rank_sci = int(np.sum(diag > rtol * diag[0]))
    A_perm_sci = A[:, piv_sci]
    recon_err_sci = norm(Q_sci[:, :rank_sci] @ R_sci[:rank_sci, :] - A_perm_sci) / norm(A_perm_sci)
    print(f"[scipy.linalg.qr] rank = {rank_sci}")
    print(f"[scipy.linalg.qr] reconstruction relative error = {recon_err_sci:e}")
    print(f"[scipy.linalg.qr] elapsed time = {t1 - t0:.3f} s\n")

    # ------------------------------------------------------------------
    # 5b) SVD (SciPy)
    # ------------------------------------------------------------------
    print("[scipy.linalg.svd] running...")
    t0 = time.perf_counter()
    U_sci, s_sci, Vt_sci = scipy_svd(A, full_matrices=False, lapack_driver='gesdd')
    t1 = time.perf_counter()
    rank_sci_svd = int(np.sum(s_sci > rtol * s_sci[0]))
    A_sci_svd = (U_sci[:, :rank_sci_svd] *
                 s_sci[:rank_sci_svd]) @ Vt_sci[:rank_sci_svd, :]
    svd_err_sci = norm(A_sci_svd - A) / norm(A)
    print(f"[scipy.linalg.svd] rank = {rank_sci_svd}")
    print(f"[scipy.linalg.svd] reconstruction relative error = {svd_err_sci:e}")
    print(f"[scipy.linalg.svd] elapsed time = {t1 - t0:.3f} s\n")

    # ------------------------------------------------------------------
    # 5c) Interpolative ID (SciPy)
    # ------------------------------------------------------------------
    print("[scipy.linalg.interpolative.interp_decomp] running...")
    t0 = time.perf_counter()
    # `eps` is an *absolute* tolerance; we turn our relative tolerance into
    # an absolute one by scaling with the Frobenius norm of A.
    eps_abs = rtol * norm(A, ord='fro')
    k, idx_sci, proj_sci = sli.interp_decomp(A, eps_abs)
    t1 = time.perf_counter()
    # Re‑construct using the skeleton columns
    A_id_sci = A[:, idx_sci[:k]] @ proj_sci
    id_err_sci = norm(A[:,idx_sci[k:]] - A_id_sci) / norm(A)
    print(f"[scipy.interpolative] rank = {k}")
    print(f"[scipy.interpolative] interp. relative error = {id_err_sci:e}")
    print(f"[scipy.interpolative] max(|proj|) = {_safe_max_abs(proj_sci):.3e}")
    print(f"[scipy.interpolative] elapsed time = {t1 - t0:.3f} s\n")

    print("-" * 70)


# ----------------------------------------------------------------------
# Main driver – builds the four test matrices and dispatches them
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------
    # Seed for reproducibility
    # --------------------------------------------------------------
    np.random.seed(0)

    # --------------------------------------------------------------
    # Define the *real* test matrices (Hilbert & Gaussian)
    # --------------------------------------------------------------
    m, n = 4000, 2000

    real_cases = [
        {
            "name": "Hilbert",
            "matrix": _hilb(m, n).astype(np.float64),
        },
        {
            "name": "Gaussian",
            "matrix": np.random.normal(size=(m, n)).astype(np.float64),
        },
    ]

    # --------------------------------------------------------------
    # Run the real‑valued cases
    # --------------------------------------------------------------
    for case in real_cases:
        run_one_test(case["matrix"], case["name"])

    # --------------------------------------------------------------
    # Build the *complex* counterparts (H + 1j·H  &  G + 1j·G)
    # --------------------------------------------------------------
    complex_cases = [
        {
            "name": f"{case['name']} (complex)",
            "matrix": case["matrix"].astype(np.complex128)
            + 1j * case["matrix"].astype(np.complex128),
        }
        for case in real_cases
    ]

    # --------------------------------------------------------------
    # Run the complex‑valued cases
    # --------------------------------------------------------------
    for case in complex_cases:
        run_one_test(case["matrix"], case["name"])

    # --------------------------------------------------------------
    # End of script
    # --------------------------------------------------------------
    print("\nAll tests completed.\n")
