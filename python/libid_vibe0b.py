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
from __future__ import annotations

import numpy as np
import time                           # <-- timing helper
from scipy import linalg
from numpy.linalg import norm


import numpy as np
from typing import Tuple


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


# --------------------------------------------------------------------- #
# b) Draw a uniform test matrix Ω  (size n × block_size, values ∈ [-1, 1))
# --------------------------------------------------------------------- #

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


def _power_iteration(A: np.ndarray, X: np.ndarray, n_iter: int) -> np.ndarray:
    """
    Apply `n_iter` subspace power‑iterations to the test matrix `X`.

    This improves the quality of the sketch when the singular values of `A`
    decay slowly.

    Parameters
    ----------
    A : (m, n) array
    X : (n, k) array – current random test matrix
    n_iter : non‑negative integer

    Returns
    -------
    X̂ : (n, k) array – power‑iterated test matrix
    """
    for _ in range(n_iter):
        # 1) orthogonalise X
        Q, _ = linalg.qr(X, mode="economic")
        # 2) multiply by Aᵀ and orthogonalise again
        Z = A.conj().T @ Q
        Qz, _ = linalg.qr(Z, mode="economic")
        # 3) new test matrix
        X = A @ Qz
    return X


def orth_sketch(
    A: np.ndarray,
    rtol: float,
    block_size: int = 42,
    flag_power: int = 0,
) -> Tuple[int, np.ndarray]:
    """
    Adaptive blocked randomized range finder (column‑space builder).

    The routine draws blocks of `block_size` random vectors, orthogonalises
    them against the basis already built, optionally applies a few power
    iterations, and stops as soon as the *relative* Frobenius‑norm error

        ‖A – Q @ (Qᴴ A)‖_F / ‖A‖_F  ≤  rtol

    falls below the user‑supplied tolerance `rtol`.  The maximal rank that can
    ever be produced is ``min(m, n)``; the block size is multiplied by four
    after each unsuccessful iteration (but never exceeds that bound).

    Parameters
    ----------
    A : (m, n) ndarray
        Input matrix (real or complex).
    rtol : float, > 0
        Desired relative error (Frobenius norm).  The algorithm stops when the
        residual norm divided by ``‖A‖_F`` is ≤ ``rtol``.
    block_size : int, default 42
        Initial number of random columns per block.  Must be at least 1.
    flag_power : int, default 0
        Number of power‑iteration steps applied to each block.  ``0`` means
        no power iterations.

    Returns
    -------
    k : int
        The actual number of basis vectors retained (``k ≤ min(m, n)``).
    Q : (m, k) ndarray
        Orthonormal basis for the (approximate) column space of ``A``.
        When ``k == 0`` an empty ``(m, 0)`` array is returned.
    """
    # ------------------------------------------------------------------ #
    # 0️⃣  sanity checks
    # ------------------------------------------------------------------ #
    if rtol <= 0:
        raise ValueError("rtol must be a positive number.")
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    if flag_power < 0 or not isinstance(flag_power, int):
        raise ValueError("flag_power must be a non‑negative integer.")

    m, n = A.shape
    max_rank = min(m, n)

    # ------------------------------------------------------------------ #
    # 1️⃣  quick exit for degenerate matrices
    # ------------------------------------------------------------------ #
    if max_rank == 0:
        return 0, np.empty((m, 0), dtype=A.dtype)

    # ------------------------------------------------------------------ #
    # 2️⃣  pre‑allocate the (potentially) full‑size basis; we will trim later
    # ------------------------------------------------------------------ #
    Q_full = np.empty((m, max_rank), dtype=A.dtype)

    # Working copy of the residual and its Frobenius norm
    A_res = A.copy()
    A_norm = np.linalg.norm(A, ord="fro")

    cols_written = 0   # total number of columns actually stored

    max_blocks = 100000
    for s in range(max_blocks):
        # ----- draw a (possibly complex) test matrix Omega (n x block) -----
        x = _gaussian_omega(A_res, n, block_size)

        Y = A @ x
        Y = _power_iteration(A_res, Y, flag_power)

        # ----- orthogonalize against the basis already built -----
        if s > 0:
            Qprev = Q_full[:, :cols_written]
            Y -= Qprev @ (Qprev.conj().T @ Y)

        # ----- QR of the new block -----
        Qblk, _ = linalg.qr(Y, mode="economic")
        Bblk = Qblk.conj().T @ A_res
        A_res -= Qblk @ Bblk

        # ----- store only the columns that fit (the last block may be truncated) -----
        col_start = cols_written
        col_end   = min(min(n,m), col_start + block_size)
        col_count = col_end - col_start

        Q_full[:, col_start:col_end] = Qblk[:, :col_count]
        cols_written += col_count
        
        # ----- stopping criterion -----
        residual_norm = np.linalg.norm(A_res, ord="fro")
        rel_err = residual_norm / A_norm
        if rel_err <= rtol:
            break

        if rel_err > rtol:
            block_size = min(block_size*4,min(m,n))

        if (block_size >= min(m,n)):
            return min(m,n), np.empty_like(A, shape=(m, 0))

    # ------------------------------------------------------------------ #
    # 3️⃣  Trim to the actual size and return
    # ------------------------------------------------------------------ #
    Q = Q_full[:, :cols_written]
    k = Q.shape[1]

    return k, Q


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



# ----------------------------------------------------------------------
# Core test runner – one matrix, one label
# ----------------------------------------------------------------------
def run_one_test(A: np.ndarray, label: str) -> None:
    """
    Execute the full suite of randomized algorithms on ``A`` and
    print a concise, colour‑coded report.

    Parameters
    ----------
    A : np.ndarray
        Input matrix (real or complex).
    label : str
        Human‑readable identifier shown in the output header.
    """
    print(f"\n=== {label} ===")
    print(f"Matrix shape: {A.shape}\n")

    # ------------------------------------------------------------------
    # 1️⃣  Orthogonal sketch
    # ------------------------------------------------------------------
    print("[orth_sketch] running...")
    t0 = time.perf_counter()
    k_range, Q_range = orth_sketch(A, rtol=1e-12)
    t1 = time.perf_counter()

    if Q_range.size == 0:  # empty sketch matrix
        print("[orth_sketch]   ⚠️  Sketch matrix is empty.")
        print(f"[orth_sketch]   k = {k_range}, basis shape = {Q_range.shape}")
    else:
        # Use conjugate transpose for complex data.
        ortho_err = norm(Q_range.conj().T @ Q_range - np.eye(k_range))
        print(f"[orth_sketch]   k = {k_range}, basis shape = {Q_range.shape}")
        print(f"[orth_sketch]   orthonormality error = {ortho_err:e}")
    print(f"[orth_sketch]   elapsed time = {t1 - t0:.3f} s\n")

    # ------------------------------------------------------------------
    # 2️⃣  Rank‑revealing QR
    # ------------------------------------------------------------------
    print("[rrqr_randomized] running...")
    t0 = time.perf_counter()
    Q_rrqr, R_rrqr, piv = rrqr_randomized(A, rtol=1e-12)
    t1 = time.perf_counter()
    rank_rrqr = R_rrqr.shape[0]
    A_perm = A[:, piv]
    recon_err = norm(Q_rrqr @ R_rrqr - A_perm) / norm(A_perm)
    print(f"[rrqr_randomized] rank = {rank_rrqr}")
    print(f"[rrqr_randomized] reconstruction relative error = {recon_err:e}")
    print(f"[rrqr_randomized] elapsed time = {t1 - t0:.3f} s\n")

    # ------------------------------------------------------------------
    # 3️⃣  Truncated SVD
    # ------------------------------------------------------------------
    print("[rrsvd_randomized] running...")
    t0 = time.perf_counter()
    U_rrsvd, s_rrsvd, Vt_rrsvd = rrsvd_randomized(A, rtol=1e-12)
    t1 = time.perf_counter()
    rank_svd = s_rrsvd.size
    A_svd = U_rrsvd @ np.diag(s_rrsvd) @ Vt_rrsvd
    svd_err = norm(A_svd - A) / norm(A)
    print(f"[rrsvd_randomized] rank = {rank_svd}")
    print(f"[rrsvd_randomized] reconstruction relative error = {svd_err:e}")
    print(f"[rrsvd_randomized] elapsed time = {t1 - t0:.3f} s\n")

    # ------------------------------------------------------------------
    # 4️⃣  Interpolative decomposition (both variants)
    # ------------------------------------------------------------------
    for use_svd, variant in [(False, "triangular‑solve"), (True, "SVD‑based")]:
        print(f"[rrid_randomized – {variant}] running...")
        t0 = time.perf_counter()
        k_id, piv_id, T_id = rrid_randomized(A, rtol=1e-12, use_svd=use_svd)
        t1 = time.perf_counter()

        # Approximation built from the selected columns.
        A_id_approx = A[:, piv_id[:k_id]] @ T_id
        # Error is measured on the *remaining* columns.
        id_err = norm(A[:, piv_id[k_id:]] - A_id_approx) / norm(A)

        print(f"[rrid_randomized – {variant}] rank = {k_id}")
        print(f"[rrid_randomized – {variant}] interp. relative error = {id_err:e}")
        print(f"[rrid_randomized – {variant}] max(|T|) = {_safe_max_abs(T_id):.3e}")
        print(f"[rrid_randomized – {variant}] elapsed time = {t1 - t0:.3f} s\n")


# ----------------------------------------------------------------------
# Main driver – builds the four test matrices and dispatches them
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Seed for reproducibility
    # ------------------------------------------------------------------
    np.random.seed(0)

    # ------------------------------------------------------------------
    # Define the *real* test matrices (Hilbert & Gaussian)
    # ------------------------------------------------------------------
    m, n = 4000, 2000

    real_cases = [
        {
            "name": "Real‑Hilbert",
            "matrix": _hilb(m, n).astype(np.float64),  # deterministic, ill‑cond.
        },
        {
            "name": "Real‑Gaussian",
            "matrix": np.random.normal(size=(m, n)).astype(np.float64),
        },
    ]

    # ------------------------------------------------------------------
    # Run the real‑valued cases
    # ------------------------------------------------------------------
    for case in real_cases:
        run_one_test(case["matrix"], case["name"])

    # ------------------------------------------------------------------
    # Build the *complex* counterparts (H + 1j·H  &  G + 1j·G)
    # ------------------------------------------------------------------
    complex_cases = [
        {
            "name": f"{case['name']} (complex)",
            "matrix": case["matrix"].astype(np.complex128)
            + 1j * case["matrix"].astype(np.complex128),
        }
        for case in real_cases
    ]

    # ------------------------------------------------------------------
    # Run the complex‑valued cases
    # ------------------------------------------------------------------
    for case in complex_cases:
        run_one_test(case["matrix"], case["name"])

    # ------------------------------------------------------------------
    # End of script
    # ------------------------------------------------------------------
    print("\nAll tests completed.\n")
