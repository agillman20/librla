#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as la
import time

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def hilbert_rect(m: int, n: int) -> np.ndarray:
    """
    Generate a rectangular Hilbert matrix.

    The Hilbert matrix is defined by

        H(i,j) = 1 / (i + j - 1)

    for i = 1,…,m and j = 1,…,n.  It is a classic example of an
    ill‑conditioned matrix and is useful for testing numerical
    algorithms.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    H : ndarray, shape (m, n)
        The m‑by‑n rectangular Hilbert matrix.
    """
    i = np.arange(1, m + 1)[:, None]   # (m,1)
    j = np.arange(1, n + 1)[None, :]   # (1,n)
    return 1.0 / (i + j - 1.0)


# ----------------------------------------------------------------------
# 1) Full column‑pivoted QR (MKL / LAPACK behind the scenes)
# ----------------------------------------------------------------------
def pivoted_qr_mkl(A: np.ndarray):
    """
    Compute a full column‑pivoted QR factorization.

    The routine calls :func:`scipy.linalg.qr` with ``pivoting=True``,
    which uses the LAPACK implementation (MKL when available).

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix.

    Returns
    -------
    Q : ndarray, shape (m, min(m,n))
        Orthogonal matrix with orthonormal columns.
    R : ndarray, shape (min(m,n), n)
        Upper‑triangular matrix.
    perm : ndarray, shape (n,)
        Permutation indices such that ``A[:, perm] = Q @ R``.
    """
    Q, R, perm = la.qr(A, mode='economic', pivoting=True)
    return Q, R, perm


# ----------------------------------------------------------------------
# 2) Rank‑k column‑pivoted QR (truncate after the full QR)
# ----------------------------------------------------------------------
def pivoted_qr_of_specified_rank(A: np.ndarray, k: int):
    """
    Compute a column‑pivoted QR factorization and truncate it to at most ``k``
    columns.

    The function also returns an estimate of the numerical rank based on a
    relative tolerance applied to the diagonal of ``R``.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix.
    k : int
        Desired maximum number of columns (rank truncation).

    Returns
    -------
    Qk : ndarray, shape (m, k)
        First ``k`` columns of the orthogonal factor.
    Rk : ndarray, shape (k, k)
        Upper‑triangular factor of the truncated decomposition.
    frank : int
        Estimated numerical rank (number of diagonal entries of ``Rk`` that
        exceed the tolerance).
    perm : ndarray, shape (n,)
        Column permutation such that ``A[:, perm] = Q @ R``.
    """
    Q, R, perm = la.qr(A, mode='economic', pivoting=True)
    Qk = Q[:, :k]          # (m,k)
    Rk = R[:k, :k]         # (k,k)

    # Simple relative tolerance on the diagonal of Rk to estimate numerical rank
    tol = max(A.shape) * np.spacing(np.linalg.norm(Rk, ord=2))
    diag_abs = np.abs(np.diag(Rk))
    frank = np.sum(diag_abs > tol)

    return Qk, Rk, frank, perm


# ----------------------------------------------------------------------
# 3) Randomized QB – single‑vector version (randQB_p)
# ----------------------------------------------------------------------
def randQB_p(A: np.ndarray,
             k: int,
             p: int = 0,
             rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a randomized QB factorization with a single block.

    The algorithm draws a Gaussian test matrix ``Omega`` of size
    ``(n, k)`` and forms ``Y = A @ Omega``.  Optional power iterations
    improve the quality of the basis.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix (real or complex).
    k : int
        Target rank (size of the block).
    p : int, optional
        Number of power iterations (default is 0, i.e. no power iterations).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.  If ``None`` a new
        default generator is created.

    Returns
    -------
    Q : ndarray, shape (m, k)
        Orthonormal basis matrix.
    B : ndarray, shape (k, n)
        Coefficient matrix such that ``A ≈ Q @ B``.
    """
    if rng is None:
        rng = np.random.default_rng()

    m, n = A.shape
    dtype = A.dtype

    # Random test matrix Omega (n x k)
    if np.iscomplexobj(A):
        Omega = (rng.standard_normal((n, k), dtype=np.float64) +
                 1j * rng.standard_normal((n, k), dtype=np.float64)).astype(dtype)
    else:
        Omega = rng.standard_normal((n, k), dtype=dtype)

    Y = A @ Omega

    # Power iterations (optional)
    for _ in range(p):
        QY, _ = la.qr(Y, mode='economic')
        Z = A.conj().T @ QY
        QZ, _ = la.qr(Z, mode='economic')
        Y = A @ QZ

    Q, _ = la.qr(Y, mode='economic')
    B = Q.conj().T @ A
    return Q, B


# ----------------------------------------------------------------------
# 4) Randomized QB – blocked version (randQB_pb)
# ----------------------------------------------------------------------
def randQB_pb(A: np.ndarray,
              block: int,
              nblocks: int,
              p: int = 0,
              rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Blocked randomized QB factorization.

    The algorithm processes ``nblocks`` blocks, each of size ``block``.
    It is a direct Python translation of the MATLAB routine
    ``randQB_pb`` and works for both real and complex data.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix.
    block : int
        Number of columns in each block (block size).
    nblocks : int
        Number of blocks to process.  The total rank is ``block * nblocks``.
    p : int, optional
        Number of power iterations per block (default 0).
    rng : np.random.Generator, optional
        Random generator; if ``None`` a new default generator is created.

    Returns
    -------
    Q : ndarray, shape (m, block * nblocks)
        Orthonormal basis matrix.
    B : ndarray, shape (block * nblocks, n)
        Coefficient matrix such that ``A ≈ Q @ B``.
    """
    if rng is None:
        rng = np.random.default_rng()

    m, n = A.shape
    l = block * nblocks
    assert l <= min(m, n), (
        f"Requested rank l={l} exceeds matrix dimensions ({m}x{n})."
    )

    Q = np.zeros((m, l), dtype=A.dtype)
    B = np.zeros((l, n), dtype=A.dtype)
    Awork = A.copy()

    for s in range(nblocks):
        col_start = s * block
        col_end   = col_start + block
        cols_block = slice(col_start, col_end)

        # Random test matrix for the current block
        if np.iscomplexobj(A):
            Omega = (rng.standard_normal((n, block), dtype=np.float64) +
                     1j * rng.standard_normal((n, block), dtype=np.float64)).astype(A.dtype)
        else:
            Omega = rng.standard_normal((n, block), dtype=A.dtype)

        Y = Awork @ Omega

        # Power iterations (optional)
        for _ in range(p):
            QY, _ = la.qr(Y, mode='economic')
            Z = Awork.conj().T @ QY
            QZ, _ = la.qr(Z, mode='economic')
            Y = Awork @ QZ

        # Orthogonalize against the basis already built
        if s > 0:
            Qprev = Q[:, :col_start]
            Y -= Qprev @ (Qprev.conj().T @ Y)

        Qblk, _ = la.qr(Y, mode='economic')
        Bblk = Qblk.conj().T @ Awork
        Awork -= Qblk @ Bblk

        Q[:, cols_block] = Qblk
        B[cols_block, :] = Bblk

    return Q, B


# ----------------------------------------------------------------------
# 5) Adaptive randomized QB
# ----------------------------------------------------------------------
def randqb_adaptive(A: np.ndarray,
                    block: int,
                    max_blocks: int,
                    p: int = 0,
                    tol: float = 1e-6,
                    rng: np.random.Generator = None,
                    return_rank: bool = False) -> tuple[np.ndarray, np.ndarray, int] | tuple[np.ndarray, np.ndarray]:
    """
    Adaptive blocked randomized QB factorization.

    The algorithm adds blocks of size ``block`` until either the
    prescribed maximum number of blocks ``max_blocks`` is reached or the
    relative Frobenius norm error falls below ``tol``.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix (real or complex).
    block : int
        Number of columns per block.
    max_blocks : int
        Upper bound on the number of blocks to process.  The
        theoretical maximum rank is ``block * max_blocks``.
    p : int, optional
        Number of power iterations per block (default 0).
    tol : float, optional
        Relative error tolerance.  The algorithm stops when

        ``||A - Q @ B||_F / ||A||_F <= tol``

        (default 1e-6).
    rng : np.random.Generator, optional
        Random generator; if ``None`` a new default generator is created.
    return_rank : bool, optional
        If ``True`` also return the actual number of columns of ``Q``
        (and rows of ``B``) that contain data.

    Returns
    -------
    Q : ndarray, shape (m, r)
        Orthonormal basis matrix, where ``r`` is the number of columns
        actually written (``r <= block * max_blocks``).
    B : ndarray, shape (r, n)
        Coefficient matrix such that ``A ≈ Q @ B``.
    r : int, optional
        The actual rank (number of columns of ``Q``) when ``return_rank`` is
        ``True``.
    """
    if rng is None:
        rng = np.random.default_rng()

    m, n = A.shape
    dtype = A.dtype

    # Upper bound on the rank we might ever need
    l_max = min(min(m, n), block * max_blocks)
    assert l_max <= min(m, n), (
        f"Maximum rank {l_max} exceeds matrix dimensions ({m}x{n})."
    )

    Q = np.zeros((m, l_max), dtype=dtype)
    B = np.zeros((l_max, n), dtype=dtype)
    Awork = A.copy()
    A_norm = np.linalg.norm(A, ord="fro")

    cols_written = 0   # total number of columns actually stored

    for s in range(max_blocks):
        # ----- draw a (possibly complex) test matrix Omega (n x block) -----
        if np.iscomplexobj(A):
            Omega = (rng.standard_normal((n, block), dtype=np.float64) +
                     1j * rng.standard_normal((n, block), dtype=np.float64)).astype(dtype)
        else:
            Omega = rng.standard_normal((n, block), dtype=dtype)

        Y = Awork @ Omega

        # ----- power iterations (optional) -----
        for _ in range(p):
            QY, _ = la.qr(Y, mode="economic")
            Z = Awork.conj().T @ QY
            QZ, _ = la.qr(Z, mode="economic")
            Y = Awork @ QZ

        # ----- orthogonalize against the basis already built -----
        if s > 0:
            Qprev = Q[:, :cols_written]
            Y -= Qprev @ (Qprev.conj().T @ Y)

        # ----- QR of the new block -----
        Qblk, _ = la.qr(Y, mode="economic")
        Bblk = Qblk.conj().T @ Awork
        Awork -= Qblk @ Bblk

        # ----- store only the columns that fit (the last block may be truncated) -----
        col_start = cols_written
        col_end   = min(l_max, col_start + block)
        col_count = col_end - col_start

        Q[:, col_start:col_end] = Qblk[:, :col_count]
        B[col_start:col_end, :] = Bblk[:col_count, :]

        cols_written += col_count

        # ----- stopping criterion -----
        residual_norm = np.linalg.norm(Awork, ord="fro")
        rel_err = residual_norm / A_norm
        if rel_err <= tol:
            break

    # Trim the outputs to the actual size
    Q = Q[:, :cols_written]
    B = B[:cols_written, :]

    if return_rank:
        return Q, B, cols_written
    else:
        return Q, B


# ----------------------------------------------------------------------
# QUICK TEST DRIVER (includes the adaptive version)
# ----------------------------------------------------------------------
def quick_test():
    """
    Run a small collection of sanity checks and performance timings.

    The driver exercises:

    * Full column‑pivoted QR
    * Fixed‑rank column‑pivoted QR
    * Randomized QB (single‑block)
    * Randomized QB (blocked)
    * Adaptive randomized QB

    Users can switch between a random Gaussian matrix and a rectangular
    Hilbert matrix by commenting/uncommenting the appropriate lines.
    """
    print("\n--- QUICK TESTS WITH ADAPTIVE QB -----------------------------------\n")

    # ------------------- Choose a test matrix -------------------
    # Random Gaussian matrix (like MATLAB's randn(120,85))
    A_rand = np.random.randn(120, 85)

    # Rectangular Hilbert matrix (much smaller than the 4000x2000 used in MATLAB,
    # otherwise the QR would be very slow on a typical laptop)
    A_hilb = hilbert_rect(4000, 2000)   # feel free to change the sizes

    # Pick which matrix you want to benchmark:
    A = A_rand          # <-- use the random Gaussian matrix
    A = A_hilb       # <-- uncomment to use the Hilbert matrix instead

    # ------------------------------------------------------------------
    # 1) Full column‑pivoted QR
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    Qfull, Rfull, perm_full = pivoted_qr_mkl(A)
    t1 = time.perf_counter()
    resid_full = la.norm(A[:, perm_full] - Qfull @ Rfull, ord='fro')
    print(f"Full QR (time = {t1 - t0:.4f}s)  residual = {resid_full:e}")

    # ------------------------------------------------------------------
    # 2) Fixed‑rank column‑pivoted QR (k = 30)
    # ------------------------------------------------------------------
    k = 30
    t0 = time.perf_counter()
    Qk, Rk, frank, perm_k = pivoted_qr_of_specified_rank(A, k)
    t1 = time.perf_counter()
    resid_k = la.norm(A[:, perm_k[:k]] - Qk @ Rk, ord='fro')
    print(f"Rank‑k QR (k={k}) (time = {t1 - t0:.4f}s)  "
          f"rank found = {frank}, residual = {resid_k:e}")

    # ------------------------------------------------------------------
    # 3) Randomized QB – single‑vector (k = 30, p = 1)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    Qrb, Brb = randQB_p(A, k=30, p=1)
    t1 = time.perf_counter()
    resid_rb = la.norm(A - Qrb @ Brb, ord='fro')
    print(f"randQB_p (k=30, p=1) (time = {t1 - t0:.4f}s)  residual = {resid_rb:e}")

    # ------------------------------------------------------------------
    # 4) Randomized QB – blocked (block = 10, steps = 3, p = 1)
    # ------------------------------------------------------------------
    block = 10
    steps = 3                # total rank = block * steps = 30
    t0 = time.perf_counter()
    QrbB, BrbB = randQB_pb(A, block=block, nblocks=steps, p=1)
    t1 = time.perf_counter()
    resid_rbB = la.norm(A - QrbB @ BrbB, ord='fro')
    print(f"randQB_pb (block={block}, steps={steps}, p=1) "
          f"(time = {t1 - t0:.4f}s)  residual = {resid_rbB:e}")

    # ------------------------------------------------------------------
    # 5) Adaptive randomized QB (stop when rel‑error <= tol)
    # ------------------------------------------------------------------
    block = 42          # block size
    max_blocks = 20     # allow up to block*max_blocks columns
    p_power = 0         # power‑iteration count
    tol_rel = 1e-12     # relative Frobenius tolerance

    t0 = time.perf_counter()
    Qad, Bad, cols_used = randqb_adaptive(A,
                                          block=block,
                                          max_blocks=max_blocks,
                                          p=p_power,
                                          tol=tol_rel,
                                          return_rank=True)
    t1 = time.perf_counter()
    resid_ad = la.norm(A - Qad @ Bad, ord='fro')
    rel_err_ad = resid_ad / la.norm(A, ord='fro')
    print(f"Adaptive QB (block={block}, max_blocks={max_blocks}, p={p_power}, tol={tol_rel:e})\n"
          f"    time = {t1 - t0:.4f}s,  rank used = {cols_used},  rel. residual = {rel_err_ad:e}")

if __name__ == "__main__":
    quick_test()
