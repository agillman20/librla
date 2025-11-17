#!/usr/bin/env python3
"""
Interpolative Decomposition using deterministic RRQR (LAPACK geqp3)

This module provides ID implementation using scipy's QR with column pivoting,
which internally calls LAPACK's dgeqp3/zgeqp3 (strong rank-revealing QR).

This serves as a comparison baseline to libid's randomized sketching approach,
representing what deterministic methods like SciPy achieve.
"""

import numpy as np
from scipy.linalg import qr, solve_triangular


def id_rrqr(A, rtol, kmax=None):
    """
    Interpolative decomposition using deterministic RRQR (geqp3).

    This function uses scipy.linalg.qr with pivoting, which calls LAPACK's
    geqp3 routine. Unlike randomized methods, this is:
    - Deterministic (same results every run)
    - Examines all columns sequentially
    - No early stopping capability (must compute full QR)

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix (real or complex)
    rtol : float
        Relative tolerance for rank determination.
        If rtol >= 1, interpreted as maximum rank.
    kmax : int, optional
        Maximum rank (defaults to min(m,n))

    Returns
    -------
    k : int
        Determined rank
    jpiv : ndarray, shape (n,)
        Column permutation indices (0-based)
        A[:, jpiv] = A[:, jpiv[:k]] @ [I; T]
    T : ndarray, shape (k, n-k)
        Interpolation matrix

    Notes
    -----
    Algorithm steps using geqp3:
    1. Compute full pivoted QR: A[:, jpiv] = Q @ R
    2. Determine rank k from diagonal of R
    3. Extract R11 (kxk) and R12 (kx(n-k))
    4. Solve R11 @ T = R12

    The key difference from randomized methods:
    - Must compute FULL QR factorization (O(mnk) where k can be large)
    - No adaptive block growth
    - No early detection of full-rank scenarios
    """
    m, n = A.shape

    # Allow rtol >= 1 to be interpreted as maximum rank
    flag_kmax = False
    if rtol >= 1:
        flag_kmax = True
        kmax_input = int(np.floor(rtol))
        rtol = max(m, n) * np.finfo(A.dtype).eps
    elif kmax is None:
        kmax_input = min(m, n)
    else:
        kmax_input = kmax

    # Compute full pivoted QR factorization
    # This calls LAPACK's dgeqp3/zgeqp3
    Q, R, jpiv = qr(A, mode='economic', pivoting=True)

    # Determine rank from diagonal of R
    # Use relative tolerance: diag(R[k]) >= rtol * diag(R[0])
    diag_abs = np.abs(np.diag(R))

    if diag_abs[0] == 0:
        # Zero matrix
        k = 0
    else:
        k = int(np.sum(diag_abs >= rtol * diag_abs[0]))

    # Apply maximum rank constraint
    k = min(k, kmax_input)

    # Handle edge cases
    if k == 0:
        # Rank 0: return empty T
        return 0, jpiv, np.zeros((0, n), dtype=A.dtype)

    if k == n:
        # Full rank: no skeleton columns to approximate
        return k, jpiv, np.zeros((k, 0), dtype=A.dtype)

    # Extract submatrices
    R11 = R[:k, :k]  # Upper triangular (k x k)
    R12 = R[:k, k:]  # Rectangular (k x (n-k))

    # Solve R11 @ T = R12 for T
    # R11 is upper triangular, so use triangular solver
    T = solve_triangular(np.triu(R11), R12, lower=False,
                         overwrite_b=False, check_finite=False)

    return k, jpiv, T


def compare_methods_on_matrix(A, rtol_or_rank):
    """
    Quick comparison of ID methods on a single matrix.

    Useful for debugging and understanding differences.
    """
    from libid import id_sketch
    import time

    print(f"Matrix: {A.shape[0]}x{A.shape[1]}, dtype={A.dtype}")
    print(f"Parameter: rtol_or_rank = {rtol_or_rank}")
    print("-" * 60)

    # Method 1: RRQR (this module)
    t0 = time.perf_counter()
    k_rrqr, piv_rrqr, T_rrqr = id_rrqr(A, rtol_or_rank)
    t_rrqr = time.perf_counter() - t0

    if T_rrqr.size > 0:
        A_skel = A[:, piv_rrqr[k_rrqr:]]
        A_basis = A[:, piv_rrqr[:k_rrqr]]
        A_approx = A_basis @ T_rrqr
        err_rrqr = np.linalg.norm(A_skel - A_approx) / np.linalg.norm(A)
        max_T_rrqr = np.max(np.abs(T_rrqr))
    else:
        err_rrqr = 0.0
        max_T_rrqr = 0.0

    print(f"RRQR (geqp3):    k={k_rrqr:4d}, err={err_rrqr:.3e}, "
          f"max|T|={max_T_rrqr:.3e}, time={t_rrqr:.4f}s")

    # Method 2: libid (randomized sketch)
    t0 = time.perf_counter()
    k_libid, piv_libid, T_libid = id_sketch(A, rtol_or_rank)
    t_libid = time.perf_counter() - t0

    if T_libid.size > 0:
        A_skel = A[:, piv_libid[k_libid:]]
        A_basis = A[:, piv_libid[:k_libid]]
        A_approx = A_basis @ T_libid
        err_libid = np.linalg.norm(A_skel - A_approx) / np.linalg.norm(A)
        max_T_libid = np.max(np.abs(T_libid))
    else:
        err_libid = 0.0
        max_T_libid = 0.0

    print(f"libid (random):  k={k_libid:4d}, err={err_libid:.3e}, "
          f"max|T|={max_T_libid:.3e}, time={t_libid:.4f}s")

    speedup = t_rrqr / t_libid if t_libid > 0 else float('inf')
    print(f"\nSpeedup: {speedup:.2f}x (libid vs RRQR)")


if __name__ == "__main__":
    """Quick test of RRQR-based ID"""
    print("Testing deterministic RRQR-based ID\n")

    # Test 1: Small low-rank matrix
    print("\n" + "="*60)
    print("Test 1: Low-rank matrix (400x250, rank~15)")
    print("="*60)
    U = np.random.randn(400, 15)
    V = np.random.randn(250, 15)
    A1 = U @ V.T + 1e-10 * np.random.randn(400, 250)
    compare_methods_on_matrix(A1, 1e-8)

    # Test 2: Hilbert matrix (ill-conditioned)
    print("\n" + "="*60)
    print("Test 2: Hilbert matrix (200x150, rank=15)")
    print("="*60)
    from libid import _hilb
    A2 = _hilb(200, 150)
    compare_methods_on_matrix(A2, 15)

    # Test 3: Full-rank scenario
    print("\n" + "="*60)
    print("Test 3: Decaying spectrum (400x300, full-rank)")
    print("="*60)
    from scipy import linalg
    A3 = np.random.randn(400, 300)
    U3, _, V3 = linalg.svd(A3, full_matrices=False)
    s3 = 1.0 / np.arange(1, 301)
    A3 = U3 @ np.diag(s3) @ V3
    compare_methods_on_matrix(A3, 1e-3)

    print("\n" + "="*60)
    print("Key Observation:")
    print("RRQR is deterministic but must compute full QR.")
    print("libid is randomized but can adapt block size.")
    print("For full-rank problems, libid's early detection wins!")
    print("="*60)
