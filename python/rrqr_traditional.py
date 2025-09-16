"""
traditional_rrqr.py

Rank-revealing QR factorisation (QR with column pivoting) implemented
with the classic Householder-reflection algorithm.

References
----------
[1] P. A. Businger and G. H. Golub,
    "Linear least squares solution by Householder transformations,"
    Numer. Math. 7, 269-276 (1965)

Author: gpt-oss-120b
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


def traditional_rrqr(
    A: np.ndarray,
    rtol: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Perform a QR factorisation with column pivoting using the
    Businger-Golub Householder algorithm.

    --------------------------------------------------------------------
    Inputs
    --------------------------------------------------------------------
    A : ndarray, shape (m, n)
        The matrix to be factorised. It may contain real or complex numbers
        and can be rectangular (any m >= 0, n >= 0). The routine works for
        tall matrices (m > n), wide matrices (m < n), and square matrices
        alike.

    rtol : float, optional
        Relative tolerance used to decide the numerical rank of A.
        If omitted the function uses a MATLAB-compatible default::

            rtol = max(m, n) * eps * max(|diag(R)|)

        where eps is the machine epsilon for the data type of A. A larger
        rtol yields a smaller estimated rank, while a smaller rtol yields a
        larger rank. The tolerance is applied to the absolute values of the
        diagonal entries of R after the factorisation is complete.

    --------------------------------------------------------------------
    Returns
    --------------------------------------------------------------------
    Q : ndarray, shape (m, m)
        Orthogonal (or unitary) matrix. The product Q @ R reproduces the
        permuted input matrix A[:, perm] up to round-off.

    R : ndarray, shape (m, n)
        Upper-triangular factor. Entries below the main diagonal are
        numerically zero.

    perm : ndarray, shape (n,)
        Permutation vector (zero-based) that records the column swaps
        performed during pivoting. The relationship

            A[:, perm] == Q @ R

        holds (subject to floating-point error).

    rank : int
        Estimated numerical rank of A based on the supplied tolerance.
    """
    # -----------------------------------------------------------------
    # 1. Validate and initialise
    # -----------------------------------------------------------------
    if A.ndim != 2:
        raise ValueError("Input matrix A must be two-dimensional.")
    m, n = A.shape

    # Work in complex arithmetic so the same code handles real & complex data.
    A = A.astype(np.complex128, copy=False)

    Q = np.eye(m, dtype=A.dtype)   # will accumulate Householder reflectors
    R = A.copy()                   # overwritten in-place
    perm = np.arange(n)            # current column ordering

    # Column 2-norms – used for pivot selection.
    col_norms = np.linalg.norm(R, axis=0)

    # -----------------------------------------------------------------
    # 2. Main loop (k = 0 … min(m,n)-1)
    # -----------------------------------------------------------------
    for k in range(min(m, n)):
        # -------------------------------------------------------------
        # 2a. Pivot selection (largest remaining column norm)
        # -------------------------------------------------------------
        max_idx = k + np.argmax(col_norms[k:])
        if max_idx != k:
            R[:, [k, max_idx]] = R[:, [max_idx, k]]
            perm[[k, max_idx]] = perm[[max_idx, k]]
            col_norms[[k, max_idx]] = col_norms[[max_idx, k]]

        # -------------------------------------------------------------
        # 2b. Build Householder reflector for column k
        # -------------------------------------------------------------
        x = R[k:, k]
        sigma = np.linalg.norm(x)
        if sigma == 0.0:
            continue

        # Sign choice follows MATLAB's convention.
        sign = 1.0 if np.real(x[0]) >= 0 else -1.0
        v = x.copy()
        v[0] = v[0] + sign * sigma
        v = v / np.linalg.norm(v)

        # -------------------------------------------------------------
        # 2c. Apply reflector to trailing submatrix of R
        # -------------------------------------------------------------
        R[k:, k:] -= 2.0 * np.outer(v, np.dot(v.conj().T, R[k:, k:]))

        # -------------------------------------------------------------
        # 2d. Accumulate reflector into Q
        # -------------------------------------------------------------
        Q[:, k:] -= 2.0 * np.outer(np.dot(Q[:, k:], v), v.conj().T)

        # -------------------------------------------------------------
        # 2e. Update column norms for the remaining columns
        # -------------------------------------------------------------
        if k + 1 < n:
            col_norms[k + 1 :] = np.linalg.norm(R[k:, k + 1 :], axis=0)

    # -----------------------------------------------------------------
    # 3. Estimate numerical rank from the diagonal of R
    # -----------------------------------------------------------------
    diag_abs = np.abs(np.diag(R))
    if rtol is None:
        eps = np.finfo(A.dtype).eps
        rtol = max(m, n) * eps * (diag_abs.max() if diag_abs.size else 0.0)

    rank = int(np.sum(diag_abs > rtol))

    return Q, R, perm, rank


def rrqr_q(H: np.ndarray, tau: np.ndarray, k: int) -> np.ndarray:
    """
    Reconstruct the orthogonal matrix Q from the compact QR data.

    Parameters
    ----------
    H   : ndarray, shape (m, n)
          Matrix that holds the Householder vectors in its strictly
          lower-triangular part (the same H that the MATLAB routine
          returns). The upper-triangular part is not needed.

    tau : ndarray, shape (k,)
          Householder scaling factors (one per reflector).

    k   : int
          Number of reflectors to use (usually min(m,n)).

    Returns
    -------
    Q : ndarray, shape (m, k)
        Orthogonal matrix built as Q = I - V * diag(tau) * V^H,
        where V contains the Householder vectors.
    """
    m = H.shape[0]

    # 1. Build the matrix V that stores the Householder vectors.
    #    - The vectors have an implicit leading 1 on the diagonal.
    #    - The strictly lower-triangular part of H already contains the
    #      remaining entries of each vector.
    V = np.eye(m, k, dtype=H.dtype)          # start with I (implicit 1s)
    V[:m, :k] += np.tril(H[:, :k], -1)       # add the stored parts

    # 2. Apply the compact WY representation:
    #    Q = I - V * diag(tau) * V^H
    #    (tau may be a scalar or a 1-D array; we broadcast it.)
    tau_mat = np.diag(tau)                   # k×k diagonal matrix
    Q = np.eye(m, k, dtype=H.dtype) - V @ tau_mat @ V.conj().T

    return Q


# -----------------------------------------------------------------
# Simple self-test (run only when executed as a script)
# -----------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(123)

    # Example: a 5-by-8 matrix with a forced linear dependency.
    A = np.random.randn(5, 8)
    A[:, -1] = A[:, 0] + 3.0 * A[:, 1]   # make last column dependent

    import hilb as hilb
    import time

    A = hilb.hilb(400,200)

    print("shape(A):", A.shape)
    
    start_time = time.perf_counter()

    # Test the basic routine.
    Q, R, perm, rank_est = traditional_rrqr(A,1e-8)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"traditional_rrqr, elapsed time: {elapsed_time:.4f} seconds")

    print("Permutation vector (zero-based):", perm)
    print("Estimated rank:", rank_est)

    recon_err = np.linalg.norm(A[:, perm] - Q @ R)
    print("Reconstruction error  ||A[:,perm] - Q*R||_F =", recon_err)

    ortho_err = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0]))
    print("Orthogonality error  ||Q.T@Q - I||_F =", ortho_err)
