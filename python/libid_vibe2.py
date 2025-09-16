#!/usr/bin/env python3
"""
Randomized linear-algebra routines.

This module implements
    * orth_sketch – build an orthonormal basis for the column space of A
    * rrqr_randomized – rank-revealing QR
    * rrsvd_randomized – truncated SVD
    * rrid_randomized – interpolative decomposition (now using SVD to compute T)
along with a helper that constructs a Hilbert matrix.

Author: Your Name
SPDX-License-Identifier: TBD
"""

import numpy as np
from scipy import linalg
from numpy.linalg import norm

# ----------------------------------------------------------------------
#  Helper: Power iteration
# ----------------------------------------------------------------------
def _power_iteration(A: np.ndarray, x: np.ndarray, flag_power: int = 0) -> np.ndarray:
    """Apply (AH @ A) repeatedly to the random vector x."""
    for _ in range(flag_power):
        x = A.conj().T @ (A @ x)
        x, _R, _p = linalg.qr(x, mode='economic', pivoting=True)
    return x


# ----------------------------------------------------------------------
#  1.  Orthogonal sketch
# ----------------------------------------------------------------------
def orth_sketch(A: np.ndarray, rtol: float, block_size: int = 42, flag_power: int = 0) -> tuple[int, np.ndarray]:
    """
    Build an orthonormal basis for the column space of A.

    Parameters
    ----------
    A : (m x n) array
    rtol : relative tolerance – when the smallest retained singular value
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
        # Random starting matrix – cast to the dtype of A
        x = (2 * np.random.uniform(size=(n, block_size)) - 1).astype(A.dtype)
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
            return min(m, n), np.empty_like(A, shape=(0, 0))


# ----------------------------------------------------------------------
#  2.  Rank-revealing QR
# ----------------------------------------------------------------------
def rrqr_randomized(A: np.ndarray, rtol: float, block_size: int = 42, flag_power: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
#  3.  Truncated SVD
# ----------------------------------------------------------------------
def rrsvd_randomized(A: np.ndarray, rtol: float, block_size: int = 42, flag_power: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
#  4.  Interpolative decomposition (uses SVD to compute T)
# ----------------------------------------------------------------------
def rrid_randomized(A: np.ndarray, rtol: float, block_size: int = 42, flag_power: int = 0) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Interpolative decomposition using a randomized QR and SVD.

    Parameters
    ----------
    A : (m x n) array
    rtol : relative tolerance used to determine the rank.
    block_size : initial sketch size.
    flag_power : number of power iterations.

    Returns
    -------
    k : rank
    piv : pivot indices
    T : coefficient matrix such that A[:,piv] @ T ~= A
    """
    Q, R, piv = rrqr_randomized(A, rtol, block_size, flag_power)
    k = R.shape[0]

    # Build the selected columns and the remaining columns
    A1 = A[:, piv[:k]]
    A2 = A[:, piv[k:]]

    # SVD of the selected columns
    U1, s1, V1 = linalg.svd(A1, full_matrices=False)

    # Pseudoinverse of A1: V1 @ diag(1/s1) @ U1ᴴ
    eps = np.finfo(np.result_type(A1.dtype, np.float64)).eps
    inv_s1 = 1.0 / np.where(s1 > eps, s1, eps)

    T = V1.conj().T @ (np.diag(inv_s1) @ (U1.conj().T @ A2))
    return k, piv, T


# ----------------------------------------------------------------------
#  5.  Hilbert matrix generator
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
#  Test harness
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    m, n = 4000, 2000
    A = _hilb(m, n)

    # --------------------------------------------------------------
    # Test orth_sketch
    # --------------------------------------------------------------
    k_range, Q_range = orth_sketch(A, rtol=1e-12)
    orth_err = norm(Q_range.T @ Q_range - np.eye(k_range))
    print(f"orth_sketch: k={k_range}, basis shape={Q_range.shape}")
    print(f"orth_sketch: k={k_range}, orthonormality error={orth_err:e}")

    # --------------------------------------------------------------
    # Test rrqr_randomized
    # --------------------------------------------------------------
    Q_rrqr, R_rrqr, piv = rrqr_randomized(A, rtol=1e-12)
    A_perm = A[:, piv]
    recon_err = norm(Q_rrqr @ R_rrqr - A_perm) / norm(A_perm)
    print(f"rrqr_randomized: reconstruction relative error={recon_err:e}")

    # --------------------------------------------------------------
    # Test rrsvd_randomized
    # --------------------------------------------------------------
    U_rrsvd, s_rrsvd, Vt_rrsvd = rrsvd_randomized(A, rtol=1e-12)
    A_svd = U_rrsvd @ np.diag(s_rrsvd) @ Vt_rrsvd
    svd_err = norm(A_svd - A) / norm(A)
    print(f"rrsvd_randomized: reconstruction relative error={svd_err:e}")
    

    # --------------------------------------------------------------
    # Test rrid_randomized
    # --------------------------------------------------------------
    k_id, piv_id, T_id = rrid_randomized(A, rtol=1e-12)
    A_id_approx = A[:, piv_id[:k_id]] @ T_id
    id_err = norm(A[:, piv_id[k_id:]] - A_id_approx) / norm(A)
    print(f"rrid_randomized: interpolation relative error={id_err:e}")
    print(f"rrid_randomized: max(abs(T))={np.max(abs(T_id)):e}")
