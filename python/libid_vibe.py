"""
Randomized linear-algebra utilities.

This module implements several randomized routines that approximate the
column space, QR factorization, singular-value decomposition (SVD), and
interpolative decomposition (ID) of a matrix A.

Public functions
----------------
orth_sketch        - Build an orthonormal basis for the column space.
rrqr_randomized    - Rank-revealing QR using a randomized basis.
rrsvd_randomized   - Truncated SVD using a randomized basis.
rrid_randomized    - Interpolative decomposition using randomized QR.

Author: Your Name
SPDX-License-Identifier: TBD
"""

import numpy as np
from scipy import linalg
from numpy.linalg import norm


def _power_iteration(A, X, flag_power=0):
    """
    Apply `flag_power` steps of power iteration to improve the sketch.

    Parameters
    ----------
    A : ndarray
        Input matrix.
    X : ndarray
        Random sketch matrix.
    flag_power : int, optional
        Number of power-iteration steps. Default is 0 (no iteration).

    Returns
    -------
    ndarray
        Updated sketch matrix.
    """
    for _ in range(flag_power):
        X = A.conj().T @ (A @ X)
        X, _, _ = linalg.qr(x, mode="economic", pivoting=True)
    return X


def orth_sketch(A, rtol, block_size=42, flag_power=0):
    """
    Build an orthonormal basis for the column space of A.

    The algorithm draws a random matrix, optionally improves it with
    power iteration, and then performs a QR factorization with column
    pivoting.  The process repeats with a larger sketch until the
    smallest diagonal element of R, relative to the column norms,
    falls below `rtol`.

    Parameters
    ----------
    A : ndarray
        Input matrix of shape (m, n).
    rtol : float
        Relative tolerance that controls the stopping criterion.
    block_size : int, optional
        Initial number of random vectors. Default is 42.
    flag_power : int, optional
        Number of power-iteration steps. Default is 0.

    Returns
    -------
    int
        Effective rank (number of basis vectors).
    ndarray
        Orthonormal basis matrix Q of shape (m, rank). If the rank
        equals min(m, n) an empty array with shape (0, 0) is returned.
    """
    m, n = A.shape

    if block_size >= min(m, n):
        return min(m, n), np.empty_like(A, shape=(0, 0))

    while True:
        # Random matrix with entries in [-1, 1].
        X = (2 * np.random.uniform(size=(n, block_size)) - 1).astype(A.dtype)
        X = _power_iteration(A, X, flag_power)
        y = A @ X

        # QR with column pivoting.
        Q, R, _ = linalg.qr(y, mode="economic", pivoting=True)
        r_diag = R.diagonal()
        # Smallest diagonal entry relative to the largest column norm.
        residual = max(abs(r_diag[-1:])) / max(norm(y, axis=0))

        if residual <= rtol:
            return block_size, Q

        # If not good enough, increase the sketch size.
        block_size = min(block_size * 4, min(m, n))

        if block_size >= min(m, n):
            return min(m, n), np.empty_like(A, shape=(0, 0))


def rrqr_randomized(A, rtol, block_size=42, flag_power=0):
    """
    Rank-revealing QR factorization using a randomized sketch.

    Parameters
    ----------
    A : ndarray
        Input matrix of shape (m, n).
    rtol : float
        Relative tolerance for determining numerical rank.
    block_size : int, optional
        Initial sketch size. Default is 42.
    flag_power : int, optional
        Number of power-iteration steps. Default is 0.

    Returns
    -------
    Q : ndarray
        Orthonormal matrix of shape (m, k) where k is the estimated rank.
    R : ndarray
        Upper-triangular factor of shape (k, n).
    p : ndarray
        Pivot indices such that A[:, p] \approx Q @ R.
    """
    m, n = A.shape
    k, q = orth_sketch(A, rtol, block_size, flag_power)

    if k >= min(m, n):
        Q, R, p = linalg.qr(A, mode="economic", pivoting=True)
        k = np.sum(norm(R, axis=1) >= rtol * norm(A))
        return Q[:, :k], R[:k, :], p

    # Project A onto the sketch subspace.
    Aproj = q.conj().T @ A
    Qproj, R, p = linalg.qr(Aproj, mode="economic", pivoting=True)
    Q = q @ Qproj
    k = np.sum(norm(R, axis=1) >= rtol * norm(Aproj))
    return Q[:, :k], R[:k, :], p


def rrsvd_randomized(A, rtol, block_size=42, flag_power=0):
    """
    Truncated singular-value decomposition using a randomized sketch.

    Parameters
    ----------
    A : ndarray
        Input matrix of shape (m, n).
    rtol : float
        Relative tolerance for determining the retained singular values.
    block_size : int, optional
        Initial sketch size. Default is 42.
    flag_power : int, optional
        Number of power-iteration steps. Default is 0.

    Returns
    -------
    U : ndarray
        Left singular vectors of shape (m, k).
    s : ndarray
        Singular values (length k).
    Vt : ndarray
        Right singular vectors transposed, shape (k, n).
    """
    m, n = A.shape
    k, q = orth_sketch(A, rtol, block_size, flag_power)

    if k >= min(m, n):
        U, s, Vt = linalg.svd(A, full_matrices=False)
        k = np.sum(np.abs(s) >= rtol * norm(A))
        return U[:, :k], s[:k], Vt[:k, :]

    Aproj = q.conj().T @ A
    Uproj, s, Vt = linalg.svd(Aproj, full_matrices=False)
    U = q @ Uproj
    k = np.sum(np.abs(s) >= rtol * norm(Aproj))
    return U[:, :k], s[:k], Vt[:k, :]


def rrid_randomized(A, rtol, block_size=42, flag_power=0):
    """
    Interpolative decomposition using a randomized QR factorization.

    Parameters
    ----------
    A : ndarray
        Input matrix of shape (m, n).
    rtol : float
        Relative tolerance for rank determination.
    block_size : int, optional
        Initial sketch size. Default is 42.
    flag_power : int, optional
        Number of power-iteration steps. Default is 0.

    Returns
    -------
    k : int
        Estimated rank.
    p : ndarray
        Pivot indices such that the first `k` columns of A[:, p] form a
        basis for the column space.
    T : ndarray
        Interpolation matrix satisfying
        A[:, p[k:]] \approx A[:, p[:k]] @ T.
    """
    Q, R, p = rrqr_randomized(A, rtol, block_size, flag_power)
    k = R.shape[0]
    # Solve R11 * T = R12 for T.
    T = linalg.solve(np.triu(R[:k, :k]), R[:k, k:])
    return k, p, T


def _hilb(n, m):
    """
    Create an n-by-m Hilbert matrix.

    Parameters
    ----------
    n : int
        Number of rows.
    m : int
        Number of columns.

    Returns
    -------
    ndarray
        Hilbert matrix of shape (n, m).
    """
    # Build first column and last row for the Hankel matrix.
    c = np.zeros(n)
    r = np.zeros(m)

    for i in range(n):
        c[i] = 1.0 / (i + 1)          # 0-based indexing

    for i in range(m):
        r[i] = 1.0 / (i + n)          # 0-based indexing

    return linalg.hankel(c, r)


if __name__ == "__main__":
    # Simple sanity checks for the public API.
    np.random.seed(0)

    # Small test matrix.
    m, n = 4000, 2000
    A = _hilb(m, n)

    # --------------------------------------------------------------
    # Test orth_sketch
    # --------------------------------------------------------------
    k_range, Q_range = orth_sketch(A, rtol=1e-12)
    orth_err = np.linalg.norm(Q_range.conj().T @ Q_range - np.eye(k_range))
    print(f"orth_sketch: k={k_range}, basis shape={Q_range.shape}")
    print(f"orth_sketch: orthonormality error={orth_err:e}")

    # --------------------------------------------------------------
    # Test rrqr_randomized
    # --------------------------------------------------------------
    Q_rrqr, R_rrqr, piv = rrqr_randomized(A, rtol=1e-12)
    A_perm = A[:, piv]
    recon_err = np.linalg.norm(Q_rrqr @ R_rrqr - A_perm) / np.linalg.norm(A_perm)
    print(f"rrqr_randomized: reconstruction relative error={recon_err:e}")

    # --------------------------------------------------------------
    # Test rrsvd_randomized
    # --------------------------------------------------------------
    U_rrsvd, s_rrsvd, Vt_rrsvd = rrsvd_randomized(A, rtol=1e-12)
    A_svd = U_rrsvd @ np.diag(s_rrsvd) @ Vt_rrsvd
    svd_err = np.linalg.norm(A_svd - A) / np.linalg.norm(A)
    print(f"rrsvd_randomized: reconstruction relative error={svd_err:e}")

    # --------------------------------------------------------------
    # Test rrid_randomized
    # --------------------------------------------------------------
    k_id, piv_id, T_id = rrid_randomized(A, rtol=1e-12)
    A_id_approx = A[:, piv_id[:k_id]] @ T_id
    id_err = np.linalg.norm(A[:, piv_id[k_id:]] - A_id_approx) / np.linalg.norm(A)
    print(f"rrid_randomized: interpolation relative error={id_err:e}")
