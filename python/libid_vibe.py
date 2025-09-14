"""
Randomized linear-algebra utilities.

Provides functions to compute a randomized orthonormal basis, a
rank-revealing QR factorization, a truncated SVD and an interpolative
decomposition (ID) of a matrix `A`.

Author: Your Name
SPDX-License-Identifier: TBD
"""

"""

gpt-oss:120b assisted summarization of libid.py, refactored

**Summary**

The module implements several randomized linear-algebra routines that
approximate the rank, QR factorization, singular-value decomposition
(SVD), and interpolative decomposition (ID) of a matrix `A`.

* `orth_randomized` builds an orthonormal basis for the column space
  of `A` using random sampling. It repeatedly draws a random test
  matrix, optionally applies a power iteration (`flag_power`), and
  checks a relative-error tolerance (`rtol`). The routine returns the
  size of the basis and the basis matrix `Q`.

* `rrqr_randomized` uses the basis from `range_randomized` to compute
  a rank-revealing QR factorization. If the full rank is needed, it
  falls back to a deterministic QR. It returns the leading `k` columns
  of `Q`, the leading `k` rows of `R`, and the pivot permutation
  vector `p`.

* `rrsvd_randomized` builds on the same basis to obtain a truncated
  SVD. When the matrix is effectively full rank it computes the full
  SVD directly. It returns the leading `k` left singular vectors,
  singular values, and right singular vectors.

* `rrid_randomized` forms an interpolative decomposition by solving a
  triangular system that extracts the interpolation matrix from the QR
  factors produced by `rrqr_randomized`. It returns the numerical rank
  `k`, the pivot vector `p`, and the interpolation matrix `proj`.

All functions accept the same optional arguments:

* `rtol` - relative tolerance that controls the truncation level;
* `block_size` - initial number of random vectors (default 42);
* `flag_power` - number of power-iteration steps (default 0).

The implementation relies on NumPy and SciPy, and it avoids any
non-ASCII symbols or special dash characters.

All three algorithms share the same workflow:

* A cheap random sketch of the column space of the input matrix `A` is built by multiplying `A` with a random matrix.
* Optional power iterations (`flag_power`) improve the sketch when the singular values decay slowly.
* The sketch is orthogonalized with a QR factorization.
* The original matrix is projected onto the sketch (`Aproj = Q.T @ A`).
* A standard deterministic factorization (QR, SVD, or QR again for ID) is performed on the tiny projected matrix.
* The result is lifted back to the original space (`Q = Q @ Qproj` for QR/SVD).

The numerical rank `k` is chosen automatically: a row (or singular
value) is kept if its norm (or absolute value) is at least `rtol *
||A||` (or `rtol * ||Aproj||`).

If the random sketch grows to the full size of the matrix, the
functions fall back to a deterministic factorization of the whole
matrix, guaranteeing correct results for small or effectively
full-rank problems.

---

**Converted Python code (American English, ASCII only)**

"""


import numpy as np
from numpy.linalg import norm
from scipy import linalg


def _power_iteration(A: np.ndarray, X: np.ndarray, flag_power: int = 0) -> np.ndarray:
    """
    Apply subspace power iteration to improve the quality of a random basis.

    Parameters
    ----------
    A : ndarray
        Input matrix.
    X : ndarray
        Random test matrix (size n x block_size).
    flag_power : int, optional
        Number of power-iteration steps (default is 0).

    Returns
    -------
    ndarray
        Updated test matrix after flag_power iterations.
    """
    for _ in range(flag_power):
        X = A.T @ (A @ X)
        X, _R, _p = linalg.qr(X, mode='economic', pivoting=True)
    return X


def orth_randomized(
    A: np.ndarray,
    rtol: float,
    block_size: int = 42,
    flag_power: int = 0,
) -> tuple[int, np.ndarray]:
    """
    Construct an orthonormal basis for the dominant column space of `A`.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix.
    rtol : float
        Relative tolerance that determines when the basis is sufficient.
    block_size : int, optional
        Initial number of random test vectors (default 42).
    flag_power : int, optional
        Number of power-iteration steps (default 0).

    Returns
    -------
    k : int
        Number of basis vectors returned (may be smaller than `block_size`).
    Q : ndarray, shape (m, k)
        Orthonormal basis matrix.
    """
    m, n = A.shape

    # If the requested block size already spans the whole matrix, quit early.
    if block_size >= min(m, n):
        return min(m, n), np.empty((m, 0), dtype=A.dtype)

    while True:
        # Random test matrix with entries in [-1, 1].
        X = 2.0 * np.random.rand(n, block_size) - 1.0
        X = _power_iteration(A, X, flag_power)

        Y = A @ X
        Q, R, piv = linalg.qr(Y, mode="economic", pivoting=True)

        # Use the last diagonal entry of R as a proxy for the residual.
        r_diag = R.diagonal()
        residual = max(abs(r_diag[-1:])) / max(norm(Y, axis=0))

        if residual <= rtol:
            return block_size, Q

        # If residual is too large, increase the block size.
        block_size = min(block_size * 4, min(m, n))

        if block_size >= min(m, n):
            return min(m, n), np.empty((0, 0), dtype=A.dtype)


def rrqr_randomized(
    A: np.ndarray,
    rtol: float,
    block_size: int = 42,
    flag_power: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rank-revealing QR factorization using a randomized basis.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix.
    rtol : float
        Relative tolerance for truncation.
    block_size : int, optional
        Initial size of the random test matrix.
    flag_power : int, optional
        Number of power-iteration steps.

    Returns
    -------
    Q : ndarray, shape (m, k)
        Orthonormal matrix.
    R : ndarray, shape (k, n)
        Upper-triangular factor.
    p : ndarray, shape (n,)
        Pivot indices such that `A[:, p] = Q @ R`.
    """
    m, n = A.shape
    k, Q_rand = orth_randomized(A, rtol, block_size, flag_power)

    if k >= min(m, n):
        # Full QR as a fallback.
        Q, R, p = linalg.qr(A, mode="economic", pivoting=True)
        keep = np.sum(norm(R, axis=1) >= rtol * norm(A))
        return Q[:, :keep], R[:keep, :], p

    # Project A onto the subspace spanned by Q_rand.
    A_proj = Q_rand.T @ A
    Q_proj, R, p = linalg.qr(A_proj, mode="economic", pivoting=True)
    Q = Q_rand @ Q_proj
    keep = np.sum(norm(R, axis=1) >= rtol * norm(A_proj))
    return Q[:, :keep], R[:keep, :], p


def rrsvd_randomized(
    A: np.ndarray,
    rtol: float,
    block_size: int = 42,
    flag_power: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncated singular-value decomposition using a randomized basis.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix.
    rtol : float
        Relative tolerance for truncation.
    block_size : int, optional
        Initial size of the random test matrix.
    flag_power : int, optional
        Number of power-iteration steps.

    Returns
    -------
    U : ndarray, shape (m, k)
        Left singular vectors.
    s : ndarray, shape (k,)
        Singular values.
    Vt : ndarray, shape (k, n)
        Right singular vectors transposed.
    """
    m, n = A.shape
    k, Q_rand = orth_randomized(A, rtol, block_size, flag_power)

    if k >= min(m, n):
        U, s, Vt = linalg.svd(A, full_matrices=False)
        keep = np.sum(np.abs(s) >= rtol * norm(A))
        return U[:, :keep], s[:keep], Vt[:keep, :]

    # Project and compute SVD in the reduced space.
    A_proj = Q_rand.T @ A
    U_proj, s, Vt = linalg.svd(A_proj, full_matrices=False)
    U = Q_rand @ U_proj
    keep = np.sum(np.abs(s) >= rtol * norm(A_proj))
    return U[:, :keep], s[:keep], Vt[:keep, :]


def rrid_randomized(
    A: np.ndarray,
    rtol: float,
    block_size: int = 42,
    flag_power: int = 0,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Interpolative decomposition using a randomized QR factorization.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix.
    rtol : float
        Relative tolerance for truncation.
    block_size : int, optional
        Initial size of the random test matrix.
    flag_power : int, optional
        Number of power-iteration steps.

    Returns
    -------
    k : int
        Numerical rank (size of the truncated factor).
    piv : ndarray, shape (n,)
        Pivot indices.
    T : ndarray, shape (k, n-k)
        Interpolation matrix such that `A ~ A[:, piv[:k]] @ T`.
    """
    Q, R, piv = rrqr_randomized(A, rtol, block_size, flag_power)
    k = R.shape[0]
    # Solve R11 * T = R12 for T.
    T = linalg.solve_triangular(R[:k, :k], R[:k, k:], lower=False)
    return k, piv, T


def _hilb(n: int, m: int) -> np.ndarray:
    """
    Return an n-by-m Hilbert matrix.

    Parameters
    ----------
    n : int
        Number of rows.
    m : int
        Number of columns.

    Returns
    -------
    ndarray
        The Hilbert matrix.
    """
    # Build first column and last row for the Hankel representation.
    c = 1.0 / (np.arange(n) + 1)          # Adjust for 0-based indexing
    r = 1.0 / (np.arange(m) + n)          # Adjust for 0-based indexing
    return linalg.hankel(c, r)


if __name__ == "__main__":
    # Simple sanity checks for the public API.
    np.random.seed(0)

    # Small test matrix (4000 x 2000 Hilbert matrix).
    m, n = 4000, 2000
    A = _hilb(m, n)

    # ------------------------------------------------------------------
    # Test orth_randomized
    # ------------------------------------------------------------------
    k_range, Q_range = orth_randomized(A, rtol=1e-12)
    orth_err = np.linalg.norm(Q_range.T @ Q_range - np.eye(k_range))
    print(f"orth_randomized: k={k_range}, basis shape={Q_range.shape}")
    print(f"orth_randomized: orthonormality error={orth_err:e}")

    # ------------------------------------------------------------------
    # Test rrqr_randomized
    # ------------------------------------------------------------------
    Q_rrqr, R_rrqr, piv = rrqr_randomized(A, rtol=1e-12)
    A_perm = A[:, piv]
    recon_err = np.linalg.norm(Q_rrqr @ R_rrqr - A_perm) / np.linalg.norm(A_perm)
    print(f"rrqr_randomized: reconstruction relative error={recon_err:e}")

    # ------------------------------------------------------------------
    # Test rrsvd_randomized
    # ------------------------------------------------------------------
    U_rrsvd, s_rrsvd, Vt_rrsvd = rrsvd_randomized(A, rtol=1e-12)
    A_svd = U_rrsvd @ np.diag(s_rrsvd) @ Vt_rrsvd
    svd_err = np.linalg.norm(A_svd - A) / np.linalg.norm(A)
    print(f"rrsvd_randomized: reconstruction relative error={svd_err:e}")

    # ------------------------------------------------------------------
    # Test rrid_randomized
    # ------------------------------------------------------------------
    k_id, piv_id, T_id = rrid_randomized(A, rtol=1e-12)
    A_id_approx = A[:, piv_id[:k_id]] @ T_id
    id_err = np.linalg.norm(A[:, piv_id[k_id:]] - A_id_approx) / np.linalg.norm(A)
    print(f"rrid_randomized: interpolation relative error={id_err:e}")
