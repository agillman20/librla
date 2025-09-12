import numpy as np
from scipy import linalg
from numpy.linalg import norm

"""
ollama run gpt-oss:120b "Must use American English. Avoid \
em-dashes. Avoid en-dashes. Avoid Unicode symbols. Summarize and \
convert to python this file: $(cat libid.py)" > libid_vibe.txt

**Summary**

The module implements several randomized linear-algebra routines that
approximate the rank, QR factorization, singular-value decomposition
(SVD), and interpolative decomposition (ID) of a matrix `A`.

* `range_randomized` builds an orthonormal basis for the column space
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

* `image_randomized` is a thin wrapper that calls `range_randomized`
  on the transpose of `A`, giving a basis for the row space.

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

The helper `range_randomized` returns the size of the sketch (`k`) and
the orthonormal basis `Q`. `image_randomized` does the same for the
row space by calling `range_randomized` on `A.T`.

---

**Converted Python code (American English, ASCII only)**
"""


def _power_iteration(A: np.ndarray, X: np.ndarray, power: int = 0) -> np.ndarray:
    """
    Apply power iteration to improve the quality of the sampling matrix.

    Parameters
    ----------
    A : np.ndarray
        Input matrix.
    X : np.ndarray
        Random test matrix.
    power : int, optional
        Number of power-iteration steps (default is 0).

    Returns
    -------
    np.ndarray
        Updated test matrix after power iteration.
    """
    for _ in range(power):
        X = A.T @ (A @ X)
        X, _, _ = linalg.qr(X, mode='economic', pivoting=True)
    return X


def range_randomized(
    A: np.ndarray,
    rtol: float,
    block_size: int = 42,
    flag_power: int = 0,
) -> tuple[int, np.ndarray]:
    """
    Compute an orthonormal basis for the column space of A using random sampling.

    Parameters
    ----------
    A : np.ndarray
        Input matrix.
    rtol : float
        Relative tolerance that determines when to stop sampling.
    block_size : int, optional
        Initial number of random vectors (default 42).
    flag_power : int, optional
        Number of power-iteration steps (default 0).

    Returns
    -------
    k : int
        Number of basis vectors found (may equal min(m, n)).
    Q : np.ndarray
        Orthonormal basis matrix with shape (m, k).  If k == 0 the array is empty.
    """
    m, n = A.shape

    # If the initial block already covers the whole space, return early.
    if block_size >= min(m, n):
        return min(m, n), np.empty((0, 0), dtype=A.dtype)

    while True:
        # Random matrix with entries in [-1, 1]
        X = 2 * np.random.uniform(size=(n, block_size)) - 1
        X = _power_iteration(A, X, flag_power)

        Y = A @ X
        Q, R, _ = linalg.qr(Y, mode='economic', pivoting=True)

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

    Returns the leading k columns of Q, the leading k rows of R,
    and the pivot permutation vector.
    """
    m, n = A.shape
    k, Q_basis = range_randomized(A, rtol, block_size, flag_power)

    if k >= min(m, n):
        Q, R, p = linalg.qr(A, mode='economic', pivoting=True)
        k = int(np.sum(norm(R, axis=1) >= rtol * norm(A)))
        return Q[:, :k], R[:k, :], p

    # Project A onto the basis and factor the small matrix.
    A_proj = Q_basis.T @ A
    Q_proj, R, p = linalg.qr(A_proj, mode='economic', pivoting=True)
    Q = Q_basis @ Q_proj
    k = int(np.sum(norm(R, axis=1) >= rtol * norm(A_proj)))
    return Q[:, :k], R[:k, :], p


def rrsvd_randomized(
    A: np.ndarray,
    rtol: float,
    block_size: int = 42,
    flag_power: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncated singular-value decomposition using a randomized basis.

    Returns the leading k left singular vectors, singular values,
    and right singular vectors.
    """
    m, n = A.shape
    k, Q_basis = range_randomized(A, rtol, block_size, flag_power)

    if k >= min(m, n):
        U, s, Vh = linalg.svd(A, full_matrices=False)
        k = int(np.sum(np.abs(s) >= rtol * norm(A)))
        return U[:, :k], s[:k], Vh[:k, :]

    A_proj = Q_basis.T @ A
    U_proj, s, Vh = linalg.svd(A_proj, full_matrices=False)
    U = Q_basis @ U_proj
    k = int(np.sum(np.abs(s) >= rtol * norm(A_proj)))
    return U[:, :k], s[:k], Vh[:k, :]


def rrid_randomized(
    A: np.ndarray,
    rtol: float,
    block_size: int = 42,
    flag_power: int = 0,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Interpolative decomposition using a randomized QR factorization.

    Returns the numerical rank, the pivot permutation vector,
    and the interpolation matrix.
    """
    Q, R, p = rrqr_randomized(A, rtol, block_size, flag_power)
    k = R.shape[0]

    # Solve R11 * X = R12 for X, where R = [R11 R12].
    R11 = np.triu(R[:k, :k])
    R12 = R[:k, k:]
    proj = linalg.solve(R11, R12)
    return k, p, proj


def image_randomized(
    A: np.ndarray,
    rtol: float,
    block_size: int = 42,
    flag_power: int = 0,
) -> tuple[int, np.ndarray]:
    """
    Compute a basis for the row space of A by applying range_randomized
    to the transpose of A.
    """
    return range_randomized(A.T, rtol, block_size, flag_power)
