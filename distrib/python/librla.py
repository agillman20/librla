"""
Randomized linear-algebra routines
==================================

Randomized algorithms for low-rank matrix approximations:
    orth_sketch  - Approximate orthonormal basis for column space
    qr_sketch    - Truncated QR factorization with column pivoting
    svd_sketch   - Truncated singular value decomposition (SVD)
    id_sketch    - Interpolative decomposition (ID)

Deterministic:
    id_qrpiv     - Interpolative decomposition via QR with pivoting

Usage::

    Q, flag, diagR = orth_sketch(A, rtol)
    Q, R, p = qr_sketch(A, rtol)
    U, s, Vh = svd_sketch(A, rtol)
    k, piv, T = id_sketch(A, rtol)
    # tolerance mode: rtol < 1, rank mode: rtol >= 1

Matrix-free operators:
    Use scipy.sparse.linalg.LinearOperator for matrix-free operators::

        from scipy.sparse.linalg import LinearOperator
        A = LinearOperator((m, n), matvec=matvec_fun, rmatvec=rmatvec_fun)
        U, s, Vh = svd_sketch(A, rank)  # rank mode only: rtol >= 1

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.1
Date: April 22, 2026
Assisted by: Claude Code (Anthropic)
"""

import numpy as np
from scipy import linalg
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator

# --------------------------------------------------------------
# 1. Orthogonal sketch
# --------------------------------------------------------------

def orth_sketch(A, rtol, *, block_size=42, power_iter=0, rng=None):
    """Approximate orthonormal basis for column space using randomized sketching.

    This function uses random test matrix multiplication (A @ Omega
    where Omega has i.i.d. uniform[-1,1] entries) followed by QR
    factorization to approximate the range of A. The approach is
    particularly efficient for matrices with rapidly decaying singular
    values.

    The algorithm has two modes:
    - Tolerance mode (rtol < 1): Adaptively grows the sketch size until the
      smallest column norm falls below rtol times the largest norm
    - Rank mode (rtol >= 1): Performs a single sketch and returns the
      requested number of columns (rtol interpreted as target rank)

    Parameters
    ----------
    A : ndarray or LinearOperator
        Input matrix (m, n) or linear operator
    rtol : float
        Relative tolerance (< 1) or target rank (>= 1)
    block_size : int, optional
        Initial number of random test vectors (default: 42)
    power_iter : int, optional
        Number of power iterations to improve accuracy (default: 0).
        Setting power_iter=1 or 2 can improve results for
        matrices with slowly decaying singular values.
    rng : Generator, optional
        Random number generator (default: None uses numpy default)

    Returns
    -------
    Q : ndarray, shape (m, k)
        Orthonormal matrix spanning approximate range of A
    flag : int
        Exit status:
        - 0: Success, Q contains valid orthonormal basis
        - 1: Early termination (tolerance mode only). Occurs when:
          (a) rtol < machine epsilon (tolerance too tight), or
          (b) sketch size grew to min(m,n) without meeting tolerance,
              indicating matrix is effectively full-rank at this tolerance
          When flag=1, Q is empty (m×0).
    diagR : ndarray
        Diagonal elements from pivoted QR factorization, representing
        column norms of the sketched matrix (sorted in decreasing order)

    Note
    ----
    Higher-level functions (qr_sketch, svd_sketch, id_sketch) automatically
    fall back to deterministic (full) QR or SVD when orth_sketch terminates
    early, so users of those functions do not need to handle flag=1 explicitly.
    """
    m, n = A.shape
    dtype = _get_dtype(A)

    # Rank mode (rtol >= 1): single sketch with rank filtering
    if rtol >= 1:
        kmax = int(np.floor(rtol))
        x = _uniform_omega(A, n, block_size, rng=rng)
        x = _power_iteration(A, x, power_iter=power_iter)
        y = _matvec(A, x)
        Q, R, _ = linalg.qr(y, mode='economic', pivoting=True)

        # Use requested rank directly (capped at available columns)
        diagR = np.abs(np.diag(R))
        rank = min(kmax, Q.shape[1])

        return Q[:, :rank], 0, diagR

    # Tolerance mode (rtol < 1): geometric growth with tolerance checking
    if rtol < np.finfo(dtype).eps:
        return np.empty((m, 0), dtype=dtype), 1, np.array([], dtype=dtype)

    if block_size >= min(m, n):
        return np.empty((m, 0), dtype=dtype), 1, np.array([], dtype=dtype)

    while True:
        x = _uniform_omega(A, n, block_size, rng=rng)
        x = _power_iteration(A, x, power_iter=power_iter)
        y = _matvec(A, x)
        Q, R, _ = linalg.qr(y, mode='economic', pivoting=True)

        # Tolerance check (cross-multiplied form of diagR[-1]/diagR[0] <= rtol,
        # avoiding the division; diagR is sorted decreasing so diagR[0] is the max)
        diagR = np.abs(R.diagonal())
        if diagR.size == 0 or diagR[-1] <= rtol * diagR[0]:
            return Q, 0, diagR

        block_size = min(block_size * 4, min(m, n))
        if block_size >= min(m, n):
            return np.empty((m, 0), dtype=dtype), 1, np.array([], dtype=dtype)


# --------------------------------------------------------------
# 2. Truncated QR with column pivoting
# --------------------------------------------------------------

def qr_sketch(A, rtol, *, block_size=42, power_iter=0, extra_samples=12, rng=None):
    """Compute truncated QR factorization with column pivoting via randomized sketching.

    The algorithm sketches an orthonormal basis for the column space
    of A, projects A onto this basis, computes the QR of the smaller
    projected matrix, and then expands back to the original space.  If
    the matrix is effectively full rank a deterministic QR is
    performed.

    This is much faster than full QR for matrices where the target
    rank k is much smaller than min(m,n).

    Parameters
    ----------
    A : ndarray or LinearOperator
        Input matrix (m, n) or linear operator
    rtol : float
        Relative tolerance (< 1) or target rank (>= 1)
        - Tolerance mode: keep columns with norm >= rtol * max_norm
        - Rank mode: return k leading columns
    block_size : int, optional
        Sketch size for tolerance mode (default: 42)
    power_iter : int, optional
        Number of power iterations for accuracy (default: 0)
    extra_samples : int, optional
        Oversampling for rank mode (default: 12).
        Rank mode uses block_size = rank + extra_samples
    rng : Generator, optional
        Random number generator (default: None uses numpy default)

    Returns
    -------
    Q : ndarray, shape (m, k)
        Orthonormal matrix, k <= min(m, n)
    R : ndarray, shape (k, n)
        Upper triangular matrix
    p : ndarray, shape (n,)
        Column permutation (0-based indexing).
        The decomposition satisfies A[:, p] ≈ Q @ R
    """
    m, n = A.shape
    dtype = _get_dtype(A)
    is_matrix_free = _is_matrix_free_linop(A)
    is_linop = _is_linop(A)

    # Rank mode vs tolerance mode
    rank_mode = False
    if rtol >= 1:
        rank_mode = True
        kmax = int(np.floor(rtol))
        block_size = kmax + extra_samples
    elif is_matrix_free:
        raise ValueError(
            "Matrix-free LinearOperator only supported in rank mode (rtol >= 1). "
            f"Got rtol={rtol}. Please specify target rank as rtol."
        )

    # Compute sketch: in rank mode, request all oversampled columns
    # for better accuracy (truncate to kmax after QR)
    orth_rtol = block_size if rank_mode else rtol
    Qs, flag, _ = orth_sketch(A, orth_rtol, block_size=block_size, power_iter=power_iter, rng=rng)

    k = Qs.shape[1] if flag == 0 else min(m, n)

    # Fallback to full QR if needed
    needs_fallback = (flag != 0 or k >= min(m, n))
    if needs_fallback and rank_mode:
        needs_fallback = False

    if needs_fallback:
        A_mat = _get_matrix(A)
        Q, R, p = linalg.qr(A_mat, mode='economic', pivoting=True)

        # Determine rank
        if rank_mode:
            rank = min(kmax, Q.shape[1])
        else:
            rank = _rank_from_diag(np.diag(R), rtol)

        return Q[:, :rank], R[:rank, :], p

    # Project and compute QR
    B = _matmat_left(Qs, A)
    Qproj, R, p = linalg.qr(B, mode='economic', pivoting=True)
    Q = Qs @ Qproj

    # Determine rank
    if rank_mode:
        rank = min(kmax, Q.shape[1])
    else:
        rank = _rank_from_diag(np.diag(R), rtol)

    return Q[:, :rank], R[:rank, :], p


# --------------------------------------------------------------
# 3. Truncated SVD
# --------------------------------------------------------------

def svd_sketch(A, rtol, *, block_size=42, power_iter=0, extra_samples=12, rng=None):
    """Compute truncated singular value decomposition (SVD) via randomized sketching.

    The algorithm sketches an orthonormal basis for the column space
    of A, projects A onto this basis, computes the SVD of the smaller
    projected matrix, and then expands back to the original space.  If
    the matrix is effectively full rank a deterministic SVD is
    performed.

    This is much faster than full SVD for matrices where the target
    rank k is much smaller than min(m,n).

    Parameters
    ----------
    A : ndarray or LinearOperator
        Input matrix (m, n) or linear operator
    rtol : float
        Relative tolerance (< 1) or target rank (>= 1)
        - Tolerance mode: keep singular values >= rtol * s[0]
        - Rank mode: return k leading singular triplets
    block_size : int, optional
        Sketch size for tolerance mode (default: 42)
    power_iter : int, optional
        Number of power iterations for accuracy (default: 0)
    extra_samples : int, optional
        Oversampling for rank mode (default: 12)
    rng : Generator, optional
        Random number generator (default: None uses numpy default)

    Returns
    -------
    U : ndarray, shape (m, k)
        Left singular vectors, orthonormal columns
    s : ndarray, shape (k,)
        Singular values, sorted descending
    Vh : ndarray, shape (k, n)
        Right singular vectors (conjugate transpose), orthonormal rows
        The decomposition satisfies A ≈ U @ np.diag(s) @ Vh
    """
    m, n = A.shape
    dtype = _get_dtype(A)
    is_matrix_free = _is_matrix_free_linop(A)
    is_linop = _is_linop(A)

    if m < n:
        A_T = _transpose_linop(A)
        V, s, U = svd_sketch(A_T, rtol, block_size=block_size, power_iter=power_iter, extra_samples=extra_samples, rng=rng)
        return U.conj().T, s, V.conj().T

    # Rank mode vs tolerance mode
    rank_mode = False
    if rtol >= 1:
        rank_mode = True
        kmax = int(np.floor(rtol))
        block_size = kmax + extra_samples
    elif is_matrix_free:
        raise ValueError(
            "Matrix-free LinearOperator only supported in rank mode (rtol >= 1). "
            f"Got rtol={rtol}. Please specify target rank as rtol."
        )

    # Compute sketch: in rank mode, request all oversampled columns
    # to get more accurate singular values (truncate to kmax after SVD)
    orth_rtol = block_size if rank_mode else rtol
    Qs, flag, _ = orth_sketch(A, orth_rtol, block_size=block_size, power_iter=power_iter, rng=rng)

    k = Qs.shape[1] if flag == 0 else min(m, n)

    # Fallback to full SVD if needed
    needs_fallback = (flag != 0 or k >= min(m, n))
    if needs_fallback and rank_mode:
        needs_fallback = False

    if needs_fallback:
        A_mat = _get_matrix(A)
        U, s, V = linalg.svd(A_mat, full_matrices=False)

        # Determine rank
        if rank_mode:
            rank = min(kmax, len(s))
        else:
            rank = _rank_from_svals(s, rtol)

        return U[:, :rank], s[:rank], V[:rank, :]

    # Project and compute SVD
    Aproj = _matmat_left(Qs, A)
    Uproj, s, V = linalg.svd(Aproj, full_matrices=False)
    U = Qs @ Uproj

    # Determine rank
    if rank_mode:
        rank = min(kmax, len(s))
    else:
        rank = _rank_from_svals(s, rtol)

    return U[:, :rank], s[:rank], V[:rank, :]


# --------------------------------------------------------------
# 4. Interpolative decomposition (ID) - randomized
# --------------------------------------------------------------

def id_sketch(A, rtol, *, block_size=42, power_iter=0, extra_samples=12, method='fast', rng=None):
    """Compute interpolative decomposition (ID) via randomized sketching.

    An ID represents a matrix A by selecting k of its columns and expressing
    the remaining columns as linear combinations of the selected ones:

        A[:, piv[k:]] ≈ A[:, piv[:k]] @ T

    where piv is a column permutation and T is a k×(n-k) interpolation matrix.
    The selected columns (skeleton) capture the essential features of A, while
    T provides the coefficients to reconstruct the other columns.

    This function uses qr_sketch() to identify the column permutation.

    Parameters
    ----------
    A : ndarray or LinearOperator
        Input matrix (m, n) or linear operator
    rtol : float
        Relative tolerance (< 1) or target rank (>= 1)
    block_size : int, optional
        Sketch size for tolerance mode (default: 42)
    power_iter : int, optional
        Number of power iterations for accuracy (default: 0)
    extra_samples : int, optional
        Oversampling for rank mode (default: 12)
    method : str, optional
        Method for computing T matrix (default: 'fast')
        - 'fast': Triangular solve R11 \\ R12 (fastest)
        - 'svd': SVD-based pseudoinverse
        - 'lstsq': Least-squares from original A (most accurate, slowest)
    rng : Generator, optional
        Random number generator (default: None uses numpy default)

    Returns
    -------
    k : int
        Rank of the approximation (number of skeleton columns)
    piv : ndarray, shape (n,)
        Column permutation (0-based indexing)
        - piv[:k] are indices of skeleton columns
        - piv[k:] are indices of interpolated columns
    T : ndarray, shape (k, n-k)
        Interpolation matrix
        The approximation is A[:, piv[k:]] ≈ A[:, piv[:k]] @ T
    """
    valid_methods = {'fast', 'svd', 'lstsq'}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    _, R, jpiv = qr_sketch(A, rtol, block_size=block_size, power_iter=power_iter, extra_samples=extra_samples, rng=rng)

    k = R.shape[0]
    piv = jpiv

    # Compute rtol for SVD filtering
    m, n = A.shape
    dtype = A.dtype if hasattr(A, 'dtype') else np.float64
    if rtol >= 1:
        # Rank mode: minimal filtering (only exact zeros)
        rtol_for_svd = 0
    else:
        # Tolerance mode: use the provided tolerance
        rtol_for_svd = rtol

    # Dispatch to shared helper functions
    if method == 'lstsq':
        T = _compute_T_lstsq(A, R, piv, k)
    elif method == 'svd':
        T = _compute_T_svd(R, k, rtol_for_svd)
    elif method == 'fast':
        T = _compute_T_fast(R, k, rtol_for_svd)

    return k, piv, T


# --------------------------------------------------------------
# 5. Interpolative decomposition (ID) - deterministic
# --------------------------------------------------------------

def id_qrpiv(A, rtol, *, method='fast'):
    """Interpolative decomposition via deterministic QR with column pivoting.

    An ID represents a matrix A by selecting k of its columns and expressing
    the remaining columns as linear combinations of the selected ones:

        A[:, piv[k:]] ≈ A[:, piv[:k]] @ T

    where piv is a column permutation and T is a k×(n-k) interpolation matrix.
    The selected columns (skeleton) capture the essential features of A, while
    T provides the coefficients to reconstruct the other columns.

    This function provides a deterministic alternative to id_sketch by
    computing the interpolative decomposition using only QR with column
    pivoting (LAPACK geqp3), without any randomized sketching. It preserves
    LinearOperator support and uses the same T matrix computation logic as
    id_sketch.

    Parameters
    ----------
    A : ndarray or LinearOperator
        Input matrix or operator
    rtol : float
        Relative tolerance (rtol < 1) or target rank (rtol >= 1)
    method : str, optional
        Method for computing T matrix (default: 'fast')
        - 'fast': Triangular solve R11 \\ R12 (fastest)
        - 'svd': SVD-based pseudoinverse
        - 'lstsq': Least-squares from original A (most accurate, slowest)

    Returns
    -------
    k : int
        Rank of the ID approximation
    piv : ndarray, shape (n,)
        Column permutation
    T : ndarray, shape (k, n-k)
        Interpolation matrix
    """
    valid_methods = {'fast', 'svd', 'lstsq'}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    is_linop = _is_linop(A)
    is_matrix_free = _is_matrix_free_linop(A)

    m, n = A.shape
    dtype = A.dtype if hasattr(A, 'dtype') else np.float64

    # Determine rank mode vs tolerance mode
    rank_mode = False
    if rtol >= 1:
        rank_mode = True
        kmax = int(np.floor(rtol))

    # Compute full QR with pivoting (deterministic)
    A_mat = _get_matrix(A)
    Q, R, jpiv = linalg.qr(A_mat, mode='economic', pivoting=True)

    # Determine rank
    if rank_mode:
        rank = min(kmax, min(m, n))
        rtol_for_svd = 0  # Minimal filtering in rank mode
    else:
        rank = _rank_from_diag(np.diag(R), rtol)
        rtol_for_svd = rtol

    k = rank
    piv = jpiv

    # Handle edge cases
    if k == 0:
        return 0, piv, np.zeros((0, n), dtype=dtype)

    if k == n:
        return k, piv, np.zeros((k, 0), dtype=dtype)

    # Dispatch to shared helper functions
    if method == 'lstsq':
        T = _compute_T_lstsq(A, R, piv, k)
    elif method == 'svd':
        T = _compute_T_svd(R, k, rtol_for_svd)
    elif method == 'fast':
        T = _compute_T_fast(R, k, rtol_for_svd)

    return k, piv, T


# ==============================================================
# Private helper functions
# ==============================================================

# --------------------------------------------------------------
# LinearOperator detection and utilities
# --------------------------------------------------------------
def _is_linop(A):
    """Check if A is a LinearOperator (not just a dense array)."""
    return isinstance(A, LinearOperator) and not isinstance(A, np.ndarray)

def _is_matrix_free_linop(A):
    """Check if A is a matrix-free LinearOperator (not backed by explicit matrix)."""
    if not _is_linop(A):
        return False
    # LinearOperators with .A attribute have explicit matrix (scipy's MatrixLinearOperator or custom)
    if hasattr(A, 'A') and A.A is not None:
        return False
    # Any other LinearOperator is assumed matrix-free
    else:
        return True

def _get_dtype(A):
    """Get dtype from either ndarray or LinearOperator."""
    if isinstance(A, np.ndarray):
        return A.dtype
    elif hasattr(A, 'dtype') and A.dtype is not None:
        return A.dtype
    elif _is_linop(A) and hasattr(A, 'A') and A.A is not None:
        return A.A.dtype
    else:
        raise ValueError(
            "Unable to determine dtype. For matrix-free LinearOperators, "
            "dtype must be specified during creation."
        )

def _is_complex(A):
    """Check if A represents complex data."""
    dtype = _get_dtype(A)
    return np.issubdtype(dtype, np.complexfloating)

def _transpose_linop(A):
    """Transpose/adjoint of a LinearOperator or ndarray.

    For LinearOperators: creates new LinearOperator with swapped matvec/rmatvec
    For arrays: uses .conj().T

    This matches MATLAB's A' operator behavior.
    """
    if _is_linop(A):
        # Create new scipy LinearOperator with swapped functions and dimensions
        m, n = A.shape
        A_T = LinearOperator(
            shape=(n, m),
            matvec=lambda x: _rmatvec(A, x),
            rmatvec=lambda x: _matvec(A, x),
            dtype=A.dtype
        )

        # Preserve custom attributes if they exist
        if hasattr(A, 'is_explicit'):
            A_T.is_explicit = A.is_explicit
        if hasattr(A, 'matrix'):
            A_T.matrix = A.matrix.conj().T if A.matrix is not None else None

        return A_T
    else:
        return A.conj().T

def _get_matrix(A):
    """Extract explicit matrix from LinearOperator or return array.

    For LinearOperators: extracts A.A if available (scipy's MatrixLinearOperator or custom)
    For arrays: returns A directly

    Raises ValueError if A is a matrix-free LinearOperator.
    """
    if _is_linop(A):
        # LinearOperators with .A attribute (scipy's MatrixLinearOperator or custom)
        if hasattr(A, 'A') and A.A is not None:
            return A.A
        else:
            raise ValueError('Cannot extract explicit matrix from matrix-free LinearOperator')
    else:
        return A

def _matvec(A, x):
    """Matrix-vector or matrix-matrix product for both ndarray and LinearOperator.

    Returns
    -------
    y : ndarray
        Result of A @ x
    """
    if x.ndim == 2:
        if isinstance(A, np.ndarray):
            return A @ x
        elif hasattr(A, 'matmat'):
            return A.matmat(x)
        else:
            m = A.shape[0]
            k = x.shape[1]
            dtype = x.dtype
            result = np.zeros((m, k), dtype=dtype)
            for i in range(k):
                result[:, i] = A @ x[:, i]
            return result
    else:
        if _is_linop(A):
            return A.apply(x) if hasattr(A, 'apply') else A @ x
        else:
            return A @ x

def _rmatvec(A, x):
    """Adjoint matrix-vector or matrix-matrix product for both ndarray and LinearOperator.

    Returns
    -------
    y : ndarray
        Result of A.conj().T @ x (Hermitian adjoint)
    """
    if x.ndim == 2:
        if isinstance(A, np.ndarray):
            return A.conj().T @ x
        elif hasattr(A, 'rmatmat'):
            return A.rmatmat(x)
        else:
            n = A.shape[1]
            k = x.shape[1]
            dtype = x.dtype
            result = np.zeros((n, k), dtype=dtype)
            for i in range(k):
                result[:, i] = A.conj().T @ x[:, i]
            return result
    else:
        if _is_linop(A):
            return A.applyT(x) if hasattr(A, 'applyT') else A.conj().T @ x
        else:
            return A.conj().T @ x

def _matmat_left(Q, A):
    """Compute Q^H @ A for both ndarray and LinearOperator A.

    Returns
    -------
    B : ndarray
        Result of Q^H @ A
    """
    if isinstance(A, np.ndarray):
        return Q.conj().T @ A
    elif hasattr(A, 'rmatmat'):
        C = A.rmatmat(Q)
        return C.conj().T
    else:
        m, n = A.shape
        k = Q.shape[1]
        dtype = Q.dtype
        C = np.zeros((n, k), dtype=dtype)

        for i in range(k):
            C[:, i] = _rmatvec(A, Q[:, i])

        return C.conj().T

# --------------------------------------------------------------
# Test-matrix generators
# --------------------------------------------------------------
def _gaussian_omega(A, n, block_size, *, rng=None):
    """Gaussian test matrix (real or complex)."""
    rng = np.random.default_rng() if rng is None else rng
    dtype = _get_dtype(A)

    if not _is_complex(A):
        return rng.standard_normal((n, block_size), dtype=dtype)

    real_dtype = np.float64 if dtype == np.complex128 else np.float32
    real = rng.standard_normal((n, block_size), dtype=real_dtype)
    imag = rng.standard_normal((n, block_size), dtype=real_dtype)
    return (real + 1j * imag).astype(dtype)


def _uniform_omega(A, n, block_size, *, rng=None):
    """Uniform[-1,1] test matrix (real or complex)."""
    rng = np.random.default_rng() if rng is None else rng
    dtype = _get_dtype(A)

    if not _is_complex(A):
        return rng.uniform(-1.0, 1.0, size=(n, block_size)).astype(dtype)

    real_dtype = np.float64 if dtype == np.complex128 else np.float32
    real = rng.uniform(-1.0, 1.0, size=(n, block_size)).astype(real_dtype)
    imag = rng.uniform(-1.0, 1.0, size=(n, block_size)).astype(real_dtype)
    return (real + 1j * imag).astype(dtype)


# --------------------------------------------------------------
# Power iteration
# --------------------------------------------------------------
def _power_iteration(A, x, power_iter=0):
    """Apply (A^H A)^n to x and orthogonalize.

    Parameters
    ----------
    A : ndarray or LinearOperator
        Input matrix or operator
    x : ndarray
        Input matrix to iterate on
    power_iter : int, optional
        Number of power iterations (default: 0)

    Returns
    -------
    x : ndarray
        Orthogonalized result after power_iter iterations
    """
    for _ in range(power_iter):
        x = _rmatvec(A, _matvec(A, x))
        x, _, _ = linalg.qr(x, mode='economic', pivoting=True)
    return x


# --------------------------------------------------------------
# Rank determination helpers
# --------------------------------------------------------------
def _rank_from_svals(s, rtol):
    """Return the numerical rank given singular values `s`."""
    if s.size == 0:
        return 0
    return int(np.sum(s >= rtol * s[0]))

def _rank_from_diag(diag_vals, rtol):
    """Return the numerical rank given diagonal elements (e.g., from QR)."""
    diag_abs = np.abs(diag_vals)
    if diag_abs.size == 0 or diag_abs[0] <= 0:
        return 0
    return int(np.sum(diag_abs >= rtol * diag_abs[0]))


# --------------------------------------------------------------
# T matrix computation helpers for ID
# --------------------------------------------------------------
def _compute_T_lstsq(A, R, piv, k):
    """Compute T using least-squares from original A columns."""
    m, n = A.shape

    if k == 0 or k >= n:
        return np.zeros((k, n - k), dtype=R.dtype)

    is_linop = _is_linop(A)
    is_matrix_free = _is_matrix_free_linop(A)

    if is_matrix_free:
        skeleton_cols = np.zeros((m, k), dtype=R.dtype)
        for j in range(k):
            e_j = np.zeros(n, dtype=R.dtype)
            e_j[piv[j]] = 1.0
            skeleton_cols[:, j] = A @ e_j

        remaining_cols = np.zeros((m, n - k), dtype=R.dtype)
        for j in range(n - k):
            e_j = np.zeros(n, dtype=R.dtype)
            e_j[piv[k + j]] = 1.0
            remaining_cols[:, j] = A @ e_j

        T, _, _, _ = linalg.lstsq(skeleton_cols, remaining_cols)
    else:
        A_mat = _get_matrix(A)
        cols = piv[:k]
        remaining = piv[k:]
        T, _, _, _ = linalg.lstsq(A_mat[:, cols], A_mat[:, remaining])

    return T


def _compute_T_svd(R, k, rtol_for_svd):
    """Compute T using SVD-based pseudoinverse of R11.

    Parameters
    ----------
    R : ndarray
        R factor from QR decomposition
    k : int
        Rank
    rtol_for_svd : float
        Tolerance for filtering small singular values (should be actual tolerance, not rank)
    """
    n = R.shape[1]

    if k == 0:
        return np.zeros((k, n - k), dtype=R.dtype)

    R11 = R[:k, :k]
    R12 = R[:k, k:]

    U, s, Vh = linalg.svd(R11, full_matrices=False)

    # Filter small singular values. Floor the relative threshold at machine
    # precision so a rank-deficient R11 (e.g. requested rank exceeds the true
    # rank) drops near-zero singular values instead of inverting them, which
    # would otherwise produce Inf/NaN. An all-zero R11 yields T = 0.
    smax = s.max() if s.size else 0.0
    if smax == 0:
        return np.zeros_like(R12)
    tol = max(rtol_for_svd, np.finfo(R.real.dtype).eps)
    keep = s >= tol * smax
    if not np.any(keep):
        T = np.zeros_like(R12)
    else:
        inv_s = 1.0 / s[keep]
        T = Vh[keep, :].conj().T @ (np.diag(inv_s) @ (U[:, keep].conj().T @ R12))

    return T


def _compute_T_fast(R, k, rtol_for_svd):
    """Compute T using fast triangular solve, falling back to the SVD-based
    minimum-norm solve when R11 is (near-)singular."""
    n = R.shape[1]

    if k == 0:
        return np.zeros((k, n - k), dtype=R.dtype)

    R11 = R[:k, :k]
    R12 = R[:k, k:]

    # R11 is upper-triangular from QR; if it is (near-)singular the fast
    # triangular solve would divide by ~0 (Inf/NaN). Detect via the diagonal
    # and fall back to the SVD-based minimum-norm solve, which stays finite
    # and lets the decomposition still reconstruct A.
    d = np.abs(np.diag(R11))
    if d.min() <= np.finfo(R.real.dtype).eps * d.max():
        return _compute_T_svd(R, k, rtol_for_svd)

    from scipy.linalg import solve_triangular
    T = solve_triangular(np.triu(R11), R12, lower=False,
                         overwrite_b=False, check_finite=False)
    return T
