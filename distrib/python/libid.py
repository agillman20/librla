"""
Randomized linear-algebra routines
==================================

Author: Your Name
License: SPDX-License-Identifier: TBD

Features
--------
* _gaussian_omega    - generate a Gaussian test matrix (real or complex)
* _uniform_omega     - generate a uniform[-1, 1] test matrix (real or complex)
* _power_iteration   - simple power iteration to improve the sketch
* orth_sketch        - sketch an orthonormal basis for the column space
* qr_sketch          - truncated QR factorization using a random sketch
* svd_sketch         - truncated SVD via a random sketch
* id_sketch          - interpolative decomposition (ID) using a random sketch
* _hilb              - generate an m x n Hilbert matrix
* _safe_max_abs      - return max(|X|) safely (returns 0.0 for an empty array)
"""

import numpy as np
from scipy import linalg
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator

# --------------------------------------------------------------
# LinearOperator detection and utilities
# --------------------------------------------------------------
def _is_linop(A):
    """Check if A is a LinearOperator (not just a dense array)."""
    return isinstance(A, ScipyLinearOperator) and not isinstance(A, np.ndarray)

def _is_matrix_free_linop(A):
    """Check if A is a matrix-free LinearOperator (not backed by explicit matrix)."""
    return _is_linop(A) and hasattr(A, 'is_explicit') and not A.is_explicit

def _get_dtype(A):
    """Get dtype from either ndarray or LinearOperator."""
    if isinstance(A, np.ndarray):
        return A.dtype
    elif hasattr(A, 'dtype') and A.dtype is not None:
        return A.dtype
    elif _is_linop(A) and hasattr(A, 'matrix') and A.matrix is not None:
        return A.matrix.dtype
    else:
        # This should not be reached - LinearOperators must have valid dtype
        raise ValueError(
            "Unable to determine dtype. For matrix-free LinearOperators, "
            "dtype must be specified during creation."
        )

def _is_complex(A):
    """Check if A represents complex data."""
    dtype = _get_dtype(A)
    return np.issubdtype(dtype, np.complexfloating)

def _matvec(A, x):
    """Matrix-vector or matrix-matrix product for both ndarray and LinearOperator.

    Automatically uses BLAS3 (matmat) when x is a matrix and A supports it.

    Parameters
    ----------
    A : ndarray or LinearOperator
        Operator to apply
    x : ndarray, shape (n,) or (n, k)
        Vector or matrix to multiply

    Returns
    -------
    y : ndarray, shape (m,) or (m, k)
        Result of A @ x
    """
    if x.ndim == 2:
        # Matrix input - use BLAS3 if available
        if isinstance(A, np.ndarray):
            return A @ x
        elif hasattr(A, 'matmat'):
            return A.matmat(x)  # BLAS3
        else:
            # Fallback: column-by-column (scipy's default behavior)
            return np.column_stack([A @ x[:, i] for i in range(x.shape[1])])
    else:
        # Vector input - use BLAS2
        if _is_linop(A):
            return A.apply(x) if hasattr(A, 'apply') else A @ x
        else:
            return A @ x

def _rmatvec(A, x):
    """Adjoint matrix-vector or matrix-matrix product for both ndarray and LinearOperator.

    Automatically uses BLAS3 (rmatmat) when x is a matrix and A supports it.

    Parameters
    ----------
    A : ndarray or LinearOperator
        Operator to apply adjoint of
    x : ndarray, shape (m,) or (m, k)
        Vector or matrix to multiply

    Returns
    -------
    y : ndarray, shape (n,) or (n, k)
        Result of A.conj().T @ x (Hermitian adjoint)
    """
    if x.ndim == 2:
        # Matrix input - use BLAS3 if available
        if isinstance(A, np.ndarray):
            return A.conj().T @ x
        elif hasattr(A, 'rmatmat'):
            return A.rmatmat(x)  # BLAS3
        else:
            # Fallback: column-by-column (scipy's default behavior)
            return np.column_stack([A.conj().T @ x[:, i] for i in range(x.shape[1])])
    else:
        # Vector input - use BLAS2
        if _is_linop(A):
            return A.applyT(x) if hasattr(A, 'applyT') else A.conj().T @ x
        else:
            return A.conj().T @ x

def _matmat_left(Q, A):
    """Compute Q^H @ A for both ndarray and LinearOperator A.

    Uses BLAS3 (rmatmat) when available for LinearOperators, otherwise
    falls back to column-by-column adjoint matvec.

    Parameters
    ----------
    Q : ndarray, shape (m, k)
        Orthonormal matrix
    A : ndarray or LinearOperator, shape (m, n)
        Matrix or operator

    Returns
    -------
    B : ndarray, shape (k, n)
        Result of Q^H @ A
    """
    if isinstance(A, np.ndarray):
        # Dense matrix: direct BLAS3 GEMM
        return Q.conj().T @ A
    elif hasattr(A, 'rmatmat'):
        # LinearOperator with BLAS3 support:
        # B = Q^H @ A  =>  B^H = A^H @ Q  =>  B = (A^H @ Q)^H
        C = A.rmatmat(Q)  # BLAS3: C = A^H @ Q, shape (n, k)
        return C.conj().T  # B = C^H, shape (k, n)
    else:
        # Fallback: column-by-column adjoint matvec (BLAS2)
        # B = Q^H @ A  =>  B^H = A^H @ Q
        # Compute C = A^H @ Q by applying adjoint to each column of Q
        # Then B = C^H
        m, n = A.shape
        k = Q.shape[1]
        dtype = Q.dtype
        C = np.zeros((n, k), dtype=dtype)

        for i in range(k):
            C[:, i] = _rmatvec(A, Q[:, i])

        # Return B = C^H
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

    # complex case: generate real and imag parts separately
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

    # complex case: generate real and imag parts separately
    real_dtype = np.float64 if dtype == np.complex128 else np.float32
    real = rng.uniform(-1.0, 1.0, size=(n, block_size)).astype(real_dtype)
    imag = rng.uniform(-1.0, 1.0, size=(n, block_size)).astype(real_dtype)
    return (real + 1j * imag).astype(dtype)


# --------------------------------------------------------------
# Power iteration (optional)
# --------------------------------------------------------------
def _power_iteration(A, x, flag_power=0):
    """
    Apply (A^H A)^n to x (n = flag_power).

    Always ensures the output is orthogonalized, even when flag_power=0.
    """
    # Orthogonalize input first (important when flag_power=0)
    if flag_power == 0:
        x, _, _ = linalg.qr(x, mode='economic', pivoting=True)
        return x

    for _ in range(flag_power):
        x = _rmatvec(A, _matvec(A, x))
        # Re-orthogonalise after each iteration
        x, _, _ = linalg.qr(x, mode='economic', pivoting=True)
    return x


# --------------------------------------------------------------
# 1. Orthogonal sketch
# --------------------------------------------------------------

def orth_sketch(A, rtol, block_size=42, flag_power=0, *, rng=None, skip_tol_check=False):
    """
    Orthonormal basis Q for col(A) via random sketch.

    Parameters
    ----------
    A : ndarray or LinearOperator
        Input matrix or operator.
    rtol : float
        Relative tolerance for sketch quality.
    block_size : int
        Initial sketch size.
    flag_power : int
        Number of power iteration steps for range estimation.
        Uses 2-step structure: alternates A and A^H operations with
        incremental orthogonalization, equivalent to (A^H A)^flag_power.
        Default: 0 (no improvement). Higher values (1-4) provide better
        accuracy at increased cost (converges to dominant subspace).
    rng : Generator, optional
        Random number generator.
    skip_tol_check : bool
        If True, skip tolerance check and geometric growth. Computes a single
        sketch of size block_size, then filters Q to remove vectors outside
        the operator's range (based on diagonal of R vs column norms of Y).
        Used for matrix-free operators in rank mode.

    Returns
    -------
    Q : ndarray, shape (m, k) or (m, 0)
        Orthonormal basis for the column space. May be empty if the routine
        exits early because the requested sketch already spans the whole space.
    flag : int
        Exit flag:
        * 0 - normal exit, Q satisfies the tolerance.
        * 1 - early exit because the sketch already covered the full space.
    err : float
        Error estimate: smallest R diagonal ratio d = |R[k,k]| / max(||y_j||).
        Indicates how much signal remains unmodeled. Values close to 0 indicate
        good approximation. For early exits, returns 0.0.
    """
    m, n = A.shape
    dtype = _get_dtype(A)

    # For matrix-free operators in rank mode, just compute a single sketch
    # without tolerance checking or geometric growth
    if skip_tol_check:
        x = _uniform_omega(A, n, block_size, rng=rng)
        x = _power_iteration(A, x, flag_power=flag_power)
        y = _matvec(A, x)
        Q, R, _ = linalg.qr(y, mode='economic', pivoting=True)

        # Filter Q using same criterion as tolerance loop:
        # Keep columns where |R[i,i]| / max(||y_j||) > rtol
        col_norms = norm(y, axis=0)
        max_col_norm = np.max(col_norms) if col_norms.size > 0 else 1.0
        diagR = np.abs(np.diag(R))

        # Use machine epsilon as tolerance (same as rank mode in qr_sketch)
        rtol_eps = max(m, n) * np.finfo(dtype).eps

        # Filter: keep column i if |R[i,i]| / max_col_norm > rtol_eps
        # This is equivalent to: |R[i,i]| > rtol_eps * max_col_norm
        if diagR.size > 0 and max_col_norm > 0:
            # Keep columns where diagonal element is significant
            # Use same criterion as in tolerance loop: d = |R[i]| / max(||y||)
            d_ratios = diagR / max_col_norm
            keep = d_ratios > rtol_eps
            rank = np.sum(keep)
        else:
            rank = 0

        # Truncate to actual rank
        Q = Q[:, :rank]

        # Compute error estimate from final diagonal ratio
        if rank > 0 and max_col_norm > 0:
            err = float(diagR[rank-1] / max_col_norm)
        else:
            err = 0.0

        return Q, 0, err

    # Guard against tolerance smaller than machine precision (normal mode only)
    if rtol < np.finfo(dtype).eps:
        return np.empty((m, 0), dtype=dtype), 1, 0.0

    # Guard against a sketch that already covers the whole matrix (normal mode only)
    if block_size >= min(m, n):
        # Early exit - the whole space is already spanned.
        return np.empty((m, 0), dtype=dtype), 1, 0.0

    # Main loop - keep enlarging the sketch until the tolerance is met.
    while True:
        # a) random test matrix
        x = _uniform_omega(A, n, block_size, rng=rng)
        x = _power_iteration(A, x, flag_power=flag_power)

        # b) sketch the column space
        y = _matvec(A, x)
        Q, R, _ = linalg.qr(y, mode='economic', pivoting=True)

        # c) stop when smallest diag(R) / max column norm(y) <= rtol
        d = abs(R.diagonal()[-1:]) / max(norm(y, axis=0))
        if d <= rtol:
            # Normal termination - we have a basis that meets the tolerance.
            err = float(d[0]) if d.size > 0 else 0.0
            return Q, 0, err

        # d) enlarge the sketch (geometric growth)
        block_size = min(block_size * 4, min(m, n))
        if block_size >= min(m, n):
            # Early exit - we have exhausted the possible sketch size.
            return np.empty((m, 0), dtype=dtype), 1, 0.0


# --------------------------------------------------------------
# 2. Truncated QR with column pivoting
# --------------------------------------------------------------
def qr_sketch(A, rtol, block_size=42, flag_power=0, extra_samples=12):
    """QR with column pivoting via a random sketch.

    Parameters
    ----------
    A : ndarray or LinearOperator
        Input matrix or linear operator.
    rtol : float
        Relative tolerance (rtol < 1) or target rank (rtol >= 1).
        For matrix-free LinearOperator, only rank mode (rtol >= 1) is supported.
    block_size : int, optional
        Sketch block size (default: 42). Interpretation depends on mode:

        - **Tolerance mode** (rtol < 1): Starting value for geometric growth.

        - **Rank mode** (rtol >= 1): Ignored. Total sketch size is always
          kmax + extra_samples. Control via extra_samples parameter.
    flag_power : int, optional
        Number of power iterations for sketch improvement.
    extra_samples : int, optional
        Extra samples for oversampling in rank mode (default: 20).

    Returns
    -------
    Q : ndarray, shape (m, rank)
        Orthonormal basis.
    R : ndarray, shape (rank, n)
        Upper triangular factor.
    p : ndarray, shape (n,)
        Column permutation.
    """
    m, n = A.shape
    dtype = _get_dtype(A)
    is_matrix_free = _is_matrix_free_linop(A)
    is_linop = _is_linop(A)

    # Allow ``rtol`` >= 1 to be interpreted as a maximum rank.
    flag_kmax = False
    rtol_for_sketch = rtol  # tolerance to use for orth_sketch
    if rtol >= 1:
        flag_kmax = True
        kmax = int(np.floor(rtol))

        # In rank mode, block_size is always kmax + extra_samples
        # User controls via extra_samples parameter
        block_size = kmax + extra_samples

        # Set tolerance to machine epsilon (effectively disables tolerance checking)
        rtol_for_sketch = max(m, n) * np.finfo(dtype).eps
    elif is_matrix_free:
        # Matrix-free LinearOperator in tolerance mode not supported
        raise ValueError(
            "Matrix-free LinearOperator only supported in rank mode (rtol >= 1). "
            f"Got rtol={rtol}. Please specify target rank as rtol."
        )

    # In rank mode, skip tolerance check (user specifies exact rank, no need for adaptive checking)
    skip_tol = flag_kmax
    Qs, flag, _ = orth_sketch(A, rtol_for_sketch, block_size, flag_power, skip_tol_check=skip_tol)

    # Determine effective sketch size k.
    k = Qs.shape[1] if flag == 0 else min(m, n)

    # Full-rank fallback: deterministic QR (triggered either by early exit
    # or by the sketch already spanning the whole space).
    needs_fallback = (flag != 0 or k >= min(m, n))

    # In rank mode: skip fallback (user requested specific rank, use sketch as-is)
    if needs_fallback and flag_kmax:
        needs_fallback = False

    if needs_fallback:

        # For explicit LinearOperator, extract the underlying matrix
        A_mat = A.matrix if (is_linop and hasattr(A, 'matrix') and A.matrix is not None) else A
        Q, R, p = linalg.qr(A_mat, mode='economic', pivoting=True)
        # Determine rank using diagonal elements (standard RRQR criterion)
        # Use machine epsilon for tolerance (already converted above)
        rtol_for_rank = max(m, n) * np.finfo(dtype).eps if flag_kmax else rtol
        diag_abs = np.abs(np.diag(R))
        if diag_abs.size > 0 and diag_abs[0] > 0:
            rank = int(np.sum(diag_abs >= rtol_for_rank * diag_abs[0]))
        else:
            rank = 0
        if flag_kmax:
            rank = min(kmax, rank)
        return Q[:, :rank], R[:rank, :], p

    # Project onto the sketch space
    B = _matmat_left(Qs, A)

    # Deterministic QR of the thin matrix
    Qproj, R, p = linalg.qr(B, mode='economic', pivoting=True)
    Q = Qs @ Qproj

    # Determine rank using diagonal elements (standard RRQR criterion)
    # Use machine epsilon for tolerance in rank mode
    rtol_for_rank = max(m, n) * np.finfo(dtype).eps if flag_kmax else rtol
    diag_abs = np.abs(np.diag(R))
    if diag_abs.size > 0 and diag_abs[0] > 0:
        rank = int(np.sum(diag_abs >= rtol_for_rank * diag_abs[0]))
    else:
        rank = 0
    if flag_kmax:
        rank = min(kmax, rank)

    return Q[:, :rank], R[:rank, :], p


# --------------------------------------------------------------
# 2. Helper for rank from singular values (used everywhere)
# --------------------------------------------------------------
def _rank_from_svals(s, rtol):
    """Return the numerical rank given singular values `s`."""
    if s.size == 0:
        return 0
    return int(np.sum(s >= rtol * s[0]))

# --------------------------------------------------------------
# 3. Truncated SVD
# --------------------------------------------------------------
def svd_sketch(A, rtol, block_size=42, flag_power=0, extra_samples=12):
    """Truncated SVD via a random sketch.

    Parameters
    ----------
    A : ndarray or LinearOperator
        Input matrix or linear operator.
    rtol : float
        Relative tolerance (rtol < 1) or target rank (rtol >= 1).
        For matrix-free LinearOperator, only rank mode (rtol >= 1) is supported.
    block_size : int, optional
        Sketch block size (default: 42). Interpretation depends on mode:

        - **Tolerance mode** (rtol < 1): Starting value for geometric growth.

        - **Rank mode** (rtol >= 1): Ignored. Total sketch size is always
          kmax + extra_samples. Control via extra_samples parameter.
    flag_power : int, optional
        Number of power iterations for sketch improvement.
    extra_samples : int, optional
        Extra samples for oversampling in rank mode (default: 12).

    Returns
    -------
    U : ndarray, shape (m, rank)
        Left singular vectors.
    s : ndarray, shape (rank,)
        Singular values.
    Vh : ndarray, shape (rank, n)
        Right singular vectors (conjugate transpose). Matches scipy convention.
        Reconstruction: A ~ U @ diag(s) @ Vh
    """
    m, n = A.shape
    dtype = _get_dtype(A)
    is_matrix_free = _is_matrix_free_linop(A)
    is_linop = _is_linop(A)

    # For wide matrices (m < n), work on transpose to minimize SVD cost.
    # The projected matrix becomes kxm instead of kxn, reducing SVD
    # from O(kn^2) to O(km^2), giving speedup of (n/m)^2.
    # For LinearOperators, we need to create a transposed operator
    if m < n:
        if is_linop:
            # Create a transposed LinearOperator
            from linop.make_linop import make_linop
            A_T = make_linop(n, m,
                           matvec=lambda x: _rmatvec(A, x),
                           rmatvec=lambda x: _matvec(A, x))
            # Mark as matrix-free if original was matrix-free
            if is_matrix_free:
                A_T.matrix = None  # Ensure it's detected as matrix-free
            # Note: For explicit LinearOperators, we don't copy the transposed matrix
            # to save memory. The matvec/rmatvec functions handle transposition.
        else:
            # Dense array
            A_T = A.conj().T
        V, s, U = svd_sketch(A_T, rtol, block_size, flag_power, extra_samples)
        return U.conj().T, s, V.conj().T

    # Allow ``rtol`` >= 1 to be interpreted as a maximum rank.
    flag_kmax = False
    rtol_for_sketch = rtol
    if rtol >= 1:
        flag_kmax = True
        kmax = int(np.floor(rtol))

        # In rank mode, block_size is always kmax + extra_samples
        # User controls via extra_samples parameter
        block_size = kmax + extra_samples

        # Set tolerance to machine epsilon (effectively disables tolerance checking)
        rtol_for_sketch = max(m, n) * np.finfo(dtype).eps
    elif is_matrix_free:
        # Matrix-free LinearOperator in tolerance mode not supported
        raise ValueError(
            "Matrix-free LinearOperator only supported in rank mode (rtol >= 1). "
            f"Got rtol={rtol}. Please specify target rank as rtol."
        )

    # In rank mode, skip tolerance check (user specifies exact rank)
    skip_tol = flag_kmax
    Qs, flag, _ = orth_sketch(A, rtol_for_sketch, block_size, flag_power, skip_tol_check=skip_tol)

    k = Qs.shape[1] if flag == 0 else min(m, n)

    # Full-rank fallback: deterministic SVD
    needs_fallback = (flag != 0 or k >= min(m, n))

    # In rank mode: skip fallback (user requested specific rank, use sketch as-is)
    if needs_fallback and flag_kmax:
        needs_fallback = False

    if needs_fallback:

        # For explicit LinearOperator, extract the underlying matrix
        A_mat = A.matrix if (is_linop and hasattr(A, 'matrix') and A.matrix is not None) else A
        U, s, V = linalg.svd(A_mat, full_matrices=False)
        rank = _rank_from_svals(s, rtol_for_sketch)
        if flag_kmax:
            rank = min(kmax, rank)
        return U[:, :rank], s[:rank], V[:rank, :]

    # Project onto the sketch space and compute SVD
    Aproj = _matmat_left(Qs, A)
    Uproj, s, V = linalg.svd(Aproj, full_matrices=False)
    U = Qs @ Uproj
    rank = _rank_from_svals(s, rtol_for_sketch)
    if flag_kmax:
        rank = min(kmax, rank)
    return U[:, :rank], s[:rank], V[:rank, :]


# --------------------------------------------------------------
# 4. Interpolative decomposition (ID)
# --------------------------------------------------------------
def id_sketch(A, rtol, block_size=42, flag_power=0, extra_samples=12, *, use_svd=False, recompute_T=False):
    """ID via random sketches using single-stage QR pipeline.

    Uses direct QR factorization with column pivoting on the sketched matrix.

    Parameters
    ----------
    A : ndarray or LinearOperator
        Input matrix or linear operator.
    rtol : float
        Relative tolerance (rtol < 1) or target rank (rtol >= 1).
        For matrix-free LinearOperator, only rank mode (rtol >= 1) is supported.
    block_size : int, optional
        Sketch block size (default: 42). Interpretation depends on mode:

        - **Tolerance mode** (rtol < 1): Starting value for geometric growth.

        - **Rank mode** (rtol >= 1): Ignored. Total sketch size is always
          kmax + extra_samples. Control via extra_samples parameter.
    flag_power : int, optional
        Number of power iterations for sketch improvement.
    extra_samples : int, optional
        Extra samples for oversampling in rank mode (default: 20).
    use_svd : bool, optional
        Use SVD-based pseudoinverse for solving (default: False).
    recompute_T : bool, optional
        Recompute interpolation matrix T from original A (default: False).

        - False (default): Compute T from R matrix (Fortran's approach).
          Fast (6-9x speedup for matrix-free), but may give error > 1.0 on
          full-rank matrices. Use when speed is critical and higher error
          is acceptable.

        - True: Recompute T via least squares on original A.
          Ensures error < 1.0 (mathematically guaranteed).

          For explicit matrices: Direct column indexing.
          For matrix-free operators: Extracts all n columns via unit vectors
          (n matvecs). Slower but guarantees accuracy.

    Returns
    -------
    k : int
        Rank of the ID approximation.
    piv : ndarray, shape (n,)
        Column permutation. First k columns are the skeleton.
    T : ndarray, shape (k, n-k)
        Interpolation matrix satisfying A[:, piv[k:]] ~ A[:, piv[:k]] @ T
    """
    is_linop = _is_linop(A)
    is_matrix_free = _is_matrix_free_linop(A)

    # Standard path: QR with column pivoting on sketched A
    _, R, jpiv = qr_sketch(A, rtol, block_size, flag_power, extra_samples)

    k = R.shape[0]
    piv = jpiv

    # Recompute T from original A for better accuracy
    if recompute_T and k > 0 and k < A.shape[1]:
        m, n = A.shape

        if is_matrix_free:
            # Matrix-free path: Extract columns via unit vectors
            # This is slower (n matvecs) but guarantees error < 1.0
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

            # Solve: skeleton_cols @ T ~ remaining_cols
            T, residuals, rank_ls, s = linalg.lstsq(skeleton_cols, remaining_cols)
            return k, piv, T

        else:
            # Explicit matrix path: Direct column indexing
            A_mat = A.matrix if (is_linop and hasattr(A, 'matrix') and A.matrix is not None) else A

            # Recompute T via least squares on original A
            # This ensures: A[:, piv[k:]] ~ A[:, piv[:k]] @ T
            cols = piv[:k]
            remaining = piv[k:]

            # Solve: A[:, piv[:k]] @ T ~ A[:, piv[k:]]
            T, residuals, rank_ls, s = linalg.lstsq(A_mat[:, cols], A_mat[:, remaining])
            return k, piv, T

    # Fall back to using T from R matrix (may have error > 1.0 for sketches)
    R11 = R[:k, :k]
    R12 = R[:k, k:]

    if use_svd:
        U, s, Vh = linalg.svd(R11, full_matrices=False)
        keep = s >= rtol * np.max(s)
        if not np.any(keep):
            T = np.zeros_like(R12)
        else:
            inv_s = 1.0 / s[keep]
            T = Vh[keep, :].conj().T @ (np.diag(inv_s) @ (U[:, keep].conj().T @ R12))
    else:
        from scipy.linalg import solve_triangular
        T = solve_triangular(np.triu(R11), R12, lower=False,
                             overwrite_b=False, check_finite=False)

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
# Helper utilities (keep them tiny - replace with the real implementations)
# ----------------------------------------------------------------------
def _safe_max_abs(X: np.ndarray) -> float:
    """Return max|X| safely - 0.0 for an empty array."""
    return float(np.max(np.abs(X))) if X.size else 0.0


import time
import statistics as _stats

def _run_and_report(name, func, *args, **kwargs):
    """
    Execute ``func(*args, **kwargs)`` five times.

    The routine *must* return a **tuple** ``(rank, error)`` - i.e. the first
    element is the integer rank (or 0 if the routine produces an empty basis)
    and the second element is the scalar error (relative reconstruction /
    orthonormality error).  For each run we print

        <name> run #i : elapsed = xx.xxx s, rank = r, error = x.xxxe-yy

    After the five repetitions we also print a tiny summary (average elapsed
    time and worst-case error).
    """
    times = []
    ranks = []
    errs  = []

    for i in range(5):                     # <-- testing count = 5
        t0 = time.perf_counter()
        rank, err = func(*args, **kwargs)   # routine must return (rank, error)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        times.append(elapsed)
        ranks.append(rank)
        errs.append(err)

        print(f"{name:<12} run #{i+1:<1}: elapsed = {elapsed:6.3f} s, "
              f"rank = {rank:<4}, error = {err:.3e}")

    avg_time = _stats.mean(times)
    max_err  = max(errs)
    # (All ranks should be identical; we just show the first one in the summary.)
    print(f"{name:<12}: avg elapsed = {avg_time:6.3f} s,  max error = {max_err:.3e}\n")
    return times, ranks, errs


# --------------------------------------------------------------
# 1. orth_sketch
# --------------------------------------------------------------
def run_orth(A):
    Q_range, flag, err_est = orth_sketch(A, rtol=1e-15)
    rank_orth = Q_range.shape[1]          # basis size
    if rank_orth == 0:
        orth_err = np.nan
    else:
        orth_err = np.linalg.norm(Q_range.conj().T @ Q_range -
                                  np.eye(rank_orth))
    return rank_orth, orth_err


# --------------------------------------------------------------
# 2. qr_sketch
# --------------------------------------------------------------
def run_qr(A):
    Q_rrqr, R_rrqr, piv = qr_sketch(A, rtol=1e-15)
    rank_qr = R_rrqr.shape[0]                 # retained columns
    A_perm = A[:, piv]                         # columns in pivot order
    recon_err = np.linalg.norm(Q_rrqr @ R_rrqr - A_perm) / np.linalg.norm(A_perm)
    return rank_qr, recon_err


# --------------------------------------------------------------
# 4. svd_ketch
# --------------------------------------------------------------
def run_svd(A):
    U_rrsvd, s_rrsvd, Vt_rrsvd = svd_sketch(A, rtol=1e-15)
    rank_svd = _rank_from_svals(s_rrsvd, 1e-15)
    A_svd = U_rrsvd @ np.diag(s_rrsvd) @ Vt_rrsvd
    svd_err = np.linalg.norm(A_svd - A) / np.linalg.norm(A)
    return rank_svd, svd_err


# --------------------------------------------------------------
# 5. id_sketch
# --------------------------------------------------------------
def run_id(A):
    k_id, piv_id, T_id = id_sketch(
        A, rtol=1e-15, use_svd=False
    )
    A_id_approx = A[:, piv_id[:k_id]] @ T_id
    id_err = norm(A[:, piv_id[k_id:]] - A_id_approx) / norm(A)
    # Rank we report = k_id (number of kept columns)
    return k_id, id_err


# --------------------------------------------------------------
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import statistics as _stats
import numpy as np
from numpy.linalg import norm, lstsq

# ----------------------------------------------------------------------
# IMPORT YOUR SKETCHING ROUTINES HERE
# ----------------------------------------------------------------------
# from my_sketch_lib import (
#     orth_sketch, qr_sketch, svd_sketch, id_sketch,
# )

# ----------------------------------------------------------------------
# tiny helpers (identical to the ones you already have)
# ----------------------------------------------------------------------
def _hilb(m, n):
    i = np.arange(1, m + 1)[:, None]
    j = np.arange(1, n + 1)[None, :]
    return 1.0 / (i + j - 1)


def _rank_from_svals(s, tol):
    return np.sum(s > tol)


def _safe_max_abs(T):
    return np.max(np.abs(T)) if T.size else 0.0


# ----------------------------------------------------------------------
# MAIN - everything is **inlined** here
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------
    # 1. Build a modest-size Hilbert matrix (quick to factorise)
    # --------------------------------------------------------------
    m, n = 4000, 2000
    A = _hilb(m, n).astype(np.float64)

    # A = A.astype(np.float32)
    # A = A.astype(np.float64)

    # A = A.astype(np.complex64)
    # A = A.astype(np.complex128)

    # from scipy.io import loadmat
    # F = loadmat('../contrib/atest.mat')
    # A = F['Ain']
    # A = A.T.copy()
    # print(A)

    ###rtol = 1e-6
    ###rtol = 1e-7
    ###rtol = 1e-12
    ###rtol = 1e-14
    ###rtol = max(m, n) * np.finfo(A.dtype).eps
    ###rtol = 10*np.sqrt(max(m, n)) * np.finfo(A.dtype).eps
    rtol = 20

    print(A.dtype)
    print(A.shape)
    print(rtol)
    
    # ``_run`` is a *single-line* lambda that executes a routine
    # and returns (rank, error).  It exists only so that the repeated
    # "for-i-in-range(5)" blocks stay compact.
    _run = lambda f, *a, **kw: f(*a, **kw)

    # --------------------------------------------------------------
    # 2. orth_sketch
    # --------------------------------------------------------------
    print("\n--- orth_sketch ------------------------------------------------")
    for i in range(5):                         # testing count = 5
        t0 = time.perf_counter()
        Q_range, flag, err_est = _run(orth_sketch, A, rtol=rtol)
        t1 = time.perf_counter()
        rank = Q_range.shape[1]
        if rank == 0:
            err = np.nan
        else:
            err = np.linalg.norm(Q_range.conj().T @ Q_range - np.eye(rank))
        print(f"orth_sketch run #{i+1:<1}: elapsed = {t1-t0:6.3f} s, "
              f"rank = {rank:<4}, error = {err:.3e}")
    # summary
    # (all runs produce the same rank for deterministic routines; we just reuse the last rank)
    print(f"orth_sketch : avg elapsed = {_stats.mean([t1-t0 for _ in range(5)]):6.3f} s, "
          f"max error = {err:.3e}\n")

    # --------------------------------------------------------------
    # 3. qr_sketch
    # --------------------------------------------------------------
    print("\n--- qr_sketch -------------------------------------------------")
    for i in range(5):
        t0 = time.perf_counter()
        Q_rrqr, R_rrqr, piv = _run(qr_sketch, A, rtol=rtol)
        t1 = time.perf_counter()
        rank = R_rrqr.shape[0]                     # retained columns
        A_perm = A[:, piv]
        err = np.linalg.norm(Q_rrqr @ R_rrqr - A_perm) / np.linalg.norm(A_perm)
        print(f"qr_sketch   run #{i+1:<1}: elapsed = {t1-t0:6.3f} s, "
              f"rank = {rank:<4}, error = {err:.3e}")
    print(f"qr_sketch   : avg elapsed = {_stats.mean([t1-t0 for _ in range(5)]):6.3f} s, "
          f"max error = {err:.3e}\n")

    # --------------------------------------------------------------
    # 4. svd_sketch
    # --------------------------------------------------------------
    print("\n--- svd_sketch ------------------------------------------------")
    for i in range(5):
        t0 = time.perf_counter()
        U_rrsvd, s_rrsvd, Vt_rrsvd = _run(svd_sketch, A, rtol=rtol)
        t1 = time.perf_counter()
        rank = s_rrsvd.shape[0]
        A_svd = U_rrsvd @ np.diag(s_rrsvd) @ Vt_rrsvd
        err = np.linalg.norm(A_svd - A) / np.linalg.norm(A)
        print(f"svd_sketch  run #{i+1:<1}: elapsed = {t1-t0:6.3f} s, "
              f"rank = {rank:<4}, error = {err:.3e}")
    print(f"svd_sketch  : avg elapsed = {_stats.mean([t1-t0 for _ in range(5)]):6.3f} s, "
          f"max error = {err:.3e}\n")

    # --------------------------------------------------------------
    # 6. id_sketch
    # --------------------------------------------------------------
    print("\n--- id_sketch -------------------------------------------------")
    for i in range(5):
        t0 = time.perf_counter()
        k_id, piv_id, T_id = _run(id_sketch, A,
                                   rtol=rtol,
                                   use_svd=False)
        t1 = time.perf_counter()
        rank = k_id                              # number of retained columns
        A_id_approx = A[:, piv_id[:k_id]] @ T_id
        err = norm(A[:, piv_id[k_id:]] - A_id_approx) / norm(A)
        print(f"id_sketch   run #{i+1:<1}: elapsed = {t1-t0:6.3f} s, "
              f"rank = {rank:<4}, error = {err:.3e}")
    print(f"id_sketch   : avg elapsed = {_stats.mean([t1-t0 for _ in range(5)]):6.3f} s, "
          f"max error = {err:.3e}\n")
    max_abs_T = _safe_max_abs(T_id)
    print(f"max|T| = {max_abs_T:.6e}")

    print("\nAll routines completed - each was run 5 times.\n")



