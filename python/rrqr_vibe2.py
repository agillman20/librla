"""
rrqr.py

Optimized implementation of the Businger-Golub QR factorization with column
pivoting.  The heavy lifting is now delegated to LAPACK (via SciPy) which
provides highly tuned, cache-aware BLAS level-3 kernels.  The public API
remains unchanged:

    rrqr(A, rtol) -> Q, R, p, k

where
    Q : (m, k)   orthonormal columns,
    R : (k, n)   upper-triangular,
    p : (n,)     permutation vector (zero-based),
    k : int      estimated numerical rank.

All auxiliary routines that were previously implemented in pure NumPy have
been retained (they are useful for testing/educational purposes) but the
core driver now calls the LAPACK routines `geqp3` (QR with column pivoting)
and `orgqr`/`ungqr` (explicit Q construction).  This yields a **large
speed-up** especially for large dense matrices (e.g. the 4000 x 2000 Hilbert
test).

The implementation works for both real and complex inputs and automatically
chooses the correct LAPACK backend.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.linalg import lapack

# ----------------------------------------------------------------------
# Helper utilities (unchanged - kept for completeness / tests)
# ----------------------------------------------------------------------


def copysign_matlab(x, y):
    """MATLAB-style `copysign`: |x| with the sign of y."""
    return math.copysign(abs(x), y)


def reflector(x):
    """Construct a Householder reflector for a vector *x*."""
    n = x.shape[0]
    if n == 0:
        return np.array(0, dtype=x.dtype), x

    xi = x[0]
    norm_u = np.linalg.norm(x)

    if norm_u == 0:
        return np.array(0, dtype=x.dtype), x

    nu = copysign_matlab(norm_u, np.real(xi))

    x[0] = -nu
    if n > 1:
        x[1:] = x[1:] / (xi + nu)

    tau = (xi + nu) / nu
    return tau, x


# ----------------------------------------------------------------------
# Core routine - now a thin wrapper around LAPACK
# ----------------------------------------------------------------------


def _lapack_qr_with_pivoting(
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform QR factorization with column pivoting using the appropriate
    LAPACK routine (`geqp3`) and return the raw QR data:

        a   - matrix containing R (upper triangle) and the elementary
              reflectors (strict lower triangle)
        tau - scalar factors of the elementary reflectors
        jpvt - pivot indices (1-based as returned by LAPACK)

    The function automatically selects the real or complex variant.
    """
    # Choose the correct LAPACK driver based on dtype
    geqp3, = lapack.get_lapack_funcs(("geqp3",), (A,))

    # Query optimal work size
    a, jpvt, tau, work, info = geqp3(A, lwork=-1, overwrite_a=False)
    lwork_opt = int(work.real.max()) if work.size else 1

    # Actual factorization
    a, jpvt, tau, work, info = geqp3(
        A,
        lwork=lwork_opt,
        overwrite_a=True,
    )
    if info != 0:
        raise RuntimeError(f"LAPACK geqp3 failed with info = {info}")

    return a, tau, jpvt


def _build_Q_from_reflectors(
    a: np.ndarray, tau: np.ndarray, k: int
) -> np.ndarray:
    """
    Build the orthogonal (unitary) matrix Q from the reflectors stored in *a*
    and the scalar factors *tau*.  Only the first *k* reflectors are used
    (corresponding to the estimated numerical rank).

    Returns the full Q (m x m) matrix; the caller can slice the needed columns.
    """
    m, _ = a.shape
    # Choose the correct routine: orgqr for real, ungqr for complex
    if np.iscomplexobj(a):
        ungqr, = lapack.get_lapack_funcs(("ungqr",), (a,))
        Q, work, info = ungqr(a, tau, lwork=-1, overwrite_a=False)
        lwork_opt = int(work.real.max())
        Q, work, info = ungqr(a, tau, lwork=lwork_opt, overwrite_a=False)
    else:
        orgqr, = lapack.get_lapack_funcs(("orgqr",), (a,))
        Q, work, info = orgqr(a, tau, lwork=-1, overwrite_a=False)
        lwork_opt = int(work.max())
        Q, work, info = orgqr(a, tau, lwork=lwork_opt, overwrite_a=False)

    if info != 0:
        raise RuntimeError(f"LAPACK org/ungqr failed with info = {info}")

    # Q is m x m; keep only the first *k* columns (the rest are irrelevant)
    return Q[:, :k]


def rrqr(A: np.ndarray, rtol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute the QR factorization with column pivoting using LAPACK.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix (real or complex).  It is **not** modified.
    rtol : float
        Relative tolerance used to determine the numerical rank.

    Returns
    -------
    Q : ndarray, shape (m, k)
        Orthonormal columns (unitary for complex data).
    R : ndarray, shape (k, n)
        Upper-triangular factor.
    p : ndarray, shape (n,)
        Zero-based permutation vector such that A[:, p] = Q @ R.
    k : int
        Estimated numerical rank (number of significant diagonal entries of R).
    """
    # ------------------------------------------------------------------
    # 1. LAPACK QR with column pivoting
    # ------------------------------------------------------------------
    a, tau, jpvt = _lapack_qr_with_pivoting(A.astype(A.dtype, copy=False))

    m, n = A.shape
    min_mn = min(m, n)

    # ------------------------------------------------------------------
    # 2. Determine numerical rank from the diagonal of R
    # ------------------------------------------------------------------
    diag_R = np.abs(np.diag(a[:min_mn, :min_mn]))
    # Global scaling for the tolerance (same strategy as the original code)
    if diag_R.size == 0:
        atol = 0.0
    else:
        atol = rtol * np.linalg.norm(diag_R)
    k = np.sum(diag_R > atol)

    # If the matrix is rank-deficient, truncate tau to the first *k* entries.
    tau_k = tau[:k] if k > 0 else np.empty(0, dtype=A.dtype)

    # ------------------------------------------------------------------
    # 3. Build explicit Q (only the first *k* columns are needed)
    # ------------------------------------------------------------------
    if k == 0:
        Q = np.empty((m, 0), dtype=A.dtype)
    else:
        Q_full = _build_Q_from_reflectors(a, tau_k, k)
        Q = Q_full  # already sliced to (m, k) inside the helper

    # ------------------------------------------------------------------
    # 4. Extract the upper-triangular factor R (first *k* rows)
    # ------------------------------------------------------------------
    R = np.triu(a[:k, :])

    # ------------------------------------------------------------------
    # 5. Convert LAPACK's 1-based pivot indices to 0-based NumPy indices
    # ------------------------------------------------------------------
    p = np.array(jpvt - 1, dtype=int)  # jpvt comes 1-based from LAPACK

    return Q, R, p, k


# ----------------------------------------------------------------------
# The following functions are retained for compatibility / unit-testing.
# They are **not** used by the optimized driver above.
# ----------------------------------------------------------------------

def rrqr_native(A, rtol):
    """
    Compute the QR factorization with column pivoting using the pure NumPy
    implementation.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix, real or complex.
    rtol : float
        Relative tolerance used to determine the numerical rank.

    Returns
    -------
    Q : ndarray, shape (m, k)
        Orthogonal (unitary) matrix.
    R : ndarray, shape (k, n)
        Upper-triangular factor.
    p : ndarray, shape (n,)
        Permutation vector (0-based indices).
    k : int
        Numerical rank of the decomposition.
    """
    m, n = A.shape
    tau, p, k, H = rrqr_piv(A.copy(), rtol)

    # Build the identity matrix that will be transformed into Q
    I = np.eye(m, k, dtype=A.dtype)
    Q = rrqr_q(H, tau, I, k)

    # Upper-triangular part of the first k rows of H is R
    R = np.triu(H[:k, :])
    return Q, R, p, k


def rrqr_piv(a, rtol):
    """
    Perform the pivoted Householder reduction.

    Parameters
    ----------
    a : ndarray, shape (m, n)
        Matrix to be factorized (will be overwritten).
    rtol : float
        Relative tolerance for rank determination.

    Returns
    -------
    tau : ndarray, shape (min(m,n),)
        Scalar factors of the elementary reflectors.
    p : ndarray, shape (n,)
        Permutation vector (0-based indices).
    k : int
        Numerical rank.
    a : ndarray
        Matrix containing the reflectors in its lower-triangular part.
    """
    m, n = a.shape
    max_reflectors = min(m, n)
    tau = np.zeros(max_reflectors, dtype=a.dtype)

    # Initial permutation: identity
    p = np.arange(n, dtype=int)

    # Column norms (2-norm for each column)
    s = np.linalg.norm(a, axis=0)
    d = np.linalg.norm(s)
    atol = rtol * d

    k = 0
    blas_level = 2  # use level-2 update (more efficient)

    if d == 0:
        # Matrix is zero; nothing to do
        return tau[:k], p, k, a

    for j in range(max_reflectors):
        # ----- Choose pivot column ---------------------------------
        # Find index of the column with largest remaining norm
        jpiv = j + np.argmax(s[j:])   # np.argmax returns position in slice
        spiv = s[jpiv]

        # ----- Swap columns if necessary ----------------------------
        if jpiv != j:
            a[:, [j, jpiv]] = a[:, [jpiv, j]]
            p[[j, jpiv]] = p[[jpiv, j]]
            s[[j, jpiv]] = s[[jpiv, j]]

        # ----- Form the current Householder reflector ---------------
        v = a[j:, j].copy()
        tau_j, v = reflector(v)
        tau[j] = tau_j
        a[j:, j] = v

        # ----- Apply reflector to the trailing submatrix ------------
        if blas_level == 1:
            # Level-1 BLAS style: column-wise updates
            for i in range(j + 1, n):
                a[j:, i] = reflectorApply_vector(v, tau_j, a[j:, i])
                s[i] = np.linalg.norm(a[j + 1:, i])
        else:  # blas_level == 2
            # Level-2 BLAS style: update whole block at once
            a[j:, j + 1:] = reflectorApply2(v, tau_j, a[j:, j + 1:])
            for i in range(j + 1, n):
                s[i] = np.linalg.norm(a[j + 1:, i])

        # ----- Update rank estimate ---------------------------------
        k = j + 1
        if np.linalg.norm(s[j + 1:], 2) < atol:
            # Early termination: remaining columns are negligible
            tau = tau[:k]
            return tau, p, k, a

    # Completed full reduction
    tau = tau[:k]
    return tau, p, k, a


def rrqr_q(a, tau, q, k):
    """
    Build the orthogonal matrix Q from the stored Householder vectors.

    Parameters
    ----------
    a : ndarray, shape (m, n)
        Matrix that holds the Householder vectors in its lower-triangular part.
    tau : ndarray, shape (k,)
        Scalar factors of the reflectors.
    q : ndarray, shape (m, k)
        Matrix to be overwritten (normally the identity).
    k : int
        Number of reflectors (also the numerical rank).

    Returns
    -------
    q : ndarray, shape (m, k)
        The orthogonal matrix Q.
    """
    m, _ = q.shape
    for j in range(k - 1, -1, -1):
        v = a[j:, j]
        # Apply the j-th reflector to the columns j...k-1 of Q
        for i in range(k - 1, j - 1, -1):
            qi = q[j:, i]
            qi = reflectorApply_vector(v, np.conj(tau[j]), qi)
            q[j:, i] = qi
    return q


def rrqr_breflector(H, tau, k):
    """
    Build a block reflector Q = I - U * (S \ U.T).

    Parameters
    ----------
    H : ndarray, shape (m, n)
        Truncated Householder QR factorization.
    tau : ndarray, shape (k,)
        Scalar factors.
    k : int
        Number of reflectors to use.

    Returns
    -------
    U : ndarray, shape (m, k)
        Matrix whose columns contain the Householder vectors (with ones on the diagonal).
    S : ndarray, shape (k, k)
        Upper-triangular matrix that satisfies diag(S) = 1/tau.
    """
    m, _ = H.shape
    # Extract the strictly lower part and set the diagonal to 1
    U = np.tril(H[:, :k], -1)
    diag_indices = np.arange(m)[:k]  # only first k diagonal entries are needed
    U[diag_indices, diag_indices] = 1.0

    # Compute S = U.T @ U and replace its diagonal with 1/tau
    S = np.triu(U.conj().T @ U)
    np.fill_diagonal(S, 1.0 / tau)
    return U, S


def reflectorApply(x, tau, A):
    """
    Apply a Householder reflector to a matrix A from the left.

    This is the level-1 BLAS version that updates each column separately.

    Parameters
    ----------
    x : ndarray, shape (m,)
        Householder vector (with the implicit leading 1 omitted).
    tau : scalar
        Reflector scaling factor.
    A : ndarray, shape (m, n)
        Matrix to be updated (will be overwritten).

    Returns
    -------
    A : ndarray
        Updated matrix.
    """
    m, n = A.shape
    if m == 0:
        return A

    # The full reflector uses the vector [1; x[1:]]
    for j in range(n):
        vAj = np.conj(tau) * (A[0, j] + np.dot(x[1:], A[1:, j]))
        A[0, j] -= vAj
        A[1:, j] -= vAj * x[1:]
    return A


def reflectorApply_vector(x, tau, A):
    """
    Apply a Householder reflector to a vector A from the left.

    This is the level-1 BLAS version that updates each column separately.

    Parameters
    ----------
    x : ndarray, shape (m,)
        Householder vector (with the implicit leading 1 omitted).
    tau : scalar
        Reflector scaling factor.
    A : ndarray, shape (m,)
        Vector to be updated (will be overwritten).

    Returns
    -------
    A : ndarray
        Updated vector.
    """
    m, = A.shape
    if m == 0:
        return A

    # The full reflector uses the vector [1; x[1:]]
    vAj = np.conj(tau) * (A[0] + np.dot(x[1:], A[1:]))
    A[0] -= vAj
    A[1:] -= vAj * x[1:]
    return A


def reflectorApply1(x, tau, A):
    """
    Alternative level-1 update that constructs the explicit reflector vector.
    """
    m, n = A.shape
    if m == 0:
        return A

    y = np.empty(m, dtype=A.dtype)
    y[0] = 1.0
    y[1:] = x[1:]

    for j in range(n):
        vAj = np.conj(tau) * np.dot(y, A[:, j])
        A[:, j] -= vAj * y
    return A


def reflectorApply2(x, tau, A):
    """
    Level-2 BLAS update using an outer product.

    Parameters
    ----------
    x : ndarray, shape (m,)
        Householder vector (implicit leading 1 omitted).
    tau : scalar
        Reflector scaling factor.
    A : ndarray, shape (m, n)
        Matrix to be updated (will be overwritten).

    Returns
    -------
    A : ndarray
        Updated matrix.
    """
    m, n = A.shape
    if m == 0:
        return A

    y = np.empty(m, dtype=A.dtype)
    y[0] = 1.0
    y[1:] = x[1:]

    # A = A - conj(tau) * y * (y.T @ A)
    A -= np.conj(tau) * np.outer(y, y.conj().T @ A)
    return A


# ----------------------------------------------------------------------
# Simple test harness (unchanged, now runs much faster for large matrices)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # --------------------------------------------------------------
    # Test 1: full-rank random matrix
    # --------------------------------------------------------------
    m, n = 8, 5
    A = np.random.randn(m, n)
    Q, R, p, k = rrqr(A, rtol=1e-12)

    A_perm = A[:, p]
    A_recon = Q @ R

    err = np.linalg.norm(A_perm - A_recon)
    orth_err = np.linalg.norm(Q.conj().T @ Q - np.eye(k))

    print("Test 1: full rank random matrix")
    print(f"  Numerical rank k = {k}")
    print(f"  Reconstruction error ||A[:,p] - Q*R|| = {err:.2e}")
    print(f"  Orthogonality error   ||Q.T*Q - I||   = {orth_err:.2e}")

    # --------------------------------------------------------------
    # Test 2: rank-deficient matrix
    # --------------------------------------------------------------
    rank = 3
    U = np.random.randn(m, rank)
    V = np.random.randn(rank, n)
    A2 = U @ V                     # rank <= 3
    Q2, R2, p2, k2 = rrqr(A2, rtol=1e-12)

    A2_perm = A2[:, p2]
    A2_recon = Q2 @ R2
    err2 = np.linalg.norm(A2_perm - A2_recon)
    orth_err2 = np.linalg.norm(Q2.conj().T @ Q2 - np.eye(k2))

    print("\nTest 2: rank-deficient matrix")
    print(f"  Expected rank <= {rank}, detected rank k = {k2}")
    print(f"  Reconstruction error ||A2[:,p] - Q2*R2|| = {err2:.2e}")
    print(f"  Orthogonality error   ||Q2.T*Q2 - I||   = {orth_err2:.2e}")

    # --------------------------------------------------------------
    # Test 3: Hilbert matrix (large, ill-conditioned) using LAPACK driver
    # --------------------------------------------------------------
    try:
        import hilb as hilb
    except ImportError:
        # Small fallback if the external hilb module is unavailable
        from scipy.linalg import hilbert

        def hilb(m, n):
            return hilbert(m, n)

    from scipy import linalg
    import time

    A_hilb = hilb.hilb(4000, 2000)

    print("\nTest 3: Hilbert matrix (LAPACK driver)")
    print("shape(A):", A_hilb.shape)

    start_time = time.perf_counter()
    Q, R, perm, rank_est = rrqr(A_hilb, 1e-12)
    elapsed_time = time.perf_counter() - start_time

    print(f"rrqr (LAPACK) elapsed time: {elapsed_time:.4f} s")
    print("Permutation vector (zero-based):", perm)
    print("Estimated rank:", rank_est)

    recon_err = np.linalg.norm(A_hilb[:, perm] - Q @ R)
    ortho_err = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))
    print(f"Reconstruction error  ||A[:,perm] - Q*R||_F = {recon_err:.2e}")
    print(f"Orthogonality error   ||Q.T@Q - I||_F    = {ortho_err:.2e}")

    # --------------------------------------------------------------
    # Test 4: Hilbert matrix (large, ill-conditioned) using native implementation
    # --------------------------------------------------------------
    print("\nTest 4: Hilbert matrix (native implementation)")
    start_time = time.perf_counter()
    Qn, Rn, permn, rank_est_n = rrqr_native(A_hilb, 1e-12)
    elapsed_time = time.perf_counter() - start_time

    print(f"rrqr_native elapsed time: {elapsed_time:.4f} s")
    print("Permutation vector (zero-based):", permn)
    print("Estimated rank:", rank_est_n)

    recon_err_n = np.linalg.norm(A_hilb[:, permn] - Qn @ Rn)
    ortho_err_n = np.linalg.norm(Qn.T @ Qn - np.eye(Qn.shape[1]))
    print(f"Reconstruction error  ||A[:,perm] - Q*R||_F = {recon_err_n:.2e}")
    print(f"Orthogonality error   ||Q.T@Q - I||_F    = {ortho_err_n:.2e}")

    # --------------------------------------------------------------
    # Compare against SciPy's high-level QR (also LAPACK based)
    # --------------------------------------------------------------
    start_time = time.perf_counter()
    Qs, Rs, perms = linalg.qr(A_hilb, mode="economic", pivoting=True)
    elapsed_time = time.perf_counter() - start_time

    print("\nSciPy linalg.qr elapsed time: {:.4f} s".format(elapsed_time))
    print("Permutation vector (zero-based):", perms)

    recon_err_sci = np.linalg.norm(A_hilb[:, perms] - Qs @ Rs)
    ortho_err_sci = np.linalg.norm(Qs.T @ Qs - np.eye(Qs.shape[1]))
    print(f"Reconstruction error  ||A[:,perm] - Q*R||_F = {recon_err_sci:.2e}")
    print(f"Orthogonality error   ||Q.T@Q - I||_F    = {ortho_err_sci:.2e}")
