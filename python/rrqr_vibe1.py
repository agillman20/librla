"""
rrqr.py

Implementation of the Businger-Golub QR factorization with column pivoting.
The code follows the structure of the original MATLAB version and works
with real or complex NumPy arrays.

Functions (in the same order as the MATLAB file):
    rrqr          - main driver, returns Q, R, p, k
    rrqr_piv      - pivoted Householder reduction
    rrqr_q        - builds Q from the stored reflectors
    rrqr_breflector - builds a block reflector (not used by rrqr)
    copysign_matlab - sign-preserving copy of a scalar
    reflector     - creates a Householder reflector (tau and vector)
    reflectorApply - level-1 update: apply a reflector to a matrix (left)
    reflectorApply1 - alternative level-1 update using an explicit vector
    reflectorApply2 - level-2 update using an outer product
"""

import numpy as np
import math


def rrqr(A, rtol):
    """
    Compute the QR factorization with column pivoting.

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
    Build a block reflector Q = I - U * (S \\ U.T).

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


def copysign_matlab(x, y):
    """
    MATLAB-style copysign: return |x| with the sign of y.
    """
    return math.copysign(abs(x), y)


def reflector(x):
    """
    Construct a Householder reflector for a vector x.

    Returns
    -------
    tau : scalar
        The reflector scaling factor.
    x : ndarray
        The vector is overwritten so that x[0] = -nu and the rest contains
        the scaled Householder vector.
    """
    n = x.shape[0]
    if n == 0:
        return np.array(0, dtype=x.dtype), x

    xi = x[0]
    norm_u = np.linalg.norm(x)

    if norm_u == 0:
        return np.array(0, dtype=x.dtype), x

    # nu has the same sign as xi (real part) and magnitude norm_u
    nu = copysign_matlab(norm_u, np.real(xi))

    # Update the first entry and scale the rest
    x[0] = -nu
    if n > 1:
        x[1:] = x[1:] / (xi + nu)

    tau = (xi + nu) / nu
    return tau, x


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
# Simple test harness
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # Test 1: full rank random matrix
    m, n = 8, 5
    A = np.random.randn(m, n)
    Q, R, p, k = rrqr(A, rtol=1e-12)

    # Reorder columns according to the permutation vector
    A_perm = A[:, p]

    # Reconstruct A from Q and R
    A_recon = Q @ R

    err = np.linalg.norm(A_perm - A_recon)
    orth_err = np.linalg.norm(Q.conj().T @ Q - np.eye(k))

    print("Test 1: full rank random matrix")
    print(f"  Numerical rank k = {k}")
    print(f"  Reconstruction error ||A[:,p] - Q*R|| = {err:.2e}")
    print(f"  Orthogonality error ||Q.T*Q - I|| = {orth_err:.2e}")

    # Test 2: rank-deficient matrix
    rank = 3
    U = np.random.randn(m, rank)
    V = np.random.randn(rank, n)
    A2 = U @ V  # rank-deficient (rank <= 3)
    Q2, R2, p2, k2 = rrqr(A2, rtol=1e-12)

    A2_perm = A2[:, p2]
    A2_recon = Q2 @ R2
    err2 = np.linalg.norm(A2_perm - A2_recon)
    orth_err2 = np.linalg.norm(Q2.conj().T @ Q2 - np.eye(k2))

    print("\nTest 2: rank-deficient matrix")
    print(f"  Expected rank <= {rank}, detected rank k = {k2}")
    print(f"  Reconstruction error ||A2[:,p] - Q2*R2|| = {err2:.2e}")
    print(f"  Orthogonality error ||Q2.T*Q2 - I|| = {orth_err2:.2e}")

    # Test 3: Hilbert matrix

    import hilb as hilb
    from scipy import linalg
    import time

    A = hilb.hilb(400, 200)

    print("\nTest 3: Hilbert matrix")
    print("shape(A):", A.shape)

    start_time = time.perf_counter()

    # Test the basic routine.
    Q, R, perm, rank_est = rrqr(A, 1e-12)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"rrqr, elapsed time: {elapsed_time:.4f} seconds")

    print("Permutation vector (zero-based):", perm)
    print("Estimated rank:", rank_est)

    recon_err = np.linalg.norm(A[:, perm] - Q @ R)
    print("Reconstruction error  ||A[:,perm] - Q*R||_F =", recon_err)

    ortho_err = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))
    print("Orthogonality error  ||Q.T@Q - I||_F =", ortho_err)

    start_time = time.perf_counter()

    # Test the basic routine.
    Q, R, perm = linalg.qr(A, mode='economic', pivoting=True)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"linalg.qr, elapsed time: {elapsed_time:.4f} seconds")

    print("Permutation vector (zero-based):", perm)

    recon_err = np.linalg.norm(A[:, perm] - Q @ R)
    print("Reconstruction error  ||A[:,perm] - Q*R||_F =", recon_err)

    ortho_err = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))
    print("Orthogonality error  ||Q.T@Q - I||_F =", ortho_err)
