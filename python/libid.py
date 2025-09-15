"""
Description
-----------

This module implements several randomized linear-algebra routines that
approximate the column space, QR factorization, singular-value decomposition
(SVD), and interpolative decomposition (ID) of a matrix ``A``.

User-callable methods
---------------------
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


def _power_iteration(A,x,flag_power=0):

    for j in range(flag_power):
        x = A.T @ (A @ x)
        x, _R, _p = linalg.qr(x, mode='economic', pivoting=True)
    return x


def orth_sketch(A,rtol,block_size=42,flag_power=0):

    m, n = A.shape

    if (block_size >= min(m,n)):
        return min(m,n), np.empty_like(A, shape=(0, 0))
    
    while 1:
        x = 2*np.random.uniform(size=(n, block_size))-1
        x = _power_iteration(A,x,flag_power)
        y = A @ x
        Q, R, p = linalg.qr(y, mode='economic', pivoting=True)
        r = R.diagonal()
        d = max(abs(r[-1:]))/max(norm(y,axis=0))

        if (d <= rtol): 
            return block_size, Q

        if (d > rtol): 
            block_size = min(block_size*4,min(m,n))

        if (block_size >= min(m,n)):
            return min(m,n), np.empty_like(A, shape=(0, 0))


def rrqr_randomized(A,rtol,block_size=42,flag_power=0):

    m, n = A.shape
    k, q = orth_sketch(A,rtol,block_size,flag_power)

    if (k >= min(m,n)):
        Q, R, p = linalg.qr(A, mode='economic', pivoting=True)
        k = sum(norm(R,axis=1) >= rtol*norm(A))
        return Q[:,:k],R[:k,:],p
  
    Aproj = q.T @ A
    Qproj, R, p = linalg.qr(Aproj, mode='economic', pivoting=True)   
    Q = q @ Qproj
    k = sum(norm(R,axis=1) >= rtol*norm(Aproj))
    return Q[:,:k],R[:k,:],p


def rrsvd_randomized(A,rtol,block_size=42,flag_power=0):

    m, n = A.shape
    k, q = orth_sketch(A,rtol,block_size,flag_power)

    if (k >= min(m,n)):
        U, s, V = linalg.svd(A,full_matrices=False)
        k = sum(abs(s) >= rtol*norm(A))
        return U[:,:k],s[:k],V[:k,:]
  
    Aproj = q.T @ A
    Uproj, s, V = linalg.svd(Aproj,full_matrices=False)
    U = q @ Uproj
    k = sum(abs(s) >= rtol*norm(Aproj))
    return U[:,:k],s[:k],V[:k,:]


def rrid_randomized(A,rtol,block_size=42,flag_power=0):

    Q, R, p = rrqr_randomized(A,rtol,block_size,flag_power)
    k = R.shape[0]
    T = linalg.solve(np.triu(R[:k,:k]), R[:,k:])
    return k, p, T




def _hilb(n: int, m: int) -> np.ndarray:
    """
    Creates an n x m Hilbert matrix using NumPy/SciPy.

    Args:
        n (int): The order of the Hilbert matrix.
        m (int): other direction

    Returns:
        numpy.ndarray: The n x m Hilbert matrix.
    """

    # Optimized version, via scipy.linalg.hankel
    c = np.zeros(n)
    r = np.zeros(m)

    for i in range(n):
        c[i] = 1.0 / (i + 1)  # Adjust for 0-based indexing

    for i in range(m):
        r[i] = 1.0 / (i + n)  # Adjust for 0-based indexing

    return linalg.hankel(c,r)


if __name__ == "__main__":
    # Simple sanity checks for the public API.
    np.random.seed(0)

    # Small test matrix.
    m, n = 4000, 2000
    A = _hilb(m, n)

    # --------------------------------------------------------------
    # Test range_randomized
    # --------------------------------------------------------------
    k_range, Q_range = orth_sketch(A, rtol=1e-12)
    orth_err = np.linalg.norm(Q_range.T @ Q_range - np.eye(k_range))
    print(f"orth_sketch: k={k_range}, basis shape={Q_range.shape}")
    print(f"orth_sketch: k={k_range}, orthonormality error={orth_err:e}")

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

