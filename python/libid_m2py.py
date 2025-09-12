"""
Description
-----------
This module provides a class ``libid`` that implements several randomized
linear‑algebra routines that approximate the rank, QR factorization,
singular‑value decomposition (SVD), and interpolative decomposition (ID)
of a matrix ``A``.

User‑callable methods
---------------------
range_randomized   - Build an orthonormal basis for the column space.
rrqr_randomized    - Rank‑revealing QR using a randomized basis.
rrsvd_randomized   - Truncated SVD using a randomized basis.
rrid_randomized    - Interpolative decomposition using randomized QR.
image_randomized   - Basis for the row space via transpose.

Author: Your Name
SPDX-License-Identifier: TBD
"""

import numpy as np
from typing import Tuple, Optional

from scipy import linalg
from numpy.linalg import norm


class libid:
    """
    Collection of static methods that perform randomized matrix factorizations.

    See also: rrid_randomized, rrqr_randomized, rrsvd_randomized,
              range_randomized, image_randomized
    """

    # -----------------------------------------------------------------
    # Public static methods
    # -----------------------------------------------------------------

    @staticmethod
    def range_randomized(A: np.ndarray,
                         rtol: float,
                         block_size: int = 42,
                         flag_power: int = 0) -> Tuple[int, np.ndarray]:
        """
        RANGE_RANDOMIZED Compute an orthonormal basis for the column space of ``A`` using random sampling.

        Description
        -----------
        Build an orthonormal basis for the column space of ``A`` using random
        sampling and optional power iterations.  The routine stops when the
        relative residual falls below ``rtol``.

        Parameters
        ----------
        A : ndarray
            Input matrix.
        rtol : float
            Relative tolerance that determines when to stop sampling.
        block_size : int, optional
            Initial number of random vectors (default = 42).
        flag_power : int, optional
            Number of power‑iteration steps (default = 0).

        Returns
        -------
        k : int
            Number of basis vectors found (may equal ``min(m,n)``).
        Q : ndarray
            Orthonormal basis matrix with size ``(m,k)``.  If ``k == 0`` the
            array is empty.

        Notes
        -----
        1. If the initial block already covers the whole space, the function
           returns early.
        2. The random matrix ``X`` has entries in ``[-1,1]``.
        3. Power iteration is performed by the private method ``_power_iteration``.

        Example
        -------
        >>> A = libid._hilb(4000, 2000)
        >>> k, Q = libid.range_randomized(A, 1e-8)
        >>> print(k, Q.shape)

        Code flow
        ---------
        1. Determine matrix dimensions.
        2. Check early‑exit condition.
        3. Loop:
           a) Generate random test matrix ``X``.
           b) Apply power iteration (if ``flag_power > 0``).
           c) Form ``Y = A @ X`` and compute its QR factorization.
           d) Estimate residual and compare with ``rtol``.
           e) Increase block size if needed.
        4. Return block size and orthonormal basis ``Q``.
        """
        m, n = A.shape

        # If the initial block already covers the whole space, return early.
        if block_size >= min(m, n):
            k = min(m, n)
            Q = np.empty((m, 0), dtype=A.dtype)
            return k, Q

        while True:
            # Random matrix with entries in [-1, 1]
            X = 2 * np.random.uniform(size=(n, block_size)) - 1
            X = libid._power_iteration(A, X, flag_power)

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
                return min(m, n), np.empty((m, 0), dtype=A.dtype)


    @staticmethod
    def rrqr_randomized(A: np.ndarray,
                        rtol: float,
                        block_size: int = 42,
                        flag_power: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        RRQR_RANDOMIZED Rank‑revealing QR factorization using a randomized basis.

        Description
        -----------
        Compute a rank‑revealing QR factorization of ``A`` by first building a
        randomized orthonormal basis for the column space.  If the matrix
        is effectively full rank, a deterministic QR is performed.

        Parameters
        ----------
        A : ndarray
            Input matrix.
        rtol : float
            Relative tolerance for rank determination.
        block_size : int, optional
            Initial number of random vectors (default = 42).
        flag_power : int, optional
            Number of power‑iteration steps (default = 0).

        Returns
        -------
        Qk : ndarray
            Leading ``k`` columns of the orthogonal factor.
        Rk : ndarray
            Leading ``k`` rows of the upper‑triangular factor.
        p : ndarray
            Pivot permutation vector (identity if pivoting not used).

        Notes
        -----
        1. The rank ``k`` is chosen as the number of rows of ``R`` whose 2‑norm
           exceeds ``rtol * ||A||`` (or ``||A_proj||`` for the projected case).
        2. The private method ``_power_iteration`` is used internally.

        Example
        -------
        >>> A = libid._hilb(4000, 2000)
        >>> Q, R, p = libid.rrqr_randomized(A, 1e-15)
        >>> rel_err = np.linalg.norm(Q @ R - A[:, p], ord='fro') / np.linalg.norm(A, ord='fro')
        >>> print(rel_err)

        Code flow
        ---------
        1. Call ``range_randomized`` to obtain basis ``Q_basis``.
        2. If full rank, compute deterministic QR of ``A``.
        3. Otherwise project ``A`` onto the basis and QR the small matrix.
        4. Determine numerical rank ``k`` from ``R``.
        5. Return truncated factors and pivot vector.
        """
        m, n = A.shape
        k, Q_basis = libid.range_randomized(A, rtol, block_size, flag_power)

        if k >= min(m, n):
            Q, R, p = np.linalg.qr(A, mode='economic', pivoting=True)
            # Determine numerical rank
            k = np.sum(np.linalg.norm(R, axis=1) >= rtol * np.linalg.norm(A, 'fro'))
            Qk = Q[:, :k]
            Rk = R[:k, :]
            return Qk, Rk, p

        # Project A onto the basis and factor the small matrix.
        A_proj = Q_basis.T @ A
        Q_proj, R, p = linalg.qr(A_proj, mode='economic', pivoting=True)
        Qk = Q_basis @ Q_proj
        k = np.sum(np.linalg.norm(R, axis=1) >= rtol * np.linalg.norm(A_proj, 'fro'))
        Qk = Qk[:, :k]
        Rk = R[:k, :]
        return Qk, Rk, p

    @staticmethod
    def rrsvd_randomized(A: np.ndarray,
                         rtol: float,
                         block_size: int = 42,
                         flag_power: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        RRSVD_RANDOMIZED Truncated singular‑value decomposition using a randomized basis.

        Description
        -----------
        Compute a truncated SVD of ``A`` by first constructing a randomized
        orthonormal basis for the column space.  If ``A`` is effectively full
        rank, the full deterministic SVD is performed.

        Parameters
        ----------
        A : ndarray
            Input matrix.
        rtol : float
            Relative tolerance for truncation.
        block_size : int, optional
            Initial number of random vectors (default = 42).
        flag_power : int, optional
            Number of power‑iteration steps (default = 0).

        Returns
        -------
        Uk : ndarray
            Leading ``k`` left singular vectors.
        sk : ndarray
            Leading ``k`` singular values.
        Vk : ndarray
            Leading ``k`` right singular vectors (rows of ``V.T``).

        Notes
        -----
        1. The rank ``k`` is the number of singular values greater than
           ``rtol * ||A||`` (or ``||A_proj||`` for the projected case).

        Example
        -------
        >>> A = libid._hilb(4000, 2000)
        >>> U, s, Vt = libid.rrsvd_randomized(A, 1e-15)
        >>> rel_err = np.linalg.norm(U @ np.diag(s) @ Vt - A, ord='fro') / np.linalg.norm(A, 'fro')
        >>> print(rel_err)

        Code flow
        ---------
        1. Obtain basis ``Q_basis`` via ``range_randomized``.
        2. If full rank, call deterministic ``svd``.
        3. Otherwise form ``A_proj = Q_basis.T @ A``.
        4. Compute ``svd`` of the small matrix.
        5. Lift left singular vectors back: ``U = Q_basis @ U_proj``.
        6. Truncate to ``k`` based on ``rtol``.
        """
        m, n = A.shape
        k, Q_basis = libid.range_randomized(A, rtol, block_size, flag_power)

        if k >= min(m, n):
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            k = np.sum(np.abs(S) >= rtol * np.linalg.norm(A, 'fro'))
            Uk = U[:, :k]
            sk = S[:k]
            Vk = Vt[:k, :]
            return Uk, sk, Vk

        A_proj = Q_basis.T @ A
        U_proj, S_proj, Vt_proj = np.linalg.svd(A_proj, full_matrices=False)
        k = np.sum(np.abs(S_proj) >= rtol * np.linalg.norm(A_proj, 'fro'))

        Uk = Q_basis @ U_proj[:, :k]
        sk = S_proj[:k]
        Vk = Vt_proj[:k, :]
        return Uk, sk, Vk

    @staticmethod
    def rrid_randomized(A: np.ndarray,
                        rtol: float,
                        block_size: int = 42,
                        flag_power: int = 0) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        RRID_RANDOMIZED Interpolative decomposition using a randomized QR factorization.

        Description
        -----------
        Form an interpolative decomposition (ID) of ``A`` by first computing a
        randomized rank‑revealing QR and then solving a triangular system to
        obtain the interpolation matrix.

        Parameters
        ----------
        A : ndarray
            Input matrix.
        rtol : float
            Relative tolerance for rank determination.
        block_size : int, optional
            Initial number of random vectors (default = 42).
        flag_power : int, optional
            Number of power‑iteration steps (default = 0).

        Returns
        -------
        k : int
            Numerical rank (size of leading block of ``R``).
        p : ndarray
            Pivot permutation vector.
        proj : ndarray
            Interpolation matrix such that ``A[:, p] ≈ A[:, p[:k]] @ proj``.

        Notes
        -----
        1. The method uses the private ``_power_iteration`` routine indirectly
           through ``rrqr_randomized``.

        Example
        -------
        >>> A = libid._hilb(4000, 2000)
        >>> k, p, proj = libid.rrid_randomized(A, 1e-8)
        >>> err = np.linalg.norm(A[:, p[k:]] - A[:, p[:k]] @ proj, ord='fro')
        >>> print(err)

        Code flow
        ---------
        1. Call ``rrqr_randomized`` to obtain ``Q``, ``R``, and pivot vector ``p``.
        2. Extract ``R11`` (upper‑triangular leading block) and ``R12``.
        3. Solve ``R11 * X = R12`` for the interpolation matrix ``X``.
        4. Return rank ``k``, pivot vector, and ``X``.
        """
        Q, R, p = libid.rrqr_randomized(A, rtol, block_size, flag_power)
        k = R.shape[0]

        # Solve R11 * X = R12 for X, where R = [R11 R12].
        R11 = np.triu(R[:k, :k])
        R12 = R[:k, k:]
        proj = np.linalg.solve(R11, R12)
        return k, p, proj

    @staticmethod
    def image_randomized(A: np.ndarray,
                         rtol: float,
                         block_size: int = 42,
                         flag_power: int = 0) -> Tuple[int, np.ndarray]:
        """
        IMAGE_RANDOMIZED Compute a basis for the row space of ``A`` by applying ``range_randomized`` to the transpose.

        Description
        -----------
        A thin wrapper that calls ``range_randomized`` on ``A.T`` to obtain
        a basis for the row space.

        Parameters
        ----------
        A : ndarray
            Input matrix.
        rtol : float
            Relative tolerance for stopping criterion.
        block_size : int, optional
            Initial number of random vectors (default = 42).
        flag_power : int, optional
            Number of power‑iteration steps (default = 0).

        Returns
        -------
        k : int
            Number of basis vectors found.
        Q : ndarray
            Orthonormal basis for the row space (size ``n-by-k``).

        Example
        -------
        >>> A = libid._hilb(4000, 2000)
        >>> k, Q = libid.image_randomized(A, 1e-8)
        >>> print(k, Q.shape)

        See also: range_randomized
        """
        k, Q = libid.range_randomized(A.T, rtol, block_size, flag_power)
        return k, Q

    # -----------------------------------------------------------------
    # Private static helper methods
    # -----------------------------------------------------------------

    @staticmethod
    def _hilb(m: int, n: Optional[int] = None) -> np.ndarray:
        """
        HILB Generate a Hilbert matrix of size ``m``‑by‑``n`` (or square if ``n`` omitted).

        Parameters
        ----------
        m : int
            Number of rows.
        n : int, optional
            Number of columns (default = ``m``).

        Returns
        -------
        a : ndarray
            Hilbert matrix.
        """
        if n is None:
            n = m
        i = np.arange(1, n + 1)
        j = np.arange(1, m + 1).reshape(-1, 1)
        a = 1.0 / (i + j - 1)
        return a

    @staticmethod
    def _power_iteration(A: np.ndarray,
                         X: np.ndarray,
                         power: int = 0) -> np.ndarray:
        """
        POWERITERATION Apply power iteration to improve the quality of the sampling matrix.

        Description
        -----------
        Multiply the test matrix ``X`` by ``A`` and ``A.T`` repeatedly to amplify the
        dominant singular directions.  After each iteration a QR factorization
        re‑orthogonalizes ``X``.

        Parameters
        ----------
        A : ndarray
            Input matrix.
        X : ndarray
            Random test matrix.
        power : int, optional
            Number of power‑iteration steps (default = 0).

        Returns
        -------
        X : ndarray
            Updated test matrix after power iteration.

        Notes
        -----
        This routine is used internally by ``range_randomized``.

        Example
        -------
        >>> A = libid._hilb(400, 200)
        >>> X = np.random.rand(200, 42) * 2 - 1
        >>> X = libid._power_iteration(A, X, 2)
        """
        for _ in range(power):
            X = A.T @ (A @ X)
            X, _, _ = np.linalg.qr(X, mode='economic', pivoting=True)
        return X



if __name__ == "__main__":
    # Simple sanity checks for the public API.
    np.random.seed(0)

    # Small test matrix.
    m, n = 4000, 2000
    A = libid._hilb(m, n)

    # --------------------------------------------------------------
    # Test range_randomized
    # --------------------------------------------------------------
    k_range, Q_range = libid.range_randomized(A, rtol=1e-12)
    orth_err = np.linalg.norm(Q_range.T @ Q_range - np.eye(k_range))
    print(f"range_randomized: k={k_range}, basis shape={Q_range.shape}")
    print(f"range_randomized: k={k_range}, orthonormality error={orth_err:e}")

    # --------------------------------------------------------------
    # Test rrqr_randomized
    # --------------------------------------------------------------
    Q_rrqr, R_rrqr, piv = libid.rrqr_randomized(A, rtol=1e-12)
    A_perm = A[:, piv]
    recon_err = np.linalg.norm(Q_rrqr @ R_rrqr - A_perm) / np.linalg.norm(A_perm)
    print(f"rrqr_randomized: reconstruction relative error={recon_err:e}")

    # --------------------------------------------------------------
    # Test rrsvd_randomized
    # --------------------------------------------------------------
    U_rrsvd, s_rrsvd, Vt_rrsvd = libid.rrsvd_randomized(A, rtol=1e-12)
    A_svd = U_rrsvd @ np.diag(s_rrsvd) @ Vt_rrsvd
    svd_err = np.linalg.norm(A_svd - A) / np.linalg.norm(A)
    print(f"rrsvd_randomized: reconstruction relative error={svd_err:e}")

    # --------------------------------------------------------------
    # Test rrid_randomized
    # --------------------------------------------------------------
    k_id, piv_id, proj_id = libid.rrid_randomized(A, rtol=1e-12)
    A_id_approx = A[:, piv_id[:k_id]] @ proj_id
    id_err = np.linalg.norm(A[:, piv_id[k_id:]] - A_id_approx) / np.linalg.norm(A)
    print(f"rrid_randomized: interpolation relative error={id_err:e}")

    # --------------------------------------------------------------
    # Test image_randomized
    # --------------------------------------------------------------
    k_img, Q_img = libid.image_randomized(A, rtol=1e-12)
    print(f"image_randomized: k={k_img}, basis shape={Q_img.shape}")
    
