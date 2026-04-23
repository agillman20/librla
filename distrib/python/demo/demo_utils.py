"""
demo_utils.py - Shared utilities for librla demos

Matrix generators and helper functions used across all demos.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: NIST-PD
Version: 1.0.1
Date: April 22, 2026
Assisted by: Claude Code (Anthropic)
"""

import numpy as np
from scipy import linalg


# =============================================================================
# Matrix Generators
# =============================================================================

def hilbert(m, n=None):
    """
    Generate m x n Hilbert matrix.

    The Hilbert matrix is severely ill-conditioned, with entries H[i,j] = 1/(i+j-1).
    Useful for testing numerical stability.

    Parameters
    ----------
    m : int
        Number of rows
    n : int, optional
        Number of columns (default: m)

    Returns
    -------
    H : ndarray, shape (m, n)
        Hilbert matrix
    """
    if n is None:
        n = m
    i = np.arange(1, m + 1)[:, None]
    j = np.arange(1, n + 1)[None, :]
    return 1.0 / (i + j - 1)


def kahan(m, n=None, theta=1.2, pert=25):
    """
    Generate m x n Kahan matrix.

    Upper triangular matrix with exponentially decaying rows.
    Classic test for QR factorization algorithms.

    Parameters
    ----------
    m : int
        Number of rows
    n : int, optional
        Number of columns (default: m)
    theta : float, optional
        Angle parameter controlling condition number (default: 1.2)
    pert : float, optional
        Diagonal perturbation parameter (default: 25)

    Returns
    -------
    K : ndarray, shape (m, n)
        Kahan matrix
    """
    if n is None:
        n = m

    s = np.sin(theta)
    c = np.cos(theta)
    eps = np.finfo(float).eps

    K = np.zeros((m, n))
    r = min(m, n)

    # Set diagonal
    for i in range(r):
        K[i, i] = 1.0

    # Set upper triangular part
    for i in range(m):
        for j in range(i + 1, n):
            K[i, j] = -c

    # Scale rows by s^i
    for i in range(m):
        K[i, :] *= (s ** i)

    # Add diagonal perturbation
    for i in range(r):
        K[i, i] += pert * eps * (r - i)

    return K


def lowrank(m, n, k, decay='exponential', gap=100.0):
    """
    Generate m x n matrix with controlled rank-k structure.

    Creates a matrix where the first k singular values are well-separated
    from the remaining ones. Useful for testing rank detection.

    Parameters
    ----------
    m : int
        Number of rows
    n : int
        Number of columns
    k : int
        Target numerical rank
    decay : str, optional
        Singular value decay pattern: 'exponential', 'polynomial', 'step'
    gap : float, optional
        Ratio between s[k-1] and s[k] (default: 100)

    Returns
    -------
    A : ndarray, shape (m, n)
        Low-rank matrix
    s : ndarray
        True singular values
    """
    r = min(m, n)

    if decay == 'exponential':
        # Exponential decay within first k, then sharp drop
        s = np.concatenate([
            np.logspace(0, -2, k),
            np.logspace(-2, -10, r - k) / gap
        ])
    elif decay == 'polynomial':
        # Polynomial decay: s[i] = 1/(i+1)^2
        s = np.concatenate([
            1.0 / (np.arange(1, k + 1) ** 2),
            1.0 / (np.arange(k + 1, r + 1) ** 2) / gap
        ])
    elif decay == 'step':
        # Step function: first k are 1, rest are 1/gap
        s = np.concatenate([
            np.ones(k),
            np.ones(r - k) / gap
        ])
    else:
        raise ValueError(f"Unknown decay type: {decay}")

    # Generate random orthogonal factors
    U = linalg.orth(np.random.randn(m, r))
    V = linalg.orth(np.random.randn(n, r))

    A = U @ np.diag(s) @ V.T
    return A, s


def random_matrix(m, n, seed=None):
    """
    Generate m x n random Gaussian matrix.

    Parameters
    ----------
    m : int
        Number of rows
    n : int
        Number of columns
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    A : ndarray, shape (m, n)
        Random matrix with entries ~ N(0,1)
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(m, n)


# =============================================================================
# Error Computation
# =============================================================================

def id_error(A, k, piv, T):
    """
    Compute relative ID reconstruction error.

    The ID approximation is: A[:, piv[k:]] ≈ A[:, piv[:k]] @ T

    Parameters
    ----------
    A : ndarray
        Original matrix
    k : int
        Rank of approximation
    piv : ndarray
        Column permutation
    T : ndarray
        Interpolation matrix

    Returns
    -------
    error : float
        Relative Frobenius norm error: ||A_skel - A_basis @ T|| / ||A||
    """
    A_basis = A[:, piv[:k]]
    A_skel = A[:, piv[k:]]
    return np.linalg.norm(A_skel - A_basis @ T, 'fro') / np.linalg.norm(A, 'fro')


def svd_error(A, U, s, Vh):
    """
    Compute relative SVD reconstruction error.

    Parameters
    ----------
    A : ndarray
        Original matrix
    U : ndarray
        Left singular vectors
    s : ndarray
        Singular values
    Vh : ndarray
        Right singular vectors (transposed)

    Returns
    -------
    error : float
        Relative Frobenius norm error: ||A - U @ diag(s) @ Vh|| / ||A||
    """
    A_approx = U @ np.diag(s) @ Vh
    return np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')


# =============================================================================
# Display Helpers
# =============================================================================

def print_header(title, width=70):
    """Print formatted section header."""
    print("=" * width)
    print(title)
    print("=" * width)


def print_subheader(title, width=70):
    """Print formatted subsection header."""
    print(f"\n{title}")
    print("-" * width)


def print_matrix_info(A, name="A"):
    """Print basic matrix information."""
    m, n = A.shape
    normA = np.linalg.norm(A, 'fro')
    print(f"  {name}: {m} x {n}, ||{name}||_F = {normA:.3e}")


def print_id_result(name, k, error, max_T, time_sec):
    """Print ID result in consistent format."""
    print(f"  {name}:")
    print(f"    Rank:     {k}")
    print(f"    Error:    {error:.3e}")
    print(f"    Max |T|:  {max_T:.3e}")
    print(f"    Time:     {time_sec:.4f} s")


def print_svd_result(name, k, error, time_sec):
    """Print SVD result in consistent format."""
    print(f"  {name}:")
    print(f"    Rank:     {k}")
    print(f"    Error:    {error:.3e}")
    print(f"    Time:     {time_sec:.4f} s")
