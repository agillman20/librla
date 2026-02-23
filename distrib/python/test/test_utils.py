"""
test_utils.py - Shared utilities for librla tests

Matrix generators and helper functions used across all test files.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: NIST-PD
Assisted by: Claude Code (Anthropic)
"""

import numpy as np
import os
from scipy import linalg
from scipy.sparse import rand as sprand


# =============================================================================
# Matrix Generators (from make_mat)
# =============================================================================

def make_mat(m, n, flag_type):
    """
    Generate test matrices for ID benchmarking.

    Parameters
    ----------
    m : int
        Number of rows
    n : int
        Number of columns
    flag_type : str
        Matrix type: 'cifar', 'mnist', 'gaussexp', 'gmm', 'snn'

    Returns
    -------
    X : ndarray
        Generated matrix (normalized by column)

    Examples
    --------
    >>> # Generate Gaussian with exponential decay
    >>> X = make_mat(500, 300, 'gaussexp')

    >>> # Generate GMM matrix
    >>> X = make_mat(500, 300, 'gmm')

    >>> # Generate sparse neural network matrix
    >>> X = make_mat(500, 500, 'snn')
    """

    if flag_type == 'cifar':
        if n > m:
            raise ValueError('columns must be less than or equal to rows for CIFAR')
        if m > 3072:
            raise ValueError('too many rows! CIFAR-10 has 3072 features (32x32x3)')

        # Try to load CIFAR-10 data
        if os.path.isfile('exampledata/cifar10.mat'):
            from scipy.io import loadmat
            MM = loadmat('exampledata/cifar10.mat')
            A = MM['full_matrix'].T
            X = A[:m, np.random.permutation(60000)[:n]]
        elif os.path.isdir('../cifar/cifar-10-batches-py'):
            # Use Python CIFAR reader
            import sys
            sys.path.insert(0, '../cifar')
            from read_cifar import load_cifar10_batch

            # Load first batch
            batch_file = '../cifar/cifar-10-batches-py/data_batch_1'
            X_full, _ = load_cifar10_batch(batch_file)
            # X_full is 10000x3072, transpose to 3072x10000
            X_full = X_full.T
            X = X_full[:m, np.random.permutation(X_full.shape[1])[:n]]
        else:
            raise FileNotFoundError(
                "CIFAR-10 data not found. Need exampledata/cifar10.mat or ../cifar/"
            )

    elif flag_type == 'mnist':
        if n > m:
            raise ValueError('columns must be less than or equal to rows for MNIST')
        if m > 784:
            raise ValueError('too many rows! MNIST has 784 features (28x28)')

        if os.path.isfile('exampledata/mnist_mat.mat'):
            from scipy.io import loadmat
            MM = loadmat('exampledata/mnist_mat.mat')
            A = MM['mnist_mat'].T
            X = A[:m, np.random.permutation(60000)[:n]]
        else:
            raise FileNotFoundError(
                "MNIST data not found. Need exampledata/mnist_mat.mat"
            )

    elif flag_type == 'gaussexp':
        X = Matrix_Gaussian_exp(m)

    elif flag_type == 'gmm':
        X = Matrix_GMM(n, m)

    elif flag_type == 'snn':
        X = Matrix_SNN(n)

    else:
        raise ValueError(
            f"Unknown flag_type: {flag_type}. "
            "Valid types: cifar, mnist, gaussexp, gmm, snn"
        )

    # Normalize by column
    X = X / np.linalg.norm(X, axis=0, keepdims=True)

    return X


def Matrix_GMM(n, d):
    """
    Generate Gaussian Mixture Model matrix.

    Creates a matrix with k=100 clusters, each with m=n/k samples.
    Each cluster has a mean scaled by cluster index.

    Parameters
    ----------
    n : int
        Number of rows (samples)
    d : int
        Number of columns (features)

    Returns
    -------
    A : ndarray, shape (d, n)
        GMM matrix (transposed)
    """
    k = 100
    m = n // k

    A = np.random.randn(n, d)

    for i in range(min(k, d)):  # Only modify columns that exist
        I = slice((i * m), ((i + 1) * m))
        A[I, i] += 10 * (i + 1)

    return A.T


def Matrix_Gaussian_exp(n):
    """
    Generate matrix with Gaussian entries and exponentially decaying singular values.

    Singular values:
    - First 100: sv = 1
    - Remaining: sv = 0.8^k (minimum 1e-5)

    Parameters
    ----------
    n : int
        Matrix size (nxn)

    Returns
    -------
    A : ndarray, shape (n, n)
        Matrix with exponential SV decay
    """
    # Singular values decay fast
    m = 100
    sv = np.zeros(n)
    sv[:m] = 1.0
    sv[m:] = 0.8 ** np.arange(1, n - m + 1)
    sv[sv < 1e-5] = 1e-5

    # Generate random orthogonal matrices
    U, _ = np.linalg.qr(np.random.randn(n, n))
    V, _ = np.linalg.qr(np.random.randn(n, n))

    A = U @ np.diag(sv) @ V.T

    return A


def Matrix_SNN(n):
    """
    Generate Sparse Neural Network matrix.

    Sparse matrices (10% density) with singular value decay:
    - First 100: sv = 10/k
    - Remaining: sv = 1/k

    Parameters
    ----------
    n : int
        Matrix size (nxn)

    Returns
    -------
    A : ndarray, shape (n, n)
        Sparse neural network matrix (converted to dense)
    """
    from scipy.sparse import diags

    m = 100
    sv = np.zeros(n)
    sv[:m] = 10.0 / np.arange(1, m + 1)
    sv[m:] = 1.0 / np.arange(m + 1, n + 1)

    # Create sparse random matrices with 10% density
    U = sprand(n, n, density=0.1, format='csr')
    V = sprand(n, n, density=0.1, format='csr')

    # Use sparse diagonal matrix to keep computation sparse
    sv_sparse = diags(sv, 0, format='csr')
    A = U @ sv_sparse @ V.T

    return A.toarray()  # Convert to dense for compatibility


# =============================================================================
# Additional Matrix Generators (from demo_utils)
# =============================================================================

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
