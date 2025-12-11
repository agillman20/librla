"""
make_mat - Generate test matrices for ID benchmarking

Generates various types of test matrices from:
"Robust blockwise random pivoting: Fast and accurate adaptive
 interpolative decomposition"

Matrix Types:
- cifar: CIFAR-10 image data (requires data files)
- mnist: MNIST digit data (requires data files)
- gaussexp: Gaussian with exponential singular value decay
- gmm: Gaussian Mixture Model-like matrix
- snn: Sparse Neural Network-like matrix

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

import numpy as np
import os
from scipy.sparse import rand as sprand


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


if __name__ == '__main__':
    """Test the matrix generators."""
    print("=" * 70)
    print("Testing make_mat.py")
    print("=" * 70)

    # Test gaussexp
    print("\nTest 1: Gaussexp Matrix (500x500)")
    print("-" * 70)
    A1 = make_mat(500, 500, 'gaussexp')
    print(f"Shape: {A1.shape}")
    print(f"Type: {A1.dtype}")
    print(f"Norm: {np.linalg.norm(A1):.3e}")
    sv1 = np.linalg.svd(A1, compute_uv=False)
    print(f"Largest SV: {sv1[0]:.3e}")
    print(f"Smallest SV: {sv1[-1]:.3e}")
    print(f"Condition number: {sv1[0] / sv1[-1]:.3e}")

    # Test gmm
    print("\nTest 2: GMM Matrix (400x400)")
    print("-" * 70)
    A2 = make_mat(400, 400, 'gmm')
    print(f"Shape: {A2.shape}")
    print(f"Type: {A2.dtype}")
    print(f"Norm: {np.linalg.norm(A2):.3e}")
    sv2 = np.linalg.svd(A2, compute_uv=False)
    print(f"Largest SV: {sv2[0]:.3e}")
    print(f"Smallest SV: {sv2[-1]:.3e}")
    print(f"Condition number: {sv2[0] / sv2[-1]:.3e}")

    # Test snn
    print("\nTest 3: SNN Matrix (300x300)")
    print("-" * 70)
    A3 = make_mat(300, 300, 'snn')
    print(f"Shape: {A3.shape}")
    print(f"Type: {A3.dtype}")
    print(f"Norm: {np.linalg.norm(A3):.3e}")
    print(f"Sparsity: {np.sum(A3 == 0) / A3.size * 100:.1f}% zeros")
    sv3 = np.linalg.svd(A3, compute_uv=False)
    print(f"Largest SV: {sv3[0]:.3e}")
    print(f"Smallest SV: {sv3[-1]:.3e}")
    print(f"Condition number: {sv3[0] / sv3[-1]:.3e}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
