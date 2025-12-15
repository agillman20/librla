"""
PyTorch-compatible wrappers for librla
======================================

Thin wrappers providing torch.svd_lowrank and torch.pca_lowrank
compatible interfaces using librla's randomized algorithms.

Usage::

    from torch_compat import svd_lowrank, pca_lowrank

    U, s, V = svd_lowrank(A, q=10, niter=2)
    U, s, V = pca_lowrank(A, q=10, center=True, niter=2)

Note: q is the oversampled rank (sketch size), not the final rank.
User is responsible for choosing q = k + oversampling where k is
the target rank. Typical oversampling is 5-10.

Reference: Halko et al., "Finding structure with randomness" (2009)
"""

import numpy as np
from librla import svd_sketch


def svd_lowrank(A, q=6, niter=2, M=None):
    """PyTorch-compatible randomized low-rank SVD.

    Parameters
    ----------
    A : ndarray
        Input matrix (m, n)
    q : int, optional
        Oversampled rank / sketch size (default: 6)
    niter : int, optional
        Power iterations (default: 2)
    M : ndarray, optional
        Matrix to subtract before decomposition

    Returns
    -------
    U : ndarray, shape (m, q)
        Left singular vectors
    s : ndarray, shape (q,)
        Singular values (1D array)
    V : ndarray, shape (n, q)
        Right singular vectors (not transposed)
    """
    if M is not None:
        A = A - M
    U, s, Vh = svd_sketch(A, rtol=q, power_iter=niter, extra_samples=0)
    return U, s, Vh.conj().T


def pca_lowrank(A, q=None, center=True, niter=2):
    """PyTorch-compatible randomized low-rank PCA.

    Parameters
    ----------
    A : ndarray
        Input matrix (m, n) - m samples, n features
    q : int, optional
        Oversampled rank / sketch size (default: min(6, m, n))
    center : bool, optional
        Subtract column means (default: True)
    niter : int, optional
        Power iterations (default: 2)

    Returns
    -------
    U : ndarray, shape (m, q)
        Left singular vectors
    s : ndarray, shape (q,)
        Singular values (1D array)
    V : ndarray, shape (n, q)
        Right singular vectors (not transposed)

    Notes
    -----
    The relation to PCA:
    - V columns are principal directions
    - A @ V[:, :k] projects data to first k principal components
    """
    m, n = A.shape
    if q is None:
        q = min(6, m, n)
    if center:
        A = A - A.mean(axis=0, keepdims=True)
    U, s, Vh = svd_sketch(A, rtol=q, power_iter=niter, extra_samples=0)
    return U, s, Vh.conj().T
