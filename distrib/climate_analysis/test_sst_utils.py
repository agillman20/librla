"""
test_sst_utils.py - Utilities for SST/EOF climate analysis

Statistical tests and helper functions for EOF analysis of climate data.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Assisted by: Claude Code (Anthropic)
"""

import numpy as np


def north_test(singular_values, n_samples, return_errors=False):
    """
    Apply North's rule of thumb to test EOF/SVD mode separability.

    North et al. (1982) showed that eigenvalues have sampling error:
        δλ_i ≈ λ_i * sqrt(2/N)

    where N is the number of independent samples. Two modes are considered
    "effectively degenerate" if their eigenvalues overlap within error bars:
        λ_i - λ_{i+1} < δλ_i + δλ_{i+1}

    For SVD, eigenvalues λ = σ² (squared singular values).

    Parameters
    ----------
    singular_values : array_like
        Singular values from SVD (σ, not σ²)
    n_samples : int
        Number of independent samples (typically number of time steps)
    return_errors : bool, optional
        If True, also return the eigenvalue errors

    Returns
    -------
    separable : ndarray of bool
        separable[i] is True if mode i is well-separated from mode i+1
    errors : ndarray (only if return_errors=True)
        Sampling errors δλ for each eigenvalue

    References
    ----------
    North, G. R., T. L. Bell, R. F. Cahalan, and F. J. Moeng, 1982:
    Sampling errors in the estimation of empirical orthogonal functions.
    Mon. Wea. Rev., 110, 699-706.

    Examples
    --------
    >>> s = np.array([10, 8, 7.9, 5, 3])  # singular values
    >>> sep = north_test(s, n_samples=100)
    >>> print(sep)  # [True, False, True, True] - modes 2,3 are degenerate
    """
    s = np.asarray(singular_values)
    eigenvalues = s ** 2

    # Sampling error: δλ_i = λ_i * sqrt(2/N)
    delta_lambda = eigenvalues * np.sqrt(2.0 / n_samples)

    # Test separation: λ_i - λ_{i+1} > δλ_i + δλ_{i+1}
    gap = eigenvalues[:-1] - eigenvalues[1:]
    error_sum = delta_lambda[:-1] + delta_lambda[1:]
    separable = gap > error_sum

    if return_errors:
        return separable, delta_lambda
    return separable


def print_north_test(singular_values, n_samples, n_modes=10):
    """
    Print North's rule of thumb results for EOF analysis.

    Parameters
    ----------
    singular_values : array_like
        Singular values from SVD
    n_samples : int
        Number of independent samples
    n_modes : int, optional
        Number of modes to display (default: 10)
    """
    s = np.asarray(singular_values)
    n_modes = min(n_modes, len(s) - 1)

    separable, errors = north_test(s, n_samples, return_errors=True)
    eigenvalues = s ** 2
    total_var = np.sum(s ** 2)
    var_pct = 100 * eigenvalues / total_var

    print(f"\nNorth's Rule of Thumb (N={n_samples} samples)")
    print("  Mode   Variance %    λ ± δλ              Separated?")
    print("  " + "-" * 55)

    for i in range(n_modes):
        status = "Yes" if separable[i] else "NO (degenerate)"
        print(f"  EOF{i+1:2d}   {var_pct[i]:6.2f}%    "
              f"{eigenvalues[i]:.2e} ± {errors[i]:.2e}   {status}")

    n_degen = np.sum(~separable[:n_modes])
    if n_degen > 0:
        print(f"\n  Warning: {n_degen} mode pair(s) may be degenerate")
