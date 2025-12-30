#!/usr/bin/env python3
"""
test_torch_compat.py - Test PyTorch-compatible wrappers for librla

Tests svd_lowrank and pca_lowrank functions:
- Reconstruction error
- Singular value accuracy
- Orthonormality of U and V
- M parameter (subtraction matrix)
- center parameter for PCA

Usage:
    python test_torch_compat.py

Requires:
    - NumPy, SciPy
    - librla.py, torch_compat.py in Python path

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Version: 0.1.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

import numpy as np
import sys
import time
from dataclasses import dataclass
from typing import List

# Import torch_compat implementations
from torch_compat import svd_lowrank, pca_lowrank


@dataclass
class TestResult:
    """Results from testing torch_compat functions."""
    name: str
    q: int

    # Rank returned
    k: int

    # Reconstruction error: ||A - U @ diag(s) @ V'|| / ||A||
    recon_err: float

    # Singular value accuracy (vs reference)
    sval_err: float

    # Orthonormality errors
    orth_U: float
    orth_V: float

    # Timing
    t_compat: float
    t_ref: float

    passed: bool


def hilbert(m, n):
    """Generate an mxn Hilbert matrix."""
    i = np.arange(1, m + 1).reshape(-1, 1)
    j = np.arange(1, n + 1).reshape(1, -1)
    return 1.0 / (i + j - 1)


def run_svd_lowrank_test(A, q, name, M=None):
    """
    Test svd_lowrank on a single matrix.

    Parameters
    ----------
    A : ndarray
        Input matrix to decompose (real or complex)
    q : int
        Oversampled rank / sketch size
    name : str
        Test case name for display
    M : ndarray, optional
        Matrix to subtract before decomposition

    Returns
    -------
    result : TestResult
        Test metrics
    """
    print("\n" + "="*70)
    print(f"Test: {name}")
    print(f"Matrix: {A.shape[0]}x{A.shape[1]}", end="")
    if np.iscomplexobj(A):
        print(", complex")
    else:
        print()
    print(f"Parameter: q = {q}")
    if M is not None:
        print("Using M parameter (subtraction matrix)")
    print("="*70)

    # Matrix to actually decompose
    A_eff = A - M if M is not None else A
    normA = np.linalg.norm(A_eff, 'fro')

    # -------------------------------------------------------------------------
    # svd_lowrank (torch_compat)
    # -------------------------------------------------------------------------
    print("\n--- svd_lowrank (torch_compat) ---")

    t0 = time.perf_counter()
    U, s, V = svd_lowrank(A, q=q, niter=2, M=M)
    t_compat = time.perf_counter() - t0

    k = len(s)

    # Reconstruction error (V is not transposed in torch_compat)
    A_recon = U @ np.diag(s) @ V.conj().T
    recon_err = np.linalg.norm(A_eff - A_recon, 'fro') / normA

    # Orthonormality checks
    orth_U = np.linalg.norm(U.conj().T @ U - np.eye(k), 'fro')
    orth_V = np.linalg.norm(V.conj().T @ V - np.eye(k), 'fro')

    print(f"Rank:       k = {k}")
    print(f"Recon Err:  ||A - U @ S @ V'|| / ||A|| = {recon_err:.3e}")
    print(f"Orth U:     ||U'U - I|| = {orth_U:.3e}")
    print(f"Orth V:     ||V'V - I|| = {orth_V:.3e}")
    print(f"Time:       {t_compat:.4f} s")

    # -------------------------------------------------------------------------
    # Reference (numpy.linalg.svd truncated)
    # -------------------------------------------------------------------------
    print("\n--- Reference (numpy.linalg.svd truncated) ---")

    t0 = time.perf_counter()
    U_ref, s_ref, Vh_ref = np.linalg.svd(A_eff, full_matrices=False)
    t_ref = time.perf_counter() - t0

    # Truncate to same rank
    U_ref = U_ref[:, :k]
    s_ref = s_ref[:k]
    Vh_ref = Vh_ref[:k, :]

    # Reconstruction error
    A_recon_ref = U_ref @ np.diag(s_ref) @ Vh_ref
    recon_err_ref = np.linalg.norm(A_eff - A_recon_ref, 'fro') / normA

    print(f"Rank:       k = {k}")
    print(f"Recon Err:  ||A - U @ S @ Vh|| / ||A|| = {recon_err_ref:.3e}")
    print(f"Time:       {t_ref:.4f} s (full SVD)")

    # Singular value accuracy
    sval_err = np.linalg.norm(s - s_ref) / np.linalg.norm(s_ref) if np.linalg.norm(s_ref) > 0 else 0.0

    print(f"\nSingular value accuracy: ||s - s_ref|| / ||s_ref|| = {sval_err:.3e}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"{'Method':<28s} {'Rank':<8s} {'Recon Err':<12s} {'SVal Err':<12s} {'Time (s)':<10s}")
    print("-"*75)
    print(f"{'svd_lowrank (torch_compat)':<28s} {k:<8d} {recon_err:<12.3e} {sval_err:<12.3e} {t_compat:<10.4f}")
    print(f"{'numpy.svd (reference)':<28s} {k:<8d} {recon_err_ref:<12.3e} {'(ref)':<12s} {t_ref:<10.4f}")

    # -------------------------------------------------------------------------
    # Determine if test passed
    # -------------------------------------------------------------------------
    # Reconstruction error should be within 4x of optimal
    # Orthonormality should be near machine precision
    error_ratio_ok = (recon_err_ref == 0) or (recon_err / max(recon_err_ref, 1e-15) < 4.0)
    passed = error_ratio_ok and sval_err < 0.5 and orth_U < 1e-10 and orth_V < 1e-10

    return TestResult(
        name=name,
        q=q,
        k=k,
        recon_err=recon_err,
        sval_err=sval_err,
        orth_U=orth_U,
        orth_V=orth_V,
        t_compat=t_compat,
        t_ref=t_ref,
        passed=passed
    )


def run_pca_lowrank_test(A, q, name, center=True):
    """
    Test pca_lowrank on a single matrix.

    Parameters
    ----------
    A : ndarray
        Input matrix (m samples, n features)
    q : int
        Oversampled rank / sketch size
    name : str
        Test case name for display
    center : bool
        Whether to subtract column means

    Returns
    -------
    result : TestResult
        Test metrics
    """
    print("\n" + "="*70)
    print(f"Test: {name}")
    print(f"Matrix: {A.shape[0]}x{A.shape[1]}")
    print(f"Parameter: q = {q}, center = {center}")
    print("="*70)

    # Centered matrix for comparison
    if center:
        A_centered = A - A.mean(axis=0, keepdims=True)
    else:
        A_centered = A
    normA = np.linalg.norm(A_centered, 'fro')

    # -------------------------------------------------------------------------
    # pca_lowrank (torch_compat)
    # -------------------------------------------------------------------------
    print("\n--- pca_lowrank (torch_compat) ---")

    t0 = time.perf_counter()
    U, s, V = pca_lowrank(A, q=q, center=center, niter=2)
    t_compat = time.perf_counter() - t0

    k = len(s)

    # Reconstruction error (V is not transposed in torch_compat)
    A_recon = U @ np.diag(s) @ V.conj().T
    recon_err = np.linalg.norm(A_centered - A_recon, 'fro') / normA

    # Orthonormality checks
    orth_U = np.linalg.norm(U.conj().T @ U - np.eye(k), 'fro')
    orth_V = np.linalg.norm(V.conj().T @ V - np.eye(k), 'fro')

    print(f"Rank:       k = {k}")
    print(f"Recon Err:  ||A_c - U @ S @ V'|| / ||A_c|| = {recon_err:.3e}")
    print(f"Orth U:     ||U'U - I|| = {orth_U:.3e}")
    print(f"Orth V:     ||V'V - I|| = {orth_V:.3e}")
    print(f"Time:       {t_compat:.4f} s")

    # -------------------------------------------------------------------------
    # Reference (numpy.linalg.svd on centered data)
    # -------------------------------------------------------------------------
    print("\n--- Reference (numpy.linalg.svd on centered data) ---")

    t0 = time.perf_counter()
    U_ref, s_ref, Vh_ref = np.linalg.svd(A_centered, full_matrices=False)
    t_ref = time.perf_counter() - t0

    # Truncate to same rank
    U_ref = U_ref[:, :k]
    s_ref = s_ref[:k]
    Vh_ref = Vh_ref[:k, :]

    # Reconstruction error
    A_recon_ref = U_ref @ np.diag(s_ref) @ Vh_ref
    recon_err_ref = np.linalg.norm(A_centered - A_recon_ref, 'fro') / normA

    print(f"Rank:       k = {k}")
    print(f"Recon Err:  ||A_c - U @ S @ Vh|| / ||A_c|| = {recon_err_ref:.3e}")
    print(f"Time:       {t_ref:.4f} s (full SVD)")

    # Singular value accuracy
    sval_err = np.linalg.norm(s - s_ref) / np.linalg.norm(s_ref) if np.linalg.norm(s_ref) > 0 else 0.0

    print(f"\nSingular value accuracy: ||s - s_ref|| / ||s_ref|| = {sval_err:.3e}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"{'Method':<28s} {'Rank':<8s} {'Recon Err':<12s} {'SVal Err':<12s} {'Time (s)':<10s}")
    print("-"*75)
    print(f"{'pca_lowrank (torch_compat)':<28s} {k:<8d} {recon_err:<12.3e} {sval_err:<12.3e} {t_compat:<10.4f}")
    print(f"{'numpy.svd (reference)':<28s} {k:<8d} {recon_err_ref:<12.3e} {'(ref)':<12s} {t_ref:<10.4f}")

    # -------------------------------------------------------------------------
    # Determine if test passed
    # -------------------------------------------------------------------------
    error_ratio_ok = (recon_err_ref == 0) or (recon_err / max(recon_err_ref, 1e-15) < 4.0)
    passed = error_ratio_ok and sval_err < 0.5 and orth_U < 1e-10 and orth_V < 1e-10

    return TestResult(
        name=name,
        q=q,
        k=k,
        recon_err=recon_err,
        sval_err=sval_err,
        orth_U=orth_U,
        orth_V=orth_V,
        t_compat=t_compat,
        t_ref=t_ref,
        passed=passed
    )


def main():
    """Run comprehensive torch_compat tests."""

    print("="*70)
    print("TORCH_COMPAT TESTS")
    print("Testing PyTorch-compatible wrappers: svd_lowrank, pca_lowrank")
    print("="*70)

    print("\nEnvironment:")
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  NumPy:      {np.__version__}")
    print("="*70)

    # Results collection
    results = []

    # -------------------------------------------------------------------------
    # SVD_LOWRANK TESTS
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("SVD_LOWRANK TESTS")
    print("Testing PyTorch-compatible randomized SVD")
    print("="*70)

    np.random.seed(42)

    # Test 1: Random matrix (well-conditioned)
    A1 = np.random.randn(500, 300)
    results.append(run_svd_lowrank_test(A1, 20, "svd_lowrank: Random Matrix"))

    # Test 2: Low-rank matrix
    U = np.random.randn(400, 15)
    V = np.random.randn(250, 15)
    A2 = U @ V.T + 1e-10 * np.random.randn(400, 250)
    results.append(run_svd_lowrank_test(A2, 20, "svd_lowrank: Low-Rank Matrix (rank~15)"))

    # Test 3: Hilbert matrix (ill-conditioned)
    A3 = hilbert(500, 300)
    results.append(run_svd_lowrank_test(A3, 15, "svd_lowrank: Hilbert Matrix"))

    # Test 4: Complex matrix
    A4 = np.random.randn(300, 200) + 1j * np.random.randn(300, 200)
    results.append(run_svd_lowrank_test(A4, 25, "svd_lowrank: Complex Matrix"))

    # Test 5: With M parameter (subtraction)
    A5 = np.random.randn(400, 300)
    M5 = np.random.randn(400, 300) * 0.1
    results.append(run_svd_lowrank_test(A5, 20, "svd_lowrank: With M parameter", M=M5))

    # Test 6: Tall matrix
    A6 = np.random.randn(1000, 100)
    results.append(run_svd_lowrank_test(A6, 30, "svd_lowrank: Tall Matrix (1000x100)"))

    # Test 7: Wide matrix
    A7 = np.random.randn(100, 1000)
    results.append(run_svd_lowrank_test(A7, 30, "svd_lowrank: Wide Matrix (100x1000)"))

    # -------------------------------------------------------------------------
    # PCA_LOWRANK TESTS
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("PCA_LOWRANK TESTS")
    print("Testing PyTorch-compatible randomized PCA")
    print("="*70)

    # Test 8: Random data with centering
    A8 = np.random.randn(500, 100) + 5.0  # Add offset to test centering
    results.append(run_pca_lowrank_test(A8, 20, "pca_lowrank: Centered Random Data", center=True))

    # Test 9: Without centering
    A9 = np.random.randn(500, 100)
    results.append(run_pca_lowrank_test(A9, 20, "pca_lowrank: Uncentered Random Data", center=False))

    # Test 10: Low-rank data with centering
    U10 = np.random.randn(400, 10)
    V10 = np.random.randn(80, 10)
    A10 = U10 @ V10.T + 3.0 + 1e-8 * np.random.randn(400, 80)
    results.append(run_pca_lowrank_test(A10, 15, "pca_lowrank: Low-Rank Centered", center=True))

    # Test 11: Default q parameter
    A11 = np.random.randn(100, 50)
    results.append(run_pca_lowrank_test(A11, 6, "pca_lowrank: Default q=6", center=True))

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print("\n\n" + "="*80)
    print(f"TEST SUMMARY - {len(results)} tests completed")
    print("="*80)

    # Overall pass/fail
    passed_tests = sum(1 for r in results if r.passed)
    total_tests = len(results)
    pass_rate = 100.0 * passed_tests / total_tests

    print()
    print(f"Pass Rate: {passed_tests}/{total_tests} ({pass_rate:.1f}%)")

    if passed_tests == total_tests:
        print("[PASS] All tests PASSED")
    else:
        print("[WARNING] Some tests FAILED")
        for r in results:
            if not r.passed:
                print(f"  [FAIL] {r.name}")

    # Performance summary
    print()
    print("="*80)
    print("Performance Summary")
    print("="*80)

    avg_time_compat = np.mean([r.t_compat for r in results])
    avg_time_ref = np.mean([r.t_ref for r in results])

    print()
    print(f"{'Method':<28s} {'Avg Time':<12s} {'vs NumPy':<15s}")
    print("-"*80)
    print(f"{'torch_compat':<28s} {avg_time_compat:>8.4f}s    {avg_time_ref/avg_time_compat:>6.1f}x ")
    print(f"{'numpy.svd (reference)':<28s} {avg_time_ref:>8.4f}s    {1.0:>6.1f}x -")

    # Accuracy summary
    print()
    print("Reconstruction Error Summary:")
    print("-"*80)

    avg_recon_err = np.mean([r.recon_err for r in results])
    max_recon_err = np.max([r.recon_err for r in results])

    print(f"  torch_compat:  mean={avg_recon_err:.3e}, max={max_recon_err:.3e}")

    # Singular value accuracy summary
    print()
    print("Singular Value Accuracy (vs reference):")
    print("-"*80)

    avg_sval_err = np.mean([r.sval_err for r in results])
    max_sval_err = np.max([r.sval_err for r in results])

    print(f"  torch_compat:  mean={avg_sval_err:.3e}, max={max_sval_err:.3e}")

    # Orthonormality summary
    print()
    print("Orthonormality Summary:")
    print("-"*80)

    max_orth_U = np.max([r.orth_U for r in results])
    max_orth_V = np.max([r.orth_V for r in results])

    print(f"  max ||U'U - I||: {max_orth_U:.3e}")
    print(f"  max ||V'V - I||: {max_orth_V:.3e}")

    print()
    print("="*80)

    # Return exit code based on pass/fail
    return 0 if all(r.passed for r in results) else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
