#!/usr/bin/env python3
"""
compare_svd.py - Compare librla SVD implementations

Compares svd_sketch (randomized) vs numpy.linalg.svd (deterministic):
- Accuracy (reconstruction error)
- Singular value accuracy
- Orthonormality of U and V
- Runtime

Usage:
    python compare_svd.py

Requires:
    - NumPy, SciPy
    - librla.py, make_mat.py in Python path

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.2.1
Date: July 30, 2026
Assisted by: Claude Code (Anthropic)
"""

import numpy as np
import sys
import time
from dataclasses import dataclass
from typing import List

sys.path.insert(0,'..')


# Import SVD implementations
from librla import svd_sketch
from test_utils import make_mat


@dataclass
class ComparisonResult:
    """Results from comparing SVD methods on a single matrix."""
    name: str
    rtol_or_rank: float

    # Ranks
    k_sketch: int
    k_ref: int

    # Reconstruction errors
    err_sketch: float
    err_ref: float

    # Singular value accuracy (sketch vs reference)
    sval_err: float

    # Orthonormality errors
    orth_U_sketch: float
    orth_V_sketch: float

    # Timing
    t_sketch: float
    t_ref: float

    passed: bool


def hilbert(m, n):
    """Generate an mxn Hilbert matrix."""
    i = np.arange(1, m + 1).reshape(-1, 1)
    j = np.arange(1, n + 1).reshape(1, -1)
    return 1.0 / (i + j - 1)


def run_test_case(A, rtol_or_rank, name):
    """
    Compare SVD implementations on a single matrix.

    Parameters
    ----------
    A : ndarray
        Input matrix to decompose (real or complex)
    rtol_or_rank : float
        Tolerance (< 1) or target rank (>= 1)
    name : str
        Test case name for display

    Returns
    -------
    result : ComparisonResult
        Comparison metrics
    """
    print("\n" + "="*70)
    print(f"Test: {name}")
    print(f"Matrix: {A.shape[0]}x{A.shape[1]}", end="")
    if np.iscomplexobj(A):
        print(", complex")
    else:
        print()
    print(f"Parameter: rtol_or_rank = {rtol_or_rank}")
    print("="*70)

    normA = np.linalg.norm(A, 'fro')

    # -------------------------------------------------------------------------
    # 1. svd_sketch (randomized)
    # -------------------------------------------------------------------------
    print("\n--- svd_sketch (randomized) ---")

    t0 = time.perf_counter()
    U_sketch, s_sketch, Vh_sketch = svd_sketch(A, rtol=float(rtol_or_rank))
    t_sketch = time.perf_counter() - t0

    k_sketch = len(s_sketch)

    # Reconstruction error (Vh is already conjugate transposed)
    A_recon_sketch = U_sketch @ np.diag(s_sketch) @ Vh_sketch
    err_sketch = np.linalg.norm(A - A_recon_sketch, 'fro') / normA

    # Orthonormality checks
    orth_U_sketch = np.linalg.norm(U_sketch.conj().T @ U_sketch - np.eye(k_sketch), 'fro')
    orth_V_sketch = np.linalg.norm(Vh_sketch @ Vh_sketch.conj().T - np.eye(k_sketch), 'fro')

    print(f"Rank:       k = {k_sketch}")
    print(f"Error:      ||A - U @ S @ Vh|| / ||A|| = {err_sketch:.3e}")
    print(f"Orth U:     ||U'U - I|| = {orth_U_sketch:.3e}")
    print(f"Orth V:     ||Vh @ Vh' - I|| = {orth_V_sketch:.3e}")
    print(f"Time:       {t_sketch:.4f} s")

    # -------------------------------------------------------------------------
    # 2. numpy.linalg.svd (deterministic, truncated)
    # -------------------------------------------------------------------------
    print("\n--- numpy.linalg.svd (deterministic) ---")

    t0 = time.perf_counter()
    U_ref_full, s_ref_full, Vh_ref_full = np.linalg.svd(A, full_matrices=False)
    t_ref = time.perf_counter() - t0

    # Determine reference rank (same as sketch for fair comparison)
    if rtol_or_rank >= 1:
        k_ref = int(rtol_or_rank)
    else:
        # For tolerance mode, use same rank as sketch found
        k_ref = k_sketch

    # Truncate to target rank
    U_ref = U_ref_full[:, :k_ref]
    s_ref = s_ref_full[:k_ref]
    Vh_ref = Vh_ref_full[:k_ref, :]

    # Reconstruction error
    A_recon_ref = U_ref @ np.diag(s_ref) @ Vh_ref
    err_ref = np.linalg.norm(A - A_recon_ref, 'fro') / normA

    print(f"Rank:       k = {k_ref}")
    print(f"Error:      ||A - U @ S @ Vh|| / ||A|| = {err_ref:.3e}")
    print(f"Time:       {t_ref:.4f} s (full SVD)")

    # -------------------------------------------------------------------------
    # Singular value accuracy
    # -------------------------------------------------------------------------
    # Compare sketch singular values to reference (truncated to sketch rank)
    k_cmp = min(k_sketch, len(s_ref_full))
    if k_cmp > 0:
        sval_err = np.linalg.norm(s_sketch[:k_cmp] - s_ref_full[:k_cmp]) / np.linalg.norm(s_ref_full[:k_cmp])
    else:
        sval_err = 0.0

    print(f"\nSingular value accuracy: ||s_sketch - s_ref|| / ||s_ref|| = {sval_err:.3e}")

    # -------------------------------------------------------------------------
    # Summary comparison
    # -------------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"{'Method':<25s} {'Rank':<8s} {'Recon Err':<12s} {'SVal Err':<12s} {'Time (s)':<10s}")
    print("-"*75)
    print(f"{'svd_sketch (randomized)':<25s} {k_sketch:<8d} {err_sketch:<12.3e} {sval_err:<12.3e} {t_sketch:<10.4f}")
    print(f"{'numpy.svd (deterministic)':<25s} {k_ref:<8d} {err_ref:<12.3e} {'(ref)':<12s} {t_ref:<10.4f}")

    # Highlight fastest method
    methods = ['svd_sketch', 'numpy.svd']
    times = [t_sketch, t_ref]
    fastest_idx = np.argmin(times)
    print(f"\nFastest method: {methods[fastest_idx]} ({times[fastest_idx]:.4f}s)")

    # Speedup
    if t_sketch > 0:
        speedup = t_ref / t_sketch
        if speedup > 1:
            print(f"Speedup: svd_sketch is {speedup:.1f}x faster")
        else:
            print(f"Speedup: numpy.svd is {1/speedup:.1f}x faster")

    # -------------------------------------------------------------------------
    # Determine if test passed
    # -------------------------------------------------------------------------
    # For tolerance mode (rtol < 1): error should be within 100x tolerance
    # For rank mode (rtol >= 1): check orthonormality and that error is close to optimal
    if rtol_or_rank < 1:
        tol_threshold = rtol_or_rank * 100
        passed = err_sketch < min(0.1, tol_threshold) and orth_U_sketch < 1e-10 and orth_V_sketch < 1e-10
    else:
        # Rank mode: sketch error should be within 4x of optimal (reference)
        # Singular value accuracy can vary more for randomized methods
        error_ratio_ok = (err_ref == 0) or (err_sketch / max(err_ref, 1e-15) < 4.0)
        passed = error_ratio_ok and sval_err < 0.5 and orth_U_sketch < 1e-10 and orth_V_sketch < 1e-10

    return ComparisonResult(
        name=name,
        rtol_or_rank=float(rtol_or_rank),
        k_sketch=k_sketch, k_ref=k_ref,
        err_sketch=err_sketch, err_ref=err_ref,
        sval_err=sval_err,
        orth_U_sketch=orth_U_sketch, orth_V_sketch=orth_V_sketch,
        t_sketch=t_sketch, t_ref=t_ref,
        passed=passed
    )


def main():
    """Run comprehensive SVD comparison tests."""

    print("="*70)
    print("SVD COMPARISON")
    print("svd_sketch (randomized) vs numpy.linalg.svd (deterministic)")
    print("="*70)

    print("\nEnvironment:")
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  NumPy:      {np.__version__}")
    print("="*70)

    # Results collection
    results = []

    # -------------------------------------------------------------------------
    # BASIC TESTS
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("BASIC TESTS")
    print("Testing fundamental tolerance and rank modes")
    print("="*70)

    # Test 1: Random matrix (well-conditioned)
    np.random.seed(42)
    A1 = np.random.randn(500, 300)
    results.append(run_test_case(A1, 20, "Random Matrix (well-conditioned)"))

    # Test 2: Low-rank matrix
    U = np.random.randn(400, 15)
    V = np.random.randn(250, 15)
    A2 = U @ V.T + 1e-10 * np.random.randn(400, 250)
    results.append(run_test_case(A2, 1e-8, "Low-Rank Matrix (rank~15)"))

    # Test 3: Hilbert matrix (extremely ill-conditioned)
    A3 = hilbert(2000, 1000)
    results.append(run_test_case(A3, 15, "Hilbert Matrix (severely ill-conditioned)"))

    # Test 4: Complex matrix
    A4 = np.random.randn(300, 200) + 1j * np.random.randn(300, 200)
    results.append(run_test_case(A4, 25, "Complex Matrix"))

    # Test 5: Decaying spectrum (tolerance mode)
    A5 = np.random.randn(400, 300)
    U5, S5, Vh5 = np.linalg.svd(A5, full_matrices=False)
    s5 = 1.0 / np.arange(1, 301)  # decaying: 1/k
    A5 = U5 @ np.diag(s5) @ Vh5
    results.append(run_test_case(A5, 1e-3, "Decaying Spectrum (1/k)"))

    # -------------------------------------------------------------------------
    # LARGE MATRIX TESTS (2x larger)
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("LARGE MATRIX TESTS (2x SCALE)")
    print("Testing scaling behavior with matrices twice as large")
    print("="*70)

    # Test 6: Large random matrix
    A6 = np.random.randn(1000, 600)
    results.append(run_test_case(A6, 20, "Large Random Matrix (1000x600)"))

    # Test 7: Large low-rank matrix
    U7 = np.random.randn(800, 15)
    V7 = np.random.randn(500, 15)
    A7 = U7 @ V7.T + 1e-10 * np.random.randn(800, 500)
    results.append(run_test_case(A7, 1e-8, "Large Low-Rank (800x500, rank~15)"))

    # Test 8: Large Hilbert matrix
    A8 = hilbert(4000, 2000)
    results.append(run_test_case(A8, 15, "Large Hilbert Matrix (4000x2000)"))

    # Test 9: Large complex matrix
    A9 = np.random.randn(600, 400) + 1j * np.random.randn(600, 400)
    results.append(run_test_case(A9, 25, "Large Complex Matrix (600x400)"))

    # Test 10: Large decaying spectrum
    A10 = np.random.randn(800, 600)
    U10, S10, Vh10 = np.linalg.svd(A10, full_matrices=False)
    s10 = 1.0 / np.arange(1, 601)  # Fast decay: 1/k
    A10 = U10 @ np.diag(s10) @ Vh10
    results.append(run_test_case(A10, 1e-3, "Large Decaying Spectrum (1/k, 800x600)"))

    # -------------------------------------------------------------------------
    # SLOW DECAYING SPECTRUM TESTS
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("SLOW DECAYING SPECTRUM TESTS")
    print("Testing harder rank-deficient problems with slow decay")
    print("="*70)

    # Test 11: Slow decay - sqrt (small)
    A11 = np.random.randn(400, 300)
    U11, S11, Vh11 = np.linalg.svd(A11, full_matrices=False)
    s11 = 1.0 / np.sqrt(np.arange(1, 301))  # Slow decay: 1/sqrt(k)
    A11 = U11 @ np.diag(s11) @ Vh11
    results.append(run_test_case(A11, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 400x300)"))

    # Test 12: Slow decay - sqrt (large)
    A12 = np.random.randn(800, 600)
    U12, S12, Vh12 = np.linalg.svd(A12, full_matrices=False)
    s12 = 1.0 / np.sqrt(np.arange(1, 601))  # Slow decay: 1/sqrt(k)
    A12 = U12 @ np.diag(s12) @ Vh12
    results.append(run_test_case(A12, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 800x600)"))

    # Test 13: Slow decay - polynomial (small)
    A13 = np.random.randn(400, 300)
    U13, S13, Vh13 = np.linalg.svd(A13, full_matrices=False)
    s13 = 1.0 / (np.arange(1, 301) ** 0.7)  # Polynomial: 1/k^0.7
    A13 = U13 @ np.diag(s13) @ Vh13
    results.append(run_test_case(A13, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 400x300)"))

    # Test 14: Slow decay - polynomial (large)
    A14 = np.random.randn(800, 600)
    U14, S14, Vh14 = np.linalg.svd(A14, full_matrices=False)
    s14 = 1.0 / (np.arange(1, 601) ** 0.7)  # Polynomial: 1/k^0.7
    A14 = U14 @ np.diag(s14) @ Vh14
    results.append(run_test_case(A14, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 800x600)"))

    # Test 15: Slow decay - exponential (small)
    A15 = np.random.randn(400, 300)
    U15, S15, Vh15 = np.linalg.svd(A15, full_matrices=False)
    s15 = np.exp(-np.arange(1, 301) / 100.0)  # Exponential: exp(-k/100)
    A15 = U15 @ np.diag(s15) @ Vh15
    results.append(run_test_case(A15, 1e-3, "Slow Decay - Exponential (exp(-k/100), 400x300)"))

    # Test 16: Slow decay - exponential (large)
    A16 = np.random.randn(800, 600)
    U16, S16, Vh16 = np.linalg.svd(A16, full_matrices=False)
    s16 = np.exp(-np.arange(1, 601) / 150.0)  # Exponential: exp(-k/150)
    A16 = U16 @ np.diag(s16) @ Vh16
    results.append(run_test_case(A16, 1e-3, "Slow Decay - Exponential (exp(-k/150), 800x600)"))

    # -------------------------------------------------------------------------
    # MAKE_MAT MATRIX TESTS (structured matrices from paper)
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("MAKE_MAT STRUCTURED MATRIX TESTS")
    print("Testing matrices from 'Robust blockwise random pivoting' paper")
    print("="*70)

    # Test 22: Gaussian Exponential Decay Matrix
    A22 = make_mat(500, 500, 'gaussexp')
    results.append(run_test_case(A22, 1e-3, "Gaussexp (Gaussian Exponential Decay, 500x500)"))

    # Test 23: Gaussian Mixture Model Matrix
    A23 = make_mat(400, 400, 'gmm')
    results.append(run_test_case(A23, 1e-3, "GMM (Gaussian Mixture Model, 400x400)"))

    # Test 24: Sparse Neural Network Matrix
    A24 = make_mat(300, 300, 'snn')
    results.append(run_test_case(A24, 1e-3, "SNN (Sparse Neural Network, 300x300)"))

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

    avg_time_sketch = np.mean([r.t_sketch for r in results])
    avg_time_ref = np.mean([r.t_ref for r in results])

    print()
    print(f"{'Method':<28s} {'Avg Time':<12s} {'vs NumPy':<15s}")
    print("-"*80)
    print(f"{'svd_sketch (randomized)':<28s} {avg_time_sketch:>8.4f}s    {avg_time_ref/avg_time_sketch:>6.1f}x ")
    print(f"{'numpy.svd (deterministic)':<28s} {avg_time_ref:>8.4f}s    {1.0:>6.1f}x -")

    # Accuracy summary
    print()
    print("Reconstruction Error Summary:")
    print("-"*80)

    avg_err_sketch = np.mean([r.err_sketch for r in results])
    max_err_sketch = np.max([r.err_sketch for r in results])

    print(f"  svd_sketch:    mean={avg_err_sketch:.3e}, max={max_err_sketch:.3e}")

    # Singular value accuracy summary
    print()
    print("Singular Value Accuracy (vs reference):")
    print("-"*80)

    avg_sval_err = np.mean([r.sval_err for r in results])
    max_sval_err = np.max([r.sval_err for r in results])

    print(f"  svd_sketch:    mean={avg_sval_err:.3e}, max={max_sval_err:.3e}")

    # Orthonormality summary
    print()
    print("Orthonormality Summary:")
    print("-"*80)

    max_orth_U = np.max([r.orth_U_sketch for r in results])
    max_orth_V = np.max([r.orth_V_sketch for r in results])

    print(f"  max ||U'U - I||:    {max_orth_U:.3e}")
    print(f"  max ||Vh Vh' - I||: {max_orth_V:.3e}")

    print()
    print("="*80)

    # Return exit code based on pass/fail
    return 0 if all(r.passed for r in results) else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
