#!/usr/bin/env python3
"""
compare_id_scipy.py - Compare librla id_sketch vs scipy interp_decomp

Compares two ID implementations:
- librla.id_sketch:  Randomized QR sketching (our implementation)
- scipy.linalg.interpolative.interp_decomp: SciPy's interpolative decomposition

Compares on metrics:
- Accuracy (reconstruction error)
- Conditioning (max|T| or max|proj|)
- Runtime
- Rank selection behavior

Usage:
    python compare_id_scipy.py [--precision {double,single}]
                               [--extra-samples N] [--power-iter N]

Options:
    --precision        Floating-point precision: double (default) or single
    --extra-samples N  Number of extra samples for oversampling (default: 12)
    --power-iter N     Number of power iterations (default: 0)

Requires:
    - NumPy, SciPy
    - librla.py from ../python/, test_utils.py from ../python/test/

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.2.1
Date: July 30, 2026
Assisted by: Claude Code (Anthropic)
"""

import argparse
import numpy as np
import sys
import os
import time
from dataclasses import dataclass
from typing import List

# Parse arguments
parser = argparse.ArgumentParser(description='Compare librla id_sketch vs scipy interp_decomp')
parser.add_argument('--precision', choices=['double', 'single'], default='double',
                    help='Floating-point precision (default: double)')
parser.add_argument('--extra-samples', type=int, default=12,
                    help='Number of extra samples for oversampling (default: 12)')
parser.add_argument('--power-iter', type=int, default=0,
                    help='Number of power iterations (default: 0)')
parser.add_argument('--verbose', action='store_true',
                    help='Show detailed results table (default: summary only)')
args = parser.parse_args()

# Set dtype and tolerances based on precision
DTYPE = np.float64 if args.precision == 'double' else np.float32
PRECISION = args.precision

# Precision-dependent constants
# Double: ~16 decimal digits, Single: ~7 decimal digits
if PRECISION == 'double':
    EPS = 1e-10      # Noise level for low-rank matrices
    RTOL_TIGHT = 1e-8   # Tight tolerance for low-rank tests
    ORTH_TOL = 1e-10    # Orthonormality tolerance
else:
    EPS = 1e-5       # Noise level for low-rank matrices (single precision)
    RTOL_TIGHT = 1e-4   # Tight tolerance for low-rank tests (single precision)
    ORTH_TOL = 1e-5     # Orthonormality tolerance (single precision)

# Add parent python directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'test'))

# Import implementations
from librla import id_sketch
import scipy.linalg.interpolative as sli
from test_utils import make_mat


@dataclass
class ComparisonResult:
    """Results from comparing ID methods on a single matrix."""
    name: str
    rtol_or_rank: float

    # Results for each method
    k_librla: int
    k_scipy: int

    err_librla: float
    err_scipy: float

    t_librla: float
    t_scipy: float

    maxT_librla: float
    maxT_scipy: float

    passed: bool


def hilbert(m, n, dtype=np.float64):
    """Generate an mxn Hilbert matrix."""
    i = np.arange(1, m + 1, dtype=dtype).reshape(-1, 1)
    j = np.arange(1, n + 1, dtype=dtype).reshape(1, -1)
    return 1.0 / (i + j - 1)


def run_test_case(A, rtol_or_rank, name, extra_samples=12, power_iter=0):
    """
    Compare ID implementations on a single matrix.

    Parameters
    ----------
    A : ndarray
        Input matrix to decompose (real or complex)
    rtol_or_rank : float
        Tolerance (< 1) or target rank (>= 1)
    name : str
        Test case name for display
    extra_samples : int
        Oversampling parameter (default: 12)
    power_iter : int
        Number of power iterations (default: 0)

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
    print(f"extra_samples={extra_samples}, power_iter={power_iter}")
    print("="*70)

    normA = np.linalg.norm(A, 'fro')

    # -------------------------------------------------------------------------
    # 1. librla id_sketch (randomized QR sketching)
    # -------------------------------------------------------------------------
    print("\n--- librla id_sketch (randomized QR) ---")

    t0 = time.perf_counter()
    k_librla, piv_librla, T_librla = id_sketch(A, rtol=float(rtol_or_rank),
                                                extra_samples=extra_samples,
                                                power_iter=power_iter)
    t_librla = time.perf_counter() - t0

    # Compute reconstruction error
    # librla: A[:, piv[k:]] = A[:, piv[:k]] @ T
    A_skel_librla = A[:, piv_librla[k_librla:]]
    A_basis_librla = A[:, piv_librla[:k_librla]]
    if T_librla.size > 0:
        err_librla = np.linalg.norm(A_skel_librla - A_basis_librla @ T_librla, 'fro') / normA
        maxT_librla = np.max(np.abs(T_librla))
    else:
        err_librla = 0.0
        maxT_librla = 0.0

    # Handle error > 1.0 (retry with lstsq method)
    if err_librla > 1.0:
        print(f"\n[NOTE] Detected error > 1.0 ({err_librla:.6f})")
        print("  Recomputing with method='lstsq' for accurate T...")

        t0 = time.perf_counter()
        k_librla, piv_librla, T_librla = id_sketch(A, rtol=float(rtol_or_rank),
                                                    extra_samples=extra_samples,
                                                    power_iter=power_iter,
                                                    method='lstsq')
        t_librla = time.perf_counter() - t0

        A_skel_librla = A[:, piv_librla[k_librla:]]
        A_basis_librla = A[:, piv_librla[:k_librla]]
        if T_librla.size > 0:
            err_librla = np.linalg.norm(A_skel_librla - A_basis_librla @ T_librla, 'fro') / normA
            maxT_librla = np.max(np.abs(T_librla))
        else:
            err_librla = 0.0
            maxT_librla = 0.0

        print(f"  -> Recomputed: error = {err_librla:.3e}")

    print(f"Rank:       k = {k_librla}")
    print(f"Error:      ||A_skel - A_basis @ T|| / ||A|| = {err_librla:.3e}")
    print(f"Condition:  max|T| = {maxT_librla:.3e}")
    print(f"Time:       {t_librla:.4f} s")

    # -------------------------------------------------------------------------
    # 2. scipy.linalg.interpolative.interp_decomp
    # -------------------------------------------------------------------------
    print("\n--- scipy interp_decomp ---")

    t0 = time.perf_counter()
    if rtol_or_rank < 1:
        # Tolerance mode: returns (k, idx, proj)
        k_scipy, idx_scipy, proj_scipy = sli.interp_decomp(A, rtol_or_rank, rand=True)
    else:
        # Rank mode: returns (idx, proj) only
        idx_scipy, proj_scipy = sli.interp_decomp(A, int(rtol_or_rank), rand=True)
        k_scipy = int(rtol_or_rank)
    t_scipy = time.perf_counter() - t0

    # Compute reconstruction error
    # scipy: A[:, idx[k:]] ≈ A[:, idx[:k]] @ proj.T
    # Note: proj is k x (n-k), so proj.T is (n-k) x k
    # Reconstruction: A_skel ≈ A_basis @ proj.T means we need to use proj (not proj.T)
    # Actually scipy's convention: A[:, idx[:k]] @ proj = A[:, idx[k:]]
    # So proj is k x (n-k)
    A_skel_scipy = A[:, idx_scipy[k_scipy:]]
    A_basis_scipy = A[:, idx_scipy[:k_scipy]]
    if proj_scipy.size > 0:
        err_scipy = np.linalg.norm(A_skel_scipy - A_basis_scipy @ proj_scipy, 'fro') / normA
        maxT_scipy = np.max(np.abs(proj_scipy))
    else:
        err_scipy = 0.0
        maxT_scipy = 0.0

    print(f"Rank:       k = {k_scipy}")
    print(f"Error:      ||A_skel - A_basis @ proj|| / ||A|| = {err_scipy:.3e}")
    print(f"Condition:  max|proj| = {maxT_scipy:.3e}")
    print(f"Time:       {t_scipy:.4f} s")

    # -------------------------------------------------------------------------
    # Summary comparison
    # -------------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"{'Method':<25s} {'Rank':<8s} {'Error':<12s} {'max|T|':<12s} {'Time (s)':<10s}")
    print("-"*75)
    print(f"{'librla id_sketch':<25s} {k_librla:<8d} {err_librla:<12.3e} {maxT_librla:<12.3e} {t_librla:<10.4f}")
    print(f"{'scipy interp_decomp':<25s} {k_scipy:<8d} {err_scipy:<12.3e} {maxT_scipy:<12.3e} {t_scipy:<10.4f}")

    # Highlight best conditioning
    methods = ['librla', 'scipy']
    max_Ts = [maxT_librla, maxT_scipy]
    times = [t_librla, t_scipy]

    best_idx = np.argmin(max_Ts)
    print(f"\nBest conditioning: {methods[best_idx]} (smallest max|T|)")

    # Highlight fastest method
    fastest_idx = np.argmin(times)
    print(f"Fastest method: {methods[fastest_idx]} ({times[fastest_idx]:.4f}s)")

    # Speedup
    if t_librla > 0 and t_scipy > 0:
        if t_librla < t_scipy:
            print(f"Speedup: librla is {t_scipy/t_librla:.1f}x faster")
        else:
            print(f"Speedup: scipy is {t_librla/t_scipy:.1f}x faster")

    # Determine if test passed
    max_error = max(err_librla, err_scipy)

    if rtol_or_rank < 1:
        # Tolerance mode: error should be within 100x tolerance
        tol_threshold = rtol_or_rank * 100
        passed = max_error < min(0.1, tol_threshold)
    else:
        # Rank mode: check reasonable error
        error_reasonable = max_error < 10.0
        passed = error_reasonable

    return ComparisonResult(
        name=name,
        rtol_or_rank=float(rtol_or_rank),
        k_librla=k_librla, k_scipy=k_scipy,
        err_librla=err_librla, err_scipy=err_scipy,
        t_librla=t_librla, t_scipy=t_scipy,
        maxT_librla=maxT_librla, maxT_scipy=maxT_scipy,
        passed=passed
    )


def main():
    """Run comprehensive ID comparison tests."""

    print("="*70)
    print("INTERPOLATIVE DECOMPOSITION COMPARISON")
    print("librla id_sketch vs scipy interp_decomp")
    print("="*70)

    print("\nEnvironment:")
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  NumPy:      {np.__version__}")
    import scipy
    print(f"  SciPy:      {scipy.__version__}")
    print(f"  Precision:  {PRECISION} ({DTYPE.__name__})")

    print(f"\nComparison settings:")
    print(f"  extra_samples={args.extra_samples}")
    print(f"  power_iter={args.power_iter}")
    print(f"  verbose={args.verbose}")
    print("="*70)

    extra_samples = args.extra_samples
    power_iter = args.power_iter

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
    A1 = np.random.randn(500, 300).astype(DTYPE)
    results.append(run_test_case(A1, 20, "Random Matrix (well-conditioned)", extra_samples, power_iter))

    # Test 2: Low-rank matrix
    U = np.random.randn(400, 15).astype(DTYPE)
    V = np.random.randn(250, 15).astype(DTYPE)
    A2 = (U @ V.T + EPS * np.random.randn(400, 250)).astype(DTYPE)
    results.append(run_test_case(A2, RTOL_TIGHT, "Low-Rank Matrix (rank~15)", extra_samples, power_iter))

    # Test 3: Hilbert matrix (extremely ill-conditioned)
    A3 = hilbert(2000, 1000, dtype=DTYPE)
    results.append(run_test_case(A3, 15, "Hilbert Matrix (severely ill-conditioned)", extra_samples, power_iter))

    # Test 4: Complex matrix
    CDTYPE = np.complex128 if DTYPE == np.float64 else np.complex64
    A4 = (np.random.randn(300, 200) + 1j * np.random.randn(300, 200)).astype(CDTYPE)
    results.append(run_test_case(A4, 25, "Complex Matrix", extra_samples, power_iter))

    # Test 5: Decaying spectrum (tolerance mode)
    A5 = np.random.randn(400, 300).astype(DTYPE)
    U5, S5, Vh5 = np.linalg.svd(A5, full_matrices=False)
    s5 = (1.0 / np.arange(1, 301)).astype(DTYPE)
    A5 = (U5 @ np.diag(s5) @ Vh5).astype(DTYPE)
    results.append(run_test_case(A5, 1e-3, "Decaying Spectrum (1/k)", extra_samples, power_iter))

    # -------------------------------------------------------------------------
    # LARGE MATRIX TESTS (2x larger)
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("LARGE MATRIX TESTS (2x SCALE)")
    print("Testing scaling behavior with matrices twice as large")
    print("="*70)

    # Test 6: Large random matrix
    A6 = np.random.randn(1000, 600).astype(DTYPE)
    results.append(run_test_case(A6, 20, "Large Random Matrix (1000x600)", extra_samples, power_iter))

    # Test 7: Large low-rank matrix
    U7 = np.random.randn(800, 15).astype(DTYPE)
    V7 = np.random.randn(500, 15).astype(DTYPE)
    A7 = (U7 @ V7.T + EPS * np.random.randn(800, 500)).astype(DTYPE)
    results.append(run_test_case(A7, RTOL_TIGHT, "Large Low-Rank (800x500, rank~15)", extra_samples, power_iter))

    # Test 8: Large Hilbert matrix
    A8 = hilbert(4000, 2000, dtype=DTYPE)
    results.append(run_test_case(A8, 15, "Large Hilbert Matrix (4000x2000)", extra_samples, power_iter))

    # Test 9: Large complex matrix
    A9 = (np.random.randn(600, 400) + 1j * np.random.randn(600, 400)).astype(CDTYPE)
    results.append(run_test_case(A9, 25, "Large Complex Matrix (600x400)", extra_samples, power_iter))

    # Test 10: Large decaying spectrum
    A10 = np.random.randn(800, 600).astype(DTYPE)
    U10, S10, Vh10 = np.linalg.svd(A10, full_matrices=False)
    s10 = (1.0 / np.arange(1, 601)).astype(DTYPE)
    A10 = (U10 @ np.diag(s10) @ Vh10).astype(DTYPE)
    results.append(run_test_case(A10, 1e-3, "Large Decaying Spectrum (1/k, 800x600)", extra_samples, power_iter))

    # -------------------------------------------------------------------------
    # SLOW DECAYING SPECTRUM TESTS
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("SLOW DECAYING SPECTRUM TESTS")
    print("Testing harder rank-deficient problems with slow decay")
    print("="*70)

    # Test 11: Slow decay - sqrt (small)
    A11 = np.random.randn(400, 300).astype(DTYPE)
    U11, S11, Vh11 = np.linalg.svd(A11, full_matrices=False)
    s11 = (1.0 / np.sqrt(np.arange(1, 301))).astype(DTYPE)
    A11 = (U11 @ np.diag(s11) @ Vh11).astype(DTYPE)
    results.append(run_test_case(A11, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 400x300)", extra_samples, power_iter))

    # Test 12: Slow decay - sqrt (large)
    A12 = np.random.randn(800, 600).astype(DTYPE)
    U12, S12, Vh12 = np.linalg.svd(A12, full_matrices=False)
    s12 = (1.0 / np.sqrt(np.arange(1, 601))).astype(DTYPE)
    A12 = (U12 @ np.diag(s12) @ Vh12).astype(DTYPE)
    results.append(run_test_case(A12, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 800x600)", extra_samples, power_iter))

    # Test 13: Slow decay - polynomial (small)
    A13 = np.random.randn(400, 300).astype(DTYPE)
    U13, S13, Vh13 = np.linalg.svd(A13, full_matrices=False)
    s13 = (1.0 / (np.arange(1, 301) ** 0.7)).astype(DTYPE)
    A13 = (U13 @ np.diag(s13) @ Vh13).astype(DTYPE)
    results.append(run_test_case(A13, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 400x300)", extra_samples, power_iter))

    # Test 14: Slow decay - polynomial (large)
    A14 = np.random.randn(800, 600).astype(DTYPE)
    U14, S14, Vh14 = np.linalg.svd(A14, full_matrices=False)
    s14 = (1.0 / (np.arange(1, 601) ** 0.7)).astype(DTYPE)
    A14 = (U14 @ np.diag(s14) @ Vh14).astype(DTYPE)
    results.append(run_test_case(A14, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 800x600)", extra_samples, power_iter))

    # Test 15: Slow decay - exponential (small)
    A15 = np.random.randn(400, 300).astype(DTYPE)
    U15, S15, Vh15 = np.linalg.svd(A15, full_matrices=False)
    s15 = np.exp(-np.arange(1, 301) / 100.0).astype(DTYPE)
    A15 = (U15 @ np.diag(s15) @ Vh15).astype(DTYPE)
    results.append(run_test_case(A15, 1e-3, "Slow Decay - Exponential (exp(-k/100), 400x300)", extra_samples, power_iter))

    # Test 16: Slow decay - exponential (large)
    A16 = np.random.randn(800, 600).astype(DTYPE)
    U16, S16, Vh16 = np.linalg.svd(A16, full_matrices=False)
    s16 = np.exp(-np.arange(1, 601) / 150.0).astype(DTYPE)
    A16 = (U16 @ np.diag(s16) @ Vh16).astype(DTYPE)
    results.append(run_test_case(A16, 1e-3, "Slow Decay - Exponential (exp(-k/150), 800x600)", extra_samples, power_iter))

    # -------------------------------------------------------------------------
    # MAKE_MAT MATRIX TESTS (structured matrices from paper)
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("MAKE_MAT STRUCTURED MATRIX TESTS")
    print("Testing matrices from 'Robust blockwise random pivoting' paper")
    print("="*70)

    # Test 17: Gaussian Exponential Decay Matrix
    A17 = make_mat(500, 500, 'gaussexp').astype(DTYPE)
    results.append(run_test_case(A17, 1e-3, "Gaussexp (Gaussian Exponential Decay, 500x500)", extra_samples, power_iter))

    # Test 18: Gaussian Mixture Model Matrix
    A18 = make_mat(400, 400, 'gmm').astype(DTYPE)
    results.append(run_test_case(A18, 1e-3, "GMM (Gaussian Mixture Model, 400x400)", extra_samples, power_iter))

    # Test 19: Sparse Neural Network Matrix
    A19 = make_mat(300, 300, 'snn').astype(DTYPE)
    results.append(run_test_case(A19, 1e-3, "SNN (Sparse Neural Network, 300x300)", extra_samples, power_iter))

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

    avg_time_librla = np.mean([r.t_librla for r in results])
    avg_time_scipy = np.mean([r.t_scipy for r in results])

    print()
    print(f"{'Method':<25s} {'Avg Time':<12s} {'Speedup':<15s}")
    print("-"*80)

    # Show speedup relative to the slower method
    if avg_time_librla < avg_time_scipy:
        speedup_librla = avg_time_scipy / avg_time_librla
        print(f"{'librla id_sketch':<25s} {avg_time_librla:>8.4f}s    {speedup_librla:>6.1f}x faster")
        print(f"{'scipy interp_decomp':<25s} {avg_time_scipy:>8.4f}s    {1.0:>6.1f}x (base)")
    else:
        speedup_scipy = avg_time_librla / avg_time_scipy
        print(f"{'librla id_sketch':<25s} {avg_time_librla:>8.4f}s    {1.0:>6.1f}x (base)")
        print(f"{'scipy interp_decomp':<25s} {avg_time_scipy:>8.4f}s    {speedup_scipy:>6.1f}x faster")

    # Accuracy summary
    print()
    print("Reconstruction Error Summary:")
    print("-"*80)

    avg_err_librla = np.mean([r.err_librla for r in results])
    max_err_librla = np.max([r.err_librla for r in results])
    avg_err_scipy = np.mean([r.err_scipy for r in results])
    max_err_scipy = np.max([r.err_scipy for r in results])

    print(f"  librla id_sketch:   mean={avg_err_librla:.3e}, max={max_err_librla:.3e}")
    print(f"  scipy interp_decomp: mean={avg_err_scipy:.3e}, max={max_err_scipy:.3e}")

    # Conditioning summary
    print()
    print("Conditioning Summary (max|T|):")
    print("-"*80)

    avg_maxT_librla = np.mean([r.maxT_librla for r in results])
    avg_maxT_scipy = np.mean([r.maxT_scipy for r in results])
    max_maxT_librla = np.max([r.maxT_librla for r in results])
    max_maxT_scipy = np.max([r.maxT_scipy for r in results])

    print(f"  librla id_sketch:    mean={avg_maxT_librla:.3e}, max={max_maxT_librla:.3e}")
    print(f"  scipy interp_decomp: mean={avg_maxT_scipy:.3e}, max={max_maxT_scipy:.3e}")

    print()
    print("="*80)

    # Return exit code based on pass/fail
    return 0 if all(r.passed for r in results) else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
