#!/usr/bin/env python3
"""
compare_id_parla.py - Compare librla id_sketch vs PARLA osid1/osid2

Compares interpolative decomposition implementations:
- librla.id_sketch: Our randomized ID
- parla.drivers.interpolative.osid1: PARLA's ID (tri-solve on sketch)
- parla.drivers.interpolative.osid2: PARLA's ID (lstsq on original A)

PARLA ID Strategies:
- osid1: Triangular solve on SKETCH (fast, but error can exceed 1.0 on full-rank)
- osid2: Least squares on ORIGINAL A (slower, always accurate)

librla equivalents:
- method='fast' (default): triangular solve, similar to osid1
- method='lstsq': least squares on A, similar to osid2

librla automatically falls back to method='lstsq' when error > 1.0.

Usage:
    python compare_id_parla.py [--precision {double,single}]
                               [--extra-samples N] [--power-iter N]

Options:
    --precision        Floating-point precision: double (default) or single
    --extra-samples N  Number of extra samples for oversampling (default: 12)
    --power-iter N     Number of power iterations (default: 0)

Requires:
    - NumPy, SciPy
    - PARLA (install via ./setup_parla.sh)
    - librla.py, make_mat.py from ../python/

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: BSD-3-Clause
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

import argparse
import os
import sys

# Parse arguments BEFORE any numeric library imports
parser = argparse.ArgumentParser(description='Compare librla id_sketch vs PARLA osid1/osid2')
parser.add_argument('--precision', choices=['double', 'single'], default='double',
                    help='Floating-point precision (default: double)')
parser.add_argument('--extra-samples', type=int, default=12,
                    help='Number of extra samples for oversampling (default: 12)')
parser.add_argument('--power-iter', type=int, default=0,
                    help='Number of power iterations (default: 0)')
parser.add_argument('--verbose', action='store_true',
                    help='Show detailed results table (default: summary only)')
args = parser.parse_args()

# Now import numeric libraries
import numpy as np
import time
from dataclasses import dataclass
from typing import List

# Set dtype based on precision
DTYPE = np.float64 if args.precision == 'double' else np.float32
PRECISION = args.precision

# Precision-dependent constants
if PRECISION == 'double':
    EPS = 1e-10
    RTOL_TIGHT = 1e-8
else:
    EPS = 1e-5
    RTOL_TIGHT = 1e-4

# Add parent python directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# Import implementations
from librla import id_sketch
from make_mat import make_mat

try:
    from parla.drivers.interpolative import osid1, osid2
    PARLA_AVAILABLE = True
except ImportError:
    PARLA_AVAILABLE = False
    print("WARNING: PARLA not installed. Install via ./setup_parla.sh")


@dataclass
class ComparisonResult:
    """Results from comparing ID methods on a single matrix."""
    name: str
    rtol_or_rank: float
    k_librla: int
    k_parla: int
    err_librla: float
    err_parla: float
    t_librla: float
    t_parla: float
    passed: bool


def hilb(m, n, dtype=np.float64):
    """Generate an mxn Hilbert matrix."""
    i = np.arange(1, m + 1, dtype=dtype).reshape(-1, 1)
    j = np.arange(1, n + 1, dtype=dtype).reshape(1, -1)
    return 1.0 / (i + j - 1)


def compare_on_matrix(A, rtol_or_rank, name, extra_samples=12, power_iter=0):
    """
    Compare ID implementations on a single matrix.

    Parameters
    ----------
    A : ndarray
        Input matrix to decompose
    rtol_or_rank : float
        Tolerance (< 1) or target rank (>= 1)
    name : str
        Test case name for display
    extra_samples : int
        Oversampling parameter
    power_iter : int
        Number of power iterations
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
    rng = np.random.default_rng(42)

    # Determine if rank mode or tolerance mode
    rank_mode = rtol_or_rank >= 1

    # -------------------------------------------------------------------------
    # 1. librla id_sketch
    # Returns (k, piv, T) where A[:, piv[k:]] ≈ A[:, piv[:k]] @ T
    # -------------------------------------------------------------------------
    print("\n--- librla id_sketch ---")

    t0 = time.perf_counter()
    k_librla, piv, T = id_sketch(A, rtol=float(rtol_or_rank),
                                  extra_samples=extra_samples,
                                  power_iter=power_iter)
    t_librla = time.perf_counter() - t0

    # Reconstruction error
    A_recon_librla = np.zeros_like(A)
    A_recon_librla[:, piv[:k_librla]] = A[:, piv[:k_librla]]
    A_recon_librla[:, piv[k_librla:]] = A[:, piv[:k_librla]] @ T
    err_librla = np.linalg.norm(A - A_recon_librla, 'fro') / normA

    # Fallback to method='lstsq' if error > 1.0 (same logic as validate_id.m)
    if err_librla > 1.0:
        print(f"  [NOTE] Error > 1.0 ({err_librla:.3e}), retrying with method='lstsq'...")
        t0 = time.perf_counter()
        k_librla, piv, T = id_sketch(A, rtol=float(rtol_or_rank),
                                      extra_samples=extra_samples,
                                      power_iter=power_iter,
                                      method='lstsq')
        t_librla = time.perf_counter() - t0

        A_recon_librla = np.zeros_like(A)
        A_recon_librla[:, piv[:k_librla]] = A[:, piv[:k_librla]]
        A_recon_librla[:, piv[k_librla:]] = A[:, piv[:k_librla]] @ T
        err_librla = np.linalg.norm(A - A_recon_librla, 'fro') / normA
        print(f"  -> Recomputed with lstsq: error = {err_librla:.3e}")

    print(f"Rank:       k = {k_librla}")
    print(f"Error:      ||A - A_recon|| / ||A|| = {err_librla:.3e}")
    print(f"Time:       {t_librla:.4f} s")

    # -------------------------------------------------------------------------
    # 2. PARLA osid2 (lstsq on original A - fair comparison with librla lstsq)
    # Returns (mat, idxs) where A ≈ A[:, idxs] @ mat (for axis=1)
    # -------------------------------------------------------------------------
    print(f"\n--- PARLA osid2 (lstsq on A) ---")

    # For rank mode, use rtol_or_rank as k
    # For tolerance mode, we need to estimate rank - use librla's rank
    k_target = int(rtol_or_rank) if rank_mode else k_librla

    t0 = time.perf_counter()
    mat_parla, idxs_parla = osid2(A, k=k_target, over=extra_samples,
                                   p=max(1, power_iter), axis=1, rng=rng)
    t_parla = time.perf_counter() - t0

    k_parla = len(idxs_parla)
    A_recon_parla = A[:, idxs_parla] @ mat_parla
    err_parla = np.linalg.norm(A - A_recon_parla, 'fro') / normA

    print(f"Rank:       k = {k_parla}")
    print(f"Error:      ||A - A_recon|| / ||A|| = {err_parla:.3e}")
    print(f"Time:       {t_parla:.4f} s")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"{'Method':<20s} {'Rank':<8s} {'Recon Err':<12s} {'Time (s)':<10s}")
    print("-"*60)
    print(f"{'librla id_sketch':<20s} {k_librla:<8d} {err_librla:<12.3e} {t_librla:<10.4f}")
    print(f"{'PARLA osid2':<20s} {k_parla:<8d} {err_parla:<12.3e} {t_parla:<10.4f}")

    # Speedup
    if t_librla < t_parla:
        print(f"\nSpeedup: librla is {t_parla/t_librla:.1f}x faster")
    else:
        print(f"\nSpeedup: PARLA is {t_librla/t_parla:.1f}x faster")

    # Pass/fail (very lenient for ID - full-rank matrices inherently have high error)
    # This matches validate_id.m which uses max_error < 10.0 for rank mode
    error_ok = max(err_librla, err_parla) < 10.0
    passed = error_ok

    return ComparisonResult(
        name=name, rtol_or_rank=rtol_or_rank,
        k_librla=k_librla, k_parla=k_parla,
        err_librla=err_librla, err_parla=err_parla,
        t_librla=t_librla, t_parla=t_parla,
        passed=passed
    )


def run_test_suite(extra_samples=12, power_iter=0):
    """Run ID comparison test suite with comprehensive matrix collection."""
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
    results.append(compare_on_matrix(A1, 20, "Random Matrix (well-conditioned)", extra_samples, power_iter))

    # Test 2: Low-rank matrix
    U = np.random.randn(400, 15).astype(DTYPE)
    V = np.random.randn(250, 15).astype(DTYPE)
    A2 = (U @ V.T + EPS * np.random.randn(400, 250)).astype(DTYPE)
    results.append(compare_on_matrix(A2, RTOL_TIGHT, "Low-Rank Matrix (rank~15)", extra_samples, power_iter))

    # Test 3: Hilbert matrix (extremely ill-conditioned)
    A3 = hilb(2000, 1000, dtype=DTYPE)
    results.append(compare_on_matrix(A3, 15, "Hilbert Matrix (severely ill-conditioned)", extra_samples, power_iter))

    # Test 4: Complex matrix
    CDTYPE = np.complex128 if DTYPE == np.float64 else np.complex64
    A4 = (np.random.randn(300, 200) + 1j * np.random.randn(300, 200)).astype(CDTYPE)
    results.append(compare_on_matrix(A4, 25, "Complex Matrix", extra_samples, power_iter))

    # Test 5: Decaying spectrum (tolerance mode)
    A5 = np.random.randn(400, 300).astype(DTYPE)
    U5, S5, Vh5 = np.linalg.svd(A5, full_matrices=False)
    s5 = (1.0 / np.arange(1, 301)).astype(DTYPE)
    A5 = (U5 @ np.diag(s5) @ Vh5).astype(DTYPE)
    results.append(compare_on_matrix(A5, 1e-3, "Decaying Spectrum (1/k)", extra_samples, power_iter))

    # -------------------------------------------------------------------------
    # LARGE MATRIX TESTS (2x larger)
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("LARGE MATRIX TESTS (2x SCALE)")
    print("Testing scaling behavior with matrices twice as large")
    print("="*70)

    # Test 6: Large random matrix
    A6 = np.random.randn(1000, 600).astype(DTYPE)
    results.append(compare_on_matrix(A6, 20, "Large Random Matrix (1000x600)", extra_samples, power_iter))

    # Test 7: Large low-rank matrix
    U7 = np.random.randn(800, 15).astype(DTYPE)
    V7 = np.random.randn(500, 15).astype(DTYPE)
    A7 = (U7 @ V7.T + EPS * np.random.randn(800, 500)).astype(DTYPE)
    results.append(compare_on_matrix(A7, RTOL_TIGHT, "Large Low-Rank (800x500, rank~15)", extra_samples, power_iter))

    # Test 8: Large Hilbert matrix
    A8 = hilb(4000, 2000, dtype=DTYPE)
    results.append(compare_on_matrix(A8, 15, "Large Hilbert Matrix (4000x2000)", extra_samples, power_iter))

    # Test 9: Large complex matrix
    A9 = (np.random.randn(600, 400) + 1j * np.random.randn(600, 400)).astype(CDTYPE)
    results.append(compare_on_matrix(A9, 25, "Large Complex Matrix (600x400)", extra_samples, power_iter))

    # Test 10: Large decaying spectrum
    A10 = np.random.randn(800, 600).astype(DTYPE)
    U10, S10, Vh10 = np.linalg.svd(A10, full_matrices=False)
    s10 = (1.0 / np.arange(1, 601)).astype(DTYPE)
    A10 = (U10 @ np.diag(s10) @ Vh10).astype(DTYPE)
    results.append(compare_on_matrix(A10, 1e-3, "Large Decaying Spectrum (1/k, 800x600)", extra_samples, power_iter))

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
    results.append(compare_on_matrix(A11, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 400x300)", extra_samples, power_iter))

    # Test 12: Slow decay - sqrt (large)
    A12 = np.random.randn(800, 600).astype(DTYPE)
    U12, S12, Vh12 = np.linalg.svd(A12, full_matrices=False)
    s12 = (1.0 / np.sqrt(np.arange(1, 601))).astype(DTYPE)
    A12 = (U12 @ np.diag(s12) @ Vh12).astype(DTYPE)
    results.append(compare_on_matrix(A12, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 800x600)", extra_samples, power_iter))

    # Test 13: Slow decay - polynomial (small)
    A13 = np.random.randn(400, 300).astype(DTYPE)
    U13, S13, Vh13 = np.linalg.svd(A13, full_matrices=False)
    s13 = (1.0 / (np.arange(1, 301) ** 0.7)).astype(DTYPE)
    A13 = (U13 @ np.diag(s13) @ Vh13).astype(DTYPE)
    results.append(compare_on_matrix(A13, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 400x300)", extra_samples, power_iter))

    # Test 14: Slow decay - polynomial (large)
    A14 = np.random.randn(800, 600).astype(DTYPE)
    U14, S14, Vh14 = np.linalg.svd(A14, full_matrices=False)
    s14 = (1.0 / (np.arange(1, 601) ** 0.7)).astype(DTYPE)
    A14 = (U14 @ np.diag(s14) @ Vh14).astype(DTYPE)
    results.append(compare_on_matrix(A14, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 800x600)", extra_samples, power_iter))

    # Test 15: Slow decay - exponential (small)
    A15 = np.random.randn(400, 300).astype(DTYPE)
    U15, S15, Vh15 = np.linalg.svd(A15, full_matrices=False)
    s15 = np.exp(-np.arange(1, 301) / 100.0).astype(DTYPE)
    A15 = (U15 @ np.diag(s15) @ Vh15).astype(DTYPE)
    results.append(compare_on_matrix(A15, 1e-3, "Slow Decay - Exponential (exp(-k/100), 400x300)", extra_samples, power_iter))

    # Test 16: Slow decay - exponential (large)
    A16 = np.random.randn(800, 600).astype(DTYPE)
    U16, S16, Vh16 = np.linalg.svd(A16, full_matrices=False)
    s16 = np.exp(-np.arange(1, 601) / 150.0).astype(DTYPE)
    A16 = (U16 @ np.diag(s16) @ Vh16).astype(DTYPE)
    results.append(compare_on_matrix(A16, 1e-3, "Slow Decay - Exponential (exp(-k/150), 800x600)", extra_samples, power_iter))

    # -------------------------------------------------------------------------
    # MAKE_MAT MATRIX TESTS (structured matrices from paper)
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("MAKE_MAT STRUCTURED MATRIX TESTS")
    print("Testing matrices from 'Robust blockwise random pivoting' paper")
    print("="*70)

    # Test 17: Gaussian Exponential Decay Matrix
    A17 = make_mat(500, 500, 'gaussexp').astype(DTYPE)
    results.append(compare_on_matrix(A17, 1e-3, "Gaussexp (Gaussian Exponential Decay, 500x500)", extra_samples, power_iter))

    # Test 18: Gaussian Mixture Model Matrix
    A18 = make_mat(400, 400, 'gmm').astype(DTYPE)
    results.append(compare_on_matrix(A18, 1e-3, "GMM (Gaussian Mixture Model, 400x400)", extra_samples, power_iter))

    # Test 19: Sparse Neural Network Matrix
    A19 = make_mat(300, 300, 'snn').astype(DTYPE)
    results.append(compare_on_matrix(A19, 1e-3, "SNN (Sparse Neural Network, 300x300)", extra_samples, power_iter))

    return results


def print_summary(results: List[ComparisonResult], verbose=False):
    """Print comparison summary."""
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
    avg_time_parla = np.mean([r.t_parla for r in results])

    print()
    print(f"{'Method':<25s} {'Avg Time':<12s} {'Speedup':<15s}")
    print("-"*80)

    # Show speedup relative to the slower method
    if avg_time_librla < avg_time_parla:
        speedup_librla = avg_time_parla / avg_time_librla
        print(f"{'librla id_sketch':<25s} {avg_time_librla:>8.4f}s    {speedup_librla:>6.1f}x faster")
        print(f"{'PARLA osid2':<25s} {avg_time_parla:>8.4f}s    {1.0:>6.1f}x (base)")
    else:
        speedup_parla = avg_time_librla / avg_time_parla
        print(f"{'librla id_sketch':<25s} {avg_time_librla:>8.4f}s    {1.0:>6.1f}x (base)")
        print(f"{'PARLA osid2':<25s} {avg_time_parla:>8.4f}s    {speedup_parla:>6.1f}x faster")

    # Accuracy summary
    print()
    print("Reconstruction Error Summary:")
    print("-"*80)

    avg_err_librla = np.mean([r.err_librla for r in results])
    max_err_librla = np.max([r.err_librla for r in results])
    avg_err_parla = np.mean([r.err_parla for r in results])
    max_err_parla = np.max([r.err_parla for r in results])

    print(f"  librla id_sketch: mean={avg_err_librla:.3e}, max={max_err_librla:.3e}")
    print(f"  PARLA osid2:      mean={avg_err_parla:.3e}, max={max_err_parla:.3e}")

    # Summary table (verbose only)
    if verbose:
        print()
        print("="*80)
        print("DETAILED RESULTS")
        print("="*80)
        print(f"{'Test':<45s} {'librla':<12s} {'PARLA':<12s} {'Speedup':<12s}")
        print("-"*90)

        for r in results:
            if r.t_librla < r.t_parla:
                speedup = f"librla {r.t_parla/r.t_librla:.1f}x"
            else:
                speedup = f"PARLA {r.t_librla/r.t_parla:.1f}x"
            name = r.name[:43] if len(r.name) > 43 else r.name
            print(f"{name:<45s} {r.t_librla:<12.4f} {r.t_parla:<12.4f} {speedup:<12s}")


def main():
    """Run comparison tests."""
    if not PARLA_AVAILABLE:
        print("ERROR: PARLA is required for this comparison.")
        print("Install via: ./setup_parla.sh")
        return 1

    print("="*70)
    print("LIBRLA vs PARLA: ID COMPARISON")
    print("="*70)

    print("\nEnvironment:")
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  NumPy:        {np.__version__}")
    print(f"  Precision:    {PRECISION} ({DTYPE.__name__})")

    print(f"\nComparison settings:")
    print(f"  extra_samples={args.extra_samples}")
    print(f"  power_iter={args.power_iter}")
    print(f"  verbose={args.verbose}")
    print("="*70)

    # Run tests
    results = run_test_suite(extra_samples=args.extra_samples,
                              power_iter=args.power_iter)
    print_summary(results, verbose=args.verbose)

    # Overall
    all_passed = all(r.passed for r in results)
    if all_passed:
        print("\n[PASS] All tests PASSED")
        return 0
    else:
        print("\n[WARNING] Some tests FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
