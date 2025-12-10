#!/usr/bin/env python3
"""
compare_svd_parla.py - Compare librla svd_sketch vs PARLA svd1

Compares randomized SVD implementations:
- librla.svd_sketch: Our randomized SVD
- parla.drivers.svd.svd1: PARLA's randomized SVD

Metrics:
- Reconstruction error
- Singular value accuracy
- Orthonormality
- Runtime

Usage:
    python compare_svd_parla.py [--precision {double,single}]
                                [--extra-samples N] [--power-iter N]

Options:
    --precision        Floating-point precision: double (default) or single
    --extra-samples N  Number of extra samples for oversampling (default: 12)
    --power-iter N     Number of power iterations (default: 0)

Requires:
    - NumPy, SciPy
    - PARLA (install via ./setup_parla.sh)
    - librla.py, make_mat.py from ../python/
"""

import argparse
import os
import sys

# Parse arguments BEFORE any numeric library imports
parser = argparse.ArgumentParser(description='Compare librla svd_sketch vs PARLA svd1')
parser.add_argument('--precision', choices=['double', 'single'], default='double',
                    help='Floating-point precision (default: double)')
parser.add_argument('--extra-samples', type=int, default=12,
                    help='Number of extra samples for oversampling (default: 12)')
parser.add_argument('--power-iter', type=int, default=0,
                    help='Number of power iterations (default: 0)')
parser.add_argument('--verbose', action='store_true',
                    help='Show detailed per-matrix results (default: summary only)')
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
    ORTH_TOL = 1e-10
else:
    EPS = 1e-5
    ORTH_TOL = 1e-5

# Add parent python directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# Import implementations
from librla import svd_sketch
from make_mat import make_mat

try:
    from parla.drivers.svd import svd1
    PARLA_AVAILABLE = True
except ImportError:
    PARLA_AVAILABLE = False
    print("WARNING: PARLA not installed. Install via ./setup_parla.sh")


@dataclass
class ComparisonResult:
    """Results from comparing SVD methods on a single matrix."""
    name: str
    rank: int
    err_librla: float
    err_parla: float
    sval_err_librla: float
    sval_err_parla: float
    orth_U_librla: float
    orth_V_librla: float
    orth_U_parla: float
    orth_V_parla: float
    t_librla: float
    t_parla: float
    passed: bool
    fail_reason: str = ""


def hilb(m, n, dtype=np.float64):
    """Generate an mxn Hilbert matrix."""
    i = np.arange(1, m + 1, dtype=dtype).reshape(-1, 1)
    j = np.arange(1, n + 1, dtype=dtype).reshape(1, -1)
    return 1.0 / (i + j - 1)


def compare_on_matrix(A, rank, name, power_iter=0, extra_samples=12):
    """
    Compare SVD implementations on a single matrix.

    Parameters
    ----------
    A : ndarray
        Input matrix to decompose
    rank : int
        Target rank for approximation
    name : str
        Test case name for display
    power_iter : int
        Number of power iterations
    extra_samples : int
        Oversampling parameter
    """
    print("\n" + "="*70)
    print(f"Test: {name}")
    print(f"Matrix: {A.shape[0]}x{A.shape[1]}, Target rank: {rank}")
    print(f"Power iterations: {power_iter}, extra_samples: {extra_samples}")
    print("="*70)

    normA = np.linalg.norm(A, 'fro')
    rng = np.random.default_rng(42)

    # Get reference singular values from numpy
    s_ref = np.linalg.svd(A, compute_uv=False)

    # -------------------------------------------------------------------------
    # 1. librla svd_sketch
    # -------------------------------------------------------------------------
    print("\n--- librla svd_sketch ---")

    t0 = time.perf_counter()
    U_librla, s_librla, Vh_librla = svd_sketch(A, rtol=float(rank),
                                                power_iter=power_iter,
                                                extra_samples=extra_samples)
    t_librla = time.perf_counter() - t0

    k_librla = len(s_librla)

    # Reconstruction error
    A_recon_librla = U_librla @ np.diag(s_librla) @ Vh_librla
    err_librla = np.linalg.norm(A - A_recon_librla, 'fro') / normA

    # Singular value accuracy
    k_cmp = min(k_librla, len(s_ref))
    sval_err_librla = np.linalg.norm(s_librla[:k_cmp] - s_ref[:k_cmp]) / np.linalg.norm(s_ref[:k_cmp])

    # Orthonormality
    orth_U_librla = np.linalg.norm(U_librla.T @ U_librla - np.eye(k_librla), 'fro')
    orth_V_librla = np.linalg.norm(Vh_librla @ Vh_librla.T - np.eye(k_librla), 'fro')

    print(f"Rank:       k = {k_librla}")
    print(f"Error:      ||A - U @ S @ Vh|| / ||A|| = {err_librla:.3e}")
    print(f"SVal Err:   ||s - s_ref|| / ||s_ref|| = {sval_err_librla:.3e}")
    print(f"Orth U:     ||U'U - I|| = {orth_U_librla:.3e}")
    print(f"Orth Vh:    ||Vh Vh' - I|| = {orth_V_librla:.3e}")
    print(f"Time:       {t_librla:.4f} s")

    # -------------------------------------------------------------------------
    # 2. PARLA svd1
    # PARLA uses inner_num_pass for power iterations (passes over A)
    # inner_num_pass >= 2 required, so use max(2, 2*power_iter + 1)
    # -------------------------------------------------------------------------
    print(f"\n--- PARLA svd1 ---")

    inner_num_pass = max(2, 2 * power_iter + 1)
    block_size = rank + extra_samples

    t0 = time.perf_counter()
    # Use np.nan for tol to skip tolerance checking (fixed rank mode)
    U_parla, s_parla, Vh_parla = svd1(A, k=rank, over=extra_samples, tol=np.nan,
                                       inner_num_pass=inner_num_pass,
                                       block_size=block_size, rng=rng)
    t_parla = time.perf_counter() - t0

    k_parla = len(s_parla)

    # Reconstruction error
    A_recon_parla = U_parla @ np.diag(s_parla) @ Vh_parla
    err_parla = np.linalg.norm(A - A_recon_parla, 'fro') / normA

    # Singular value accuracy
    k_cmp = min(k_parla, len(s_ref))
    sval_err_parla = np.linalg.norm(s_parla[:k_cmp] - s_ref[:k_cmp]) / np.linalg.norm(s_ref[:k_cmp])

    # Orthonormality
    orth_U_parla = np.linalg.norm(U_parla.T @ U_parla - np.eye(k_parla), 'fro')
    orth_V_parla = np.linalg.norm(Vh_parla @ Vh_parla.T - np.eye(k_parla), 'fro')

    print(f"Rank:       k = {k_parla}")
    print(f"Error:      ||A - U @ S @ Vh|| / ||A|| = {err_parla:.3e}")
    print(f"SVal Err:   ||s - s_ref|| / ||s_ref|| = {sval_err_parla:.3e}")
    print(f"Orth U:     ||U'U - I|| = {orth_U_parla:.3e}")
    print(f"Orth Vh:    ||Vh Vh' - I|| = {orth_V_parla:.3e}")
    print(f"Time:       {t_parla:.4f} s")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"{'Method':<20s} {'Rank':<8s} {'Recon Err':<12s} {'SVal Err':<12s} {'Time (s)':<10s}")
    print("-"*70)
    print(f"{'librla svd_sketch':<20s} {k_librla:<8d} {err_librla:<12.3e} {sval_err_librla:<12.3e} {t_librla:<10.4f}")
    print(f"{'PARLA svd1':<20s} {k_parla:<8d} {err_parla:<12.3e} {sval_err_parla:<12.3e} {t_parla:<10.4f}")

    # Speedup
    if t_librla < t_parla:
        print(f"\nSpeedup: librla is {t_parla/t_librla:.1f}x faster")
    else:
        print(f"\nSpeedup: PARLA is {t_librla/t_parla:.1f}x faster")

    # Pass/fail - track which library failed
    fail_reasons = []

    # Check librla
    if orth_U_librla >= ORTH_TOL or orth_V_librla >= ORTH_TOL:
        fail_reasons.append("librla: orth")
    if err_librla >= 1.0:
        fail_reasons.append(f"librla: err={err_librla:.2e}")

    # Check PARLA
    if orth_U_parla >= ORTH_TOL or orth_V_parla >= ORTH_TOL:
        fail_reasons.append("PARLA: orth")
    if err_parla >= 1.0:
        fail_reasons.append(f"PARLA: err={err_parla:.2e}")

    passed = len(fail_reasons) == 0
    fail_reason = ", ".join(fail_reasons)

    return ComparisonResult(
        name=name, rank=rank,
        err_librla=err_librla, err_parla=err_parla,
        sval_err_librla=sval_err_librla, sval_err_parla=sval_err_parla,
        orth_U_librla=orth_U_librla, orth_V_librla=orth_V_librla,
        orth_U_parla=orth_U_parla, orth_V_parla=orth_V_parla,
        t_librla=t_librla, t_parla=t_parla,
        passed=passed, fail_reason=fail_reason
    )


def run_test_suite(power_iter=0, extra_samples=12):
    """Run SVD comparison test suite."""
    results = []

    # -------------------------------------------------------------------------
    # BASIC TESTS
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("BASIC TESTS")
    print("="*70)

    # Test 1: Random matrix
    np.random.seed(42)
    A1 = np.random.randn(500, 300).astype(DTYPE)
    results.append(compare_on_matrix(A1, 20, "Random Matrix (500x300)", power_iter, extra_samples))

    # Test 2: Low-rank matrix
    U = np.random.randn(400, 15).astype(DTYPE)
    V = np.random.randn(250, 15).astype(DTYPE)
    A2 = (U @ V.T + EPS * np.random.randn(400, 250)).astype(DTYPE)
    results.append(compare_on_matrix(A2, 15, "Low-Rank Matrix (rank~15)", power_iter, extra_samples))

    # Test 3: Hilbert matrix
    A3 = hilb(1000, 500, dtype=DTYPE)
    results.append(compare_on_matrix(A3, 15, "Hilbert Matrix (1000x500)", power_iter, extra_samples))

    # Test 4: Complex matrix
    CDTYPE = np.complex128 if DTYPE == np.float64 else np.complex64
    A4 = (np.random.randn(300, 200) + 1j * np.random.randn(300, 200)).astype(CDTYPE)
    results.append(compare_on_matrix(A4, 25, "Complex Matrix (300x200)", power_iter, extra_samples))

    # Test 5: Decaying spectrum (1/k)
    A5 = np.random.randn(400, 300).astype(DTYPE)
    U5, S5, Vh5 = np.linalg.svd(A5, full_matrices=False)
    s5 = (1.0 / np.arange(1, 301)).astype(DTYPE)
    A5 = (U5 @ np.diag(s5) @ Vh5).astype(DTYPE)
    results.append(compare_on_matrix(A5, 50, "Decaying Spectrum (1/k)", power_iter, extra_samples))

    # -------------------------------------------------------------------------
    # LARGE MATRIX TESTS
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("LARGE MATRIX TESTS")
    print("="*70)

    # Test 6: Large random matrix
    A6 = np.random.randn(1000, 600).astype(DTYPE)
    results.append(compare_on_matrix(A6, 30, "Large Random (1000x600)", power_iter, extra_samples))

    # Test 7: Large low-rank
    U7 = np.random.randn(800, 15).astype(DTYPE)
    V7 = np.random.randn(500, 15).astype(DTYPE)
    A7 = (U7 @ V7.T + EPS * np.random.randn(800, 500)).astype(DTYPE)
    results.append(compare_on_matrix(A7, 15, "Large Low-Rank (800x500)", power_iter, extra_samples))

    # Test 8: Large Hilbert
    A8 = hilb(2000, 1000, dtype=DTYPE)
    results.append(compare_on_matrix(A8, 15, "Large Hilbert (2000x1000)", power_iter, extra_samples))

    # -------------------------------------------------------------------------
    # SLOW DECAY TESTS
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("SLOW DECAY TESTS")
    print("="*70)

    # Test 9: Sqrt decay
    A9 = np.random.randn(400, 300).astype(DTYPE)
    U9, S9, Vh9 = np.linalg.svd(A9, full_matrices=False)
    s9 = (1.0 / np.sqrt(np.arange(1, 301))).astype(DTYPE)
    A9 = (U9 @ np.diag(s9) @ Vh9).astype(DTYPE)
    results.append(compare_on_matrix(A9, 50, "Slow Decay - Sqrt (1/sqrt(k))", power_iter, extra_samples))

    # Test 10: Polynomial decay
    A10 = np.random.randn(400, 300).astype(DTYPE)
    U10, S10, Vh10 = np.linalg.svd(A10, full_matrices=False)
    s10 = (1.0 / (np.arange(1, 301) ** 0.7)).astype(DTYPE)
    A10 = (U10 @ np.diag(s10) @ Vh10).astype(DTYPE)
    results.append(compare_on_matrix(A10, 50, "Slow Decay - Polynomial (1/k^0.7)", power_iter, extra_samples))

    # -------------------------------------------------------------------------
    # STRUCTURED MATRIX TESTS
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("STRUCTURED MATRIX TESTS")
    print("="*70)

    # Test 11: Gaussexp
    A11 = make_mat(500, 500, 'gaussexp').astype(DTYPE)
    results.append(compare_on_matrix(A11, 50, "Gaussexp (500x500)", power_iter, extra_samples))

    # Test 12: GMM
    A12 = make_mat(400, 400, 'gmm').astype(DTYPE)
    results.append(compare_on_matrix(A12, 50, "GMM (400x400)", power_iter, extra_samples))

    # Test 13: SNN
    A13 = make_mat(300, 300, 'snn').astype(DTYPE)
    results.append(compare_on_matrix(A13, 50, "SNN (300x300)", power_iter, extra_samples))

    return results


def print_summary(results: List[ComparisonResult], verbose=False):
    """Print comparison summary."""
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Pass rate
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\nPass Rate: {passed}/{total} ({100*passed/total:.1f}%)")

    if passed < total:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  [FAIL] {r.name}: {r.fail_reason}")

    # Average times
    avg_librla = np.mean([r.t_librla for r in results])
    avg_parla = np.mean([r.t_parla for r in results])

    print("\nAverage Times:")
    print(f"  librla svd_sketch: {avg_librla:.4f}s")
    print(f"  PARLA svd1:        {avg_parla:.4f}s")

    if avg_librla < avg_parla:
        print(f"  -> librla is {avg_parla/avg_librla:.1f}x faster on average")
    else:
        print(f"  -> PARLA is {avg_librla/avg_parla:.1f}x faster on average")

    # Average errors
    avg_err_librla = np.mean([r.err_librla for r in results])
    avg_err_parla = np.mean([r.err_parla for r in results])

    print("\nAverage Reconstruction Error:")
    print(f"  librla: {avg_err_librla:.3e}")
    print(f"  PARLA:  {avg_err_parla:.3e}")

    # Summary table (verbose only)
    if verbose:
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        print(f"{'Test':<35s} {'librla':<12s} {'PARLA':<12s} {'Speedup':<12s}")
        print("-"*80)

        for r in results:
            if r.t_librla < r.t_parla:
                speedup = f"librla {r.t_parla/r.t_librla:.1f}x"
            else:
                speedup = f"PARLA {r.t_librla/r.t_parla:.1f}x"
            print(f"{r.name:<35s} {r.t_librla:<12.4f} {r.t_parla:<12.4f} {speedup:<12s}")


def main():
    """Run comparison tests."""
    if not PARLA_AVAILABLE:
        print("ERROR: PARLA is required for this comparison.")
        print("Install via: ./setup_parla.sh")
        return 1

    print("="*70)
    print("LIBRLA vs PARLA: SVD COMPARISON")
    print("="*70)

    print("\nEnvironment:")
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  NumPy:        {np.__version__}")
    print(f"  Precision:    {PRECISION} ({DTYPE.__name__})")

    print(f"\nComparison settings:")
    print(f"  power_iter={args.power_iter}")
    print(f"  extra_samples={args.extra_samples}")
    print(f"  verbose={args.verbose}")
    print("="*70)

    # Run tests
    results = run_test_suite(power_iter=args.power_iter,
                              extra_samples=args.extra_samples)
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
