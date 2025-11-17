#!/usr/bin/env python3
"""
compare_id.py - Compare libid interpolative decomposition implementations

Compares two ID implementations:
- libid (id_sketch):  Randomized QR sketching (default, recommended)
- libid_rrqr:         Deterministic RRQR via LAPACK geqp3

Compares on metrics:
- Accuracy (reconstruction error)
- Conditioning (max|T|)
- Runtime
- Rank selection behavior

Usage:
    python compare_id.py

Requires:
    - NumPy, SciPy
    - libid.py, libid_rrqr.py, make_mat.py in Python path
"""

import numpy as np
import sys
import time
from dataclasses import dataclass
from typing import List

# No path additions needed - all files in same directory

# Import ID implementations
from libid import id_sketch
from libid_rrqr import id_rrqr
from make_mat import make_mat


@dataclass
class ComparisonResult:
    """Results from comparing ID methods on a single matrix."""
    name: str
    rtol_or_rank: float

    # Results for each method (2 methods x 4 metrics)
    k_sketch: int
    k_rrqr: int

    err_sketch: float
    err_rrqr: float

    t_sketch: float
    t_rrqr: float

    maxT_sketch: float
    maxT_rrqr: float

    passed: bool


def hilb(m, n):
    """Generate an mxn Hilbert matrix."""
    i = np.arange(1, m + 1).reshape(-1, 1)
    j = np.arange(1, n + 1).reshape(1, -1)
    return 1.0 / (i + j - 1)


def compare_on_matrix(A, rtol_or_rank, name):
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
    # 1. libid_sketch (randomized QR sketching)
    # -------------------------------------------------------------------------
    print("\n--- libid_sketch (randomized QR) ---")

    t0 = time.perf_counter()
    k_sketch, piv_sketch, T_sketch = id_sketch(A, rtol=float(rtol_or_rank))
    t_sketch = time.perf_counter() - t0

    # Compute reconstruction error
    A_skel_sketch = A[:, piv_sketch[k_sketch:]]
    A_basis_sketch = A[:, piv_sketch[:k_sketch]]
    if T_sketch.size > 0:
        err_sketch = np.linalg.norm(A_skel_sketch - A_basis_sketch @ T_sketch, 'fro') / normA
        maxT_sketch = np.max(np.abs(T_sketch))
    else:
        err_sketch = 0.0
        maxT_sketch = 0.0

    # CHECK: Error > 1.0 can occur for (nearly) full-rank matrices with fast T computation
    if err_sketch > 1.0:
        print(f"\n[NOTE] Detected error > 1.0 ({err_sketch:.6f})")
        print("  This can occur for (nearly) full-rank matrices with fast T computation.")
        print("  Recomputing with recompute_T=True for accurate lstsq-based T...")

        # Retry with recompute_T=True for accurate T computation via lstsq
        t0 = time.perf_counter()
        k_sketch, piv_sketch, T_sketch = id_sketch(A, rtol=float(rtol_or_rank), recompute_T=True)
        t_sketch = time.perf_counter() - t0

        # Recompute error with new T
        A_skel_sketch = A[:, piv_sketch[k_sketch:]]
        A_basis_sketch = A[:, piv_sketch[:k_sketch]]
        if T_sketch.size > 0:
            err_sketch = np.linalg.norm(A_skel_sketch - A_basis_sketch @ T_sketch, 'fro') / normA
            maxT_sketch = np.max(np.abs(T_sketch))
        else:
            err_sketch = 0.0
            maxT_sketch = 0.0

        print(f"  -> Recomputed: error = {err_sketch:.3e} (recompute_T=True)")

        if err_sketch > 1.0:
            raise ValueError(f"[ERROR] Error still > 1.0 even with recompute_T=True!\n"
                           f"   Error = {err_sketch:.6f}, Test: {name}, rtol_or_rank={rtol_or_rank}")

    print(f"Rank:       k = {k_sketch}")
    print(f"Error:      ||A_skel - A_basis @ T|| / ||A|| = {err_sketch:.3e}")
    print(f"Condition:  max|T| = {maxT_sketch:.3e}")
    print(f"Time:       {t_sketch:.4f} s")

    # -------------------------------------------------------------------------
    # 2. libid_rrqr (deterministic RRQR)
    # -------------------------------------------------------------------------
    print("\n--- libid_rrqr (deterministic RRQR) ---")

    t0 = time.perf_counter()
    k_rrqr, piv_rrqr, T_rrqr = id_rrqr(A, rtol=float(rtol_or_rank))
    t_rrqr = time.perf_counter() - t0

    # Compute reconstruction error
    A_skel_rrqr = A[:, piv_rrqr[k_rrqr:]]
    A_basis_rrqr = A[:, piv_rrqr[:k_rrqr]]
    if T_rrqr.size > 0:
        err_rrqr = np.linalg.norm(A_skel_rrqr - A_basis_rrqr @ T_rrqr, 'fro') / normA
        maxT_rrqr = np.max(np.abs(T_rrqr))
    else:
        err_rrqr = 0.0
        maxT_rrqr = 0.0

    print(f"Rank:       k = {k_rrqr}")
    print(f"Error:      ||A_skel - A_basis @ T|| / ||A|| = {err_rrqr:.3e}")
    print(f"Condition:  max|T| = {maxT_rrqr:.3e}")
    print(f"Time:       {t_rrqr:.4f} s")

    # -------------------------------------------------------------------------
    # Summary comparison
    # -------------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"{'Method':<22s} {'Rank':<8s} {'Error':<12s} {'max|T|':<12s} {'Time (s)':<10s}")
    print("-"*72)
    print(f"{'libid (QR sketch)':<22s} {k_sketch:<8d} {err_sketch:<12.3e} {maxT_sketch:<12.3e} {t_sketch:<10.4f}")
    print(f"{'RRQR (geqp3)':<22s} {k_rrqr:<8d} {err_rrqr:<12.3e} {maxT_rrqr:<12.3e} {t_rrqr:<10.4f}")

    # Highlight best conditioning
    max_Ts = [maxT_sketch, maxT_rrqr]
    methods = ['libid', 'RRQR']
    times = [t_sketch, t_rrqr]

    best_idx = np.argmin(max_Ts)
    print(f"\nBest conditioning: {methods[best_idx]} (smallest max|T|)")

    # Highlight fastest method
    fastest_idx = np.argmin(times)
    print(f"Fastest method: {methods[fastest_idx]} ({times[fastest_idx]:.4f}s)")

    # Determine if test passed
    max_error = max(err_sketch, err_rrqr)

    # For tolerance mode (rtol < 1): expect error ~ rtol
    # For rank mode (rtol >= 1): check consistency and reasonable error
    if rtol_or_rank < 1:
        # Tolerance mode: error should be within 100x tolerance
        tol_threshold = rtol_or_rank * 100
        passed = max_error < min(0.1, tol_threshold)
    else:
        # Rank mode: check deterministic methods agree on rank
        # For full-rank matrices with small k, error can be large (e.g., 90%)
        # This is expected - just verify methods are consistent
        ranks_match = (k_sketch == k_rrqr)
        error_reasonable = max_error < 10.0  # Very lenient for rank mode
        passed = ranks_match and error_reasonable

    # Return ComparisonResult
    return ComparisonResult(
        name=name,
        rtol_or_rank=float(rtol_or_rank),
        k_sketch=k_sketch, k_rrqr=k_rrqr,
        err_sketch=err_sketch, err_rrqr=err_rrqr,
        t_sketch=t_sketch, t_rrqr=t_rrqr,
        maxT_sketch=maxT_sketch, maxT_rrqr=maxT_rrqr,
        passed=passed
    )


def main():
    """Run comprehensive ID comparison tests."""

    print("="*70)
    print("INTERPOLATIVE DECOMPOSITION COMPARISON")
    print("LibID implementations in Python")
    print("="*70)

    print("\nEnvironment:")
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  NumPy:      {np.__version__}")
    print("="*70)

    # Results collection
    results = []

    # -------------------------------------------------------------------------
    # Test 1: Random matrix (well-conditioned)
    # -------------------------------------------------------------------------
    np.random.seed(42)
    A1 = np.random.randn(500, 300)
    results.append(compare_on_matrix(A1, 20, "Random Matrix (well-conditioned)"))

    # -------------------------------------------------------------------------
    # Test 2: Low-rank matrix
    # -------------------------------------------------------------------------
    U = np.random.randn(400, 15)
    V = np.random.randn(250, 15)
    A2 = U @ V.T + 1e-10 * np.random.randn(400, 250)
    results.append(compare_on_matrix(A2, 1e-8, "Low-Rank Matrix (rank~15)"))

    # -------------------------------------------------------------------------
    # Test 3: Hilbert matrix (extremely ill-conditioned)
    # -------------------------------------------------------------------------
    A3 = hilb(2000, 1000)
    results.append(compare_on_matrix(A3, 15, "Hilbert Matrix (severely ill-conditioned)"))

    # -------------------------------------------------------------------------
    # Test 4: Complex matrix
    # -------------------------------------------------------------------------
    A4 = np.random.randn(300, 200) + 1j * np.random.randn(300, 200)
    results.append(compare_on_matrix(A4, 25, "Complex Matrix"))

    # -------------------------------------------------------------------------
    # Test 5: Decaying spectrum (tolerance mode)
    # -------------------------------------------------------------------------
    A5 = np.random.randn(400, 300)
    U5, S5, Vh5 = np.linalg.svd(A5, full_matrices=False)
    s5 = 1.0 / np.arange(1, 301)  # decaying: 1/k
    A5 = U5 @ np.diag(s5) @ Vh5
    results.append(compare_on_matrix(A5, 1e-3, "Decaying Spectrum (1/k)"))

    # -------------------------------------------------------------------------
    # LARGE MATRIX TESTS (2x larger)
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("LARGE MATRIX TESTS (2x SCALE)")
    print("Testing scaling behavior with matrices twice as large")
    print("="*70)

    # Test 6: Large random matrix
    A6 = np.random.randn(1000, 600)
    results.append(compare_on_matrix(A6, 20, "Large Random Matrix (1000x600)"))

    # Test 7: Large low-rank matrix
    U7 = np.random.randn(800, 15)
    V7 = np.random.randn(500, 15)
    A7 = U7 @ V7.T + 1e-10 * np.random.randn(800, 500)
    results.append(compare_on_matrix(A7, 1e-8, "Large Low-Rank (800x500, rank~15)"))

    # Test 8: Large Hilbert matrix
    A8 = hilb(4000, 2000)
    results.append(compare_on_matrix(A8, 15, "Large Hilbert Matrix (4000x2000)"))

    # Test 9: Large complex matrix
    A9 = np.random.randn(600, 400) + 1j * np.random.randn(600, 400)
    results.append(compare_on_matrix(A9, 25, "Large Complex Matrix (600x400)"))

    # Test 10: Large decaying spectrum
    A10 = np.random.randn(800, 600)
    U10, S10, Vh10 = np.linalg.svd(A10, full_matrices=False)
    s10 = 1.0 / np.arange(1, 601)  # Fast decay: 1/k
    A10 = U10 @ np.diag(s10) @ Vh10
    results.append(compare_on_matrix(A10, 1e-3, "Large Decaying Spectrum (1/k, 800x600)"))

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
    results.append(compare_on_matrix(A11, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 400x300)"))

    # Test 12: Slow decay - sqrt (large)
    A12 = np.random.randn(800, 600)
    U12, S12, Vh12 = np.linalg.svd(A12, full_matrices=False)
    s12 = 1.0 / np.sqrt(np.arange(1, 601))  # Slow decay: 1/sqrt(k)
    A12 = U12 @ np.diag(s12) @ Vh12
    results.append(compare_on_matrix(A12, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 800x600)"))

    # Test 13: Slow decay - polynomial (small)
    A13 = np.random.randn(400, 300)
    U13, S13, Vh13 = np.linalg.svd(A13, full_matrices=False)
    s13 = 1.0 / (np.arange(1, 301) ** 0.7)  # Polynomial: 1/k^0.7
    A13 = U13 @ np.diag(s13) @ Vh13
    results.append(compare_on_matrix(A13, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 400x300)"))

    # Test 14: Slow decay - polynomial (large)
    A14 = np.random.randn(800, 600)
    U14, S14, Vh14 = np.linalg.svd(A14, full_matrices=False)
    s14 = 1.0 / (np.arange(1, 601) ** 0.7)  # Polynomial: 1/k^0.7
    A14 = U14 @ np.diag(s14) @ Vh14
    results.append(compare_on_matrix(A14, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 800x600)"))

    # Test 15: Slow decay - exponential (small)
    A15 = np.random.randn(400, 300)
    U15, S15, Vh15 = np.linalg.svd(A15, full_matrices=False)
    s15 = np.exp(-np.arange(1, 301) / 100.0)  # Exponential: exp(-k/100)
    A15 = U15 @ np.diag(s15) @ Vh15
    results.append(compare_on_matrix(A15, 1e-3, "Slow Decay - Exponential (exp(-k/100), 400x300)"))

    # Test 16: Slow decay - exponential (large)
    A16 = np.random.randn(800, 600)
    U16, S16, Vh16 = np.linalg.svd(A16, full_matrices=False)
    s16 = np.exp(-np.arange(1, 601) / 150.0)  # Exponential: exp(-k/150)
    A16 = U16 @ np.diag(s16) @ Vh16
    results.append(compare_on_matrix(A16, 1e-3, "Slow Decay - Exponential (exp(-k/150), 800x600)"))

    # -------------------------------------------------------------------------
    # EXTRA LARGE MATRIX TESTS (4x SCALE)
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("EXTRA LARGE MATRIX TESTS (4x SCALE)")
    print("Testing scaling behavior with matrices 4x larger than base")
    print("="*70)

    # Test 17: 4x Random matrix
    A17 = np.random.randn(2000, 1200)
    results.append(compare_on_matrix(A17, 20, "XL Random Matrix (2000x1200)"))

    # Test 18: 4x Low-rank matrix
    U18 = np.random.randn(1600, 15)
    V18 = np.random.randn(1000, 15)
    A18 = U18 @ V18.T + 1e-10 * np.random.randn(1600, 1000)
    results.append(compare_on_matrix(A18, 1e-8, "XL Low-Rank Matrix (1600x1000, rank~15)"))

    # Test 19: 4x Hilbert matrix - WARNING: VERY SLOW!
    A19 = hilb(8000, 4000)
    results.append(compare_on_matrix(A19, 15, "XL Hilbert Matrix (8000x4000)"))

    # Test 20: 4x Complex matrix
    A20 = np.random.randn(1200, 800) + 1j * np.random.randn(1200, 800)
    results.append(compare_on_matrix(A20, 25, "XL Complex Matrix (1200x800)"))

    # Test 21: 4x Decaying spectrum
    A21 = np.random.randn(1600, 1200)
    U21, S21, Vh21 = np.linalg.svd(A21, full_matrices=False)
    s21 = 1.0 / np.arange(1, 1201)  # Fast decay: 1/k
    A21 = U21 @ np.diag(s21) @ Vh21
    results.append(compare_on_matrix(A21, 1e-3, "XL Decaying Spectrum (1/k, 1600x1200)"))

    # -------------------------------------------------------------------------
    # MAKE_MAT MATRIX TESTS (structured matrices from paper)
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("MAKE_MAT STRUCTURED MATRIX TESTS")
    print("Testing matrices from 'Robust blockwise random pivoting' paper")
    print("="*70)

    # Test 22: Gaussian Exponential Decay Matrix
    A22 = make_mat(500, 500, 'gaussexp')
    results.append(compare_on_matrix(A22, 1e-3, "Gaussexp (Gaussian Exponential Decay, 500x500)"))

    # Test 23: Gaussian Mixture Model Matrix
    A23 = make_mat(400, 400, 'gmm')
    results.append(compare_on_matrix(A23, 1e-3, "GMM (Gaussian Mixture Model, 400x400)"))

    # Test 24: Sparse Neural Network Matrix
    A24 = make_mat(300, 300, 'snn')
    results.append(compare_on_matrix(A24, 1e-3, "SNN (Sparse Neural Network, 300x300)"))

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
    avg_time_rrqr = np.mean([r.t_rrqr for r in results])

    print()
    print(f"{'Method':<15s} {'Avg Time':<12s} {'vs RRQR':<12s}")
    print("-"*80)
    print(f"{'sketch':<15s} {avg_time_sketch:>8.4f}s    {avg_time_rrqr/avg_time_sketch:>6.1f}x ")
    print(f"{'RRQR':<15s} {avg_time_rrqr:>8.4f}s    {1.0:>6.1f}x -")

    print()
    print("Conditioning Summary (max|T|):")
    print("-"*80)

    avg_maxT_sketch = np.mean([r.maxT_sketch for r in results])
    avg_maxT_rrqr = np.mean([r.maxT_rrqr for r in results])
    min_maxT_sketch = np.min([r.maxT_sketch for r in results])
    min_maxT_rrqr = np.min([r.maxT_rrqr for r in results])
    max_maxT_sketch = np.max([r.maxT_sketch for r in results])
    max_maxT_rrqr = np.max([r.maxT_rrqr for r in results])

    print(f"  id_sketch:   mean={avg_maxT_sketch:.3e}, min={min_maxT_sketch:.3e}, max={max_maxT_sketch:.3e}")
    print(f"  id_rrqr:     mean={avg_maxT_rrqr:.3e}, min={min_maxT_rrqr:.3e}, max={max_maxT_rrqr:.3e}")

    print()
    print("="*80)

    # Return exit code based on pass/fail
    return 0 if all(r.passed for r in results) else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
