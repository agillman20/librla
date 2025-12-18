#!/usr/bin/env python3
"""
compare_orth.py - Test librla orthonormal basis computation

Tests orth_sketch (randomized orthonormal basis for column space):
- Column space accuracy (how well Q spans A's column space)
- Orthonormality of Q
- diagR values (sorted column norms, conditioning indicator)
- Runtime

Usage:
    python compare_orth.py

Requires:
    - NumPy, SciPy
    - librla.py, make_mat.py in Python path

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

# Import orth_sketch implementation
from librla import orth_sketch
from test_utils import make_mat


@dataclass
class ComparisonResult:
    """Results from testing orth_sketch on a single matrix."""
    name: str
    rtol_or_rank: float

    # Rank
    k: int

    # Column space error: ||A - Q @ Q' @ A|| / ||A||
    span_err: float

    # Orthonormality error: ||Q'Q - I||
    orth_err: float

    # diagR ratio: smallest/largest (conditioning indicator)
    diagR_ratio: float

    # Flag from orth_sketch (0=success, 1=early termination)
    flag: int

    # Timing
    t_sketch: float

    passed: bool


def hilbert(m, n):
    """Generate an mxn Hilbert matrix."""
    i = np.arange(1, m + 1).reshape(-1, 1)
    j = np.arange(1, n + 1).reshape(1, -1)
    return 1.0 / (i + j - 1)


def run_test_case(A, rtol_or_rank, name):
    """
    Test orth_sketch on a single matrix.

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
        Test metrics
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
    # orth_sketch (randomized)
    # -------------------------------------------------------------------------
    print("\n--- orth_sketch (randomized) ---")

    t0 = time.perf_counter()
    Q, flag, diagR = orth_sketch(A, rtol=float(rtol_or_rank))
    t_sketch = time.perf_counter() - t0

    k = Q.shape[1]

    # Column space error: how well Q spans A's column space
    # ||A - Q @ Q' @ A|| / ||A|| measures the approximation quality
    QQtA = Q @ (Q.conj().T @ A)
    span_err = np.linalg.norm(A - QQtA, 'fro') / normA

    # Orthonormality check
    orth_err = np.linalg.norm(Q.conj().T @ Q - np.eye(k), 'fro')

    # diagR ratio (conditioning indicator)
    # diagR contains sorted diagonal elements of R from internal QR
    # Use absolute values since diagonal elements can be negative
    abs_diagR = np.abs(diagR)
    if len(abs_diagR) > 0 and abs_diagR[0] != 0:
        diagR_ratio = abs_diagR[-1] / abs_diagR[0]
    else:
        diagR_ratio = 0.0

    print(f"Rank:       k = {k}")
    print(f"Flag:       {flag} ({'success' if flag == 0 else 'early termination'})")
    print(f"Span Err:   ||A - Q @ Q' @ A|| / ||A|| = {span_err:.3e}")
    print(f"Orth Err:   ||Q'Q - I|| = {orth_err:.3e}")
    print(f"|diagR[0]|: {abs_diagR[0]:.3e}" if len(abs_diagR) > 0 else "|diagR[0]|: N/A")
    print(f"|diagR[-1]|:{abs_diagR[-1]:.3e}" if len(abs_diagR) > 0 else "|diagR[-1]|:N/A")
    print(f"diagR ratio: {diagR_ratio:.3e}")
    print(f"Time:       {t_sketch:.4f} s")

    # -------------------------------------------------------------------------
    # Compare with full SVD (reference)
    # -------------------------------------------------------------------------
    print("\n--- Reference (full SVD truncated) ---")

    t0 = time.perf_counter()
    U_ref, s_ref, Vh_ref = np.linalg.svd(A, full_matrices=False)
    t_ref = time.perf_counter() - t0

    # Truncate to same rank
    U_k = U_ref[:, :k]

    # Reference column space error
    UUtA = U_k @ (U_k.conj().T @ A)
    span_err_ref = np.linalg.norm(A - UUtA, 'fro') / normA

    print(f"Rank:       k = {k}")
    print(f"Span Err:   ||A - U_k @ U_k' @ A|| / ||A|| = {span_err_ref:.3e}")
    print(f"Time:       {t_ref:.4f} s (full SVD)")

    # -------------------------------------------------------------------------
    # Summary comparison
    # -------------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"{'Method':<28s} {'Rank':<8s} {'Span Err':<12s} {'Orth Err':<12s} {'Time (s)':<10s}")
    print("-"*75)
    print(f"{'orth_sketch (randomized)':<28s} {k:<8d} {span_err:<12.3e} {orth_err:<12.3e} {t_sketch:<10.4f}")
    print(f"{'SVD (optimal)':<28s} {k:<8d} {span_err_ref:<12.3e} {'(ref)':<12s} {t_ref:<10.4f}")

    # Quality comparison
    if span_err_ref > 0:
        quality_ratio = span_err / span_err_ref
        print(f"\nQuality ratio: orth_sketch error / optimal error = {quality_ratio:.2f}x")

    # Speedup
    if t_sketch > 0:
        speedup = t_ref / t_sketch
        if speedup > 1:
            print(f"Speedup: orth_sketch is {speedup:.1f}x faster than full SVD")
        else:
            print(f"Speedup: full SVD is {1/speedup:.1f}x faster")

    # -------------------------------------------------------------------------
    # Determine if test passed
    # -------------------------------------------------------------------------
    # Orthonormality should be near machine precision (or 0 if k=0)
    # Early termination (flag=1) is OK - it's expected for some matrices
    # Key criterion: span error should be close to optimal (SVD reference)

    orth_ok = (k == 0) or (orth_err < 1e-10)

    # Quality threshold: randomized methods typically achieve within 8x of optimal
    # (slightly relaxed to account for randomness in ill-conditioned cases)
    quality_threshold = 8.0

    if rtol_or_rank < 1:
        # Tolerance mode: span error should be within threshold of optimal
        # (optimal is what truncated SVD achieves at the same rank)
        if span_err_ref == 0:
            quality_ok = (span_err < 1e-10)
        else:
            quality_ok = (span_err / max(span_err_ref, 1e-15) < quality_threshold)
        passed = quality_ok and orth_ok
    else:
        # Rank mode: span error should be within threshold of optimal
        passed = (span_err_ref == 0 or span_err / max(span_err_ref, 1e-15) < quality_threshold) and orth_ok

    return ComparisonResult(
        name=name,
        rtol_or_rank=float(rtol_or_rank),
        k=k,
        span_err=span_err,
        orth_err=orth_err,
        diagR_ratio=diagR_ratio,
        flag=flag,
        t_sketch=t_sketch,
        passed=passed
    )


def main():
    """Run comprehensive orth_sketch tests."""

    print("="*70)
    print("ORTH_SKETCH TESTS")
    print("Testing orthonormal basis computation via randomized sketching")
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

    avg_time = np.mean([r.t_sketch for r in results])
    min_time = np.min([r.t_sketch for r in results])
    max_time = np.max([r.t_sketch for r in results])

    print()
    print(f"orth_sketch timing: mean={avg_time:.4f}s, min={min_time:.4f}s, max={max_time:.4f}s")

    # Accuracy summary
    print()
    print("Column Space Error Summary (||A - Q Q' A|| / ||A||):")
    print("-"*80)

    avg_span_err = np.mean([r.span_err for r in results])
    max_span_err = np.max([r.span_err for r in results])

    print(f"  orth_sketch:    mean={avg_span_err:.3e}, max={max_span_err:.3e}")

    # Orthonormality summary
    print()
    print("Orthonormality Summary (||Q'Q - I||):")
    print("-"*80)

    max_orth_err = np.max([r.orth_err for r in results])

    print(f"  orth_sketch:    max={max_orth_err:.3e}")

    # Flag summary
    early_term_count = sum(1 for r in results if r.flag == 1)
    print()
    print(f"Early termination flags: {early_term_count}/{total_tests}")

    print()
    print("="*80)

    # Return exit code based on pass/fail
    return 0 if all(r.passed for r in results) else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
