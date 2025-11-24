#!/usr/bin/env python3
"""
test5_method_comparison.py - Compare all three T computation methods

Tests all three T computation methods on a full-rank random matrix:
1. method='fast' - Triangular solve (fastest, may have error > 1.0)
2. method='svd' - SVD-based pseudoinverse (stable)
3. method='lstsq' - Least-squares from original A (most accurate)
"""

import numpy as np
import time
import sys
sys.path.insert(0, '.')

from librla import id_sketch


def test_method_comparison():
    """Compare all four T computation methods."""

    print("="*70)
    print("TEST 5: T Computation Method Comparison")
    print("="*70)

    # Create full-rank random matrix
    np.random.seed(42)
    m = 400
    n = 300
    print(f"\nMatrix size: {m} x {n}")
    print(f"Matrix type: Full-rank random (all {n} columns independent)")

    # Create full-rank matrix
    A = np.random.randn(m, n)
    normA = np.linalg.norm(A, 'fro')

    # Target rank (low compared to matrix rank)
    k_target = 20
    print(f"Target rank: {k_target} ({100*k_target/n:.1f}% of columns)")
    print("="*70)

    # =========================================================================
    # Test 1: method='fast' (fastest, may have error > 1.0)
    # =========================================================================
    print("\n1. method='fast' (triangular solve)")
    print("-"*70)

    t0 = time.perf_counter()
    k1, piv1, T1 = id_sketch(A, rtol=float(k_target), method='fast')
    t1 = time.perf_counter() - t0

    # Compute error
    A_skel1 = A[:, piv1[k1:]]
    A_basis1 = A[:, piv1[:k1]]
    err1 = np.linalg.norm(A_skel1 - A_basis1 @ T1, 'fro') / normA
    maxT1 = np.max(np.abs(T1)) if T1.size > 0 else 0.0

    print(f"  Rank:      k = {k1}")
    print(f"  Error:     ||A_skel - A_basis @ T|| / ||A|| = {err1:.3e}")
    print(f"  Max |T|:   {maxT1:.3e}")
    print(f"  Time:      {t1:.4f} s")
    if err1 > 1.0:
        print(f"  [NOTE] Error > 1.0 is expected for full-rank matrices with method='fast'")

    # =========================================================================
    # Test 2: method='svd' (stable for ill-conditioned)
    # =========================================================================
    print("\n2. method='svd' (SVD-based pseudoinverse)")
    print("-"*70)

    t0 = time.perf_counter()
    k2, piv2, T2 = id_sketch(A, rtol=float(k_target), method='svd')
    t2 = time.perf_counter() - t0

    # Compute error
    A_skel2 = A[:, piv2[k2:]]
    A_basis2 = A[:, piv2[:k2]]
    err2 = np.linalg.norm(A_skel2 - A_basis2 @ T2, 'fro') / normA
    maxT2 = np.max(np.abs(T2)) if T2.size > 0 else 0.0

    print(f"  Rank:      k = {k2}")
    print(f"  Error:     ||A_skel - A_basis @ T|| / ||A|| = {err2:.3e}")
    print(f"  Max |T|:   {maxT2:.3e}")
    print(f"  Time:      {t2:.4f} s")

    # =========================================================================
    # Test 3: method='lstsq' (most accurate, slowest)
    # =========================================================================
    print("\n3. method='lstsq' (least-squares from original A)")
    print("-"*70)

    t0 = time.perf_counter()
    k3, piv3, T3 = id_sketch(A, rtol=float(k_target), method='lstsq')
    t3 = time.perf_counter() - t0

    # Compute error
    A_skel3 = A[:, piv3[k3:]]
    A_basis3 = A[:, piv3[:k3]]
    err3 = np.linalg.norm(A_skel3 - A_basis3 @ T3, 'fro') / normA
    maxT3 = np.max(np.abs(T3)) if T3.size > 0 else 0.0

    print(f"  Rank:      k = {k3}")
    print(f"  Error:     ||A_skel - A_basis @ T|| / ||A|| = {err3:.3e}")
    print(f"  Max |T|:   {maxT3:.3e}")
    print(f"  Time:      {t3:.4f} s")
    if err3 < 1.0:
        print(f"  [OK] method='lstsq' guarantees error < 1.0")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Method     Rank    Error        Max|T|       Time      Notes")
    print("-"*70)
    print(f"  fast       {k1:4d}    {err1:.3e}    {maxT1:.3e}    {t1:.4f}s   Fastest")
    print(f"  svd        {k2:4d}    {err2:.3e}    {maxT2:.3e}    {t2:.4f}s   Stable")
    print(f"  lstsq      {k3:4d}    {err3:.3e}    {maxT3:.3e}    {t3:.4f}s   Most accurate")
    print("="*70)

    # Validate
    success = True

    if err3 > 1.0:
        print("\n[FAIL] method='lstsq' should guarantee error < 1.0!")
        success = False

    if k1 != k_target or k2 != k_target or k3 != k_target:
        print(f"\n[FAIL] All methods should return rank k={k_target}!")
        success = False

    if success:
        print("\n[PASS] All method tests passed!")

    return success


if __name__ == '__main__':
    success = test_method_comparison()
    sys.exit(0 if success else 1)
