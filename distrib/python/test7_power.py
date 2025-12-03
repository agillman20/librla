#!/usr/bin/env python3
"""
test7_power.py - Test power iteration in svd_sketch (librla version)

Tests power iteration in the svd_sketch function.
Power iteration applies (A^H A)^n to improve sketch quality by amplifying
the dominant subspace.

This version uses librla instead of libid.

Usage
-----
    python test7_power.py

Tests
-----
Test 1: Power iteration in svd_sketch
    - Tests power_iter: 0-6
    - Measures reconstruction error and singular value accuracy

Author: Power iteration tests (librla version)
"""

import numpy as np
import time
from librla import svd_sketch
from scipy import linalg


def test_svd_sketch_power_iter():
    """
    Test 1: Power iteration in svd_sketch.

    Tests how power iteration improves singular value accuracy in svd_sketch.
    """
    print("\n" + "="*70)
    print("TEST 1: Power iteration in svd_sketch")
    print("="*70)

    # Test matrix with prescribed singular values
    m, n = 350, 200
    k = 40

    U_full = linalg.orth(np.random.randn(m, m))
    V_full = linalg.orth(np.random.randn(n, n))
    s = np.logspace(0, -6, n)  # Exponential decay
    # Use first n columns of U to match dimensions
    U = U_full[:, :n]
    V = V_full
    A = U @ np.diag(s) @ V.T

    print(f"\nMatrix: {m}x{n}, target rank: {k}")
    print(f"True singular values: s[0]={s[0]:.12e}, s[{k}]={s[k]:.12e}")

    # Test with different power_iter values (extended to 6)
    for power_iter in range(7):
        print(f"\n--- power_iter = {power_iter} ---")

        # Run svd_sketch
        t0 = time.perf_counter()
        U_sketch, s_sketch, Vh_sketch = svd_sketch(A, rtol=float(k), power_iter=power_iter)
        t_total = time.perf_counter() - t0

        # Compute reconstruction error
        A_approx = U_sketch @ np.diag(s_sketch) @ Vh_sketch
        err = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')

        # Singular value accuracy
        s_ref = s[:k]
        sval_err = np.linalg.norm(s_sketch - s_ref) / np.linalg.norm(s_ref)

        print(f"  Rank:         k = {len(s_sketch)}")
        print(f"  Error:        {err:.12e}")
        print(f"  SVal error:   {sval_err:.12e}")
        print(f"  Time:         {t_total:.6f}s")

        # Sanity check
        assert len(s_sketch) == k, f"Expected rank {k}, got {len(s_sketch)}"
        assert err < 0.1, f"Error {err:.3e} too large"

    print("\n[PASS] Test 1 complete")


def main():
    """
    Run power iteration in svd_sketch tests.
    """
    print("="*70)
    print("POWER ITERATION IN SVD_SKETCH TESTS (librla)")
    print("="*70)

    np.random.seed(42)  # For reproducibility

    # Test 1: Power iteration in svd_sketch
    test_svd_sketch_power_iter()

    print("\n" + "="*70)
    print("ALL TESTS PASSED [PASS]")
    print("="*70)


if __name__ == "__main__":
    main()
