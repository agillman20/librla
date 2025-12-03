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
    - Tests extra_samples: 24, 18, 12, 6, 3
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

    # Test different extra_samples and power_iter values
    extra_samples_list = [24, 18, 12, 6, 3]
    power_iter_list = list(range(7))

    # Store results for summary
    errors = np.zeros((len(extra_samples_list), len(power_iter_list)))
    sval_errors = np.zeros((len(extra_samples_list), len(power_iter_list)))

    for idx, extra_samples in enumerate(extra_samples_list):
        print(f"\n" + "="*70)
        print(f"extra_samples = {extra_samples}")
        print("="*70)

        for jdx, power_iter in enumerate(power_iter_list):
            print(f"\n--- power_iter = {power_iter} ---")

            # Run svd_sketch
            t0 = time.perf_counter()
            U_sketch, s_sketch, Vh_sketch = svd_sketch(A, rtol=float(k),
                power_iter=power_iter, extra_samples=extra_samples)
            t_total = time.perf_counter() - t0

            # Compute reconstruction error
            A_approx = U_sketch @ np.diag(s_sketch) @ Vh_sketch
            err = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
            errors[idx, jdx] = err

            # Singular value accuracy
            s_ref = s[:k]
            sval_err = np.linalg.norm(s_sketch - s_ref) / np.linalg.norm(s_ref)
            sval_errors[idx, jdx] = sval_err

            print(f"  Rank:         k = {len(s_sketch)}")
            print(f"  Error:        {err:.12e}")
            print(f"  SVal error:   {sval_err:.12e}")
            print(f"  Time:         {t_total:.6f}s")

            # Sanity check
            assert len(s_sketch) == k, f"Expected rank {k}, got {len(s_sketch)}"

    # Print summary table for reconstruction error
    print("\n" + "="*70)
    print("SUMMARY: Reconstruction error (Frobenius norm)")
    print("="*70)
    header = "extra_samples |"
    for power_iter in power_iter_list:
        header += f"  iter={power_iter}  |"
    print(header)
    print("--------------+" + "----------+" * len(power_iter_list))
    for idx, extra_samples in enumerate(extra_samples_list):
        row = f"{extra_samples:>13} |"
        for jdx in range(len(power_iter_list)):
            row += f" {errors[idx, jdx]:.2e} |"
        print(row)

    # Print summary table for singular value accuracy
    print("\n" + "="*70)
    print("SUMMARY: Singular value error (relative 2-norm)")
    print("="*70)
    header = "extra_samples |"
    for power_iter in power_iter_list:
        header += f"  iter={power_iter}  |"
    print(header)
    print("--------------+" + "----------+" * len(power_iter_list))
    for idx, extra_samples in enumerate(extra_samples_list):
        row = f"{extra_samples:>13} |"
        for jdx in range(len(power_iter_list)):
            row += f" {sval_errors[idx, jdx]:.2e} |"
        print(row)

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
