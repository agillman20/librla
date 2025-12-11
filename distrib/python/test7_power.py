#!/usr/bin/env python3
"""
test7_power.py - Test power iteration in svd_sketch (librla version)

Tests power iteration in the svd_sketch function.
Power iteration applies (A^H A)^n to improve sketch quality by amplifying
the dominant subspace.

This version uses librla instead of libid.

Usage
-----
Default (structured matrix only):
    python test7_power.py

Random matrix only:
    python test7_power.py --random

Tests
-----
Test 1: Power iteration in svd_sketch
    - Tests extra_samples: 24, 18, 12, 6, 3
    - Tests power_iter: 0-6
    - Measures reconstruction error and singular value accuracy
    - Can use structured or random matrix

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

import numpy as np
import time
from librla import svd_sketch
from scipy import linalg


def test_svd_sketch_power_iter(use_random_matrix=False):
    """
    Test 1: Power iteration in svd_sketch.

    Tests how power iteration improves singular value accuracy in svd_sketch.
    """
    print("\n" + "="*70)
    print("TEST 1: Power iteration in svd_sketch")
    print("="*70)

    # Test matrix configuration (same as test6_power)
    m, n = 500, 300
    k = 50

    if use_random_matrix:
        print("\nMatrix type: RANDOM (no prescribed singular values)")
        # Simple random matrix
        A = np.random.randn(m, n)

        # Compute SVD to get true singular values
        _, s, _ = linalg.svd(A, full_matrices=False)
    else:
        print("\nMatrix type: STRUCTURED (prescribed singular values)")
        # Create matrix with decaying spectrum
        U_full = linalg.orth(np.random.randn(m, m))
        V_full = linalg.orth(np.random.randn(n, n))
        s = np.concatenate([
            np.logspace(0, -2, k),  # Fast decay in first k singular values
            np.logspace(-2, -10, n-k)  # Slow decay after
        ])
        # Use first n columns of U to match dimensions
        U = U_full[:, :n]
        V = V_full
        A = U @ np.diag(s) @ V.T

    # Compute detailed matrix properties
    cond_number = s[0] / s[-1]
    spectral_gap_k = s[k-1] / s[k]
    decay_rate_k = s[0] / s[k-1]

    print(f"\nMatrix Properties:")
    print(f"  Dimensions:       {m}x{n}")
    print(f"  Target rank:      {k} (first {k} singular values)")
    print(f"  Condition number: {cond_number:.2e}")
    print(f"  Spectral gap at k={k}: {spectral_gap_k:.2f}x (s[{k-1}]/s[{k}])")
    print(f"  Decay rate (s[0]/s[{k-1}]): {decay_rate_k:.2f}x")
    print(f"\nSingular value distribution:")
    print(f"  s[0]    = {s[0]:.6e} (largest)")
    print(f"  s[{k-1:>3}]  = {s[k-1]:.6e} (target cutoff)")
    print(f"  s[{k:>3}]  = {s[k]:.6e} (first neglected)")
    print(f"  s[{n-1:>3}]  = {s[n-1]:.6e} (smallest)")

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

    # Print matrix info before summary
    print("\n" + "="*70)
    if use_random_matrix:
        print("Matrix type: RANDOM")
        print("Singular values: from SVD of randn(m,n)")
    else:
        print("Matrix type: STRUCTURED")
        print("Singular values: logspace(0,-2,k) + logspace(-2,-10,n-k)")
    print(f"Matrix: {m}x{n}, target rank: {k}")
    print(f"Condition number: {cond_number:.2e}, Spectral gap: {spectral_gap_k:.2f}x")

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


def main(test_random=False):
    """
    Run power iteration in svd_sketch tests.

    Parameters
    ----------
    test_random : bool, optional
        If True, run test with both structured and random matrices.
        If False (default), run test only with structured matrix.
    """
    print("="*70)
    print("POWER ITERATION IN SVD_SKETCH TESTS (librla)")
    print("="*70)

    np.random.seed(42)  # For reproducibility

    # Test 1: Power iteration in svd_sketch
    test_svd_sketch_power_iter(use_random_matrix=test_random)

    print("\n" + "="*70)
    print("ALL TESTS PASSED [PASS]")
    print("="*70)


if __name__ == "__main__":
    import sys
    # Check for --help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    # Check for --random flag
    test_random = "--random" in sys.argv
    main(test_random=test_random)
