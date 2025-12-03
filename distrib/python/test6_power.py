#!/usr/bin/env python3
"""
test6_power.py - Test power iteration for range estimation (librla version)

Tests simple power iteration for range estimation in sketching algorithms.
Power iteration applies (A^H A)^n to improve sketch quality by amplifying
the dominant subspace.

This version uses librla instead of libid.

Usage
-----
Default (structured matrix only):
    python test6_power.py

With random matrix comparison:
    python test6_power.py --random

Tests
-----
Test 1: Range estimation quality
    - Measures subspace angles to dominant subspace
    - Tests extra_samples: 24, 18, 12, 6, 3
    - Tests iterations: 0-6
    - Can use structured or random matrix

Author: Power iteration range estimator tests (librla version)
"""

import numpy as np
import time
from librla import svd_sketch, _power_iteration, _uniform_omega
from scipy import linalg


def subspace_angle(Q1, Q2):
    """
    Compute maximum principal angle between subspaces span(Q1) and span(Q2).

    Returns angle in degrees.
    """
    if Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return 90.0

    # Compute singular values of Q1^H @ Q2
    M = Q1.conj().T @ Q2
    s = linalg.svdvals(M)

    # Principal angles: theta_i = arccos(s_i)
    # Maximum angle (worst alignment)
    theta_max = np.arccos(np.clip(s[-1], 0, 1))
    return np.degrees(theta_max)


def test_range_estimation_quality(use_random_matrix=False):
    """
    Test 1: Range estimation quality with power iteration.

    Measures subspace angles to true dominant subspace across different iteration counts
    and different oversampling parameters (extra_samples).

    Parameters
    ----------
    use_random_matrix : bool, optional
        If True, use a random matrix without prescribed singular values.
        If False (default), use structured matrix with controlled spectrum.
    """
    print("\n" + "="*70)
    print("TEST 1: Range Estimation Quality")
    print("="*70)

    # Test matrix configuration
    m, n = 500, 300
    k = 50  # True rank we want to capture

    if use_random_matrix:
        print("\nMatrix type: RANDOM (no prescribed singular values)")
        # Simple random matrix
        A = np.random.randn(m, n)

        # Compute SVD to get true dominant subspace
        U_full, s, Vh_full = linalg.svd(A, full_matrices=False)
        V = Vh_full.T
        V_true = V[:, :k]

        # Note: Random matrices have clustered singular values with minimal
        # spectral gap, so power iteration converges much more slowly than
        # for structured matrices with prescribed decay.

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

        # True dominant subspace (right singular vectors)
        V_true = V[:, :k]

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
    print(f"  s[0]    = {s[0]:.12e} (largest)")
    print(f"  s[{k-1:>3}]  = {s[k-1]:.12e} (target cutoff)")
    print(f"  s[{k:>3}]  = {s[k]:.12e} (first neglected)")
    print(f"  s[{n-1:>3}] = {s[n-1]:.12e} (smallest)")

    # Test different extra_samples values
    extra_samples_list = [24, 18, 12, 6, 3]
    num_iters_list = list(range(7))

    # Store results for summary
    angles = np.zeros((len(extra_samples_list), len(num_iters_list)))

    for idx, extra_samples in enumerate(extra_samples_list):
        block_size = k + extra_samples
        print(f"\n" + "="*70)
        print(f"extra_samples = {extra_samples} (block_size = {block_size})")
        print("="*70)

        # Test different iteration counts (0-6)
        for jdx, num_iters in enumerate(num_iters_list):
            print(f"\n--- num_iters = {num_iters} ---")

            # Generate same random test matrix for fair comparison
            np.random.seed(42)
            X_init = _uniform_omega(A, n, block_size)

            # Power iteration
            t0 = time.perf_counter()
            X_power = X_init.copy()
            # Orthogonalize X_init when num_iters=0
            if num_iters == 0:
                X_power, _, _ = linalg.qr(X_power, mode='economic', pivoting=True)
            else:
                X_power = _power_iteration(A, X_power, power_iter=num_iters)
            t_power = time.perf_counter() - t0
            angle_power = subspace_angle(V_true, X_power)
            angles[idx, jdx] = angle_power

            # Orthogonality check
            orth_power = np.linalg.norm(X_power.conj().T @ X_power - np.eye(X_power.shape[1]), 'fro')

            # Compute alignment with dominant singular vectors (quality metric)
            # How well does the subspace capture the top-k singular directions?
            M_power = V_true.T @ X_power
            svals_power = linalg.svdvals(M_power)
            capture_quality_power = np.mean(svals_power)  # Average alignment

            print(f"Power iteration:")
            print(f"  Subspace angle:   {angle_power:18.12f}deg")
            print(f"  Capture quality:  {capture_quality_power:.12f} (mean singular value)")
            print(f"  Orthogonality:    ||Q^H Q - I||_F = {orth_power:.12e}")
            print(f"  Time:             {t_power:.6f}s")
            print(f"  Basis size:       {X_power.shape[1]}")

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

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Subspace angles (degrees)")
    print("="*70)
    header = "extra_samples |"
    for num_iters in num_iters_list:
        header += f" iter={num_iters} |"
    print(header)
    print("--------------+" + "--------+" * len(num_iters_list))
    for idx, extra_samples in enumerate(extra_samples_list):
        row = f"{extra_samples:>13} |"
        for jdx in range(len(num_iters_list)):
            row += f" {angles[idx, jdx]:6.2f} |"
        print(row)

    print("\n[PASS] Test 1 complete")


def main(test_random=False):
    """
    Run power iteration range estimation tests.

    Parameters
    ----------
    test_random : bool, optional
        If True, run Test 1 with both structured and random matrices.
        If False (default), run Test 1 only with structured matrix.
    """
    print("="*70)
    print("POWER ITERATION RANGE ESTIMATION TESTS (librla)")
    print("="*70)

    np.random.seed(42)  # For reproducibility

    # Test 1: Range estimation quality
    if test_random:
        # Run with structured matrix first
        test_range_estimation_quality(use_random_matrix=False)
        # Then run with random matrix
        test_range_estimation_quality(use_random_matrix=True)
    else:
        # Default: structured matrix only
        test_range_estimation_quality(use_random_matrix=False)

    print("\n" + "="*70)
    print("ALL TESTS PASSED [PASS]")
    print("="*70)


if __name__ == "__main__":
    import sys
    # Check for --random flag
    test_random = "--random" in sys.argv
    main(test_random=test_random)
