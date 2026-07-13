#!/usr/bin/env python3
"""
demo04_power.py - Power Iteration for Improved Accuracy

This demo shows how power iteration improves sketching accuracy.

Power iteration applies (A'A)^q to the random sketch, which amplifies
the dominant singular components. This is especially helpful when:
  - The spectral gap is small
  - High accuracy is needed
  - The matrix has slowly decaying singular values

The demo tests a 2D grid of parameters:
  - extra_samples: How much oversampling (more = better accuracy)
  - power_iter: Number of power iterations (more = better accuracy)

Trade-off: More iterations/samples = better accuracy but more computation.

Try changing the CONFIGURATION parameters below to experiment!

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.1.0
Date: July 13, 2026
Assisted by: Claude Code (Anthropic)
"""

# =============================================================================
# CONFIGURATION - Modify these to experiment
# =============================================================================

MATRIX_SIZE = (2000, 1000)    # (rows, columns)
TARGET_RANK = 50            # Number of singular values to compute
RANDOM_SEED = 42            # For reproducibility

# Power iteration settings - test grid of values
EXTRA_SAMPLES_LIST = [24, 18, 12, 6, 3]  # Oversampling values to test
POWER_ITER_LIST = [0, 1, 2, 3, 4]  # Power iteration counts to test

# Matrix type: 'structured' (clear spectral gap) or 'random' (no gap)
MATRIX_TYPE = 'structured'

# =============================================================================
# Demo code below
# =============================================================================

import numpy as np
import time
import sys
sys.path.insert(0, '.')
sys.path.insert(0,'..')

from scipy import linalg
from librla import svd_sketch
from demo_utils import print_header, print_subheader


def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    m, n = MATRIX_SIZE
    k = TARGET_RANK

    print_header("Demo 04: Power Iteration")
    print(f"\nMatrix: {m} x {n}")
    print(f"Target rank: {k}")

    # -------------------------------------------------------------------------
    # Create test matrix
    # -------------------------------------------------------------------------
    if MATRIX_TYPE == 'structured':
        print("Matrix type: STRUCTURED")
        print("Singular values: logspace(0,-2,k) + logspace(-2,-10,n-k)")
        # Create matrix with decaying spectrum and clear gap at rank k
        U_full = linalg.orth(np.random.randn(m, m))
        V_full = linalg.orth(np.random.randn(n, n))
        s_true = np.concatenate([
            np.logspace(0, -2, k),      # Fast decay in first k singular values
            np.logspace(-2, -10, n-k)   # Slow decay after
        ])
        U = U_full[:, :n]
        A = U @ np.diag(s_true) @ V_full.T
    else:
        print("Matrix type: RANDOM (no spectral gap)")
        A = np.random.randn(m, n)
        s_true = linalg.svd(A, compute_uv=False)

    # Matrix properties
    cond = s_true[0] / s_true[-1]
    gap = s_true[k-1] / s_true[k] if k < len(s_true) else float('inf')

    print(f"\nSpectral properties:")
    print(f"   s[0]     = {s_true[0]:.6e} (largest)")
    print(f"   s[{k-1}]   = {s_true[k-1]:.6e} (at target rank)")
    print(f"   s[{k}]   = {s_true[k]:.6e} (first neglected)")
    print(f"   s[{min(m-1,n-1)}] = {s_true[min(m-1,n-1)]:.6e} (smallest)")
    print(f"   Condition number: {cond:.2e}")
    print(f"   Spectral gap at k={k}: {gap:.1f}x")

    # -------------------------------------------------------------------------
    # Test grid of extra_samples and power_iter values
    # -------------------------------------------------------------------------
    print_subheader("Testing Parameter Grid")
    print("   power_iter=0 means no power iteration (baseline)")
    print("   Each power iteration costs 2 extra matrix-vector products")
    print("   extra_samples controls oversampling (block_size = k + extra_samples)")

    # Store results in 2D arrays
    errors = np.zeros((len(EXTRA_SAMPLES_LIST), len(POWER_ITER_LIST)))
    sval_errors = np.zeros((len(EXTRA_SAMPLES_LIST), len(POWER_ITER_LIST)))
    s_ref = s_true[:k]

    for idx, extra_samples in enumerate(EXTRA_SAMPLES_LIST):
        block_size = k + extra_samples
        print(f"\n--- extra_samples = {extra_samples} (block_size = {block_size}) ---")

        for jdx, power_iter in enumerate(POWER_ITER_LIST):
            t0 = time.perf_counter()
            U, s, Vh = svd_sketch(A, rtol=float(k),
                                  power_iter=power_iter,
                                  extra_samples=extra_samples)
            elapsed = time.perf_counter() - t0

            # Reconstruction error
            A_approx = U @ np.diag(s) @ Vh
            recon_err = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
            errors[idx, jdx] = recon_err

            # Singular value accuracy
            sval_err = np.linalg.norm(s - s_ref) / np.linalg.norm(s_ref)
            sval_errors[idx, jdx] = sval_err

            print(f"   power_iter={power_iter}: err={recon_err:.2e}, sval_err={sval_err:.2e}, time={elapsed:.4f}s")

    # -------------------------------------------------------------------------
    # Summary tables
    # -------------------------------------------------------------------------
    print_subheader("Summary: Reconstruction Error")

    # Header row
    header = "extra_samples |"
    for p in POWER_ITER_LIST:
        header += f"  iter={p}  |"
    print(header)
    print("-" * 14 + "+" + ("-" * 10 + "+") * len(POWER_ITER_LIST))

    # Data rows
    for idx, extra_samples in enumerate(EXTRA_SAMPLES_LIST):
        row = f"{extra_samples:>13} |"
        for jdx in range(len(POWER_ITER_LIST)):
            row += f" {errors[idx, jdx]:.2e} |"
        print(row)

    print_subheader("Summary: Singular Value Error")

    # Header row
    header = "extra_samples |"
    for p in POWER_ITER_LIST:
        header += f"  iter={p}  |"
    print(header)
    print("-" * 14 + "+" + ("-" * 10 + "+") * len(POWER_ITER_LIST))

    # Data rows
    for idx, extra_samples in enumerate(EXTRA_SAMPLES_LIST):
        row = f"{extra_samples:>13} |"
        for jdx in range(len(POWER_ITER_LIST)):
            row += f" {sval_errors[idx, jdx]:.2e} |"
        print(row)

    print("\nNotes:")
    print("  - Power iteration amplifies dominant singular components")
    print("  - Extra samples (oversampling) improves subspace capture")

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
