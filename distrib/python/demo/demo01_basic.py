#!/usr/bin/env python3
"""
demo01_basic.py - Introduction to Interpolative Decomposition

This demo introduces the two core ID algorithms:
  - id_sketch: Randomized ID using QR sketching (fast, approximate)
  - id_qrpiv:  Deterministic ID using column-pivoted QR (exact, slower)

The ID factorizes a matrix A as:
  A[:, piv[k:]] = A[:, piv[:k]] @ T

where piv[:k] selects the "skeleton" columns and T is the interpolation matrix.

Try changing the CONFIGURATION parameters below to experiment!

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: NIST-PD
Assisted by: Claude Code (Anthropic)
"""

# =============================================================================
# CONFIGURATION - Modify these to experiment
# =============================================================================

MATRIX_SIZE = (1000, 2000)  # (rows, columns)
TARGET_RANK = 15            # Number of skeleton columns to select
RANDOM_SEED = 42            # For reproducibility (set to None for random)

# Matrix type: 'hilbert' or 'kahan'
MATRIX_TYPE = 'hilbert'
KAHAN_THETA = 0.8           # Kahan matrix parameter (only used if MATRIX_TYPE='kahan')

# =============================================================================
# Demo code below
# =============================================================================

import numpy as np
import time
import sys
sys.path.insert(0, '.')

sys.path.insert(0,'..')


from librla import id_sketch, id_qrpiv
from demo_utils import hilbert, kahan, id_error, print_header, print_subheader


def main():
    # Set random seed
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    m, n = MATRIX_SIZE
    k = TARGET_RANK

    print_header("Demo 01: Basic Interpolative Decomposition")
    print(f"\nMatrix: {m} x {n}")
    print(f"Target rank: {k}")

    # Generate test matrix
    if MATRIX_TYPE == 'hilbert':
        print("Matrix type: Hilbert (ill-conditioned)")
        A = hilbert(m, n)
    else:  # kahan
        print(f"Matrix type: Kahan (theta={KAHAN_THETA})")
        A = kahan(m, n, theta=KAHAN_THETA)

    normA = np.linalg.norm(A, 'fro')
    print(f"||A||_F = {normA:.3e}")

    # -------------------------------------------------------------------------
    # Method 1: id_sketch (randomized)
    # -------------------------------------------------------------------------
    print_subheader("1. id_sketch (randomized)")
    print("   Uses random projections + QR. Fast but approximate.")

    t0 = time.perf_counter()
    k1, piv1, T1 = id_sketch(A, rtol=float(k))
    elapsed1 = time.perf_counter() - t0

    err1 = id_error(A, k1, piv1, T1)
    maxT1 = np.max(np.abs(T1)) if T1.size > 0 else 0.0

    print(f"   Rank:     {k1}")
    print(f"   Error:    {err1:.3e}")
    print(f"   Max |T|:  {maxT1:.3e}")
    print(f"   Time:     {elapsed1:.4f} s")

    # -------------------------------------------------------------------------
    # Method 2: id_qrpiv (deterministic)
    # -------------------------------------------------------------------------
    print_subheader("2. id_qrpiv (deterministic)")
    print("   Uses LAPACK column-pivoted QR. More accurate but slower.")

    t0 = time.perf_counter()
    k2, piv2, T2 = id_qrpiv(A, rtol=float(k))
    elapsed2 = time.perf_counter() - t0

    err2 = id_error(A, k2, piv2, T2)
    maxT2 = np.max(np.abs(T2)) if T2.size > 0 else 0.0

    print(f"   Rank:     {k2}")
    print(f"   Error:    {err2:.3e}")
    print(f"   Max |T|:  {maxT2:.3e}")
    print(f"   Time:     {elapsed2:.4f} s")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_subheader("Summary")
    print(f"   {'Method':<12} {'Rank':>6} {'Error':>12} {'Max|T|':>12} {'Time':>10}")
    print(f"   {'-'*12} {'-'*6} {'-'*12} {'-'*12} {'-'*10}")
    print(f"   {'id_sketch':<12} {k1:>6} {err1:>12.3e} {maxT1:>12.3e} {elapsed1:>9.4f}s")
    print(f"   {'id_qrpiv':<12} {k2:>6} {err2:>12.3e} {maxT2:>12.3e} {elapsed2:>9.4f}s")

    # Validate
    if err1 < 1.0 and err2 < 1.0:
        print("\n   [PASS] Both methods produced valid decompositions.")
        return True
    else:
        print("\n   [FAIL] Error > 1.0 detected!")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
