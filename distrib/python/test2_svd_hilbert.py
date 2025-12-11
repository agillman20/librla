#!/usr/bin/env python3
"""
test2_svd_hilbert.py - Simple test with medium-size Hilbert matrix for SVD

Tests SVD algorithms (svd_sketch vs standard SVD) on an ill-conditioned
Hilbert matrix.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: BSD-3-Clause
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

import numpy as np
import time
import sys
sys.path.insert(0, '.')

from librla import svd_sketch

def hilb(m, n=None):
    """Generate Hilbert matrix."""
    if n is None:
        n = m
    i = np.arange(1, m+1).reshape(-1, 1)
    j = np.arange(1, n+1).reshape(1, -1)
    return 1.0 / (i + j - 1)


def test_svd_hilbert():
    """Test SVD algorithms on Hilbert matrix."""

    print("="*70)
    print("TEST 2: Medium Hilbert Matrix - SVD")
    print("="*70)

    # Create medium-size Hilbert matrix (severely ill-conditioned)
    m, n = 300, 200
    print(f"\nMatrix size: {m} x {n}")
    print("Matrix type: Hilbert (severely ill-conditioned)")

    A = hilb(m, n)
    normA = np.linalg.norm(A, 'fro')

    # Target rank
    k_target = 15
    print(f"Target rank: {k_target}")
    print("="*70)

    # -------------------------------------------------------------------------
    # Method 1: svd_sketch (randomized)
    # -------------------------------------------------------------------------
    print("\n1. svd_sketch (randomized SVD via sketching)")
    print("-"*70)

    t0 = time.perf_counter()
    U1, s1, Vh1 = svd_sketch(A, rtol=float(k_target))
    t1 = time.perf_counter() - t0

    k1 = len(s1)

    # Compute reconstruction error (Vh is already conjugate transposed)
    A1_recon = U1 @ np.diag(s1) @ Vh1
    err1 = np.linalg.norm(A - A1_recon, 'fro') / normA

    # Singular value accuracy (compare to reference)
    U_ref, s_ref, Vh_ref = np.linalg.svd(A, full_matrices=False)
    sval_err1 = np.linalg.norm(s1 - s_ref[:k1]) / np.linalg.norm(s_ref[:k1])

    print(f"  Rank:      k = {k1}")
    print(f"  Error:     ||A - U @ S @ Vh|| / ||A|| = {err1:.3e}")
    print(f"  SVal Err:  ||s - s_ref|| / ||s_ref|| = {sval_err1:.3e}")
    print(f"  Time:      {t1:.4f} s")

    # -------------------------------------------------------------------------
    # Method 2: svd (LAPACK, truncated)
    # -------------------------------------------------------------------------
    print("\n2. svd (LAPACK, deterministic, truncated)")
    print("-"*70)

    t0 = time.perf_counter()
    U2, s2_full, Vh2 = np.linalg.svd(A, full_matrices=False)
    t2 = time.perf_counter() - t0

    # Truncate to target rank
    k2 = k_target
    U2_k = U2[:, :k2]
    s2 = s2_full[:k2]
    Vh2_k = Vh2[:k2, :]

    # Reconstruction error
    A2_recon = U2_k @ np.diag(s2) @ Vh2_k
    err2 = np.linalg.norm(A - A2_recon, 'fro') / normA

    # Singular value accuracy
    sval_err2 = np.linalg.norm(s2 - s_ref[:k2]) / np.linalg.norm(s_ref[:k2])

    print(f"  Rank:      k = {k2}")
    print(f"  Error:     ||A - U @ S @ Vh|| / ||A|| = {err2:.3e}")
    print(f"  SVal Err:  ||s - s_ref|| / ||s_ref|| = {sval_err2:.3e}")
    print(f"  Time:      {t2:.4f} s")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Method         Rank    Recon Error   SVal Error    Time")
    print("-"*70)
    print(f"  svd_sketch     {k1:4d}    {err1:.3e}    {sval_err1:.3e}    {t1:.4f}s")
    print(f"  svd (LAPACK)   {k2:4d}    {err2:.3e}    {sval_err2:.3e}    {t2:.4f}s")
    print("="*70)

    # Validate
    if err1 > 1.0 or err2 > 1.0:
        print("\n[FAIL] Reconstruction error > 1.0 detected!")
        return False

    if sval_err1 > 1e-6 or sval_err2 > 1e-10:
        print("\n[FAIL] Singular value error too large!")
        return False

    print("\n[PASS] Test completed successfully!")
    return True


if __name__ == '__main__':
    success = test_svd_hilbert()
    sys.exit(0 if success else 1)
