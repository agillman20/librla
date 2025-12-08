#!/usr/bin/env python3
"""
test1_hilbert.py - Simple test with medium-size Hilbert matrix

Tests basic ID algorithms (id_sketch, id_qrpiv) on an ill-conditioned
Hilbert matrix.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '.')

from librla import id_sketch, id_qrpiv

def hilb(m, n=None):
    """Generate Hilbert matrix."""
    if n is None:
        n = m
    i = np.arange(1, m+1).reshape(-1, 1)
    j = np.arange(1, n+1).reshape(1, -1)
    return 1.0 / (i + j - 1)


def test_hilbert():
    """Test basic ID algorithms on Hilbert matrix."""

    print("="*70)
    print("TEST 1: Medium Hilbert Matrix")
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
    # Method 1: id_sketch (randomized)
    # -------------------------------------------------------------------------
    print("\n1. id_sketch (randomized QR sketching)")
    print("-"*70)

    t0 = time.perf_counter()
    k1, piv1, T1 = id_sketch(A, rtol=float(k_target))
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

    # -------------------------------------------------------------------------
    # Method 2: id_qrpiv (deterministic QR)
    # -------------------------------------------------------------------------
    print("\n2. id_qrpiv (deterministic QR via LAPACK)")
    print("-"*70)

    t0 = time.perf_counter()
    k2, piv2, T2 = id_qrpiv(A, rtol=float(k_target))
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

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Method         Rank    Error        Max|T|       Time")
    print("-"*70)
    print(f"  id_sketch      {k1:4d}    {err1:.3e}    {maxT1:.3e}    {t1:.4f}s")
    print(f"  id_qrpiv       {k2:4d}    {err2:.3e}    {maxT2:.3e}    {t2:.4f}s")
    print("="*70)

    # Validate
    if err1 > 1.0 or err2 > 1.0:
        print("\n[FAIL] Error > 1.0 detected!")
        return False

    print("\n[PASS] Test completed successfully!")
    return True


if __name__ == '__main__':
    success = test_hilbert()
    sys.exit(0 if success else 1)
