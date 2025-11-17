#!/usr/bin/env python3
"""
test3_linop_hilbert.py - Simple test with LinearOperators

Tests id_sketch with LinearOperators on a medium-size Hilbert matrix:
1. Dense matrix (baseline)
2. Explicit LinearOperator (matrix wrapper)
3. Matrix-free LinearOperator (function handles - rank mode only)
"""

import numpy as np
import time
import sys
sys.path.insert(0, '.')

from libid import id_sketch, _hilb as hilb
from make_linop import make_linop


def test_linop_hilbert():
    """Test id_sketch with LinearOperators on Hilbert matrix."""

    print("="*70)
    print("TEST 3: LinearOperators - Hilbert Matrix")
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

    # =========================================================================
    # Test 1: Dense Matrix (Baseline)
    # =========================================================================
    print("\n1. Dense Matrix (baseline)")
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

    # =========================================================================
    # Test 2: Explicit LinearOperator (Matrix Wrapper)
    # =========================================================================
    print("\n2. Explicit LinearOperator (matrix wrapper)")
    print("-"*70)

    A_linop_explicit = make_linop(A)
    print(f"  Operator: {A_linop_explicit.m} x {A_linop_explicit.n}")
    print(f"  Is explicit: {A_linop_explicit.is_explicit}")

    t0 = time.perf_counter()
    k2, piv2, T2 = id_sketch(A_linop_explicit, rtol=float(k_target))
    t2 = time.perf_counter() - t0

    # Compute error using explicit matrix access
    A_skel2 = A[:, piv2[k2:]]
    A_basis2 = A[:, piv2[:k2]]
    err2 = np.linalg.norm(A_skel2 - A_basis2 @ T2, 'fro') / normA
    maxT2 = np.max(np.abs(T2)) if T2.size > 0 else 0.0

    print(f"  Rank:      k = {k2}")
    print(f"  Error:     ||A_skel - A_basis @ T|| / ||A|| = {err2:.3e}")
    print(f"  Max |T|:   {maxT2:.3e}")
    print(f"  Time:      {t2:.4f} s")

    # Verify explicit matches dense (rank and error should be similar)
    if k1 == k2 and abs(err1 - err2) < 1e-12:
        print("  [OK] Explicit LinearOperator produces same rank and error as dense!")
    else:
        print(f"  [NOTE] k_dense={k1}, k_linop={k2}, err_diff={abs(err1-err2):.3e}")
        print("  (Pivots may differ due to randomness, but results should be similar)")

    # =========================================================================
    # Test 3: Matrix-Free LinearOperator (Function Handles - Rank Mode Only)
    # =========================================================================
    print("\n3. Matrix-free LinearOperator (function handles)")
    print("-"*70)

    # Create matrix-free operator with function handles
    def matvec(x):
        """Forward operation: y = A*x"""
        return A @ x

    def rmatvec(x):
        """Adjoint operation: y = A^H*x"""
        return A.conj().T @ x

    A_linop_mf = make_linop(m, n, matvec, rmatvec, dtype=A.dtype)
    print(f"  Operator: {A_linop_mf.m} x {A_linop_mf.n}")
    print(f"  Is explicit: {A_linop_mf.is_explicit}")
    print(f"  Mode: Rank mode only (rtol >= 1)")

    t0 = time.perf_counter()
    k3, piv3, T3 = id_sketch(A_linop_mf, rtol=float(k_target))
    t3 = time.perf_counter() - t0

    # Compute error using explicit matrix (for validation)
    A_skel3 = A[:, piv3[k3:]]
    A_basis3 = A[:, piv3[:k3]]
    err3 = np.linalg.norm(A_skel3 - A_basis3 @ T3, 'fro') / normA
    maxT3 = np.max(np.abs(T3)) if T3.size > 0 else 0.0

    print(f"  Rank:      k = {k3}")
    print(f"  Error:     ||A_skel - A_basis @ T|| / ||A|| = {err3:.3e}")
    print(f"  Max |T|:   {maxT3:.3e}")
    print(f"  Time:      {t3:.4f} s")

    if k3 == k_target:
        print(f"  [OK] Matrix-free returns target rank k={k_target}")
    else:
        print(f"  [WARNING] Expected k={k_target}, got k={k3}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Method              Rank    Error        Max|T|       Time")
    print("-"*70)
    print(f"  Dense (baseline)    {k1:4d}    {err1:.3e}    {maxT1:.3e}    {t1:.4f}s")
    print(f"  Explicit LinOp      {k2:4d}    {err2:.3e}    {maxT2:.3e}    {t2:.4f}s")
    print(f"  Matrix-free LinOp   {k3:4d}    {err3:.3e}    {maxT3:.3e}    {t3:.4f}s")
    print("="*70)

    # Validate
    success = True

    if err1 > 1.0 or err2 > 1.0 or err3 > 1.0:
        print("\n[FAIL] Error > 1.0 detected!")
        success = False

    if k1 != k2 or abs(err1 - err2) > 1e-10:
        print(f"\n[FAIL] Explicit LinearOperator should match dense! k1={k1}, k2={k2}, err_diff={abs(err1-err2):.3e}")
        success = False

    if k3 != k_target:
        print(f"\n[FAIL] Matrix-free should return rank k={k_target}!")
        success = False

    if success:
        print("\n[PASS] All LinearOperator tests passed!")

    return success


if __name__ == '__main__':
    success = test_linop_hilbert()
    sys.exit(0 if success else 1)
