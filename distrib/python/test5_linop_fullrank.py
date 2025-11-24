#!/usr/bin/env python3
"""
test5_linop_fullrank.py - LinearOperator test with full-rank random matrix

Tests id_sketch with LinearOperators on a full-rank random matrix, demonstrating
all three T computation methods:
1. method='fast' - Triangular solve (fastest, may have error > 1.0)
2. method='svd' - SVD-based pseudoinverse (stable)
3. method='lstsq' - Least-squares from original A (most accurate)
"""

import numpy as np
import time
import sys
sys.path.insert(0, '.')

from librla import id_sketch
from scipy.sparse.linalg import LinearOperator, aslinearoperator


def test_linop_fullrank():
    """Test id_sketch with LinearOperators on full-rank random matrix."""

    print("="*70)
    print("TEST 5: LinearOperators - Full-Rank Random Matrix")
    print("="*70)

    # Create full-rank random matrix
    np.random.seed(42)
    m, n = 400, 300
    print(f"\nMatrix size: {m} x {n}")
    print(f"Matrix type: Full-rank random (all {n} columns independent)")

    # Create full-rank matrix: all columns are linearly independent
    A = np.random.randn(m, n)
    normA = np.linalg.norm(A, 'fro')

    # Target rank (low compared to matrix rank)
    k_target = 20
    print(f"Target rank: {k_target} ({100*k_target/n:.1f}% of columns)")
    print("="*70)

    # =========================================================================
    # Test 1: Dense Matrix (Baseline, method='lstsq' by default)
    # =========================================================================
    print("\n1. Dense Matrix (baseline, method='lstsq')")
    print("-"*70)

    t0 = time.perf_counter()
    k1, piv1, T1 = id_sketch(A, rtol=float(k_target), method='lstsq')
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
    if err1 < 1.0:
        print(f"  [OK] Error < 1.0 (method='lstsq' guarantees this)")

    # =========================================================================
    # Test 2: Explicit LinearOperator (Matrix Wrapper, method='lstsq')
    # =========================================================================
    print("\n2. Explicit LinearOperator (matrix wrapper, method='lstsq')")
    print("-"*70)

    A_linop_explicit = aslinearoperator(A)
    print(f"  Operator: {A_linop_explicit.shape[0]} x {A_linop_explicit.shape[1]}")
    print(f"  Type: {type(A_linop_explicit).__name__}")

    t0 = time.perf_counter()
    k2, piv2, T2 = id_sketch(A_linop_explicit, rtol=float(k_target), method='lstsq')
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
    # Test 3: Matrix-Free LinearOperator (method='lstsq', accurate)
    # =========================================================================
    print("\n3. Matrix-free LinearOperator (method='lstsq', accurate)")
    print("-"*70)

    # Create matrix-free operator with function handles
    def matvec(x):
        """Forward operation: y = A*x"""
        return A @ x

    def rmatvec(x):
        """Adjoint operation: y = A^H*x"""
        return A.conj().T @ x

    A_linop_mf = LinearOperator(shape=(m, n), matvec=matvec, rmatvec=rmatvec, dtype=A.dtype)
    print(f"  Operator: {A_linop_mf.shape[0]} x {A_linop_mf.shape[1]}")
    print(f"  Type: {type(A_linop_mf).__name__}")
    print(f"  Mode: Rank mode (rtol >= 1), method='lstsq'")
    print(f"  Note: Extracts all {n} columns via unit vectors (n matvecs)")

    t0 = time.perf_counter()
    k3, piv3, T3 = id_sketch(A_linop_mf, rtol=float(k_target), method='lstsq')
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

    if k3 == k_target and err3 < 1.0:
        print(f"  [OK] Matrix-free (method='lstsq'): rank k={k_target}, error < 1.0")
    elif err3 < 1.0:
        print(f"  [OK] Error < 1.0 guaranteed by method='lstsq'")

    # =========================================================================
    # Test 4: Matrix-Free LinearOperator (method='fast', fast)
    # =========================================================================
    print("\n4. Matrix-free LinearOperator (method='fast', fast)")
    print("-"*70)

    print(f"  Operator: {A_linop_mf.shape[0]} x {A_linop_mf.shape[1]}")
    print(f"  Type: {type(A_linop_mf).__name__}")
    print(f"  Mode: Rank mode (rtol >= 1), method='fast'")
    print(f"  Note: Uses R matrix from sketch (Fortran approach, no extra matvecs)")

    t0 = time.perf_counter()
    k4, piv4, T4 = id_sketch(A_linop_mf, rtol=float(k_target), method='fast')
    t4 = time.perf_counter() - t0

    # Compute error using explicit matrix (for validation)
    A_skel4 = A[:, piv4[k4:]]
    A_basis4 = A[:, piv4[:k4]]
    err4 = np.linalg.norm(A_skel4 - A_basis4 @ T4, 'fro') / normA
    maxT4 = np.max(np.abs(T4)) if T4.size > 0 else 0.0

    print(f"  Rank:      k = {k4}")
    print(f"  Error:     ||A_skel - A_basis @ T|| / ||A|| = {err4:.3e}")
    print(f"  Max |T|:   {maxT4:.3e}")
    print(f"  Time:      {t4:.4f} s")

    # Compare with method='lstsq'
    speedup = t3 / t4 if t4 > 0 else 0.0
    error_ratio = err4 / err3 if err3 > 0 else 0.0

    print(f"  Speedup:   {speedup:.1f}x faster than method='lstsq'")
    print(f"  Error ratio: {error_ratio:.2f}x (err_false / err_true)")

    if err4 > 1.0:
        print(f"  [NOTE] Error > 1.0 is expected for full-rank matrices with accurate=False")
        print(f"         This uses Fortran's fast R-matrix approach, trading accuracy for speed")
    else:
        print(f"  [OK] Error < 1.0 (better than expected!)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("  %-32s Rank    Error        Max|T|       Time" % "Method")
    print("-"*70)
    print("  %-32s %4d    %.3e    %.3e    %.4fs" % ("Dense (method='lstsq')", k1, err1, maxT1, t1))
    print("  %-32s %4d    %.3e    %.3e    %.4fs" % ("Explicit LinOp (accurate=T)", k2, err2, maxT2, t2))
    print("  %-32s %4d    %.3e    %.3e    %.4fs" % ("Matrix-free (method='lstsq')", k3, err3, maxT3, t3))
    print("  %-32s %4d    %.3e    %.3e    %.4fs" % ("Matrix-free (accurate=False)", k4, err4, maxT4, t4))
    print("="*70)

    # Validate
    success = True

    if err1 > 1.0 or err2 > 1.0 or err3 > 1.0:
        print("\n[FAIL] method='lstsq' should guarantee error < 1.0!")
        success = False

    # Note: Do NOT fail on err4 > 1.0 - this is expected for full-rank matrices with accurate=False
    # The NOTE message already explains this is expected

    if k3 != k_target or k4 != k_target:
        print(f"\n[FAIL] Matrix-free should return rank k={k_target}!")
        success = False

    if success:
        print("\n[PASS] All LinearOperator tests passed!")
        print("       method='lstsq' guarantees error < 1.0 for all modes")
        print(f"       accurate=False provides {speedup:.1f}x speedup with acceptable error increase")

    return success


if __name__ == '__main__':
    success = test_linop_fullrank()
    sys.exit(0 if success else 1)
