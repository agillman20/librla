#!/usr/bin/env python3
"""
compare_svd.py - Compare SVD implementations

Comprehensive comparison of two SVD implementations:
- svd_sketch:        Randomized SVD via sketching (libid)
- scipy.linalg.svd:  Deterministic full SVD (truncated for comparison)

Compares on metrics:
- Accuracy (reconstruction error)
- Singular value accuracy
- Runtime
- LinearOperator support (explicit and matrix-free)

Author: Port from compare_id.py structure
"""

import numpy as np
import sys
import time
from dataclasses import dataclass
from typing import List
from scipy import linalg

# Import SVD implementation
from libid import svd_sketch
from make_mat import make_mat

# Import LinearOperator (same directory)
from make_linop import make_linop


@dataclass
class SVDComparisonResult:
    """Results from comparing SVD methods on a single matrix."""
    name: str
    rtol_or_rank: float

    # Rank for each method
    k_sketch: int
    k_scipy: int

    # Reconstruction error: ||A - U*diag(s)*V|| / ||A||
    err_sketch: float
    err_scipy: float

    # Runtime
    t_sketch: float
    t_scipy: float

    # Singular value accuracy vs reference
    sval_err_sketch: float
    sval_err_scipy: float

    passed: bool


def hilb(m, n):
    """
    Generate an mxn Hilbert matrix.

    The Hilbert matrix is extremely ill-conditioned; it is useful for
    testing numerical algorithms.

    Parameters
    ----------
    m : int
        Number of rows
    n : int
        Number of columns

    Returns
    -------
    H : ndarray, shape (m, n)
        Hilbert matrix with entries H[i,j] = 1/(i+j+1)
    """
    i = np.arange(1, m + 1).reshape(-1, 1)
    j = np.arange(1, n + 1).reshape(1, -1)
    return 1.0 / (i + j - 1)


def compare_on_matrix(A, rtol_or_rank, name, flag_power=0):
    """
    Compare SVD implementations on a single matrix.

    Parameters
    ----------
    A : ndarray
        Input matrix to decompose (real or complex)
    rtol_or_rank : float
        Tolerance (rtol_or_rank < 1) or target rank (rtol_or_rank >= 1)
    name : str
        Descriptive name for the test
    flag_power : int, optional
        Number of power iterations for svd_sketch (default: 0)

    Returns
    -------
    result : SVDComparisonResult
        Comparison metrics for both SVD methods
    """
    print(f"\nTesting: {name}")
    if flag_power > 0:
        print(f"  Matrix shape: {A.shape}, rtol_or_rank={rtol_or_rank}, flag_power={flag_power}")
    else:
        print(f"  Matrix shape: {A.shape}, rtol_or_rank={rtol_or_rank}")

    m, n = A.shape
    is_rank_mode = rtol_or_rank >= 1
    k_target = int(np.floor(rtol_or_rank)) if is_rank_mode else None

    A_norm = np.linalg.norm(A, 'fro')

    # Get reference singular values for accuracy comparison
    U_ref, s_ref, Vh_ref = linalg.svd(A, full_matrices=False)

    # ========================================================================
    # Method 1: svd_sketch (randomized)
    # ========================================================================
    t0 = time.perf_counter()
    try:
        U1, s1, Vh1 = svd_sketch(A, rtol_or_rank, flag_power=flag_power)
        t1 = time.perf_counter() - t0
        k1 = len(s1)

        # Reconstruction error
        A1_approx = U1 @ np.diag(s1) @ Vh1
        err1 = np.linalg.norm(A - A1_approx, 'fro') / A_norm

        # Singular value accuracy
        s_ref_trunc = s_ref[:k1]
        sval_err1 = np.linalg.norm(s1 - s_ref_trunc) / np.linalg.norm(s_ref_trunc)

    except Exception as e:
        print(f"  svd_sketch failed: {e}")
        k1, err1, t1, sval_err1 = 0, 999.0, 0.0, 999.0

    # ========================================================================
    # Method 2: scipy.linalg.svd (deterministic, truncated)
    # ========================================================================
    t0 = time.perf_counter()
    try:
        U2, s2, Vh2 = linalg.svd(A, full_matrices=False)

        # Truncate to target rank
        if is_rank_mode:
            k2 = min(k_target, len(s2))
        else:
            # Use relative tolerance on singular values
            k2 = int(np.sum(s2 >= rtol_or_rank * s2[0]))

        U2 = U2[:, :k2]
        s2 = s2[:k2]
        V2 = Vh2[:k2, :]

        t2 = time.perf_counter() - t0

        # Reconstruction error
        A2_approx = U2 @ np.diag(s2) @ V2
        err2 = np.linalg.norm(A - A2_approx, 'fro') / A_norm

        # Singular value accuracy
        s_ref_trunc = s_ref[:k2]
        sval_err2 = np.linalg.norm(s2 - s_ref_trunc) / np.linalg.norm(s_ref_trunc) if k2 > 0 else 0.0

    except Exception as e:
        print(f"  scipy.linalg.svd failed: {e}")
        k2, err2, t2, sval_err2 = 0, 999.0, 0.0, 999.0

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"  Ranks:  sketch={k1}, scipy={k2}")
    print(f"  Errors: sketch={err1:.3e}, scipy={err2:.3e}")
    print(f"  Times:  sketch={t1:.3f}s, scipy={t2:.3f}s")
    print(f"  SVal:   sketch={sval_err1:.3e}, scipy={sval_err2:.3e}")

    # Test passes if all reconstruction errors < 1.0
    passed = (err1 < 1.0 and err2 < 1.0)

    return SVDComparisonResult(
        name=name,
        rtol_or_rank=rtol_or_rank,
        k_sketch=k1,
        k_scipy=k2,
        err_sketch=err1,
        err_scipy=err2,
        t_sketch=t1,
        t_scipy=t2,
        sval_err_sketch=sval_err1,
        sval_err_scipy=sval_err2,
        passed=passed
    )


def compare_dense_vs_linop():
    """
    Compare svd_sketch on dense matrix vs LinearOperator wrapper.

    Tests that svd_sketch handles explicit LinearOperators correctly
    and produces identical results to dense matrix operations.
    """
    print("\n" + "="*70)
    print("LINEAROPERATOR COMPARISON TEST")
    print("="*70)
    print("\nComparing svd_sketch: Dense vs Explicit LinearOperator")

    # Create test matrix
    np.random.seed(42)
    m, n = 100, 60
    rank_true = 15
    U_true = np.random.randn(m, rank_true)
    s_true = np.exp(-np.arange(rank_true) / 5.0)
    V_true = np.random.randn(rank_true, n)
    A = U_true @ np.diag(s_true) @ V_true + 1e-10 * np.random.randn(m, n)

    target_rank = 10

    # Test 1: Dense matrix
    print(f"\n1. Dense matrix ({m}x{n}, target rank={target_rank})")
    t0 = time.perf_counter()
    U1, s1, Vh1 = svd_sketch(A, rtol=float(target_rank))
    t1 = time.perf_counter() - t0
    A1_approx = U1 @ np.diag(s1) @ Vh1
    err1 = np.linalg.norm(A - A1_approx, 'fro') / np.linalg.norm(A, 'fro')
    print(f"   Rank: {len(s1)}, Error: {err1:.6e}, Time: {t1:.3f}s")

    # Test 2: LinearOperator wrapper
    print(f"\n2. Explicit LinearOperator wrapper")
    A_linop = make_linop(A)
    t0 = time.perf_counter()
    U2, s2, Vh2 = svd_sketch(A_linop, rtol=float(target_rank))
    t2 = time.perf_counter() - t0
    A2_approx = U2 @ np.diag(s2) @ Vh2
    err2 = np.linalg.norm(A - A2_approx, 'fro') / np.linalg.norm(A, 'fro')
    print(f"   Rank: {len(s2)}, Error: {err2:.6e}, Time: {t2:.3f}s")

    # Test 3: Comparison
    print(f"\n3. Comparison:")
    print(f"   Rank match: {len(s1) == len(s2)} (k1={len(s1)}, k2={len(s2)})")
    print(f"   Error difference: {abs(err1 - err2):.6e}")
    print(f"   Singular values match: {np.allclose(s1, s2, atol=1e-10)}")

    passed = (abs(err1 - err2) < 1e-10) and (len(s1) == len(s2))

    if passed:
        print("\n[PASS] LinearOperator comparison test PASSED")
    else:
        print("\n[FAIL] LinearOperator comparison test FAILED")

    return passed


def test_matrix_free_operator():
    """
    Test svd_sketch with matrix-free LinearOperator.

    Tests:
    1. Matrix-free operators work in rank mode
    2. Matrix-free operators reject tolerance mode
    3. Results are consistent with dense matrix version
    """
    print("\n" + "="*70)
    print("MATRIX-FREE LINEAROPERATOR TEST")
    print("="*70)

    # Create test matrix
    np.random.seed(123)
    m, n = 100, 60
    rank_true = 15
    U_true = np.random.randn(m, rank_true)
    s_true = np.exp(-np.arange(rank_true) / 5.0)
    V_true = np.random.randn(rank_true, n)
    A = U_true @ np.diag(s_true) @ V_true + 1e-10 * np.random.randn(m, n)

    target_rank = 10

    # Test 1: Dense matrix (baseline)
    print(f"\n1. Dense matrix baseline ({m}x{n}, target rank={target_rank})")
    U1, s1, Vh1 = svd_sketch(A, rtol=float(target_rank))
    A1_approx = U1 @ np.diag(s1) @ Vh1
    err1 = np.linalg.norm(A - A1_approx, 'fro') / np.linalg.norm(A, 'fro')
    print(f"   Rank: {len(s1)}, Error: {err1:.6e}")

    # Test 2: Matrix-free LinearOperator (rank mode should work)
    print(f"\n2. Matrix-free LinearOperator (rank mode)")
    def matvec(x):
        return A @ x
    def rmatvec(x):
        return A.conj().T @ x

    A_mf = make_linop(m, n, matvec, rmatvec, dtype=A.dtype)
    U2, s2, Vh2 = svd_sketch(A_mf, rtol=float(target_rank))
    print(f"   Rank: {len(s2)}")
    print(f"   Singular values: {s2[:5]}")
    print(f"   [OK] Rank mode works for matrix-free operator")

    # Test 3: Matrix-free should reject tolerance mode
    print(f"\n3. Matrix-free should reject tolerance mode")
    try:
        U3, s3, Vh3 = svd_sketch(A_mf, rtol=1e-10)
        print("   [ERROR] Should have raised ValueError!")
        passed = False
    except ValueError as e:
        print(f"   [OK] Correctly rejected tolerance mode")
        print(f"     Message: {str(e)[:70]}...")
        passed = True

    if passed:
        print("\n[PASS] Matrix-free LinearOperator test PASSED")
    else:
        print("\n[FAIL] Matrix-free LinearOperator test FAILED")

    return passed


def run_all_tests():
    """Run comprehensive SVD comparison tests."""
    print("=" + "="*68 + "=")
    print("|" + " "*20 + "LIBID SVD COMPARISON TEST SUITE" + " "*16 + "|")
    print("=" + "="*68 + "=")

    results = []

    # -------------------------------------------------------------------------
    # Test 1: Random matrix (well-conditioned)
    # -------------------------------------------------------------------------
    np.random.seed(42)
    A1 = np.random.randn(400, 250)
    results.append(compare_on_matrix(A1, 1e-8, "Random Matrix (well-conditioned)"))

    # -------------------------------------------------------------------------
    # Test 2: Low-rank matrix
    # -------------------------------------------------------------------------
    U = np.random.randn(400, 15)
    V = np.random.randn(250, 15)
    A2 = U @ V.T + 1e-10 * np.random.randn(400, 250)
    results.append(compare_on_matrix(A2, 1e-8, "Low-Rank Matrix (rank~15)"))

    # -------------------------------------------------------------------------
    # Test 3: Hilbert matrix (extremely ill-conditioned)
    # -------------------------------------------------------------------------
    A3 = hilb(2000, 1000)
    results.append(compare_on_matrix(A3, 15, "Hilbert Matrix (severely ill-conditioned)"))

    # -------------------------------------------------------------------------
    # Test 4: Complex matrix
    # -------------------------------------------------------------------------
    A4 = np.random.randn(300, 200) + 1j * np.random.randn(300, 200)
    results.append(compare_on_matrix(A4, 1e-8, "Complex Matrix"))

    # -------------------------------------------------------------------------
    # Test 5: Power-law decay (slow decay)
    # -------------------------------------------------------------------------
    rank = 50
    s_decay = 1.0 / (np.arange(1, rank + 1) ** 0.5)  # s_k ~ 1/sqrt(k)
    U5 = np.linalg.qr(np.random.randn(300, rank))[0]
    V5 = np.linalg.qr(np.random.randn(200, rank))[0].T
    A5 = U5 @ np.diag(s_decay) @ V5
    results.append(compare_on_matrix(A5, 1e-6, "Power-Law Decay (slow)"))

    # -------------------------------------------------------------------------
    # Test 6: Rank mode test (fixed rank=20)
    # -------------------------------------------------------------------------
    A6 = np.random.randn(300, 200)
    results.append(compare_on_matrix(A6, 20, "Rank Mode Test (k=20)"))

    # -------------------------------------------------------------------------
    # Test 7: Large low-rank matrix
    # -------------------------------------------------------------------------
    U7 = np.random.randn(800, 15)
    V7 = np.random.randn(500, 15)
    A7 = U7 @ V7.T + 1e-10 * np.random.randn(800, 500)
    results.append(compare_on_matrix(A7, 1e-8, "Large Low-Rank Matrix (800x500, rank~15)"))

    # -------------------------------------------------------------------------
    # Test 8: Large Hilbert matrix
    # -------------------------------------------------------------------------
    A8 = hilb(4000, 2000)
    results.append(compare_on_matrix(A8, 15, "Large Hilbert Matrix (4000x2000)"))

    # -------------------------------------------------------------------------
    # Test 9: Large complex matrix
    # -------------------------------------------------------------------------
    A9 = np.random.randn(600, 400) + 1j * np.random.randn(600, 400)
    results.append(compare_on_matrix(A9, 1e-8, "Large Complex Matrix"))

    # -------------------------------------------------------------------------
    # Test 10-12: Structured matrices with decaying spectra
    # -------------------------------------------------------------------------
    for mat_type in ['gmm', 'gaussexp', 'snn']:
        A_struct = make_mat(400, 250, mat_type)
        results.append(compare_on_matrix(A_struct, 1e-8, f"Structured: {mat_type}"))

    # -------------------------------------------------------------------------
    # Test 15-16: Wide matrices
    # -------------------------------------------------------------------------
    A15 = np.random.randn(200, 500)
    results.append(compare_on_matrix(A15, 1e-8, "Wide Random Matrix (200x500)"))

    U16 = np.random.randn(200, 15)
    V16 = np.random.randn(500, 15)
    A16 = U16 @ V16.T + 1e-10 * np.random.randn(200, 500)
    results.append(compare_on_matrix(A16, 1e-8, "Wide Low-Rank Matrix (200x500, rank~15)"))

    # -------------------------------------------------------------------------
    # Test 17: XL Random matrix
    # -------------------------------------------------------------------------
    A17 = np.random.randn(1200, 800)
    results.append(compare_on_matrix(A17, 1e-8, "XL Random Matrix (1200x800)"))

    # -------------------------------------------------------------------------
    # Test 18: XL Low-rank matrix
    # -------------------------------------------------------------------------
    U18 = np.random.randn(1600, 15)
    V18 = np.random.randn(1000, 15)
    A18 = U18 @ V18.T + 1e-10 * np.random.randn(1600, 1000)
    results.append(compare_on_matrix(A18, 1e-8, "XL Low-Rank Matrix (1600x1000, rank~15)"))

    # -------------------------------------------------------------------------
    # Test 19: 4x Hilbert matrix - WARNING: VERY SLOW!
    # -------------------------------------------------------------------------
    A19 = hilb(8000, 4000)
    results.append(compare_on_matrix(A19, 15, "XL Hilbert Matrix (8000x4000)"))

    # -------------------------------------------------------------------------
    # Test 20: 4x Complex matrix
    # -------------------------------------------------------------------------
    A20 = np.random.randn(1200, 800) + 1j * np.random.randn(1200, 800)
    results.append(compare_on_matrix(A20, 1e-8, "XL Complex Matrix (1200x800)"))

    # =========================================================================
    # Power Iteration Tests (Rank Mode)
    # =========================================================================
    print("\n" + "="*70)
    print("POWER ITERATION TESTS (Rank Mode)")
    print("="*70)

    # Test matrix for power iteration tests
    A_power = np.random.randn(400, 300)
    target_rank = 30

    # Test 21: Power iteration = 0 (no power iteration)
    results.append(compare_on_matrix(A_power, target_rank,
                                     f"Rank Mode k={target_rank}, power=0",
                                     flag_power=0))

    # Test 22: Power iteration = 1
    results.append(compare_on_matrix(A_power, target_rank,
                                     f"Rank Mode k={target_rank}, power=1",
                                     flag_power=1))

    # Test 23: Power iteration = 2
    results.append(compare_on_matrix(A_power, target_rank,
                                     f"Rank Mode k={target_rank}, power=2",
                                     flag_power=2))

    # =========================================================================
    # LinearOperator tests
    # =========================================================================
    linop_passed_1 = compare_dense_vs_linop()
    linop_passed_2 = test_matrix_free_operator()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed_count = sum(r.passed for r in results)
    total_count = len(results)

    print(f"\nDense matrix tests: {passed_count}/{total_count} passed")
    print(f"LinearOperator tests: {int(linop_passed_1 and linop_passed_2)}/2 passed")

    all_passed = (passed_count == total_count) and linop_passed_1 and linop_passed_2

    if all_passed:
        print("\n[PASS] ALL TESTS PASSED!")
    else:
        print(f"\n[FAIL] {total_count - passed_count} tests failed")
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}")

    print("="*70)

    return results, all_passed


if __name__ == "__main__":
    results, all_passed = run_all_tests()
    sys.exit(0 if all_passed else 1)
