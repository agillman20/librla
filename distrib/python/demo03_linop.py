#!/usr/bin/env python3
"""
demo03_linop.py - Matrix-Free Computation with LinearOperator

This demo shows how to use librla with LinearOperators for matrix-free computation.
This is essential for large-scale problems where the matrix doesn't fit in memory.

Three modes are demonstrated:
  1. Dense matrix (baseline)
  2. Explicit LinearOperator (matrix stored, accessed via matvec)
  3. Matrix-free LinearOperator (only matvec/rmatvec functions provided)

Note: Matrix-free mode only supports rank mode (rtol >= 1), not tolerance mode.

Try changing the CONFIGURATION parameters below to experiment!

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: NIST-PD
Assisted by: Claude Code (Anthropic)
"""

# =============================================================================
# CONFIGURATION - Modify these to experiment
# =============================================================================

MATRIX_SIZE = (300, 200)    # (rows, columns)
TARGET_RANK = 15            # Must be >= 1 for matrix-free mode
RANDOM_SEED = 42            # For reproducibility

# =============================================================================
# Demo code below
# =============================================================================

import numpy as np
import time
import sys
sys.path.insert(0, '.')

from scipy.sparse.linalg import LinearOperator
from librla import id_sketch, svd_sketch
from demo_utils import hilbert, id_error, print_header, print_subheader


def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    m, n = MATRIX_SIZE
    k = TARGET_RANK

    print_header("Demo 03: Matrix-Free Computation")
    print(f"\nMatrix: {m} x {n} Hilbert matrix")
    print(f"Target rank: {k}")

    # Generate test matrix (we'll wrap it in LinearOperators)
    A = hilbert(m, n)
    normA = np.linalg.norm(A, 'fro')

    # =========================================================================
    # Part 1: ID with different input types
    # =========================================================================
    print_subheader("Part 1: id_sketch with LinearOperator")

    # -------------------------------------------------------------------------
    # Test 1: Dense matrix (baseline)
    # -------------------------------------------------------------------------
    print("\n   1a. Dense matrix (baseline)")

    t0 = time.perf_counter()
    k1, piv1, T1 = id_sketch(A, rtol=float(k))
    elapsed1 = time.perf_counter() - t0

    err1 = id_error(A, k1, piv1, T1)
    print(f"       Rank: {k1}, Error: {err1:.3e}, Time: {elapsed1:.4f}s")

    # -------------------------------------------------------------------------
    # Test 2: Explicit LinearOperator
    # -------------------------------------------------------------------------
    print("\n   1b. Explicit LinearOperator (matrix wrapper)")
    print("       Matrix is stored; LinearOperator wraps it.")

    A_explicit = LinearOperator(
        shape=(m, n),
        matvec=lambda x: A @ x,
        rmatvec=lambda x: A.T @ x,
        dtype=A.dtype
    )

    t0 = time.perf_counter()
    k2, piv2, T2 = id_sketch(A_explicit, rtol=float(k))
    elapsed2 = time.perf_counter() - t0

    err2 = id_error(A, k2, piv2, T2)
    print(f"       Rank: {k2}, Error: {err2:.3e}, Time: {elapsed2:.4f}s")

    # Verify same result as dense
    if k1 == k2 and abs(err1 - err2) < 1e-12:
        print("       [OK] Same result as dense matrix")

    # -------------------------------------------------------------------------
    # Test 3: Matrix-free LinearOperator
    # -------------------------------------------------------------------------
    print("\n   1c. Matrix-free LinearOperator")
    print("       Only matvec/rmatvec functions provided.")
    print("       Requires rank mode (rtol >= 1).")

    # Define matvec functions (in real applications, these would compute
    # matrix-vector products without storing the full matrix)
    def my_matvec(x):
        """Forward: y = A @ x"""
        return A @ x

    def my_rmatvec(x):
        """Adjoint: y = A.T @ x"""
        return A.T @ x

    A_matfree = LinearOperator(
        shape=(m, n),
        matvec=my_matvec,
        rmatvec=my_rmatvec,
        dtype=float
    )

    t0 = time.perf_counter()
    k3, piv3, T3 = id_sketch(A_matfree, rtol=float(k))
    elapsed3 = time.perf_counter() - t0

    err3 = id_error(A, k3, piv3, T3)
    print(f"       Rank: {k3}, Error: {err3:.3e}, Time: {elapsed3:.4f}s")

    # =========================================================================
    # Part 2: SVD with LinearOperator
    # =========================================================================
    print_subheader("Part 2: svd_sketch with LinearOperator")

    # Dense baseline
    print("\n   2a. Dense matrix")
    t0 = time.perf_counter()
    U1, s1, Vh1 = svd_sketch(A, rtol=float(k))
    elapsed_svd1 = time.perf_counter() - t0

    A_approx1 = U1 @ np.diag(s1) @ Vh1
    err_svd1 = np.linalg.norm(A - A_approx1, 'fro') / normA
    print(f"       Rank: {len(s1)}, Error: {err_svd1:.3e}, Time: {elapsed_svd1:.4f}s")

    # Matrix-free
    print("\n   2b. Matrix-free LinearOperator")
    t0 = time.perf_counter()
    U2, s2, Vh2 = svd_sketch(A_matfree, rtol=float(k))
    elapsed_svd2 = time.perf_counter() - t0

    A_approx2 = U2 @ np.diag(s2) @ Vh2
    err_svd2 = np.linalg.norm(A - A_approx2, 'fro') / normA
    print(f"       Rank: {len(s2)}, Error: {err_svd2:.3e}, Time: {elapsed_svd2:.4f}s")

    # =========================================================================
    # Summary
    # =========================================================================
    print_subheader("Summary: id_sketch")
    print(f"   {'Input Type':<28} {'Rank':>6} {'Error':>12} {'Time':>10}")
    print(f"   {'-'*28} {'-'*6} {'-'*12} {'-'*10}")
    print(f"   {'Dense matrix':<28} {k1:>6} {err1:>12.3e} {elapsed1:>9.4f}s")
    print(f"   {'Explicit LinearOperator':<28} {k2:>6} {err2:>12.3e} {elapsed2:>9.4f}s")
    print(f"   {'Matrix-free LinearOperator':<28} {k3:>6} {err3:>12.3e} {elapsed3:>9.4f}s")

    print("\nNotes:")
    print("  - LinearOperator allows matrix-free computation")
    print("  - Essential for large-scale problems (matrix doesn't fit in memory)")
    print("  - Matrix-free mode requires rank mode (rtol >= 1)")
    print("  - Define only matvec(x) = A @ x and rmatvec(x) = A.T @ x")

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
