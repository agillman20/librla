#!/usr/bin/env python3
"""
demo02_svd.py - Truncated SVD via Randomized Sketching

This demo shows how to compute truncated SVD using librla:
  - svd_sketch: Randomized truncated SVD
  - qr_sketch:  Truncated QR factorization

Both functions use randomized sketching for efficiency on large matrices.

The SVD factorizes A as: A ≈ U @ diag(s) @ Vh
The QR factorizes A as: A[:, p] ≈ Q @ R

Try changing the CONFIGURATION parameters below to experiment!

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.2.1
Date: July 30, 2026
Assisted by: Claude Code (Anthropic)
"""

# =============================================================================
# CONFIGURATION - Modify these to experiment
# =============================================================================

MATRIX_SIZE = (1000, 2000)  # (rows, columns)
TARGET_RANK = 30            # Number of singular values to compute
RANDOM_SEED = 42            # For reproducibility

# Tolerance mode example (uncomment to use)
# TARGET_RANK = 1e-6        # When rtol < 1, adaptive rank selection

# =============================================================================
# Demo code below
# =============================================================================

import numpy as np
import time
import sys
sys.path.insert(0, '.')
sys.path.insert(0,'..')


from librla import svd_sketch, qr_sketch
from demo_utils import hilbert, svd_error, print_header, print_subheader


def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    m, n = MATRIX_SIZE
    k = TARGET_RANK

    print_header("Demo 02: Truncated SVD and QR")
    print(f"\nMatrix: {m} x {n} Hilbert matrix")
    print(f"Target rank: {k}")

    # Generate test matrix
    A = hilbert(m, n)
    normA = np.linalg.norm(A, 'fro')

    # Compute reference SVD for comparison
    s_true = np.linalg.svd(A, compute_uv=False)
    print(f"\nTrue singular values:")
    print(f"   s[0] = {s_true[0]:.6e}")
    print(f"   s[{int(k)-1}] = {s_true[int(k)-1]:.6e}")
    print(f"   s[{int(k)}] = {s_true[int(k)]:.6e}")
    print(f"   s[{min(m-1,n-1)}] = {s_true[min(m-1,n-1)]:.6e}")

    # -------------------------------------------------------------------------
    # Method 1: svd_sketch
    # -------------------------------------------------------------------------
    print_subheader("1. svd_sketch (truncated SVD)")
    print("   Returns U, s, Vh where A ≈ U @ diag(s) @ Vh")

    t0 = time.perf_counter()
    U, s, Vh = svd_sketch(A, rtol=float(k))
    elapsed = time.perf_counter() - t0

    err = svd_error(A, U, s, Vh)
    k_out = len(s)

    print(f"   Rank:      {k_out}")
    print(f"   Error:     {err:.6e}")
    print(f"   Time:      {elapsed:.4f} s")

    # Compare singular values
    s_err = np.linalg.norm(s - s_true[:k_out]) / np.linalg.norm(s_true[:k_out])
    print(f"   SVal err:  {s_err:.6e} (relative)")

    # Check orthogonality
    orth_U = np.linalg.norm(U.T @ U - np.eye(k_out), 'fro')
    orth_V = np.linalg.norm(Vh @ Vh.T - np.eye(k_out), 'fro')
    print(f"   ||U'U-I||: {orth_U:.2e}")
    print(f"   ||VV'-I||: {orth_V:.2e}")

    # -------------------------------------------------------------------------
    # Method 2: qr_sketch
    # -------------------------------------------------------------------------
    print_subheader("2. qr_sketch (truncated QR)")
    print("   Returns Q, R, piv where A[:, piv] ≈ Q @ R")

    t0 = time.perf_counter()
    Q, R, piv = qr_sketch(A, rtol=float(k))
    elapsed2 = time.perf_counter() - t0

    # Reconstruct: A[:, piv] ≈ Q @ R, so A ≈ Q @ R @ inv(P)
    A_qr = np.zeros_like(A)
    A_qr[:, piv] = Q @ R
    err2 = np.linalg.norm(A - A_qr, 'fro') / normA
    k2 = Q.shape[1]

    print(f"   Rank:      {k2}")
    print(f"   Error:     {err2:.6e}")
    print(f"   Time:      {elapsed2:.4f} s")

    # Check orthogonality
    orth_Q = np.linalg.norm(Q.T @ Q - np.eye(k2), 'fro')
    print(f"   ||Q'Q-I||: {orth_Q:.2e}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_subheader("Summary")
    print(f"   {'Method':<14} {'Rank':>6} {'Error':>12} {'Time':>10}")
    print(f"   {'-'*14} {'-'*6} {'-'*12} {'-'*10}")
    print(f"   {'svd_sketch':<14} {k_out:>6} {err:>12.3e} {elapsed:>9.4f}s")
    print(f"   {'qr_sketch':<14} {k2:>6} {err2:>12.3e} {elapsed2:>9.4f}s")

    print("\nNotes:")
    print("  - svd_sketch gives U, s, Vh for best rank-k approximation")
    print("  - qr_sketch gives Q, R factorization with column pivoting")

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
