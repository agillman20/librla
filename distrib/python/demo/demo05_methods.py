#!/usr/bin/env python3
"""
demo05_methods.py - T Matrix Computation Methods

This demo compares the three methods for computing the interpolation matrix T
in the ID factorization: A[:, piv[k:]] = A[:, piv[:k]] @ T

Methods:
  - 'fast':   Triangular solve (fastest, may have large T entries)
  - 'svd':    SVD-based pseudoinverse
  - 'lstsq':  Least squares from original A (most accurate, slowest)

The choice of method affects:
  - Speed (fast < svd < lstsq)
  - Stability (fast may produce large T, svd/lstsq are stable)
  - Accuracy (lstsq gives best reconstruction)

Try changing the CONFIGURATION parameters below to experiment!

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.2.0
Date: July 26, 2026
Assisted by: Claude Code (Anthropic)
"""

# =============================================================================
# CONFIGURATION - Modify these to experiment
# =============================================================================

MATRIX_SIZE = (2000, 1000)    # (rows, columns)
TARGET_RANK = 50            # Number of skeleton columns
RANDOM_SEED = 42            # For reproducibility

# Matrix type: 'lowrank', 'fullrank', or 'hilbert'
MATRIX_TYPE = 'fullrank'

# =============================================================================
# Demo code below
# =============================================================================

import numpy as np
import time
import sys
sys.path.insert(0, '.')
sys.path.insert(0,'..')

from librla import id_sketch
from demo_utils import hilbert, lowrank, random_matrix, id_error
from demo_utils import print_header, print_subheader


def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    m, n = MATRIX_SIZE
    k = TARGET_RANK

    print_header("Demo 05: T Matrix Computation Methods")
    print(f"\nMatrix: {m} x {n}")
    print(f"Target rank: {k}")

    # -------------------------------------------------------------------------
    # Create test matrix
    # -------------------------------------------------------------------------
    if MATRIX_TYPE == 'lowrank':
        print("Matrix type: LOW-RANK (true rank = 30)")
        A, _ = lowrank(m, n, 30, decay='exponential', gap=100.0)
    elif MATRIX_TYPE == 'fullrank':
        print("Matrix type: FULL-RANK RANDOM")
        print("(All columns are linearly independent)")
        A = random_matrix(m, n)
    else:  # hilbert
        print("Matrix type: HILBERT (ill-conditioned)")
        A = hilbert(m, n)

    normA = np.linalg.norm(A, 'fro')
    s = np.linalg.svd(A, compute_uv=False)
    cond = s[0] / s[-1]
    print(f"Condition number: {cond:.2e}")

    # -------------------------------------------------------------------------
    # Test all three methods
    # -------------------------------------------------------------------------
    methods = ['fast', 'svd', 'lstsq']
    results = []

    for method in methods:
        print_subheader(f"Method: '{method}'")

        if method == 'fast':
            print("   Triangular solve on R factor. Fastest but may be unstable.")
        elif method == 'svd':
            print("   SVD-based pseudoinverse. Stable for ill-conditioned R.")
        else:
            print("   Least squares from original A. Most accurate, slowest.")

        t0 = time.perf_counter()
        k_out, piv, T = id_sketch(A, rtol=float(k), method=method)
        elapsed = time.perf_counter() - t0

        err = id_error(A, k_out, piv, T)
        maxT = np.max(np.abs(T)) if T.size > 0 else 0.0

        results.append({
            'method': method,
            'k': k_out,
            'error': err,
            'maxT': maxT,
            'time': elapsed
        })

        print(f"   Rank:     {k_out}")
        print(f"   Error:    {err:.6e}")
        print(f"   Max |T|:  {maxT:.3e}")
        print(f"   Time:     {elapsed:.4f} s")

        # Warn about large T entries
        if maxT > 10.0:
            print(f"   [NOTE] Max|T| > 10 indicates potential instability")
        if err > 1.0:
            print(f"   [NOTE] Error > 1.0: relative error exceeds 100%")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_subheader("Summary")
    print(f"   {'Method':<8} {'Rank':>6} {'Error':>14} {'Max|T|':>12} {'Time':>10}")
    print(f"   {'-'*8} {'-'*6} {'-'*14} {'-'*12} {'-'*10}")
    for r in results:
        print(f"   {r['method']:<8} {r['k']:>6} {r['error']:>14.6e} {r['maxT']:>12.3e} {r['time']:>9.4f}s")

    # Analysis
    print("\nAnalysis:")

    fast_err = results[0]['error']
    lstsq_err = results[2]['error']

    if fast_err > 1.0 and lstsq_err < 1.0:
        print("  - 'fast' failed (error > 1) but 'lstsq' succeeded")
        print("  - This happens with full-rank matrices: skeleton columns")
        print("    cannot exactly represent other columns")
        print("  - Use method='lstsq' for best least-squares approximation")
    elif results[0]['maxT'] > 100 * results[2]['maxT']:
        print("  - 'fast' produced much larger T entries than 'lstsq'")
        print("  - This indicates numerical instability in triangular solve")
        print("  - Consider using method='svd' or 'lstsq' for stability")
    else:
        print("  - All methods performed similarly")
        print("  - 'fast' is recommended for speed")

    print("\nRecommendations:")
    print("  - Use 'fast' (default) for low-rank matrices")
    print("  - Use 'svd' when R factor is ill-conditioned")
    print("  - Use 'lstsq' when best accuracy is needed")
    print("  - Use 'lstsq' for full-rank matrices (guarantees error < 1)")

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
