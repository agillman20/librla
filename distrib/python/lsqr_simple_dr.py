"""LinearOperator framework validation test for Python/NumPy.

PURPOSE:
  Verify that LinearOperator abstraction gives IDENTICAL results to
  explicit matrices for LSQR algorithm

LINEAROPERATOR FRAMEWORK:
  Provides uniform interface for matrix operations:
    - Explicit matrices (A)
    - Function handles (A@x, A.T@y)
    - Implicit operators (e.g., FFT, wavelets)

METHODS TESTED:
  1. LSQR with explicit matrix A
  2. LSQR with LinearOperator from matrix (make_linop(A))
  3. LSQR with matrix-free LinearOperator (function handles)

PROBLEM:
  Matrix: A is 2400 x 4800 Hilbert matrix (underdetermined)
  Tolerance: 1e-10
  Solution: Minimum norm least squares

KEY INSIGHTS:
  - LinearOperator provides abstraction for matrix-free methods
  - Enables large-scale problems where forming A is impractical
  - Must give bit-for-bit identical results (same Krylov sequence)
  - Tests interface correctness and numerical reproducibility

VALIDATION CRITERIA:
  - All three methods must give IDENTICAL solutions
  - Match error: ||x_method - x_explicit|| / ||x_explicit|| < 1e-14
  - Same iteration counts
  - Status: PASSED if match error < 1e-14
"""

import numpy as np
import time
from hilb import hilb
from make_linop import make_linop
from lsqr_simple import lsqr_simple


def main():
    np.random.seed(1)

    m = 120 * 2 * 10  # 2400 (rows of A)
    k = 240 * 2 * 10  # 4800 (cols of A)

    # Create rectangular Hilbert matrix (ill-conditioned test case)
    a = hilb(m, k)

    print('========================================================')
    print('Test: LinearOperator Framework Validation (Python)')
    print('========================================================')
    print(f'Matrix A: {m} x {k} (underdetermined)')
    print('Tolerance: 1e-10')
    print('Problem: min ||A*x - y|| (minimum norm solution)\n')

    ## Test: Single right-hand side - LinearOperator validation

    print('------------------------------------------------------------')
    print('Test: Single RHS - LinearOperator Validation')
    print('------------------------------------------------------------')

    x0 = np.random.randn(k)
    y = a @ x0 + np.random.rand(m) * 1e-16  # Add small noise

    # Direct solve (ground truth - minimum norm solution)
    t_start = time.time()
    x_direct = np.linalg.lstsq(a, y, rcond=None)[0]
    t_direct = time.time() - t_start
    err_direct = np.linalg.norm(a @ x_direct - y) / np.linalg.norm(y)
    print('Direct solve (lstsq):')
    print(f'  Time: {t_direct:.4f} seconds')
    print(f'  Relative error: {err_direct:.4e}\n')

    ## Method 1: LSQR with explicit matrix

    print('Method 1: LSQR with Explicit Matrix')
    print('  Algorithm: Golub-Kahan bidiagonalization')
    print('  Operations: A*v and A.T*u per iteration')
    print('  Memory: O(k) - short recurrence\n')

    t_start = time.time()
    x_explicit, flag_explicit, relres_explicit, iter_explicit = lsqr_simple(
        a, y, 1e-10, 100
    )
    t_explicit = time.time() - t_start

    err_explicit = np.linalg.norm(a @ x_explicit - y) / np.linalg.norm(y)
    print('LSQR (explicit matrix) results:')
    print(f'  Time: {t_explicit:.4f} seconds')
    print(f'  Iterations: {iter_explicit}')
    print(f'  Relative error: {err_explicit:.4e}')
    print(f'  Flag: {flag_explicit} (0=converged)')

    ## Method 2: LSQR with LinearOperator from matrix

    print('\nMethod 2: LSQR with LinearOperator from Matrix')
    print('  Algorithm: Same as Method 1')
    print('  Input: LinearOperator structure (from matrix)')
    print('  Purpose: Verify LinearOperator interface\n')

    # Create LinearOperator from matrix
    op_from_matrix = make_linop(a)

    t_start = time.time()
    x_operator, flag_operator, relres_operator, iter_operator = lsqr_simple(
        op_from_matrix, y, 1e-10, 100
    )
    t_operator = time.time() - t_start

    err_operator = np.linalg.norm(a @ x_operator - y) / np.linalg.norm(y)
    match_explicit = np.linalg.norm(x_explicit - x_operator) / np.linalg.norm(x_explicit)

    print('LSQR (LinearOperator) results:')
    print(f'  Time: {t_operator:.4f} seconds')
    print(f'  Iterations: {iter_operator}')
    print(f'  Relative error: {err_operator:.4e}')
    print(f'  Flag: {flag_operator} (0=converged)')
    print(f'  Match with Method 1: {match_explicit:.4e} (should be ~0)')

    ## Method 3: LSQR with matrix-free LinearOperator

    print('\nMethod 3: LSQR with Matrix-Free LinearOperator')
    print('  Algorithm: Same as Method 1')
    print('  Input: LinearOperator from function handles')
    print('  Purpose: Verify matrix-free operations\n')

    # Create matrix-free LinearOperator
    A_forward = lambda x: a @ x
    A_adjoint = lambda y: a.conj().T @ y  # Hermitian adjoint for complex matrices
    op_matfree = make_linop(m, k, A_forward, A_adjoint, dtype=a.dtype)

    t_start = time.time()
    x_matfree, flag_matfree, relres_matfree, iter_matfree = lsqr_simple(
        op_matfree, y, 1e-10, 100
    )
    t_matfree = time.time() - t_start

    err_matfree = np.linalg.norm(a @ x_matfree - y) / np.linalg.norm(y)
    match_matfree = np.linalg.norm(x_explicit - x_matfree) / np.linalg.norm(x_explicit)

    print('LSQR (matrix-free) results:')
    print(f'  Time: {t_matfree:.4f} seconds')
    print(f'  Iterations: {iter_matfree}')
    print(f'  Relative error: {err_matfree:.4e}')
    print(f'  Flag: {flag_matfree} (0=converged)')
    print(f'  Match with Method 1: {match_matfree:.4e} (should be ~0)')

    ## Comparison

    print('\n------------------------------------------------------------')
    print('Single RHS Comparison Summary')
    print('------------------------------------------------------------')
    print(f'{"Method":<30} {"Time (s)":>10} {"Iters":>12} {"Match Error":>15}')
    print(f'{"LSQR (explicit)":<30} {t_explicit:>10.4f} {iter_explicit:>12} {"-":>15}')
    print(f'{"LSQR (LinearOperator)":<30} {t_operator:>10.4f} {iter_operator:>12} {match_explicit:>15.4e}')
    print(f'{"LSQR (matrix-free)":<30} {t_matfree:>10.4f} {iter_matfree:>12} {match_matfree:>15.4e}')

    if match_explicit < 1e-14 and match_matfree < 1e-14:
        print('\n[OK] VALIDATION PASSED: All three methods give IDENTICAL results')
    else:
        print('\n[FAIL] VALIDATION FAILED: Methods do not match')


if __name__ == '__main__':
    main()
