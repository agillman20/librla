#!/usr/bin/env python3
"""
test_libid.py - Test and demonstrate libid interpolative decomposition

Quick test script to validate id_sketch and libid_rrqr.id_rrqr
implementations. Runs several test cases and prints results.

Usage:
    python test_libid.py

Author: Your Name
SPDX-License-Identifier: TBD
"""

import numpy as np
import time
from libid import id_sketch, _hilb
from libid_rrqr import id_rrqr


def test_libid():
    """Run all tests."""
    print('=' * 65)
    print('Testing libid Interpolative Decomposition')
    print('=' * 65)
    print()

    # Test 1: Random matrix
    print('Test 1: Random Matrix (500x300, rank=20)')
    print('-' * 65)
    A1 = np.random.randn(500, 300)
    run_test(A1, 20, 'Random 500x300')

    # Test 2: Low-rank matrix
    print('\nTest 2: Low-Rank Matrix (400x250, true rank~15)')
    print('-' * 65)
    U = np.random.randn(400, 15)
    V = np.random.randn(250, 15)
    A2 = U @ V.T + 1e-10 * np.random.randn(400, 250)
    run_test(A2, 1e-8, 'Low-Rank 400x250')

    # Test 3: Hilbert matrix (ill-conditioned)
    print('\nTest 3: Hilbert Matrix (200x100, ill-conditioned)')
    print('-' * 65)
    A3 = _hilb(200, 100)
    run_test(A3, 15, 'Hilbert 200x100')

    # Test 4: Complex matrix
    print('\nTest 4: Complex Matrix (300x200, rank=25)')
    print('-' * 65)
    A4 = np.random.randn(300, 200) + 1j * np.random.randn(300, 200)
    run_test(A4, 25, 'Complex 300x200')

    print('\n' + '=' * 65)
    print('All tests completed!')
    print('=' * 65)


def run_test(A, rtol, name):
    """
    Helper function to run both methods and compare results.

    Parameters
    ----------
    A : ndarray
        Input matrix to decompose
    rtol : float or int
        Tolerance or target rank
    name : str
        Descriptive name for this test
    """
    m, n = A.shape
    normA = np.linalg.norm(A, 'fro')

    # Test libid randomized version
    t0 = time.perf_counter()
    k_libid, piv_libid, T_libid = id_sketch(A, rtol)
    t_libid = time.perf_counter() - t0

    # Compute error
    A_skel_libid = A[:, piv_libid[k_libid:]]
    A_basis_libid = A[:, piv_libid[:k_libid]]
    if T_libid.size > 0:
        err_libid = np.linalg.norm(A_skel_libid - A_basis_libid @ T_libid, 'fro') / normA
        max_T_libid = np.max(np.abs(T_libid))
    else:
        err_libid = 0.0
        max_T_libid = 0.0

    # Test RRQR deterministic version
    t0 = time.perf_counter()
    k_rrqr, piv_rrqr, T_rrqr = id_rrqr(A, rtol)
    t_rrqr = time.perf_counter() - t0

    # Compute error
    A_skel_rrqr = A[:, piv_rrqr[k_rrqr:]]
    A_basis_rrqr = A[:, piv_rrqr[:k_rrqr]]
    if T_rrqr.size > 0:
        err_rrqr = np.linalg.norm(A_skel_rrqr - A_basis_rrqr @ T_rrqr, 'fro') / normA
        max_T_rrqr = np.max(np.abs(T_rrqr))
    else:
        err_rrqr = 0.0
        max_T_rrqr = 0.0

    # Print results
    complex_str = ', complex' if np.iscomplexobj(A) else ''
    print(f'Matrix: {name} ({m}x{n}{complex_str})')
    print(f'Target: rtol = {rtol}\n')

    print(f'{"Method":<20} {"Rank":<10} {"Error":<15} {"max|T|":<15} {"Time (s)":<10}')
    print(f'{"-"*20} {"-"*10} {"-"*15} {"-"*15} {"-"*10}')
    print(f'{"libid (randomized)":<20} {k_libid:<10} {err_libid:<15.3e} {max_T_libid:<15.3e} {t_libid:<10.4f}')
    print(f'{"RRQR (deterministic)":<20} {k_rrqr:<10} {err_rrqr:<15.3e} {max_T_rrqr:<15.3e} {t_rrqr:<10.4f}')

    # Compare ranks
    if k_libid == k_rrqr:
        print(f'\n[OK] Ranks match (k={k_libid})')
    else:
        print(f'\n[WARNING] Different ranks: libid={k_libid}, RRQR={k_rrqr} (difference={abs(k_libid - k_rrqr)})')

    # Compare speeds
    if t_libid < t_rrqr:
        print(f'[OK] libid {t_rrqr/t_libid:.2f}x faster than RRQR')
    else:
        print(f'[WARNING] RRQR {t_libid/t_rrqr:.2f}x faster than libid')


if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility
    test_libid()
