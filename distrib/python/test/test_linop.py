#!/usr/bin/env python3
"""
test_linop.py - LinearOperator, wide-matrix, and method='svd' regression tests

Covers code paths not exercised by test_id/test_svd/test_qr/test_orth:
  - Explicit LinearOperator (aslinearoperator — has .A attribute)
  - Matrix-free LinearOperator (scipy.sparse.linalg.LinearOperator with
    matvec/rmatvec only)
  - Wide matrices (m < n), which trigger svd_sketch's transpose branch
  - id_sketch / id_qrpiv with method='svd' and method='lstsq'

Returns exit code 0 if all tests pass, 1 otherwise.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.2
Date: June 22, 2026
Assisted by: Claude Code (Anthropic)
"""
import sys
import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator

sys.path.insert(0, '..')

from librla import orth_sketch, qr_sketch, svd_sketch, id_sketch, id_qrpiv


def wrap_matfree(M):
    """Return a matrix-free scipy LinearOperator for M (no .A attribute)."""
    m, n = M.shape
    return LinearOperator(
        shape=(m, n),
        matvec=lambda x: M @ x,
        rmatvec=lambda x: M.conj().T @ x,
        dtype=M.dtype,
    )


def check_svd(M, k, label, errors, *, rng=None):
    """Run svd_sketch on M and its wrappers; record failures into errors."""
    normM = np.linalg.norm(M, 'fro')
    s_true = np.linalg.svd(M, compute_uv=False)
    err_opt = np.linalg.norm(s_true[k:], 2) / normM if k < len(s_true) else 0.0
    # 4x slack over optimal, with a floor for tiny noise
    tol = max(4.0 * err_opt, 1e-10)

    for variant, A in [
        ('dense',     M),
        ('explicit',  aslinearoperator(M)),
        ('matfree',   wrap_matfree(M)),
    ]:
        U, s, Vh = svd_sketch(A, float(k))
        Arec = U @ np.diag(s) @ Vh
        err = np.linalg.norm(M - Arec, 'fro') / normM
        ortho_U = np.linalg.norm(U.conj().T @ U - np.eye(U.shape[1]), 'fro')
        ortho_V = np.linalg.norm(Vh @ Vh.conj().T - np.eye(Vh.shape[0]), 'fro')
        ok = (err < tol) and (ortho_U < 1e-10) and (ortho_V < 1e-10) and (len(s) == k)
        status = 'PASS' if ok else 'FAIL'
        print(f"  [{status}] svd_sketch  {label:<26s} {variant:<9s}"
              f" err={err:.2e} opt={err_opt:.2e} orthU={ortho_U:.1e} k={len(s)}")
        if not ok:
            errors.append(f"svd_sketch {label} {variant}")


def check_qr(M, k, label, errors):
    normM = np.linalg.norm(M, 'fro')
    for variant, A in [('dense', M),
                       ('explicit', aslinearoperator(M)),
                       ('matfree',  wrap_matfree(M))]:
        Q, R, p = qr_sketch(A, float(k))
        err = np.linalg.norm(M[:, p] - Q @ R, 'fro') / normM
        ortho = np.linalg.norm(Q.conj().T @ Q - np.eye(Q.shape[1]), 'fro')
        ok = (Q.shape[1] == k) and (ortho < 1e-10)
        status = 'PASS' if ok else 'FAIL'
        print(f"  [{status}] qr_sketch   {label:<26s} {variant:<9s}"
              f" err={err:.2e} ortho={ortho:.1e} k={Q.shape[1]}")
        if not ok:
            errors.append(f"qr_sketch {label} {variant}")


def check_orth(M, k, label, errors):
    for variant, A in [('dense', M),
                       ('explicit', aslinearoperator(M)),
                       ('matfree',  wrap_matfree(M))]:
        Q, flag, _ = orth_sketch(A, float(k))
        ortho = np.linalg.norm(Q.conj().T @ Q - np.eye(Q.shape[1]), 'fro') \
                if Q.shape[1] > 0 else 0.0
        ok = (flag == 0) and (Q.shape[1] == k) and (ortho < 1e-10)
        status = 'PASS' if ok else 'FAIL'
        print(f"  [{status}] orth_sketch {label:<26s} {variant:<9s}"
              f" flag={flag} ortho={ortho:.1e} k={Q.shape[1]}")
        if not ok:
            errors.append(f"orth_sketch {label} {variant}")


def check_id(M, k, label, errors):
    """Exercise id_sketch and id_qrpiv with all three T-methods."""
    normM = np.linalg.norm(M, 'fro')
    for fn_name, fn in [('id_sketch', id_sketch), ('id_qrpiv', id_qrpiv)]:
        for method in ['fast', 'svd', 'lstsq']:
            # id_qrpiv does not take block_size/extra_samples, but accepts method.
            kk, piv, T = fn(M, float(k), method=method)
            A_skel = M[:, piv[kk:]]
            A_basis = M[:, piv[:kk]]
            if T.size > 0:
                err = np.linalg.norm(A_skel - A_basis @ T, 'fro') / normM
            else:
                err = 0.0
            # Rank mode with reasonable T-method keeps err bounded; for noisy
            # full-rank inputs relative err can approach 1 so we use 1.5 as a
            # generous sanity gate — we mainly want to catch crashes and
            # wrong-shape returns.
            ok = (kk == k) and (err < 1.5) and (T.shape == (k, M.shape[1] - k))
            status = 'PASS' if ok else 'FAIL'
            print(f"  [{status}] {fn_name:<10s} method={method:<6s}"
                  f" {label:<20s} err={err:.2e} k={kk}")
            if not ok:
                errors.append(f"{fn_name} method={method} {label}")


def check_power_iter(errors):
    """Regression: rank mode with power_iter >= 1 where block_size =
    rank + extra_samples exceeds n. The intermediate sketch inside
    _power_iteration has n rows, so block_size > n must not overrun
    its QR Q-factor (a BoundsError in the Julia port before the fix)."""
    print("\n--- power_iter, rank + extra_samples > n ---")
    m, n, k = 50, 10, 6              # block_size = k + 12 = 18 > n = 10
    M = np.random.randn(m, k) @ np.random.randn(k, n)   # exact rank k
    normM = np.linalg.norm(M, 'fro')
    for variant, A in [('dense', M),
                       ('explicit', aslinearoperator(M)),
                       ('matfree',  wrap_matfree(M))]:
        ok = False
        err = float('nan')
        msg = ''
        try:
            Q0, flag, _ = orth_sketch(A, float(k), power_iter=2)
            U, s, Vh = svd_sketch(A, float(k), power_iter=2)
            err = np.linalg.norm(M - U @ np.diag(s) @ Vh, 'fro') / normM
            Q, R, p = qr_sketch(A, float(k), power_iter=2)
            kk, piv, T = id_sketch(A, float(k), power_iter=2)
            ok = (Q0.shape[1] == k and len(s) == k and err < 1e-8
                  and Q.shape[1] == k and kk == k)
            if not ok:
                msg = 'wrong shape/err'
        except Exception as e:
            msg = type(e).__name__
        status = 'PASS' if ok else 'FAIL'
        print(f"  [{status}] power_iter=2 orth/svd/qr/id rank={k} (n={n}) "
              f"{variant:<9s} err={err:.1e}")
        if not ok:
            errors.append(f"power_iter largeblock {variant} ({msg})")


def check_zero_tol_rank(errors):
    """Regression: in tolerance mode a zero matrix has rank 0. svd_sketch must
    agree with qr_sketch / id_sketch (rank_from_svals needs the same
    s[0] <= 0 guard that rank_from_diag has)."""
    print("\n--- zero matrix rank in tolerance mode ---")
    A = np.zeros((100, 60))
    U, s, Vh = svd_sketch(A, 1e-6)
    Q, R, p = qr_sketch(A, 1e-6)
    k, piv, T = id_sketch(A, 1e-6)
    ksvd, kqr, kid = len(s), Q.shape[1], k
    ok = (ksvd == 0) and (kqr == 0) and (kid == 0)
    status = 'PASS' if ok else 'FAIL'
    print(f"  [{status}] zero 100x60 rtol=1e-6: svd k={ksvd}, qr k={kqr}, id k={kid}")
    if not ok:
        errors.append(f"zero-tol-rank svd={ksvd} qr={kqr} id={kid}")


def check_overrank_id(errors):
    """Regression: ID in rank mode with the requested rank exceeding the true
    rank (singular R11). Every T-method must stay finite, return shape
    (k, n-k), and still reconstruct A -- in particular the all-zero matrix,
    where R11 is exactly singular (fast used to crash, svd used to give NaN)."""
    print("\n--- ID over-rank (requested rank > true rank) ---")
    cases = [("zeros", np.zeros((40, 20)), 5),
             ("rank3", np.random.randn(40, 3) @ np.random.randn(3, 20), 8)]
    for lbl, A, r in cases:
        m, n = A.shape
        normA = max(np.linalg.norm(A, 'fro'), 1.0)
        for fname, fn in [("id_sketch", id_sketch), ("id_qrpiv", id_qrpiv)]:
            for meth in ['fast', 'svd', 'lstsq']:
                ok = False
                err = float('nan')
                try:
                    k, piv, T = fn(A, float(r), method=meth)
                    Arec = np.zeros((m, n), dtype=A.dtype)
                    Arec[:, piv[:k]] = A[:, piv[:k]]
                    Arec[:, piv[k:]] = A[:, piv[:k]] @ T
                    err = np.linalg.norm(A - Arec, 'fro') / normA
                    ok = (T.shape == (k, n - k)) and np.all(np.isfinite(T)) and (err < 1e-8)
                except Exception:
                    ok = False
                status = 'PASS' if ok else 'FAIL'
                print(f"  [{status}] {lbl:<6s} {fname:<9s} {meth:<5s} rec_err={err:.1e}")
                if not ok:
                    errors.append(f"overrank-id {lbl} {fname} {meth}")


def main():
    np.random.seed(17)
    errors = []

    print("=" * 72)
    print("LINEAROPERATOR / WIDE-MATRIX / METHOD REGRESSION TESTS")
    print("=" * 72)

    # (m, n, k, label) — cover tall/wide, real/complex
    shapes = [
        (200, 100,  8, "tall real  200x100"),
        (100, 200,  8, "wide real  100x200"),
        (150,  80,  6, "tall cplx  150x80"),
        ( 80, 150,  6, "wide cplx   80x150"),
    ]

    for i, (m, n, k, label) in enumerate(shapes):
        is_complex = 'cplx' in label
        dtype = np.complex128 if is_complex else np.float64
        if is_complex:
            M = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(dtype)
        else:
            M = np.random.randn(m, n).astype(dtype)

        print(f"\n--- {label} ---")
        check_orth(M, k, label, errors)
        check_qr(M, k, label, errors)
        check_svd(M, k, label, errors)
        # id_sketch needs method parameter, id_qrpiv only on square-ish m>=n cases
        if m >= n:
            check_id(M, k, label, errors)

    check_power_iter(errors)
    check_overrank_id(errors)
    check_zero_tol_rank(errors)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    if errors:
        print(f"[FAIL] {len(errors)} check(s) failed:")
        for e in errors:
            print(f"   - {e}")
        print("=" * 72)
        return 1
    print("[PASS] All LinearOperator / wide / method regression checks passed.")
    print("=" * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
