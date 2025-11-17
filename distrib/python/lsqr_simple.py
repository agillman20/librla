"""Simple LSQR implementation for Python/NumPy.

LSQR algorithm for solving least squares problems using Golub-Kahan
bidiagonalization. Compatible with both explicit matrices and matrix-free
linear operators.
"""

import numpy as np
from typing import Union, Tuple
from parse_linop import parse_linop, LinearOperator


def lsqr_simple(A: Union[np.ndarray, LinearOperator],
                b: np.ndarray,
                tol: float = 1e-6,
                maxit: int = None) -> Tuple[np.ndarray, int, float, int]:
    """Simple LSQR implementation.

    Solves the least squares problem:
        min ||A*x - b||^2
    using the LSQR algorithm (Golub-Kahan bidiagonalization).

    Args:
        A: Matrix (mxn numpy array) OR LinearOperator structure
        b: Right-hand side (mx1 array)
        tol: Convergence tolerance (default 1e-6)
        maxit: Max iterations (default min(m,n))

    Returns:
        x: Solution (nx1 array)
        flag: 0 = converged, 1 = max iterations
        relres: Relative residual norm
        iter: Iterations performed

    Reference:
        Paige & Saunders (1982), LSQR algorithm

    Examples:
        >>> # With explicit matrix
        >>> A = np.random.randn(100, 50)
        >>> b = np.random.randn(100)
        >>> x, flag, relres, niter = lsqr_simple(A, b)

        >>> # With LinearOperator
        >>> from make_linop import make_linop
        >>> op = make_linop(100, 50, lambda x: A @ x, lambda y: A.conj().T @ y)
        >>> x, flag, relres, niter = lsqr_simple(op, b)
    """

    # Parse linear operator
    op = parse_linop(A)
    m = op.m
    n = op.n
    A_apply = op.apply
    AT_apply = op.applyT

    # Ensure b is 1D
    b = np.asarray(b).ravel()
    if len(b) != m:
        raise ValueError(f"b must have length {m}, got {len(b)}")

    if maxit is None:
        maxit = min(m, n)

    # Initialize
    u = b.copy()
    beta = np.linalg.norm(u)
    if beta > 0:
        u = u / beta

    v = AT_apply(u)
    alpha = np.linalg.norm(v)
    if alpha > 0:
        v = v / alpha

    # Initialize bidiagonal matrix and solution
    w = v.copy()
    x = np.zeros(n)
    phi_bar = beta
    rho_bar = alpha

    bnorm = np.linalg.norm(b)
    if bnorm == 0:
        bnorm = 1.0

    # Main iteration
    for iter_num in range(1, maxit + 1):
        # Continue bidiagonalization
        u = A_apply(v) - alpha * u
        beta = np.linalg.norm(u)
        if beta > 0:
            u = u / beta

        v = AT_apply(u) - beta * v
        alpha = np.linalg.norm(v)
        if alpha > 0:
            v = v / alpha

        # Update QR factorization of B_k (using Givens rotations)
        rho = np.sqrt(rho_bar**2 + beta**2)
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar

        # Update solution
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w

        # Check convergence
        resid_norm = abs(phi_bar)
        relres = resid_norm / bnorm
        print(f"  Iter {iter_num:3d}: {relres:.2e}")

        if relres < tol:
            flag = 0
            print(f"  Converged at iteration {iter_num}")
            return x, flag, relres, iter_num

    # Did not converge
    flag = 1
    relres = abs(phi_bar) / bnorm
    print("\n  Warning: Did not converge (max iterations reached)")
    return x, flag, relres, maxit
