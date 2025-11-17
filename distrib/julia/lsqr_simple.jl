"""
Simple LSQR implementation for Julia.

LSQR algorithm for solving least squares problems using Golub-Kahan
bidiagonalization. Compatible with both explicit matrices and matrix-free
linear operators.
"""

using LinearAlgebra
using Printf

# Include parse_linop.jl if not already loaded
if !@isdefined(parse_linop)
    include("parse_linop.jl")
end

"""
    lsqr_simple(A, b; tol=1e-6, maxit=nothing)

Simple LSQR implementation.

Solves the least squares problem:
    min ||A*x - b||^2
using the LSQR algorithm (Golub-Kahan bidiagonalization).

# Arguments
- `A`: Matrix (mxn) OR LinearOperator structure
- `b`: Right-hand side (mx1 vector)
- `tol`: Convergence tolerance (default 1e-6)
- `maxit`: Max iterations (default min(m,n))

# Returns
- `x`: Solution (nx1 vector)
- `flag`: 0 = converged, 1 = max iterations
- `relres`: Relative residual norm
- `iter`: Iterations performed

# Reference
Paige & Saunders (1982), LSQR algorithm

# Examples
```julia
# With explicit matrix
A = randn(100, 50)
b = randn(100)
x, flag, relres, niter = lsqr_simple(A, b)

# With LinearOperator
include("make_linop.jl")
op = make_linop(100, 50, x -> A * x, y -> A' * y)
x, flag, relres, niter = lsqr_simple(op, b)
```
"""
function lsqr_simple(A, b::AbstractVector; tol::Float64=1e-6, maxit=nothing)
    # Parse linear operator
    op = parse_linop(A)
    m = op.m
    n = op.n
    A_apply = op.apply
    AT_apply = op.applyT

    # Ensure b is a vector
    b = vec(b)
    if length(b) != m
        error("b must have length $m, got $(length(b))")
    end

    if isnothing(maxit)
        maxit = min(m, n)
    end

    # Initialize
    u = copy(b)
    beta = norm(u)
    if beta > 0
        u = u / beta
    end

    v = AT_apply(u)
    alpha = norm(v)
    if alpha > 0
        v = v / alpha
    end

    # Initialize bidiagonal matrix and solution
    w = copy(v)
    x = zeros(n)
    phi_bar = beta
    rho_bar = alpha

    bnorm = norm(b)
    if bnorm == 0
        bnorm = 1.0
    end

    # Main iteration
    for iter_num in 1:maxit
        # Continue bidiagonalization
        u = A_apply(v) - alpha * u
        beta = norm(u)
        if beta > 0
            u = u / beta
        end

        v = AT_apply(u) - beta * v
        alpha = norm(v)
        if alpha > 0
            v = v / alpha
        end

        # Update QR factorization of B_k (using Givens rotations)
        rho = sqrt(rho_bar^2 + beta^2)
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
        @printf("  Iter %3d: %.2e\n", iter_num, relres)

        if relres < tol
            flag = 0
            println("  Converged at iteration $iter_num")
            return x, flag, relres, iter_num
        end
    end

    # Did not converge
    flag = 1
    relres = abs(phi_bar) / bnorm
    println("\n  Warning: Did not converge (max iterations reached)")
    return x, flag, relres, maxit
end
