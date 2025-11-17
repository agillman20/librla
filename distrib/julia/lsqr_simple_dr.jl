"""
LinearOperator framework validation test for Julia.

PURPOSE:
  Verify that LinearOperator abstraction gives IDENTICAL results to
  explicit matrices for LSQR algorithm

LINEAROPERATOR FRAMEWORK:
  Provides uniform interface for matrix operations:
    - Explicit matrices (A)
    - Function handles (A*x, A'*y)
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

using LinearAlgebra
using Random
using Printf

# Include required files
include("hilb.jl")
include("make_linop.jl")
include("parse_linop.jl")
include("lsqr_simple.jl")

function main()
    Random.seed!(1)

    m = 120 * 2 * 10  # 2400 (rows of A)
    k = 240 * 2 * 10  # 4800 (cols of A)

    # Create rectangular Hilbert matrix (ill-conditioned test case)
    a = hilb(m, k)

    println("========================================================")
    println("Test: LinearOperator Framework Validation (Julia)")
    println("========================================================")
    println("Matrix A: $m x $k (underdetermined)")
    println("Tolerance: 1e-10")
    println("Problem: min ||A*x - y|| (minimum norm solution)\n")

    ## Test: Single right-hand side - LinearOperator validation

    println("------------------------------------------------------------")
    println("Test: Single RHS - LinearOperator Validation")
    println("------------------------------------------------------------")

    x0 = randn(k)
    y = a * x0 + rand(m) * 1e-16  # Add small noise

    # Direct solve (ground truth - minimum norm solution)
    t_start = time()
    x_direct = a \ y
    t_direct = time() - t_start
    err_direct = norm(a * x_direct - y) / norm(y)
    println("Direct solve (backslash):")
    @printf("  Time: %.4f seconds\n", t_direct)
    @printf("  Relative error: %.4e\n\n", err_direct)

    ## Method 1: LSQR with explicit matrix

    println("Method 1: LSQR with Explicit Matrix")
    println("  Algorithm: Golub-Kahan bidiagonalization")
    println("  Operations: A*v and A'*u per iteration")
    println("  Memory: O(k) - short recurrence\n")

    t_start = time()
    x_explicit, flag_explicit, relres_explicit, iter_explicit = lsqr_simple(
        a, y, tol=1e-10, maxit=100
    )
    t_explicit = time() - t_start

    err_explicit = norm(a * x_explicit - y) / norm(y)
    println("LSQR (explicit matrix) results:")
    @printf("  Time: %.4f seconds\n", t_explicit)
    @printf("  Iterations: %d\n", iter_explicit)
    @printf("  Relative error: %.4e\n", err_explicit)
    @printf("  Flag: %d (0=converged)\n", flag_explicit)

    ## Method 2: LSQR with LinearOperator from matrix

    println("\nMethod 2: LSQR with LinearOperator from Matrix")
    println("  Algorithm: Same as Method 1")
    println("  Input: LinearOperator structure (from matrix)")
    println("  Purpose: Verify LinearOperator interface\n")

    # Create LinearOperator from matrix
    op_from_matrix = make_linop(a)

    t_start = time()
    x_operator, flag_operator, relres_operator, iter_operator = lsqr_simple(
        op_from_matrix, y, tol=1e-10, maxit=100
    )
    t_operator = time() - t_start

    err_operator = norm(a * x_operator - y) / norm(y)
    match_explicit = norm(x_explicit - x_operator) / norm(x_explicit)

    println("LSQR (LinearOperator) results:")
    @printf("  Time: %.4f seconds\n", t_operator)
    @printf("  Iterations: %d\n", iter_operator)
    @printf("  Relative error: %.4e\n", err_operator)
    @printf("  Flag: %d (0=converged)\n", flag_operator)
    @printf("  Match with Method 1: %.4e (should be ~0)\n", match_explicit)

    ## Method 3: LSQR with matrix-free LinearOperator

    println("\nMethod 3: LSQR with Matrix-Free LinearOperator")
    println("  Algorithm: Same as Method 1")
    println("  Input: LinearOperator from function handles")
    println("  Purpose: Verify matrix-free operations\n")

    # Create matrix-free LinearOperator
    A_forward = x -> a * x
    A_adjoint = y -> a' * y
    op_matfree = make_linop(eltype(a), m, k, A_forward, A_adjoint)

    t_start = time()
    x_matfree, flag_matfree, relres_matfree, iter_matfree = lsqr_simple(
        op_matfree, y, tol=1e-10, maxit=100
    )
    t_matfree = time() - t_start

    err_matfree = norm(a * x_matfree - y) / norm(y)
    match_matfree = norm(x_explicit - x_matfree) / norm(x_explicit)

    println("LSQR (matrix-free) results:")
    @printf("  Time: %.4f seconds\n", t_matfree)
    @printf("  Iterations: %d\n", iter_matfree)
    @printf("  Relative error: %.4e\n", err_matfree)
    @printf("  Flag: %d (0=converged)\n", flag_matfree)
    @printf("  Match with Method 1: %.4e (should be ~0)\n", match_matfree)

    ## Comparison

    println("\n------------------------------------------------------------")
    println("Single RHS Comparison Summary")
    println("------------------------------------------------------------")
    @printf("%-30s %10s %12s %15s\n", "Method", "Time (s)", "Iters", "Match Error")
    @printf("%-30s %10.4f %12d %15s\n", "LSQR (explicit)", t_explicit, iter_explicit, "-")
    @printf("%-30s %10.4f %12d %15.4e\n", "LSQR (LinearOperator)", t_operator, iter_operator, match_explicit)
    @printf("%-30s %10.4f %12d %15.4e\n", "LSQR (matrix-free)", t_matfree, iter_matfree, match_matfree)

    if match_explicit < 1e-14 && match_matfree < 1e-14
        println("\n[OK] VALIDATION PASSED: All three methods give IDENTICAL results")
    else
        println("\n[FAIL] VALIDATION FAILED: Methods do not match")
    end
end

# Run the test
main()
