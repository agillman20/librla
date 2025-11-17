"""
test3_linop_hilbert.jl - Simple test with LinearOperators

Tests id_sketch with LinearOperators on a medium-size Hilbert matrix:
1. Dense matrix (baseline)
2. Explicit LinearOperator (matrix wrapper)
3. Matrix-free LinearOperator (function handles - rank mode only)
"""

using LinearAlgebra
using Printf

include("LibIDSketch.jl")
include("make_linop.jl")

using .LibIDSketch: id_sketch


"""
    hilb(m::Int, n::Int) -> Matrix{Float64}

Generate an m x n Hilbert matrix.
"""
function hilb(m::Int, n::Int)
    i = reshape(1:m, m, 1)
    j = reshape(1:n, 1, n)
    return 1.0 ./ (i .+ j .- 1)
end


function test_linop_hilbert()
    """Test id_sketch with LinearOperators on Hilbert matrix."""

    println("="^70)
    println("TEST 3: LinearOperators - Hilbert Matrix")
    println("="^70)

    # Create medium-size Hilbert matrix (severely ill-conditioned)
    m, n = 300, 200
    @printf("\nMatrix size: %d x %d\n", m, n)
    println("Matrix type: Hilbert (severely ill-conditioned)")

    A = hilb(m, n)
    normA = norm(A)

    # Target rank
    k_target = 15
    @printf("Target rank: %d\n", k_target)

    # -------------------------------------------------------------------------
    # JIT warm-up: compile all methods before timing
    # -------------------------------------------------------------------------
    println("\nJIT warm-up (compiling methods)...")
    A_warmup = randn(50, 30)
    try
        # Warm up dense
        id_sketch(A_warmup, rtol=10.0)
        # Warm up explicit LinearOperator
        op_warmup = make_linop(A_warmup)
        id_sketch(op_warmup, rtol=10.0)
        # Warm up matrix-free LinearOperator
        op_mf_warmup = make_linop(Float64, 50, 30, x -> A_warmup * x, y -> A_warmup' * y)
        id_sketch(op_mf_warmup, rtol=10.0)
    catch
        # Ignore warm-up errors
    end
    println("JIT warm-up complete.")
    println("="^70)

    # =========================================================================
    # Test 1: Dense Matrix (Baseline)
    # =========================================================================
    println("\n1. Dense Matrix (baseline)")
    println("-"^70)

    t1 = @elapsed k1, piv1, T1 = id_sketch(A, rtol=Float64(k_target))

    # Compute error
    A_skel1 = A[:, piv1[k1+1:end]]
    A_basis1 = A[:, piv1[1:k1]]
    if !isempty(T1)
        err1 = norm(A_skel1 - A_basis1 * T1) / normA
        maxT1 = maximum(abs.(T1))
    else
        err1 = 0.0
        maxT1 = 0.0
    end

    @printf("  Rank:      k = %d\n", k1)
    @printf("  Error:     ||A_skel - A_basis @ T|| / ||A|| = %.3e\n", err1)
    @printf("  Max |T|:   %.3e\n", maxT1)
    @printf("  Time:      %.4f s\n", t1)

    # =========================================================================
    # Test 2: Explicit LinearOperator (Matrix Wrapper)
    # =========================================================================
    println("\n2. Explicit LinearOperator (matrix wrapper)")
    println("-"^70)

    A_linop_explicit = make_linop(A)
    @printf("  Operator: %d x %d\n", A_linop_explicit.m, A_linop_explicit.n)
    @printf("  Is explicit: %s\n", A_linop_explicit.is_explicit)

    t2 = @elapsed k2, piv2, T2 = id_sketch(A_linop_explicit, rtol=Float64(k_target))

    # Compute error using explicit matrix access
    A_skel2 = A[:, piv2[k2+1:end]]
    A_basis2 = A[:, piv2[1:k2]]
    if !isempty(T2)
        err2 = norm(A_skel2 - A_basis2 * T2) / normA
        maxT2 = maximum(abs.(T2))
    else
        err2 = 0.0
        maxT2 = 0.0
    end

    @printf("  Rank:      k = %d\n", k2)
    @printf("  Error:     ||A_skel - A_basis @ T|| / ||A|| = %.3e\n", err2)
    @printf("  Max |T|:   %.3e\n", maxT2)
    @printf("  Time:      %.4f s\n", t2)

    # Verify explicit matches dense (rank and error should be similar)
    if k1 == k2 && abs(err1 - err2) < 1e-12
        println("  [OK] Explicit LinearOperator produces same rank and error as dense!")
    else
        @printf("  [NOTE] k_dense=%d, k_linop=%d, err_diff=%.3e\n", k1, k2, abs(err1-err2))
        println("  (Pivots may differ due to randomness, but results should be similar)")
    end

    # =========================================================================
    # Test 3: Matrix-Free LinearOperator (Function Handles - Rank Mode Only)
    # =========================================================================
    println("\n3. Matrix-free LinearOperator (function handles)")
    println("-"^70)

    # Create matrix-free operator with function handles
    Afun(x) = A * x        # Forward operation: y = A*x
    ATfun(y) = A' * y      # Adjoint operation: y = A'*x

    A_linop_mf = make_linop(Float64, m, n, Afun, ATfun)
    @printf("  Operator: %d x %d\n", A_linop_mf.m, A_linop_mf.n)
    @printf("  Is explicit: %s\n", A_linop_mf.is_explicit)
    println("  Mode: Rank mode only (rtol >= 1)")

    t3 = @elapsed k3, piv3, T3 = id_sketch(A_linop_mf, rtol=Float64(k_target))

    # Compute error using explicit matrix (for validation)
    A_skel3 = A[:, piv3[k3+1:end]]
    A_basis3 = A[:, piv3[1:k3]]
    if !isempty(T3)
        err3 = norm(A_skel3 - A_basis3 * T3) / normA
        maxT3 = maximum(abs.(T3))
    else
        err3 = 0.0
        maxT3 = 0.0
    end

    @printf("  Rank:      k = %d\n", k3)
    @printf("  Error:     ||A_skel - A_basis @ T|| / ||A|| = %.3e\n", err3)
    @printf("  Max |T|:   %.3e\n", maxT3)
    @printf("  Time:      %.4f s\n", t3)

    if k3 == k_target
        @printf("  [OK] Matrix-free returns target rank k=%d\n", k_target)
    else
        @printf("  [WARNING] Expected k=%d, got k=%d\n", k_target, k3)
    end

    # =========================================================================
    # Summary
    # =========================================================================
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    println("  Method              Rank    Error        Max|T|       Time")
    println("-"^70)
    @printf("  Dense (baseline)    %4d    %.3e    %.3e    %.4fs\n", k1, err1, maxT1, t1)
    @printf("  Explicit LinOp      %4d    %.3e    %.3e    %.4fs\n", k2, err2, maxT2, t2)
    @printf("  Matrix-free LinOp   %4d    %.3e    %.3e    %.4fs\n", k3, err3, maxT3, t3)
    println("="^70)

    # Validate
    success = true

    if err1 > 1.0 || err2 > 1.0 || err3 > 1.0
        println("\n[FAIL] Error > 1.0 detected!")
        success = false
    end

    if k1 != k2 || abs(err1 - err2) > 1e-10
        @printf("\n[FAIL] Explicit LinearOperator should match dense! k1=%d, k2=%d, err_diff=%.3e\n",
                k1, k2, abs(err1-err2))
        success = false
    end

    if k3 != k_target
        @printf("\n[FAIL] Matrix-free should return rank k=%d!\n", k_target)
        success = false
    end

    if success
        println("\n[PASS] All LinearOperator tests passed!")
    end

    return success
end


# Run test
success = test_linop_hilbert()
exit(success ? 0 : 1)
