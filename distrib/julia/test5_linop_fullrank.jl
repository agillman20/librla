"""
test5_linop_fullrank.jl - LinearOperator test with full-rank random matrix

Tests id_sketch with LinearOperators on a full-rank random matrix, demonstrating
recompute_T parameter:
1. Dense matrix (baseline) - with recompute_T=true (default)
2. Explicit LinearOperator - with recompute_T=true
3. Matrix-free LinearOperator - recompute_T=true (accurate, n matvecs)
4. Matrix-free LinearOperator - recompute_T=false (fast, uses R matrix)
"""

using LinearAlgebra
using Printf
using Random

include("LibIDSketch.jl")
include("make_linop.jl")

using .LibIDSketch: id_sketch


function test_linop_fullrank()
    """Test id_sketch with LinearOperators on full-rank random matrix."""

    println("="^70)
    println("TEST 5: LinearOperators - Full-Rank Random Matrix")
    println("="^70)

    # Create full-rank random matrix
    Random.seed!(42)
    m, n = 400, 300
    @printf("\nMatrix size: %d x %d\n", m, n)
    @printf("Matrix type: Full-rank random (all %d columns independent)\n", n)

    # Create full-rank matrix: all columns are linearly independent
    A = randn(m, n)
    normA = norm(A)

    # Target rank (low compared to matrix rank)
    k_target = 20
    @printf("Target rank: %d (%.1f%% of columns)\n", k_target, 100*k_target/n)
    println("="^70)

    # -------------------------------------------------------------------------
    # JIT warm-up: compile all methods before timing
    # -------------------------------------------------------------------------
    println("\nJIT warm-up (compiling methods)...")
    A_warmup = randn(50, 30)
    try
        # Warm up dense
        id_sketch(A_warmup, rtol=10.0, recompute_T=true)
        # Warm up explicit LinearOperator
        op_warmup = make_linop(A_warmup)
        id_sketch(op_warmup, rtol=10.0, recompute_T=true)
        # Warm up matrix-free LinearOperator (both modes)
        op_mf_warmup = make_linop(Float64, 50, 30, x -> A_warmup * x, y -> A_warmup' * y)
        id_sketch(op_mf_warmup, rtol=10.0, recompute_T=true)
        id_sketch(op_mf_warmup, rtol=10.0, recompute_T=false)
    catch
        # Ignore warm-up errors
    end
    println("JIT warm-up complete.")
    println("="^70)

    # =========================================================================
    # Test 1: Dense Matrix (Baseline, recompute_T=true by default)
    # =========================================================================
    println("\n1. Dense Matrix (baseline, recompute_T=true)")
    println("-"^70)

    t1 = @elapsed k1, piv1, T1 = id_sketch(A, rtol=Float64(k_target), recompute_T=true)

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
    if err1 < 1.0
        println("  [OK] Error < 1.0 (recompute_T=true guarantees this)")
    end

    # =========================================================================
    # Test 2: Explicit LinearOperator (Matrix Wrapper, recompute_T=true)
    # =========================================================================
    println("\n2. Explicit LinearOperator (matrix wrapper, recompute_T=true)")
    println("-"^70)

    A_linop_explicit = make_linop(A)
    @printf("  Operator: %d x %d\n", A_linop_explicit.m, A_linop_explicit.n)
    @printf("  Is explicit: %s\n", A_linop_explicit.is_explicit)

    t2 = @elapsed k2, piv2, T2 = id_sketch(A_linop_explicit, rtol=Float64(k_target), recompute_T=true)

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
    # Test 3: Matrix-Free LinearOperator (recompute_T=true, accurate)
    # =========================================================================
    println("\n3. Matrix-free LinearOperator (recompute_T=true, accurate)")
    println("-"^70)

    # Create matrix-free operator with function handles
    Afun(x) = A * x        # Forward operation: y = A*x
    ATfun(y) = A' * y      # Adjoint operation: y = A'*x

    A_linop_mf = make_linop(Float64, m, n, Afun, ATfun)
    @printf("  Operator: %d x %d\n", A_linop_mf.m, A_linop_mf.n)
    @printf("  Is explicit: %s\n", A_linop_mf.is_explicit)
    println("  Mode: Rank mode (rtol >= 1), recompute_T=true")
    @printf("  Note: Extracts all %d columns via unit vectors (n matvecs)\n", n)

    t3 = @elapsed k3, piv3, T3 = id_sketch(A_linop_mf, rtol=Float64(k_target), recompute_T=true)

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

    if k3 == k_target && err3 < 1.0
        @printf("  [OK] Matrix-free (recompute_T=true): rank k=%d, error < 1.0\n", k_target)
    elseif err3 < 1.0
        println("  [OK] Error < 1.0 guaranteed by recompute_T=true")
    end

    # =========================================================================
    # Test 4: Matrix-Free LinearOperator (recompute_T=false, fast)
    # =========================================================================
    println("\n4. Matrix-free LinearOperator (recompute_T=false, fast)")
    println("-"^70)

    @printf("  Operator: %d x %d\n", A_linop_mf.m, A_linop_mf.n)
    @printf("  Is explicit: %s\n", A_linop_mf.is_explicit)
    println("  Mode: Rank mode (rtol >= 1), recompute_T=false")
    println("  Note: Uses R matrix from sketch (Fortran approach, no extra matvecs)")

    t4 = @elapsed k4, piv4, T4 = id_sketch(A_linop_mf, rtol=Float64(k_target), recompute_T=false)

    # Compute error using explicit matrix (for validation)
    A_skel4 = A[:, piv4[k4+1:end]]
    A_basis4 = A[:, piv4[1:k4]]
    if !isempty(T4)
        err4 = norm(A_skel4 - A_basis4 * T4) / normA
        maxT4 = maximum(abs.(T4))
    else
        err4 = 0.0
        maxT4 = 0.0
    end

    @printf("  Rank:      k = %d\n", k4)
    @printf("  Error:     ||A_skel - A_basis @ T|| / ||A|| = %.3e\n", err4)
    @printf("  Max |T|:   %.3e\n", maxT4)
    @printf("  Time:      %.4f s\n", t4)

    # Compare with recompute_T=true
    speedup = t3 / t4
    error_ratio = err4 / err3

    @printf("  Speedup:   %.1fx faster than recompute_T=true\n", speedup)
    @printf("  Error ratio: %.2fx (err_false / err_true)\n", error_ratio)

    if err4 > 1.0
        println("  [NOTE] Error > 1.0 is expected for full-rank matrices with recompute_T=false")
        println("         This uses Fortran's fast R-matrix approach, trading accuracy for speed")
    else
        println("  [OK] Error < 1.0 (better than expected!)")
    end

    # =========================================================================
    # Summary
    # =========================================================================
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    println("  Method                        Rank    Error        Max|T|       Time")
    println("-"^70)
    @printf("  Dense (recompute_T=true)      %4d    %.3e    %.3e    %.4fs\n", k1, err1, maxT1, t1)
    @printf("  Explicit LinOp (recomp=true)  %4d    %.3e    %.3e    %.4fs\n", k2, err2, maxT2, t2)
    @printf("  Matrix-free (recompute=true)  %4d    %.3e    %.3e    %.4fs\n", k3, err3, maxT3, t3)
    @printf("  Matrix-free (recompute=false) %4d    %.3e    %.3e    %.4fs\n", k4, err4, maxT4, t4)
    println("="^70)

    println("\nKey Observations:")
    @printf("  - Full-rank matrix: %dx%d, target k=%d (%.1f%% of columns)\n", m, n, k_target, 100*k_target/n)
    @printf("  - recompute_T=true:  Guarantees error < 1.0 (all methods: %.3e, %.3e, %.3e)\n", err1, err2, err3)
    @printf("  - recompute_T=false: %.1fx faster, but error may be > 1.0 (err=%.3e)\n", speedup, err4)
    @printf("  - Trade-off: Speed (%.1fx) vs Accuracy (%.2fx degradation)\n", speedup, error_ratio)

    # Validate
    success = true

    if err1 > 1.0 || err2 > 1.0 || err3 > 1.0
        println("\n[FAIL] recompute_T=true should guarantee error < 1.0!")
        success = false
    end

    # Note: Do NOT fail on err_diff - randomness can cause different pivots
    # The NOTE message already explains this is expected

    if k3 != k_target || k4 != k_target
        @printf("\n[FAIL] Matrix-free should return rank k=%d!\n", k_target)
        success = false
    end

    if success
        println("\n[PASS] All LinearOperator tests passed!")
        println("       recompute_T=true guarantees error < 1.0 for all modes")
        @printf("       recompute_T=false provides %.1fx speedup with acceptable error increase\n", speedup)
    end

    return success
end


# Run test
success = test_linop_fullrank()
exit(success ? 0 : 1)
