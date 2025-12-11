"""
test5_method_comparison.jl - Compare all three T computation methods

Tests all three T computation methods on a full-rank random matrix:
1. method="fast" - Triangular solve (fastest, may have error > 1.0)
2. method="svd" - SVD-based pseudoinverse (stable)
3. method="lstsq" - Least-squares from original A (most accurate)

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: BSD-3-Clause
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

using LinearAlgebra
using Printf
using Random

include("librla.jl")

using .librla: id_sketch


function test_method_comparison()
    """Compare all three T computation methods."""

    println("="^70)
    println("TEST 5: T Computation Method Comparison")
    println("="^70)

    # Create full-rank random matrix
    Random.seed!(42)
    m, n = 400, 300
    @printf("\nMatrix size: %d x %d\n", m, n)
    @printf("Matrix type: Full-rank random (all %d columns independent)\n", n)

    # Create full-rank matrix
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
        id_sketch(A_warmup, 10.0, method="fast")
        id_sketch(A_warmup, 10.0, method="svd")
        id_sketch(A_warmup, 10.0, method="lstsq")
    catch
        # Ignore warm-up errors
    end
    println("JIT warm-up complete.")
    println("="^70)

    # =========================================================================
    # Test 1: method="fast" (fastest, may have error > 1.0)
    # =========================================================================
    println("\n1. method=\"fast\" (triangular solve)")
    println("-"^70)

    t1 = @elapsed k1, piv1, T1 = id_sketch(A, Float64(k_target), method="fast")

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
    if err1 > 1.0
        println("  [NOTE] Error > 1.0 is expected for full-rank matrices with method=\"fast\"")
    end

    # =========================================================================
    # Test 2: method="svd" (stable for ill-conditioned)
    # =========================================================================
    println("\n2. method=\"svd\" (SVD-based pseudoinverse)")
    println("-"^70)

    t2 = @elapsed k2, piv2, T2 = id_sketch(A, Float64(k_target), method="svd")

    # Compute error
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

    # =========================================================================
    # Test 3: method="lstsq" (most accurate, slowest)
    # =========================================================================
    println("\n3. method=\"lstsq\" (least-squares from original A)")
    println("-"^70)

    t3 = @elapsed k3, piv3, T3 = id_sketch(A, Float64(k_target), method="lstsq")

    # Compute error
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
    if err3 < 1.0
        println("  [OK] method=\"lstsq\" guarantees error < 1.0")
    end

    # =========================================================================
    # Summary
    # =========================================================================
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    println("  Method     Rank    Error        Max|T|       Time      Notes")
    println("-"^70)
    @printf("  fast       %4d    %.3e    %.3e    %.4fs   Fastest\n", k1, err1, maxT1, t1)
    @printf("  svd        %4d    %.3e    %.3e    %.4fs   Stable\n", k2, err2, maxT2, t2)
    @printf("  lstsq      %4d    %.3e    %.3e    %.4fs   Most accurate\n", k3, err3, maxT3, t3)
    println("="^70)

    # Validate
    success = true

    if err3 > 1.0
        println("\n[FAIL] method=\"lstsq\" should guarantee error < 1.0!")
        success = false
    end

    if k1 != k_target || k2 != k_target || k3 != k_target
        @printf("\n[FAIL] All methods should return rank k=%d!\n", k_target)
        success = false
    end

    if success
        println("\n[PASS] All method tests passed!")
    end

    return success
end


# Run test
success = test_method_comparison()
exit(success ? 0 : 1)
