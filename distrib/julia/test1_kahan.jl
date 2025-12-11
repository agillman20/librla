"""
test1_kahan.jl - Test with Kahan matrix (384x384, theta=0.8)

Tests basic ID algorithms (id_sketch, id_qrpiv) on a Kahan matrix.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

using LinearAlgebra
using Printf

include("librla.jl")
include("kahan.jl")

using .librla: id_sketch, id_qrpiv


function test_kahan()
    """Test basic ID algorithms on Kahan matrix."""

    println("="^70)
    println("TEST 1: Kahan Matrix")
    println("="^70)

    # Create Kahan matrix with specified parameters
    n = 384
    theta = 0.8
    @printf("\nMatrix size: %d x %d\n", n, n)
    @printf("Matrix type: Kahan (theta=%.1f)\n", theta)

    A = kahan(n, theta=theta)
    normA = norm(A)

    # Compute condition number
    cond_A = cond(A)
    @printf("Condition number: %.3e\n", cond_A)

    # Target rank
    k_target = 15
    @printf("Target rank: %d\n", k_target)

    # -------------------------------------------------------------------------
    # JIT warm-up: compile all methods before timing
    # -------------------------------------------------------------------------
    println("\nJIT warm-up (compiling methods)...")
    A_warmup = randn(50, 50)
    try
        id_sketch(A_warmup, 10.0)
        id_qrpiv(A_warmup, 10.0)
    catch
        # Ignore warm-up errors
    end
    println("JIT warm-up complete.")
    println("="^70)

    # -------------------------------------------------------------------------
    # Method 1: id_sketch (randomized)
    # -------------------------------------------------------------------------
    println("\n1. librla.id_sketch (randomized QR sketching)")
    println("-"^70)

    t1 = @elapsed k1, piv1, T1 = id_sketch(A, Float64(k_target))

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

    # -------------------------------------------------------------------------
    # Method 2: id_qrpiv (deterministic QR with column pivoting)
    # -------------------------------------------------------------------------
    println("\n2. librla.id_qrpiv (deterministic QR with column pivoting via LAPACK)")
    println("-"^70)

    t2 = @elapsed k2, piv2, T2 = id_qrpiv(A, Float64(k_target))

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

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    println("  Method         Rank    Error        Max|T|       Time")
    println("-"^70)
    @printf("  id_sketch      %4d    %.3e    %.3e    %.4fs\n", k1, err1, maxT1, t1)
    @printf("  id_qrpiv       %4d    %.3e    %.3e    %.4fs\n", k2, err2, maxT2, t2)
    println("="^70)

    # Validate
    if err1 > 1.0 || err2 > 1.0
        println("\n[FAIL] Error > 1.0 detected!")
        return false
    end

    println("\n[PASS] Test completed successfully!")
    return true
end


# Run test
success = test_kahan()
exit(success ? 0 : 1)
