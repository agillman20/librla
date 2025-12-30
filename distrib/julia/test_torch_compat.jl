# test_torch_compat.jl - Test PyTorch-compatible wrappers for librla
#
# Tests svd_lowrank and pca_lowrank functions:
# - Reconstruction error
# - Singular value accuracy
# - Orthonormality of U and V
# - M parameter (subtraction matrix)
# - center parameter for PCA
#
# Usage:
#     julia test_torch_compat.jl
#
# Author: Adrianna Gillman, Zydrunas Gimbutas
# SPDX-License-Identifier: TBD
# Version: 0.1.0
# Date: TBD
# Assisted by: Claude Code (Anthropic)

module TestTorchCompat

using LinearAlgebra
using Printf
using Statistics
using Random

include(joinpath(@__DIR__, "torch_compat.jl"))
using .torch_compat: svd_lowrank, pca_lowrank

export test


mutable struct TestResult
    name::String
    q::Int

    k::Int

    recon_err::Float64
    sval_err::Float64

    orth_U::Float64
    orth_V::Float64

    t_compat::Float64
    t_ref::Float64

    passed::Bool
end


function hilb_matrix(m::Int, n::Int)
    """Generate an mxn Hilbert matrix."""
    i = (1:m)
    j = (1:n)'
    return 1.0 ./ (i .+ j .- 1)
end


function run_svd_lowrank_test(A::Matrix{T}, q::Int, name::String; M=nothing) where T
    """Test svd_lowrank on a single matrix."""

    println("\n", "="^70)
    println("Test: ", name)
    m, n = size(A)
    print("Matrix: ", m, "x", n)
    if eltype(A) <: Complex
        print(", complex")
    end
    println()
    println("Parameter: q = ", q)
    if M !== nothing
        println("Using M parameter (subtraction matrix)")
    end
    println("="^70)

    # Matrix to actually decompose
    A_eff = M !== nothing ? A - M : A
    normA = norm(A_eff)

    # -------------------------------------------------------------------------
    # svd_lowrank (torch_compat)
    # -------------------------------------------------------------------------
    println("\n--- svd_lowrank (torch_compat) ---")

    t_compat = @elapsed U, s, V = svd_lowrank(A; q=q, niter=2, M=M)

    k = length(s)

    # Reconstruction error (V is not transposed in torch_compat)
    A_recon = U * Diagonal(s) * V'
    recon_err = norm(A_eff - A_recon) / normA

    # Orthonormality checks
    orth_U = norm(U' * U - I(k))
    orth_V = norm(V' * V - I(k))

    @printf("Rank:       k = %d\n", k)
    @printf("Recon Err:  ||A - U @ S @ V'|| / ||A|| = %.3e\n", recon_err)
    @printf("Orth U:     ||U'U - I|| = %.3e\n", orth_U)
    @printf("Orth V:     ||V'V - I|| = %.3e\n", orth_V)
    @printf("Time:       %.4f s\n", t_compat)

    # -------------------------------------------------------------------------
    # Reference (svd truncated)
    # -------------------------------------------------------------------------
    println("\n--- Reference (svd truncated) ---")

    t_ref = @elapsed F = svd(A_eff)

    # Truncate to same rank
    U_ref = F.U[:, 1:k]
    s_ref = F.S[1:k]
    V_ref = F.V[:, 1:k]

    # Reconstruction error
    A_recon_ref = U_ref * Diagonal(s_ref) * V_ref'
    recon_err_ref = norm(A_eff - A_recon_ref) / normA

    @printf("Rank:       k = %d\n", k)
    @printf("Recon Err:  ||A - U @ S @ V'|| / ||A|| = %.3e\n", recon_err_ref)
    @printf("Time:       %.4f s (full SVD)\n", t_ref)

    # Singular value accuracy
    sval_err = norm(s_ref) > 0 ? norm(s - s_ref) / norm(s_ref) : 0.0

    @printf("\nSingular value accuracy: ||s - s_ref|| / ||s_ref|| = %.3e\n", sval_err)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    println("\n--- Summary ---")
    @printf("%-28s %-8s %-12s %-12s %-10s\n", "Method", "Rank", "Recon Err", "SVal Err", "Time (s)")
    println("-"^75)
    @printf("%-28s %-8d %-12.3e %-12.3e %-10.4f\n", "svd_lowrank (torch_compat)", k, recon_err, sval_err, t_compat)
    @printf("%-28s %-8d %-12.3e %-12s %-10.4f\n", "svd (reference)", k, recon_err_ref, "(ref)", t_ref)

    # -------------------------------------------------------------------------
    # Determine if test passed
    # -------------------------------------------------------------------------
    error_ratio_ok = recon_err_ref == 0 || (recon_err / max(recon_err_ref, 1e-15) < 4.0)
    passed = error_ratio_ok && sval_err < 0.5 && orth_U < 1e-10 && orth_V < 1e-10

    return TestResult(name, q, k, recon_err, sval_err, orth_U, orth_V, t_compat, t_ref, passed)
end


function run_pca_lowrank_test(A::Matrix{T}, q::Int, name::String; center::Bool=true) where T
    """Test pca_lowrank on a single matrix."""

    println("\n", "="^70)
    println("Test: ", name)
    m, n = size(A)
    println("Matrix: ", m, "x", n)
    println("Parameter: q = ", q, ", center = ", center)
    println("="^70)

    # Centered matrix for comparison
    A_centered = center ? A .- mean(A, dims=1) : A
    normA = norm(A_centered)

    # -------------------------------------------------------------------------
    # pca_lowrank (torch_compat)
    # -------------------------------------------------------------------------
    println("\n--- pca_lowrank (torch_compat) ---")

    t_compat = @elapsed U, s, V = pca_lowrank(A; q=q, center=center, niter=2)

    k = length(s)

    # Reconstruction error (V is not transposed in torch_compat)
    A_recon = U * Diagonal(s) * V'
    recon_err = norm(A_centered - A_recon) / normA

    # Orthonormality checks
    orth_U = norm(U' * U - I(k))
    orth_V = norm(V' * V - I(k))

    @printf("Rank:       k = %d\n", k)
    @printf("Recon Err:  ||A_c - U @ S @ V'|| / ||A_c|| = %.3e\n", recon_err)
    @printf("Orth U:     ||U'U - I|| = %.3e\n", orth_U)
    @printf("Orth V:     ||V'V - I|| = %.3e\n", orth_V)
    @printf("Time:       %.4f s\n", t_compat)

    # -------------------------------------------------------------------------
    # Reference (svd on centered data)
    # -------------------------------------------------------------------------
    println("\n--- Reference (svd on centered data) ---")

    t_ref = @elapsed F = svd(A_centered)

    # Truncate to same rank
    U_ref = F.U[:, 1:k]
    s_ref = F.S[1:k]
    V_ref = F.V[:, 1:k]

    # Reconstruction error
    A_recon_ref = U_ref * Diagonal(s_ref) * V_ref'
    recon_err_ref = norm(A_centered - A_recon_ref) / normA

    @printf("Rank:       k = %d\n", k)
    @printf("Recon Err:  ||A_c - U @ S @ V'|| / ||A_c|| = %.3e\n", recon_err_ref)
    @printf("Time:       %.4f s (full SVD)\n", t_ref)

    # Singular value accuracy
    sval_err = norm(s_ref) > 0 ? norm(s - s_ref) / norm(s_ref) : 0.0

    @printf("\nSingular value accuracy: ||s - s_ref|| / ||s_ref|| = %.3e\n", sval_err)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    println("\n--- Summary ---")
    @printf("%-28s %-8s %-12s %-12s %-10s\n", "Method", "Rank", "Recon Err", "SVal Err", "Time (s)")
    println("-"^75)
    @printf("%-28s %-8d %-12.3e %-12.3e %-10.4f\n", "pca_lowrank (torch_compat)", k, recon_err, sval_err, t_compat)
    @printf("%-28s %-8d %-12.3e %-12s %-10.4f\n", "svd (reference)", k, recon_err_ref, "(ref)", t_ref)

    # -------------------------------------------------------------------------
    # Determine if test passed
    # -------------------------------------------------------------------------
    error_ratio_ok = recon_err_ref == 0 || (recon_err / max(recon_err_ref, 1e-15) < 4.0)
    passed = error_ratio_ok && sval_err < 0.5 && orth_U < 1e-10 && orth_V < 1e-10

    return TestResult(name, q, k, recon_err, sval_err, orth_U, orth_V, t_compat, t_ref, passed)
end


function test()
    """Run comprehensive torch_compat tests."""

    println("="^70)
    println("TORCH_COMPAT TESTS")
    println("Testing PyTorch-compatible wrappers: svd_lowrank, pca_lowrank")
    println("="^70)
    println("\nEnvironment:")
    println("  Julia:      ", VERSION)
    println("="^70)

    # Results collection
    results = TestResult[]

    # -------------------------------------------------------------------------
    # SVD_LOWRANK TESTS
    # -------------------------------------------------------------------------
    println("\n\n", "="^70)
    println("SVD_LOWRANK TESTS")
    println("Testing PyTorch-compatible randomized SVD")
    println("="^70)

    Random.seed!(42)

    # Test 1: Random matrix (well-conditioned)
    A1 = randn(500, 300)
    push!(results, run_svd_lowrank_test(A1, 20, "svd_lowrank: Random Matrix"))

    # Test 2: Low-rank matrix
    U = randn(400, 15)
    V = randn(250, 15)
    A2 = U * V' + 1e-10 * randn(400, 250)
    push!(results, run_svd_lowrank_test(A2, 20, "svd_lowrank: Low-Rank Matrix (rank~15)"))

    # Test 3: Hilbert matrix (ill-conditioned)
    A3 = hilb_matrix(500, 300)
    push!(results, run_svd_lowrank_test(A3, 15, "svd_lowrank: Hilbert Matrix"))

    # Test 4: Complex matrix
    A4 = randn(300, 200) + im * randn(300, 200)
    push!(results, run_svd_lowrank_test(A4, 25, "svd_lowrank: Complex Matrix"))

    # Test 5: With M parameter (subtraction)
    A5 = randn(400, 300)
    M5 = randn(400, 300) * 0.1
    push!(results, run_svd_lowrank_test(A5, 20, "svd_lowrank: With M parameter"; M=M5))

    # Test 6: Tall matrix
    A6 = randn(1000, 100)
    push!(results, run_svd_lowrank_test(A6, 30, "svd_lowrank: Tall Matrix (1000x100)"))

    # Test 7: Wide matrix
    A7 = randn(100, 1000)
    push!(results, run_svd_lowrank_test(A7, 30, "svd_lowrank: Wide Matrix (100x1000)"))

    # -------------------------------------------------------------------------
    # PCA_LOWRANK TESTS
    # -------------------------------------------------------------------------
    println("\n\n", "="^70)
    println("PCA_LOWRANK TESTS")
    println("Testing PyTorch-compatible randomized PCA")
    println("="^70)

    # Test 8: Random data with centering
    A8 = randn(500, 100) .+ 5.0  # Add offset to test centering
    push!(results, run_pca_lowrank_test(A8, 20, "pca_lowrank: Centered Random Data"; center=true))

    # Test 9: Without centering
    A9 = randn(500, 100)
    push!(results, run_pca_lowrank_test(A9, 20, "pca_lowrank: Uncentered Random Data"; center=false))

    # Test 10: Low-rank data with centering
    U10 = randn(400, 10)
    V10 = randn(80, 10)
    A10 = U10 * V10' .+ 3.0 + 1e-8 * randn(400, 80)
    push!(results, run_pca_lowrank_test(A10, 15, "pca_lowrank: Low-Rank Centered"; center=true))

    # Test 11: Default q parameter
    A11 = randn(100, 50)
    push!(results, run_pca_lowrank_test(A11, 6, "pca_lowrank: Default q=6"; center=true))

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    println("\n\n", "="^80)
    @printf("TEST SUMMARY - %d tests completed\n", length(results))
    println("="^80)

    # Overall pass/fail
    passed_tests = sum(r.passed for r in results)
    total_tests = length(results)
    pass_rate = 100.0 * passed_tests / total_tests

    println()
    @printf("Pass Rate: %d/%d (%.1f%%)\n", passed_tests, total_tests, pass_rate)

    if passed_tests == total_tests
        println("[PASS] All tests PASSED")
    else
        println("[WARNING] Some tests FAILED")
        for r in results
            if !r.passed
                println("  [FAIL] ", r.name)
            end
        end
    end

    # Performance summary
    println()
    println("="^80)
    println("Performance Summary")
    println("="^80)

    avg_time_compat = mean([r.t_compat for r in results])
    avg_time_ref = mean([r.t_ref for r in results])

    println()
    @printf("%-28s %-12s %-15s\n", "Method", "Avg Time", "vs Reference")
    println("-"^80)
    @printf("%-28s %8.4fs    %6.1fx \n", "torch_compat", avg_time_compat, avg_time_ref/avg_time_compat)
    @printf("%-28s %8.4fs    %6.1fx -\n", "svd (reference)", avg_time_ref, 1.0)

    # Accuracy summary
    println()
    println("Reconstruction Error Summary:")
    println("-"^80)

    avg_recon_err = mean([r.recon_err for r in results])
    max_recon_err = maximum([r.recon_err for r in results])

    @printf("  torch_compat:  mean=%.3e, max=%.3e\n", avg_recon_err, max_recon_err)

    # Singular value accuracy summary
    println()
    println("Singular Value Accuracy (vs reference):")
    println("-"^80)

    avg_sval_err = mean([r.sval_err for r in results])
    max_sval_err = maximum([r.sval_err for r in results])

    @printf("  torch_compat:  mean=%.3e, max=%.3e\n", avg_sval_err, max_sval_err)

    # Orthonormality summary
    println()
    println("Orthonormality Summary:")
    println("-"^80)

    max_orth_U = maximum([r.orth_U for r in results])
    max_orth_V = maximum([r.orth_V for r in results])

    @printf("  max ||U'U - I||: %.3e\n", max_orth_U)
    @printf("  max ||V'V - I||: %.3e\n", max_orth_V)

    println()
    println("="^80)

    # Return exit code based on pass/fail
    return all(r.passed for r in results) ? 0 : 1
end

end  # module


# Run tests if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit_code = TestTorchCompat.test()
    exit(exit_code)
end
