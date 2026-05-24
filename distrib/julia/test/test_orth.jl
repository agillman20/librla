# test_orth.jl - Test librla orthonormal basis computation
#
# Tests orth_sketch (randomized orthonormal basis for column space):
# - Column space accuracy (how well Q spans A's column space)
# - Orthonormality of Q
# - diagR values (sorted column norms, conditioning indicator)
# - Runtime
#
# Usage:
#     julia test_orth.jl
#
# Author: Adrianna Gillman, Zydrunas Gimbutas
# SPDX-License-Identifier: MIT
# Version: 1.0.1
# Date: April 22, 2026
# Assisted by: Claude Code (Anthropic)

module TestOrth

using LinearAlgebra
using Printf
using Statistics
using Random

include(joinpath(@__DIR__, "..", "librla.jl"))
include(joinpath(@__DIR__, "test_utils.jl"))
using .TestUtils: make_mat

using .librla: orth_sketch

export test


mutable struct ComparisonResult
    name::String
    rtol_or_rank::Float64

    k::Int

    span_err::Float64
    orth_err::Float64
    diagR_ratio::Float64

    flag::Int

    t_sketch::Float64

    passed::Bool
end


function hilb_matrix(m::Int, n::Int)
    """Generate an mxn Hilbert matrix."""
    i = (1:m)
    j = (1:n)'
    return 1.0 ./ (i .+ j .- 1)
end


function run_test_case(A::Matrix{T}, rtol_or_rank::Float64, name::String) where T
    """Test orth_sketch on a single matrix."""

    println("\n", "="^70)
    println("Test: ", name)
    m, n = size(A)
    print("Matrix: ", m, "x", n)
    if eltype(A) <: Complex
        print(", complex")
    end
    println()
    println("Parameter: rtol_or_rank = ", rtol_or_rank)
    println("="^70)

    normA = norm(A)

    # -------------------------------------------------------------------------
    # orth_sketch (randomized)
    # -------------------------------------------------------------------------
    println("\n--- orth_sketch (randomized) ---")

    t_sketch = @elapsed Q, flag, diagR = orth_sketch(A, rtol_or_rank)

    k = size(Q, 2)

    # Column space error: how well Q spans A's column space
    QQtA = Q * (Q' * A)
    span_err = norm(A - QQtA) / normA

    # Orthonormality check
    if k > 0
        orth_err = norm(Q' * Q - I(k))
    else
        orth_err = 0.0
    end

    # diagR ratio (conditioning indicator)
    abs_diagR = abs.(diagR)
    if !isempty(abs_diagR) && abs_diagR[1] != 0
        diagR_ratio = abs_diagR[end] / abs_diagR[1]
    else
        diagR_ratio = 0.0
    end

    @printf("Rank:       k = %d\n", k)
    if flag == 0
        @printf("Flag:       %d (success)\n", flag)
    else
        @printf("Flag:       %d (early termination)\n", flag)
    end
    @printf("Span Err:   ||A - Q*Q'*A|| / ||A|| = %.3e\n", span_err)
    @printf("Orth Err:   ||Q'Q - I|| = %.3e\n", orth_err)
    if !isempty(abs_diagR)
        @printf("|diagR[1]|: %.3e\n", abs_diagR[1])
        @printf("|diagR[end]|:%.3e\n", abs_diagR[end])
    else
        println("|diagR[1]|: N/A")
        println("|diagR[end]|:N/A")
    end
    @printf("diagR ratio: %.3e\n", diagR_ratio)
    @printf("Time:       %.4f s\n", t_sketch)

    # -------------------------------------------------------------------------
    # Compare with full SVD (reference)
    # -------------------------------------------------------------------------
    println("\n--- Reference (full SVD truncated) ---")

    t_ref = @elapsed F = svd(A)

    U_ref = F.U

    # Truncate to same rank
    if k > 0
        U_k = U_ref[:, 1:k]
        UUtA = U_k * (U_k' * A)
        span_err_ref = norm(A - UUtA) / normA
    else
        span_err_ref = 1.0
    end

    @printf("Rank:       k = %d\n", k)
    @printf("Span Err:   ||A - U_k*U_k'*A|| / ||A|| = %.3e\n", span_err_ref)
    @printf("Time:       %.4f s (full SVD)\n", t_ref)

    # -------------------------------------------------------------------------
    # Summary comparison
    # -------------------------------------------------------------------------
    println("\n--- Summary ---")
    @printf("%-28s %-8s %-12s %-12s %-10s\n", "Method", "Rank", "Span Err", "Orth Err", "Time (s)")
    println("-"^75)
    @printf("%-28s %-8d %-12.3e %-12.3e %-10.4f\n", "orth_sketch (randomized)", k, span_err, orth_err, t_sketch)
    @printf("%-28s %-8d %-12.3e %-12s %-10.4f\n", "SVD (optimal)", k, span_err_ref, "(ref)", t_ref)

    # Quality comparison
    if span_err_ref > 0
        quality_ratio = span_err / span_err_ref
        @printf("\nQuality ratio: orth_sketch error / optimal error = %.2fx\n", quality_ratio)
    end

    # Speedup
    if t_sketch > 0
        speedup = t_ref / t_sketch
        if speedup > 1
            @printf("Speedup: orth_sketch is %.1fx faster than full SVD\n", speedup)
        else
            @printf("Speedup: full SVD is %.1fx faster\n", 1/speedup)
        end
    end

    # -------------------------------------------------------------------------
    # Determine if test passed
    # -------------------------------------------------------------------------
    # Orthonormality should be near machine precision (or 0 if k=0)
    # Early termination (flag=1) is OK - it's expected for some matrices
    # Key criterion: span error should be close to optimal (SVD reference)

    orth_ok = (k == 0) || (orth_err < 1e-10)

    # Quality threshold: randomized methods typically achieve within 8x of optimal
    # (slightly relaxed to account for randomness in ill-conditioned cases)
    quality_threshold = 8.0

    if rtol_or_rank < 1
        # Tolerance mode: span error should be within threshold of optimal
        if span_err_ref == 0
            quality_ok = span_err < 1e-10
        else
            quality_ok = (span_err / max(span_err_ref, 1e-15)) < quality_threshold
        end
        passed = quality_ok && orth_ok
    else
        # Rank mode: span error should be within threshold of optimal
        if span_err_ref == 0
            passed = orth_ok
        else
            passed = ((span_err / max(span_err_ref, 1e-15)) < quality_threshold) && orth_ok
        end
    end

    return ComparisonResult(
        name,
        rtol_or_rank,
        k,
        span_err,
        orth_err,
        diagR_ratio,
        flag,
        t_sketch,
        passed
    )
end


function print_summary(results::Vector{ComparisonResult})
    println()
    println("="^80)
    println("TEST SUMMARY - ", length(results), " tests completed")
    println("="^80)

    passed_count = sum([r.passed for r in results])
    total_count = length(results)
    pass_rate = 100.0 * passed_count / total_count

    println()
    @printf("Pass Rate: %d/%d (%.1f%%)\n", passed_count, total_count, pass_rate)

    if passed_count == total_count
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

    avg_time = mean([r.t_sketch for r in results])
    min_time = minimum([r.t_sketch for r in results])
    max_time = maximum([r.t_sketch for r in results])

    @printf("\north_sketch timing: mean=%.4fs, min=%.4fs, max=%.4fs\n", avg_time, min_time, max_time)

    # Accuracy summary
    println("\nColumn Space Error Summary (||A - Q*Q'*A|| / ||A||):")
    println("-"^80)
    avg_span_err = mean([r.span_err for r in results])
    max_span_err = maximum([r.span_err for r in results])
    @printf("  orth_sketch:    mean=%.3e, max=%.3e\n", avg_span_err, max_span_err)

    # Orthonormality summary
    println("\nOrthonormality Summary (||Q'Q - I||):")
    println("-"^80)
    max_orth_err = maximum([r.orth_err for r in results])
    @printf("  orth_sketch:    max=%.3e\n", max_orth_err)

    # Flag summary
    early_term_count = sum([r.flag == 1 for r in results])
    @printf("\nEarly termination flags: %d/%d\n", early_term_count, total_count)

    println()
    println("="^80)
end


function test()
    println()
    println("="^70)
    println("ORTH_SKETCH TESTS")
    println("Testing orthonormal basis computation via randomized sketching")
    println("="^70)
    println("\nEnvironment:")
    println("  Julia:      ", VERSION)
    println("="^70)

    # JIT warm-up
    println("\nJIT warm-up (compiling methods)...")
    A_warmup = randn(50, 30)
    try
        orth_sketch(A_warmup, 10.0)
        svd(A_warmup)
    catch
    end
    println("JIT warm-up complete.")

    results = ComparisonResult[]

    # -------------------------------------------------------------------------
    # BASIC TESTS
    # -------------------------------------------------------------------------
    println("\n\n", "="^70)
    println("BASIC TESTS")
    println("Testing fundamental tolerance and rank modes")
    println("="^70)

    Random.seed!(42)
    A1 = randn(500, 300)
    push!(results, run_test_case(A1, 20.0, "Random Matrix (well-conditioned)"))

    U = randn(400, 15)
    V = randn(250, 15)
    A2 = U * V' + 1e-10 * randn(400, 250)
    push!(results, run_test_case(A2, 1e-8, "Low-Rank Matrix (rank~15)"))

    A3 = hilb_matrix(2000, 1000)
    push!(results, run_test_case(A3, 15.0, "Hilbert Matrix (severely ill-conditioned)"))

    A4 = randn(300, 200) + im * randn(300, 200)
    push!(results, run_test_case(A4, 25.0, "Complex Matrix"))

    A5 = randn(400, 300)
    F5 = svd(A5)
    s5 = 1.0 ./ (1:300)
    A5 = F5.U * Diagonal(s5) * F5.Vt
    push!(results, run_test_case(A5, 1e-3, "Decaying Spectrum (1/k)"))

    # -------------------------------------------------------------------------
    # LARGE MATRIX TESTS (2x SCALE)
    # -------------------------------------------------------------------------
    println("\n\n", "="^70)
    println("LARGE MATRIX TESTS (2x SCALE)")
    println("Testing scaling behavior with matrices 2x larger than base")
    println("="^70)

    A6 = randn(1000, 600)
    push!(results, run_test_case(A6, 20.0, "Large Random Matrix (1000x600)"))

    U7 = randn(800, 15)
    V7 = randn(500, 15)
    A7 = U7 * V7' + 1e-10 * randn(800, 500)
    push!(results, run_test_case(A7, 1e-8, "Large Low-Rank (800x500, rank~15)"))

    A8 = hilb_matrix(4000, 2000)
    push!(results, run_test_case(A8, 15.0, "Large Hilbert Matrix (4000x2000)"))

    A9 = randn(600, 400) + im * randn(600, 400)
    push!(results, run_test_case(A9, 25.0, "Large Complex Matrix (600x400)"))

    A10 = randn(800, 600)
    F10 = svd(A10)
    s10 = 1.0 ./ (1:600)
    A10 = F10.U * Diagonal(s10) * F10.Vt
    push!(results, run_test_case(A10, 1e-3, "Large Decaying Spectrum (1/k, 800x600)"))

    # -------------------------------------------------------------------------
    # SLOW DECAYING SPECTRUM TESTS
    # -------------------------------------------------------------------------
    println("\n\n", "="^70)
    println("SLOW DECAYING SPECTRUM TESTS")
    println("Testing harder rank-deficient problems with slow decay")
    println("="^70)

    A11 = randn(400, 300)
    F11 = svd(A11)
    s11 = 1.0 ./ sqrt.(1:300)
    A11 = F11.U * Diagonal(s11) * F11.Vt
    push!(results, run_test_case(A11, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 400x300)"))

    A12 = randn(800, 600)
    F12 = svd(A12)
    s12 = 1.0 ./ sqrt.(1:600)
    A12 = F12.U * Diagonal(s12) * F12.Vt
    push!(results, run_test_case(A12, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 800x600)"))

    A13 = randn(400, 300)
    F13 = svd(A13)
    s13 = 1.0 ./ ((1:300) .^ 0.7)
    A13 = F13.U * Diagonal(s13) * F13.Vt
    push!(results, run_test_case(A13, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 400x300)"))

    A14 = randn(800, 600)
    F14 = svd(A14)
    s14 = 1.0 ./ ((1:600) .^ 0.7)
    A14 = F14.U * Diagonal(s14) * F14.Vt
    push!(results, run_test_case(A14, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 800x600)"))

    A15 = randn(400, 300)
    F15 = svd(A15)
    s15 = exp.(-(1:300) / 100.0)
    A15 = F15.U * Diagonal(s15) * F15.Vt
    push!(results, run_test_case(A15, 1e-3, "Slow Decay - Exponential (exp(-k/100), 400x300)"))

    A16 = randn(800, 600)
    F16 = svd(A16)
    s16 = exp.(-(1:600) / 150.0)
    A16 = F16.U * Diagonal(s16) * F16.Vt
    push!(results, run_test_case(A16, 1e-3, "Slow Decay - Exponential (exp(-k/150), 800x600)"))

    # -------------------------------------------------------------------------
    # STRUCTURED MATRICES FROM MAKE_MAT
    # -------------------------------------------------------------------------
    println("\n\n", "="^70)
    println("MAKE_MAT TESTS (Structured Matrices)")
    println("Testing matrices from \"Robust blockwise random pivoting\" paper")
    println("="^70)

    A22 = make_mat(500, 500, "gaussexp")
    push!(results, run_test_case(A22, 1e-3, "Gaussexp (Gaussian Exponential Decay, 500x500)"))

    A23 = make_mat(400, 400, "gmm")
    push!(results, run_test_case(A23, 1e-3, "GMM (Gaussian Mixture Model, 400x400)"))

    A24 = make_mat(300, 300, "snn")
    push!(results, run_test_case(A24, 1e-3, "SNN (Sparse Neural Network, 300x300)"))

    # Summary
    print_summary(results)

    # Final status
    passed_count = sum([r.passed for r in results])
    total_count = length(results)
    if passed_count < total_count
        println("\n[FAIL] ", total_count - passed_count, " tests failed")
        println("\nFailed tests:")
        for r in results
            if !r.passed
                println("  - ", r.name)
            end
        end
        return 1
    else
        println("\n[PASS] ALL TESTS PASSED!")
        return 0
    end
end

end # module TestOrth


if abspath(PROGRAM_FILE) == @__FILE__
    exit(TestOrth.test())
end
