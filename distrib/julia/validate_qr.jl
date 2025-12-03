# compare_qr.jl - Compare librla QR implementations
#
# Compares qr_sketch (randomized) vs qr with column pivoting:
# - Accuracy (reconstruction error)
# - Orthonormality of Q
# - Runtime
#
# Usage:
#     julia compare_qr.jl

using LinearAlgebra
using Printf
using Statistics
using Random

include("librla.jl")
include("make_mat.jl")

using .librla: qr_sketch


mutable struct ComparisonResult
    name::String
    rtol_or_rank::Float64

    k_sketch::Int
    k_ref::Int

    err_sketch::Float64
    err_ref::Float64

    orth_Q_sketch::Float64
    orth_Q_ref::Float64

    t_sketch::Float64
    t_ref::Float64

    passed::Bool
end


function hilb_matrix(m::Int, n::Int)
    """Generate an mxn Hilbert matrix."""
    i = (1:m)
    j = (1:n)'
    return 1.0 ./ (i .+ j .- 1)
end


function compare_on_matrix(A::Matrix{T}, rtol_or_rank::Float64, name::String) where T
    """Compare QR implementations on a single matrix."""

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
    # 1. qr_sketch (randomized)
    # -------------------------------------------------------------------------
    println("\n--- qr_sketch (randomized) ---")

    t_sketch = @elapsed Q_sketch, R_sketch, p_sketch = qr_sketch(A, rtol_or_rank)

    k_sketch = size(Q_sketch, 2)

    # Reconstruction error: A[:, p] = Q * R
    A_perm_sketch = A[:, p_sketch]
    err_sketch = norm(A_perm_sketch - Q_sketch * R_sketch) / normA

    # Orthonormality check
    orth_Q_sketch = norm(Q_sketch' * Q_sketch - I(k_sketch))

    # R diagonal (conditioning indicator)
    diag_R_sketch = abs.(diag(R_sketch[1:min(k_sketch,n), 1:min(k_sketch,n)]))
    if !isempty(diag_R_sketch) && diag_R_sketch[1] != 0
        r_ratio_sketch = diag_R_sketch[end] / diag_R_sketch[1]
    else
        r_ratio_sketch = 0.0
    end

    @printf("Rank:       k = %d\n", k_sketch)
    @printf("Error:      ||A[:,p] - Q*R|| / ||A|| = %.3e\n", err_sketch)
    @printf("Orth Q:     ||Q'Q - I|| = %.3e\n", orth_Q_sketch)
    @printf("R ratio:    R[k,k]/R[1,1] = %.3e\n", r_ratio_sketch)
    @printf("Time:       %.4f s\n", t_sketch)

    # -------------------------------------------------------------------------
    # 2. qr with column pivoting (deterministic)
    # -------------------------------------------------------------------------
    println("\n--- qr (pivoted, deterministic) ---")

    t_ref = @elapsed F = qr(A, ColumnNorm())

    Q_ref_full = Matrix(F.Q)
    R_ref_full = F.R
    p_ref = F.p

    # Determine reference rank (same as sketch for fair comparison)
    if rtol_or_rank >= 1
        k_ref = Int(floor(rtol_or_rank))
    else
        k_ref = k_sketch
    end

    # Truncate Q and R
    Q_ref = Q_ref_full[:, 1:k_ref]
    R_ref = R_ref_full[1:k_ref, :]

    # Reconstruction error
    A_perm_ref = A[:, p_ref]
    err_ref = norm(A_perm_ref - Q_ref * R_ref) / normA

    # Orthonormality check
    orth_Q_ref = norm(Q_ref' * Q_ref - I(k_ref))

    @printf("Rank:       k = %d\n", k_ref)
    @printf("Error:      ||A[:,p] - Q*R|| / ||A|| = %.3e\n", err_ref)
    @printf("Orth Q:     ||Q'Q - I|| = %.3e\n", orth_Q_ref)
    @printf("Time:       %.4f s (full QR)\n", t_ref)

    # -------------------------------------------------------------------------
    # Summary comparison
    # -------------------------------------------------------------------------
    println("\n--- Summary ---")
    @printf("%-28s %-8s %-12s %-12s %-10s\n", "Method", "Rank", "Recon Err", "Orth Err", "Time (s)")
    println("-"^75)
    @printf("%-28s %-8d %-12.3e %-12.3e %-10.4f\n", "qr_sketch (randomized)", k_sketch, err_sketch, orth_Q_sketch, t_sketch)
    @printf("%-28s %-8d %-12.3e %-12.3e %-10.4f\n", "qr (deterministic)", k_ref, err_ref, orth_Q_ref, t_ref)

    # Highlight fastest method
    methods = ["qr_sketch", "qr"]
    times = [t_sketch, t_ref]
    fastest_idx = argmin(times)
    @printf("\nFastest method: %s (%.4fs)\n", methods[fastest_idx], times[fastest_idx])

    # Speedup
    if t_sketch > 0
        speedup = t_ref / t_sketch
        if speedup > 1
            @printf("Speedup: qr_sketch is %.1fx faster\n", speedup)
        else
            @printf("Speedup: qr is %.1fx faster\n", 1/speedup)
        end
    end

    # -------------------------------------------------------------------------
    # Determine if test passed
    # -------------------------------------------------------------------------
    if rtol_or_rank < 1
        tol_threshold = rtol_or_rank * 100
        passed = err_sketch < min(0.1, tol_threshold) && orth_Q_sketch < 1e-10
    else
        # Rank mode: sketch error should be within 2x of reference
        error_ratio_ok = (err_ref == 0) || (err_sketch / max(err_ref, 1e-15) < 2.0)
        passed = error_ratio_ok && orth_Q_sketch < 1e-10
    end

    return ComparisonResult(
        name,
        rtol_or_rank,
        k_sketch, k_ref,
        err_sketch, err_ref,
        orth_Q_sketch, orth_Q_ref,
        t_sketch, t_ref,
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

    avg_time_sketch = mean([r.t_sketch for r in results])
    avg_time_ref = mean([r.t_ref for r in results])

    println()
    @printf("%-28s %-12s %-15s\n", "Method", "Avg Time", "vs QR")
    println("-"^80)
    @printf("%-28s %8.4fs    %6.1fx\n", "qr_sketch (randomized)", avg_time_sketch, avg_time_ref/avg_time_sketch)
    @printf("%-28s %8.4fs    %6.1fx -\n", "qr (deterministic)", avg_time_ref, 1.0)

    # Accuracy summary
    println("\nReconstruction Error Summary:")
    println("-"^80)
    avg_err_sketch = mean([r.err_sketch for r in results])
    max_err_sketch = maximum([r.err_sketch for r in results])
    @printf("  qr_sketch:    mean=%.3e, max=%.3e\n", avg_err_sketch, max_err_sketch)

    # Orthonormality summary
    println("\nOrthonormality Summary:")
    println("-"^80)
    max_orth_Q_sketch = maximum([r.orth_Q_sketch for r in results])
    max_orth_Q_ref = maximum([r.orth_Q_ref for r in results])
    @printf("  qr_sketch:    max ||Q'Q - I|| = %.3e\n", max_orth_Q_sketch)
    @printf("  qr:           max ||Q'Q - I|| = %.3e\n", max_orth_Q_ref)

    println()
    println("="^80)
end


function main()
    println()
    println("="^70)
    println("QR COMPARISON")
    println("qr_sketch (randomized) vs qr (deterministic)")
    println("="^70)
    println("\nEnvironment:")
    println("  Julia:      ", VERSION)
    println("="^70)

    # JIT warm-up
    println("\nJIT warm-up (compiling methods)...")
    A_warmup = randn(50, 30)
    try
        qr_sketch(A_warmup, 10.0)
        qr(A_warmup, ColumnNorm())
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
    push!(results, compare_on_matrix(A1, 20.0, "Random Matrix (well-conditioned)"))

    U = randn(400, 15)
    V = randn(250, 15)
    A2 = U * V' + 1e-10 * randn(400, 250)
    push!(results, compare_on_matrix(A2, 1e-8, "Low-Rank Matrix (rank~15)"))

    A3 = hilb_matrix(2000, 1000)
    push!(results, compare_on_matrix(A3, 15.0, "Hilbert Matrix (severely ill-conditioned)"))

    A4 = randn(300, 200) + im * randn(300, 200)
    push!(results, compare_on_matrix(A4, 25.0, "Complex Matrix"))

    A5 = randn(400, 300)
    F5 = svd(A5)
    s5 = 1.0 ./ (1:300)
    A5 = F5.U * Diagonal(s5) * F5.Vt
    push!(results, compare_on_matrix(A5, 1e-3, "Decaying Spectrum (1/k)"))

    # -------------------------------------------------------------------------
    # LARGE MATRIX TESTS (2x SCALE)
    # -------------------------------------------------------------------------
    println("\n\n", "="^70)
    println("LARGE MATRIX TESTS (2x SCALE)")
    println("Testing scaling behavior with matrices 2x larger than base")
    println("="^70)

    A6 = randn(1000, 600)
    push!(results, compare_on_matrix(A6, 20.0, "Large Random Matrix (1000x600)"))

    U7 = randn(800, 15)
    V7 = randn(500, 15)
    A7 = U7 * V7' + 1e-10 * randn(800, 500)
    push!(results, compare_on_matrix(A7, 1e-8, "Large Low-Rank (800x500, rank~15)"))

    A8 = hilb_matrix(4000, 2000)
    push!(results, compare_on_matrix(A8, 15.0, "Large Hilbert Matrix (4000x2000)"))

    A9 = randn(600, 400) + im * randn(600, 400)
    push!(results, compare_on_matrix(A9, 25.0, "Large Complex Matrix (600x400)"))

    A10 = randn(800, 600)
    F10 = svd(A10)
    s10 = 1.0 ./ (1:600)
    A10 = F10.U * Diagonal(s10) * F10.Vt
    push!(results, compare_on_matrix(A10, 1e-3, "Large Decaying Spectrum (1/k, 800x600)"))

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
    push!(results, compare_on_matrix(A11, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 400x300)"))

    A12 = randn(800, 600)
    F12 = svd(A12)
    s12 = 1.0 ./ sqrt.(1:600)
    A12 = F12.U * Diagonal(s12) * F12.Vt
    push!(results, compare_on_matrix(A12, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 800x600)"))

    A13 = randn(400, 300)
    F13 = svd(A13)
    s13 = 1.0 ./ ((1:300) .^ 0.7)
    A13 = F13.U * Diagonal(s13) * F13.Vt
    push!(results, compare_on_matrix(A13, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 400x300)"))

    A14 = randn(800, 600)
    F14 = svd(A14)
    s14 = 1.0 ./ ((1:600) .^ 0.7)
    A14 = F14.U * Diagonal(s14) * F14.Vt
    push!(results, compare_on_matrix(A14, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 800x600)"))

    A15 = randn(400, 300)
    F15 = svd(A15)
    s15 = exp.(-(1:300) / 100.0)
    A15 = F15.U * Diagonal(s15) * F15.Vt
    push!(results, compare_on_matrix(A15, 1e-3, "Slow Decay - Exponential (exp(-k/100), 400x300)"))

    A16 = randn(800, 600)
    F16 = svd(A16)
    s16 = exp.(-(1:600) / 150.0)
    A16 = F16.U * Diagonal(s16) * F16.Vt
    push!(results, compare_on_matrix(A16, 1e-3, "Slow Decay - Exponential (exp(-k/150), 800x600)"))

    # -------------------------------------------------------------------------
    # EXTRA LARGE MATRIX TESTS (4x SCALE)
    # -------------------------------------------------------------------------
    println("\n\n", "="^70)
    println("EXTRA LARGE MATRIX TESTS (4x SCALE)")
    println("Testing scaling behavior with matrices 4x larger than base")
    println("="^70)

    A17 = randn(2000, 1200)
    push!(results, compare_on_matrix(A17, 20.0, "XL Random Matrix (2000x1200)"))

    U18 = randn(1600, 15)
    V18 = randn(1000, 15)
    A18 = U18 * V18' + 1e-10 * randn(1600, 1000)
    push!(results, compare_on_matrix(A18, 1e-8, "XL Low-Rank Matrix (1600x1000, rank~15)"))

    A19 = hilb_matrix(8000, 4000)
    push!(results, compare_on_matrix(A19, 15.0, "XL Hilbert Matrix (8000x4000)"))

    A20 = randn(1200, 800) + im * randn(1200, 800)
    push!(results, compare_on_matrix(A20, 25.0, "XL Complex Matrix (1200x800)"))

    A21 = randn(1600, 1200)
    F21 = svd(A21)
    s21 = 1.0 ./ (1:1200)
    A21 = F21.U * Diagonal(s21) * F21.Vt
    push!(results, compare_on_matrix(A21, 1e-3, "XL Decaying Spectrum (1/k, 1600x1200)"))

    # -------------------------------------------------------------------------
    # STRUCTURED MATRICES FROM MAKE_MAT
    # -------------------------------------------------------------------------
    println("\n\n", "="^70)
    println("MAKE_MAT TESTS (Structured Matrices)")
    println("Testing matrices from \"Robust blockwise random pivoting\" paper")
    println("="^70)

    A22 = make_mat(500, 500, "gaussexp")
    push!(results, compare_on_matrix(A22, 1e-3, "Gaussexp (Gaussian Exponential Decay, 500x500)"))

    A23 = make_mat(400, 400, "gmm")
    push!(results, compare_on_matrix(A23, 1e-3, "GMM (Gaussian Mixture Model, 400x400)"))

    A24 = make_mat(300, 300, "snn")
    push!(results, compare_on_matrix(A24, 1e-3, "SNN (Sparse Neural Network, 300x300)"))

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


if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
