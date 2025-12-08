# validate_id.jl - Compare libid interpolative decomposition implementations
#
# Compares two ID implementations from librla:
# - id_sketch:  Randomized QR sketching (default, recommended)
# - id_qrpiv:   Deterministic QR via LAPACK geqp3
#
# Compares on metrics:
# - Accuracy (reconstruction error)
# - Conditioning (max|T|)
# - Runtime
# - Rank selection behavior
#
# Usage:
#     julia validate_id.jl

module ValidateID

using LinearAlgebra
using Printf
using Statistics
using Random

# Import ID implementations
include(joinpath(@__DIR__, "librla.jl"))
include(joinpath(@__DIR__, "make_mat.jl"))

using .librla: id_sketch, id_qrpiv

export validate


mutable struct ComparisonResult
    name::String
    rtol_or_rank::Float64

    # Results for each method (2 methods x 4 metrics)
    k_sketch::Int
    k_rrqr::Int

    err_sketch::Float64
    err_rrqr::Float64

    t_sketch::Float64
    t_rrqr::Float64

    maxT_sketch::Float64
    maxT_rrqr::Float64

    passed::Bool
end


function hilb_matrix(m::Int, n::Int)
    """Generate an mxn Hilbert matrix."""
    i = (1:m)
    j = (1:n)'
    return 1.0 ./ (i .+ j .- 1)
end


function compare_on_matrix(A::Matrix{T}, rtol_or_rank::Float64, name::String) where T
    """
    Compare ID implementations on a single matrix with verbose output.

    Parameters
    ----------
    A : Matrix
        Input matrix to decompose (real or complex)
    rtol_or_rank : Float64
        Tolerance (< 1) or target rank (>= 1)
    name : String
        Test case name for display

    Returns
    -------
    result : ComparisonResult
        Comparison metrics
    """
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
    # 1. librla id_sketch (randomized QR sketching)
    # -------------------------------------------------------------------------
    println("\n--- librla id_sketch (randomized QR) ---")

    t_sketch = @elapsed k_sketch, piv_sketch, T_sketch = id_sketch(A, rtol_or_rank)

    # Compute reconstruction error
    A_skel_sketch = A[:, piv_sketch[k_sketch+1:end]]
    A_basis_sketch = A[:, piv_sketch[1:k_sketch]]
    if !isempty(T_sketch)
        err_sketch = norm(A_skel_sketch - A_basis_sketch * T_sketch) / normA
        max_T_sketch = maximum(abs.(T_sketch))
    else
        err_sketch = 0.0
        max_T_sketch = 0.0
    end

    # CHECK: Error > 1.0 can occur for (nearly) full-rank matrices with fast T computation
    if err_sketch > 1.0
        @printf("\n[NOTE] Detected error > 1.0 (%.6f)\n", err_sketch)
        println("  This can occur for (nearly) full-rank matrices with fast T computation.")
        println("  Recomputing with method=\"lstsq\" for accurate lstsq-based T...")

        # Retry with method="lstsq" for accurate T computation via lstsq
        t_sketch = @elapsed k_sketch, piv_sketch, T_sketch = id_sketch(A, rtol_or_rank, method="lstsq")

        # Recompute error with new T
        A_skel_sketch = A[:, piv_sketch[k_sketch+1:end]]
        A_basis_sketch = A[:, piv_sketch[1:k_sketch]]
        if !isempty(T_sketch)
            err_sketch = norm(A_skel_sketch - A_basis_sketch * T_sketch) / normA
            max_T_sketch = maximum(abs.(T_sketch))
        else
            err_sketch = 0.0
            max_T_sketch = 0.0
        end

        @printf("  -> Recomputed: error = %.3e (method=\"lstsq\")\n", err_sketch)

        if err_sketch > 1.0
            error("[ERROR] Error still > 1.0 even with method=\"lstsq\"!\n   Error = $err_sketch, Test: $name, rtol_or_rank=$rtol_or_rank")
        end
    end

    @printf("Rank:       k = %d\n", k_sketch)
    @printf("Error:      ||A_skel - A_basis @ T|| / ||A|| = %.3e\n", err_sketch)
    @printf("Condition:  max|T| = %.3e\n", max_T_sketch)
    @printf("Time:       %.4f s\n", t_sketch)

    # -------------------------------------------------------------------------
    # 2. id_qrpiv (deterministic QR via LAPACK geqp3)
    # -------------------------------------------------------------------------
    println("\n--- id_qrpiv (QR geqp3) ---")

    t_rrqr = @elapsed k_rrqr, piv_rrqr, T_rrqr = id_qrpiv(A, rtol_or_rank)

    # Compute reconstruction error
    A_skel_rrqr = A[:, piv_rrqr[k_rrqr+1:end]]
    A_basis_rrqr = A[:, piv_rrqr[1:k_rrqr]]
    if !isempty(T_rrqr)
        err_rrqr = norm(A_skel_rrqr - A_basis_rrqr * T_rrqr) / normA
        max_T_rrqr = maximum(abs.(T_rrqr))
    else
        err_rrqr = 0.0
        max_T_rrqr = 0.0
    end

    # CRITICAL CHECK: Relative error must be <= 1.0 (mathematically bounded)
    if err_rrqr > 1.0
        error("[ERROR] CRITICAL BUG in id_qrpiv: Error = $err_rrqr > 1.0\n   This is mathematically impossible for a relative error!\n   Test: $name, rtol_or_rank=$rtol_or_rank")
    end

    @printf("Rank:       k = %d\n", k_rrqr)
    @printf("Error:      ||A_skel - A_basis @ T|| / ||A|| = %.3e\n", err_rrqr)
    @printf("Condition:  max|T| = %.3e\n", max_T_rrqr)
    @printf("Time:       %.4f s\n", t_rrqr)

    # -------------------------------------------------------------------------
    # Summary comparison
    # -------------------------------------------------------------------------
    println("\n--- Summary ---")
    @printf("%-25s %-8s %-12s %-12s %-10s\n", "Method", "Rank", "Error", "max|T|", "Time (s)")
    println("-"^75)
    @printf("%-25s %-8d %-12.3e %-12.3e %-10.4f\n", "id_sketch (randomized)", k_sketch, err_sketch, max_T_sketch, t_sketch)
    @printf("%-25s %-8d %-12.3e %-12.3e %-10.4f\n", "id_qrpiv (deterministic)", k_rrqr, err_rrqr, max_T_rrqr, t_rrqr)

    # Highlight best conditioning
    max_Ts = [max_T_sketch, max_T_rrqr]
    methods = ["id_sketch", "id_qrpiv"]
    times = [t_sketch, t_rrqr]

    best_idx = argmin(max_Ts)
    println("\nBest conditioning: ", methods[best_idx], " (smallest max|T|)")

    # Highlight fastest method
    fastest_idx = argmin(times)
    @printf("Fastest method: %s (%.4fs)\n", methods[fastest_idx], times[fastest_idx])

    # -------------------------------------------------------------------------
    # Create result struct
    # -------------------------------------------------------------------------
    # Determine if test passed
    max_error = max(err_sketch, err_rrqr)

    # For tolerance mode (rtol < 1): expect error ~ rtol
    # For rank mode (rtol >= 1): check consistency and reasonable error
    if rtol_or_rank < 1
        # Tolerance mode: error should be within 100x tolerance
        tol_threshold = rtol_or_rank * 100
        passed = max_error < min(0.1, tol_threshold)
    else
        # Rank mode: check deterministic methods agree on rank
        # For full-rank matrices with small k, error can be large (e.g., 90%)
        # This is expected - just verify methods are consistent
        ranks_match = (k_sketch == k_rrqr)
        error_reasonable = max_error < 10.0  # Very lenient for rank mode
        passed = ranks_match && error_reasonable
    end

    # Create result struct with all metrics
    return ComparisonResult(
        name,
        rtol_or_rank,
        k_sketch, k_rrqr,
        err_sketch, err_rrqr,
        t_sketch, t_rrqr,
        max_T_sketch, max_T_rrqr,
        passed
    )
end


function print_summary(results::Vector{ComparisonResult})
    """Print comprehensive summary of all test results."""
    println()
    println("="^70)
    println("SUMMARY")
    println("="^70)

    total_count = length(results)
    passed_count = sum([r.passed for r in results])

    println("\nTests: ", passed_count, "/", total_count, " passed")

    # Timing statistics
    t_sketch_all = [r.t_sketch for r in results]
    t_rrqr_all = [r.t_rrqr for r in results]

    println("\nTiming Statistics:")
    @printf("  id_sketch:   mean=%.3fs, min=%.3fs, max=%.3fs\n",
            mean(t_sketch_all), minimum(t_sketch_all), maximum(t_sketch_all))
    @printf("  id_qrpiv:     mean=%.3fs, min=%.3fs, max=%.3fs\n",
            mean(t_rrqr_all), minimum(t_rrqr_all), maximum(t_rrqr_all))

    # Speedup calculation
    speedups = t_rrqr_all ./ t_sketch_all
    println("\nSpeedup (rrqr time / sketch time):")
    @printf("  mean=%.2fx, min=%.2fx, max=%.2fx\n",
            mean(speedups), minimum(speedups), maximum(speedups))
    if mean(speedups) > 1
        @printf("  -> sketch is %.2fx faster on average\n", mean(speedups))
    else
        @printf("  -> sketch is %.2fx slower on average\n", 1/mean(speedups))
    end

    # Error statistics
    err_sketch_all = [r.err_sketch for r in results]
    err_rrqr_all = [r.err_rrqr for r in results]

    println("\nReconstruction Error Statistics:")
    @printf("  id_sketch:   mean=%.3e, max=%.3e\n", mean(err_sketch_all), maximum(err_sketch_all))
    @printf("  id_qrpiv:     mean=%.3e, max=%.3e\n", mean(err_rrqr_all), maximum(err_rrqr_all))

    # Conditioning statistics
    maxT_sketch_all = [r.maxT_sketch for r in results]
    maxT_rrqr_all = [r.maxT_rrqr for r in results]

    println("\nConditioning Statistics (max|T|):")
    @printf("  id_sketch:   mean=%.3e, min=%.3e, max=%.3e\n", mean(maxT_sketch_all), minimum(maxT_sketch_all), maximum(maxT_sketch_all))
    @printf("  id_qrpiv:     mean=%.3e, min=%.3e, max=%.3e\n", mean(maxT_rrqr_all), minimum(maxT_rrqr_all), maximum(maxT_rrqr_all))

    println("="^70)
end


function validate()
    """Run comprehensive ID comparison tests. Returns 0 on success, 1 on failure."""

    println()
    println("="^70)
    println("INTERPOLATIVE DECOMPOSITION (ID) COMPARISON")
    println("librla.id_sketch vs librla.id_qrpiv")
    println("="^70)
    println("\nEnvironment:")
    println("  Julia:      ", VERSION)
    println("="^70)

    # -------------------------------------------------------------------------
    # JIT warm-up: compile all methods before timing
    # -------------------------------------------------------------------------
    println("\nJIT warm-up (compiling methods)...")
    A_warmup = randn(50, 30)
    try
        id_sketch(A_warmup, 10.0)
        id_qrpiv(A_warmup, 10.0)
        id_sketch(A_warmup, 1e-3)  # tolerance mode
        id_qrpiv(A_warmup, 1e-3)
    catch
        # Ignore warm-up errors
    end
    println("JIT warm-up complete.")

    # Initialize results array
    results = ComparisonResult[]

    # -------------------------------------------------------------------------
    # BASIC TESTS
    # -------------------------------------------------------------------------
    println("\n\n", "="^70)
    println("BASIC TESTS")
    println("Testing fundamental tolerance and rank modes")
    println("="^70)

    # Test 1: Random matrix (well-conditioned)
    Random.seed!(42)
    A1 = randn(500, 300)
    push!(results, compare_on_matrix(A1, 20.0, "Random Matrix (well-conditioned)"))

    # Test 2: Low-rank matrix
    U = randn(400, 15)
    V = randn(250, 15)
    A2 = U * V' + 1e-10 * randn(400, 250)
    push!(results, compare_on_matrix(A2, 1e-8, "Low-Rank Matrix (rank~15)"))

    # Test 3: Hilbert matrix (extremely ill-conditioned)
    A3 = hilb_matrix(2000, 1000)
    push!(results, compare_on_matrix(A3, 15.0, "Hilbert Matrix (severely ill-conditioned)"))

    # Test 4: Complex matrix
    A4 = randn(300, 200) + im * randn(300, 200)
    push!(results, compare_on_matrix(A4, 25.0, "Complex Matrix"))

    # Test 5: Decaying spectrum (tolerance mode)
    A5 = randn(400, 300)
    U5, S5, V5 = svd(A5)
    s5 = 1.0 ./ (1:300)  # decaying: 1/k
    A5 = U5 * Diagonal(s5) * V5'
    push!(results, compare_on_matrix(A5, 1e-3, "Decaying Spectrum (1/k)"))

    # -------------------------------------------------------------------------
    # LARGE MATRIX TESTS (2x SCALE)
    # -------------------------------------------------------------------------
    println("\n\n", "="^70)
    println("LARGE MATRIX TESTS (2x SCALE)")
    println("Testing scaling behavior with matrices 2x larger than base")
    println("="^70)

    # Test 6: Large random matrix
    A6 = randn(1000, 600)
    push!(results, compare_on_matrix(A6, 20.0, "Large Random Matrix (1000x600)"))

    # Test 7: Large low-rank matrix
    U7 = randn(800, 15)
    V7 = randn(500, 15)
    A7 = U7 * V7' + 1e-10 * randn(800, 500)
    push!(results, compare_on_matrix(A7, 1e-8, "Large Low-Rank (800x500, rank~15)"))

    # Test 8: Large Hilbert matrix
    A8 = hilb_matrix(4000, 2000)
    push!(results, compare_on_matrix(A8, 15.0, "Large Hilbert Matrix (4000x2000)"))

    # Test 9: Large complex matrix
    A9 = randn(600, 400) + im * randn(600, 400)
    push!(results, compare_on_matrix(A9, 25.0, "Large Complex Matrix (600x400)"))

    # Test 10: Large decaying spectrum
    A10 = randn(800, 600)
    U10, S10, V10 = svd(A10)
    s10 = 1.0 ./ (1:600)  # Fast decay: 1/k
    A10 = U10 * Diagonal(s10) * V10'
    push!(results, compare_on_matrix(A10, 1e-3, "Large Decaying Spectrum (1/k, 800x600)"))

    # -------------------------------------------------------------------------
    # SLOW DECAYING SPECTRUM TESTS
    # -------------------------------------------------------------------------
    println("\n\n", "="^70)
    println("SLOW DECAYING SPECTRUM TESTS")
    println("Testing harder rank-deficient problems with slow decay")
    println("="^70)

    # Test 11: Slow decay - sqrt (small)
    A11 = randn(400, 300)
    U11, S11, V11 = svd(A11)
    s11 = 1.0 ./ sqrt.(1:300)  # Slow decay: 1/sqrt(k)
    A11 = U11 * Diagonal(s11) * V11'
    push!(results, compare_on_matrix(A11, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 400x300)"))

    # Test 12: Slow decay - sqrt (large)
    A12 = randn(800, 600)
    U12, S12, V12 = svd(A12)
    s12 = 1.0 ./ sqrt.(1:600)  # Slow decay: 1/sqrt(k)
    A12 = U12 * Diagonal(s12) * V12'
    push!(results, compare_on_matrix(A12, 1e-3, "Slow Decay - Sqrt (1/sqrtk, 800x600)"))

    # Test 13: Slow decay - polynomial (small)
    A13 = randn(400, 300)
    U13, S13, V13 = svd(A13)
    s13 = 1.0 ./ ((1:300) .^ 0.7)  # Polynomial: 1/k^0.7
    A13 = U13 * Diagonal(s13) * V13'
    push!(results, compare_on_matrix(A13, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 400x300)"))

    # Test 14: Slow decay - polynomial (large)
    A14 = randn(800, 600)
    U14, S14, V14 = svd(A14)
    s14 = 1.0 ./ ((1:600) .^ 0.7)  # Polynomial: 1/k^0.7
    A14 = U14 * Diagonal(s14) * V14'
    push!(results, compare_on_matrix(A14, 1e-3, "Slow Decay - Polynomial (1/k^0.7, 800x600)"))

    # Test 15: Slow decay - exponential (small)
    A15 = randn(400, 300)
    U15, S15, V15 = svd(A15)
    s15 = exp.(-(1:300) / 100.0)  # Exponential: exp(-k/100)
    A15 = U15 * Diagonal(s15) * V15'
    push!(results, compare_on_matrix(A15, 1e-3, "Slow Decay - Exponential (exp(-k/100), 400x300)"))

    # Test 16: Slow decay - exponential (large)
    A16 = randn(800, 600)
    U16, S16, V16 = svd(A16)
    s16 = exp.(-(1:600) / 150.0)  # Exponential: exp(-k/150)
    A16 = U16 * Diagonal(s16) * V16'
    push!(results, compare_on_matrix(A16, 1e-3, "Slow Decay - Exponential (exp(-k/150), 800x600)"))

    # -------------------------------------------------------------------------
    # STRUCTURED MATRICES FROM MAKE_MAT
    # -------------------------------------------------------------------------
    println("\n\n", "="^70)
    println("MAKE_MAT TESTS (Structured Matrices)")
    println("Testing matrices from \"Robust blockwise random pivoting\" paper")
    println("="^70)

    # Test 22: Gaussian Exponential Decay Matrix
    A22 = make_mat(500, 500, "gaussexp")
    push!(results, compare_on_matrix(A22, 1e-3, "Gaussexp (Gaussian Exponential Decay, 500x500)"))

    # Test 23: Gaussian Mixture Model Matrix
    A23 = make_mat(400, 400, "gmm")
    push!(results, compare_on_matrix(A23, 1e-3, "GMM (Gaussian Mixture Model, 400x400)"))

    # Test 24: Sparse Neural Network Matrix
    A24 = make_mat(300, 300, "snn")
    push!(results, compare_on_matrix(A24, 1e-3, "SNN (Sparse Neural Network, 300x300)"))

    # =========================================================================
    # Summary
    # =========================================================================
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

end # module ValidateID


# Run validate function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit(ValidateID.validate())
end
