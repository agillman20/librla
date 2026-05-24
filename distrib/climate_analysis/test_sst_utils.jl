"""
test_sst_utils.jl - Utilities for SST/EOF climate analysis

Statistical tests and helper functions for EOF analysis of climate data.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Assisted by: Claude Code (Anthropic)
"""

module TestSstUtils

using Printf

export north_test, print_north_test

"""
    north_test(singular_values, n_samples; return_errors=false)

Apply North's rule of thumb to test EOF/SVD mode separability.

North et al. (1982) showed that eigenvalues have sampling error:
    δλ_i ≈ λ_i * sqrt(2/N)

where N is the number of independent samples. Two modes are considered
"effectively degenerate" if their eigenvalues overlap within error bars:
    λ_i - λ_{i+1} < δλ_i + δλ_{i+1}

For SVD, eigenvalues λ = σ² (squared singular values).

# Arguments
- `singular_values`: Singular values from SVD (σ, not σ²)
- `n_samples::Int`: Number of independent samples (typically number of time steps)
- `return_errors::Bool=false`: If true, also return the eigenvalue errors

# Returns
- `separable::Vector{Bool}`: separable[i] is true if mode i is well-separated from mode i+1
- `errors::Vector{Float64}` (only if return_errors=true): Sampling errors δλ

# Reference
North, G. R., T. L. Bell, R. F. Cahalan, and F. J. Moeng, 1982:
Sampling errors in the estimation of empirical orthogonal functions.
Mon. Wea. Rev., 110, 699-706.

# Examples
```julia
s = [10.0, 8.0, 7.9, 5.0, 3.0]  # singular values
sep = north_test(s, 100)
# sep = [true, false, true, true] - modes 2,3 are degenerate
```
"""
function north_test(singular_values::AbstractVector, n_samples::Int; return_errors::Bool=false)
    s = collect(singular_values)
    eigenvalues = s .^ 2

    # Sampling error: δλ_i = λ_i * sqrt(2/N)
    delta_lambda = eigenvalues .* sqrt(2.0 / n_samples)

    # Test separation: λ_i - λ_{i+1} > δλ_i + δλ_{i+1}
    gap = eigenvalues[1:end-1] .- eigenvalues[2:end]
    error_sum = delta_lambda[1:end-1] .+ delta_lambda[2:end]
    separable = gap .> error_sum

    if return_errors
        return separable, delta_lambda
    end
    return separable
end


"""
    print_north_test(singular_values, n_samples; n_modes=10)

Print North's rule of thumb results for EOF analysis.

# Arguments
- `singular_values`: Singular values from SVD
- `n_samples::Int`: Number of independent samples
- `n_modes::Int=10`: Number of modes to display
"""
function print_north_test(singular_values::AbstractVector, n_samples::Int; n_modes::Int=10)
    s = collect(singular_values)
    n_modes = min(n_modes, length(s) - 1)

    separable, errors = north_test(s, n_samples; return_errors=true)
    eigenvalues = s .^ 2
    total_var = sum(s .^ 2)
    var_pct = 100 .* eigenvalues ./ total_var

    @printf("\nNorth's Rule of Thumb (N=%d samples)\n", n_samples)
    println("  Mode   Variance %    λ ± δλ              Separated?")
    println("  " * "-"^55)

    for i in 1:n_modes
        status = separable[i] ? "Yes" : "NO (degenerate)"
        @printf("  EOF%2d   %6.2f%%    %.2e ± %.2e   %s\n",
                i, var_pct[i], eigenvalues[i], errors[i], status)
    end

    n_degen = sum(.!separable[1:n_modes])
    if n_degen > 0
        @printf("\n  Warning: %d mode pair(s) may be degenerate\n", n_degen)
    end
end

end  # module TestSstUtils
