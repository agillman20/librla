"""
demo_utils.jl - Shared utilities for librla demos

Matrix generators and helper functions used across all demos.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: NIST-PD
Assisted by: Claude Code (Anthropic)
"""

module demo_utils

using LinearAlgebra
using Printf
using Random

export hilbert, kahan, lowrank, random_matrix
export id_error, svd_error
export print_header, print_subheader

#=============================================================================
# Matrix Generators
=============================================================================#

"""
    hilbert(m, n=m) -> Matrix{Float64}

Generate m x n Hilbert matrix.

The Hilbert matrix is severely ill-conditioned, with entries H[i,j] = 1/(i+j-1).
Useful for testing numerical stability.
"""
function hilbert(m::Int, n::Int=m)
    i = reshape(1:m, m, 1)
    j = reshape(1:n, 1, n)
    return 1.0 ./ (i .+ j .- 1)
end


"""
    kahan(m, n=m, theta=1.2, pert=25.0) -> Matrix{Float64}

Generate m x n Kahan matrix.

Upper triangular matrix with exponentially decaying rows.
Classic test for QR factorization algorithms.
"""
function kahan(m::Int, n::Int=m; theta::Float64=1.2, pert::Float64=25.0)
    s = sin(theta)
    c = cos(theta)
    ep = eps(Float64)
    r = min(m, n)

    K = zeros(m, n)

    # Set diagonal
    for i = 1:r
        K[i, i] = 1.0
    end

    # Set upper triangular part
    for i = 1:m
        for j = i+1:n
            K[i, j] = -c
        end
    end

    # Scale rows by s^(i-1)
    for i = 1:m
        K[i, :] .*= s^(i-1)
    end

    # Add diagonal perturbation
    for i = 1:r
        K[i, i] += pert * ep * (r - i + 1)
    end

    return K
end


"""
    lowrank(m, n, k; decay="exponential", gap=100.0) -> (A, s)

Generate m x n matrix with controlled rank-k structure.

Creates a matrix where the first k singular values are well-separated
from the remaining ones. Returns the matrix A and true singular values s.
"""
function lowrank(m::Int, n::Int, k::Int; decay::String="exponential", gap::Float64=100.0)
    r = min(m, n)

    if decay == "exponential"
        s = vcat(10.0 .^ range(0, -2, length=k),
                 10.0 .^ range(-2, -10, length=r-k) ./ gap)
    elseif decay == "polynomial"
        s = vcat(1.0 ./ (1:k).^2,
                 1.0 ./ ((k+1):r).^2 ./ gap)
    elseif decay == "step"
        s = vcat(ones(k), ones(r-k) ./ gap)
    else
        error("Unknown decay type: $decay")
    end

    U = Matrix(qr(randn(m, r)).Q)
    V = Matrix(qr(randn(n, r)).Q)

    A = U * Diagonal(s) * V'
    return A, s
end


"""
    random_matrix(m, n; seed=nothing) -> Matrix{Float64}

Generate m x n random Gaussian matrix.
"""
function random_matrix(m::Int, n::Int; seed::Union{Int,Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    return randn(m, n)
end


#=============================================================================
# Error Computation
=============================================================================#

"""
    id_error(A, k, piv, T) -> Float64

Compute relative ID reconstruction error.

The ID approximation is: A[:, piv[k+1:end]] ≈ A[:, piv[1:k]] * T
"""
function id_error(A::AbstractMatrix, k::Int, piv::Vector{Int}, T::AbstractMatrix)
    A_basis = A[:, piv[1:k]]
    A_skel = A[:, piv[k+1:end]]
    return norm(A_skel - A_basis * T) / norm(A)
end


"""
    svd_error(A, U, s, Vt) -> Float64

Compute relative SVD reconstruction error.

Note: Julia convention - Vt is conjugate-transposed (like Python's Vh).
"""
function svd_error(A::AbstractMatrix, U::AbstractMatrix, s::AbstractVector, Vt::AbstractMatrix)
    A_approx = U * Diagonal(s) * Vt
    return norm(A - A_approx) / norm(A)
end


#=============================================================================
# Display Helpers
=============================================================================#

"""Print formatted section header."""
function print_header(title::String)
    println("="^70)
    println(title)
    println("="^70)
end

"""Print formatted subsection header."""
function print_subheader(title::String)
    println()
    println(title)
    println("-"^70)
end

end  # module demo_utils
