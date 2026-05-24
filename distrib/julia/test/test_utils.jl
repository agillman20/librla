"""
test_utils.jl - Shared utilities for librla tests

Matrix generators and helper functions used across all test files.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Assisted by: Claude Code (Anthropic)
"""

module TestUtils

using LinearAlgebra
using SparseArrays
using Printf
using Random

export make_mat, Matrix_GMM, Matrix_Gaussian_exp, Matrix_SNN
export lowrank, random_matrix
export id_error, svd_error
export print_header, print_subheader

#=============================================================================
# Matrix Generators (from make_mat)
=============================================================================#

"""
    make_mat(m, n, flag_type)

Generate test matrices for ID benchmarking.

# Arguments
- `m::Int`: Number of rows
- `n::Int`: Number of columns
- `flag_type::String`: Matrix type: "cifar", "mnist", "gaussexp", "gmm", "snn"

# Returns
- `X::Matrix`: Generated matrix (normalized by column)

# Examples
```julia
X = make_mat(500, 300, "gaussexp")
X = make_mat(500, 300, "gmm")
X = make_mat(500, 500, "snn")
```
"""
function make_mat(m::Int, n::Int, flag_type::String)
    if flag_type == "cifar"
        if n > m
            error("columns must be less than or equal to rows for CIFAR")
        end
        if m > 3072
            error("too many rows! CIFAR-10 has 3072 features (32x32x3)")
        end
        error("CIFAR loading not implemented. Requires MAT.jl package and exampledata/cifar10.mat file.")

    elseif flag_type == "mnist"
        if n > m
            error("columns must be less than or equal to rows for MNIST")
        end
        if m > 784
            error("too many rows! MNIST has 784 features (28x28)")
        end
        error("MNIST loading not implemented. Requires MAT.jl package and exampledata/mnist_mat.mat file.")

    elseif flag_type == "gaussexp"
        X = Matrix_Gaussian_exp(m)

    elseif flag_type == "gmm"
        X = Matrix_GMM(n, m)

    elseif flag_type == "snn"
        X = Matrix_SNN(n)

    else
        error("Unknown flag_type: $flag_type. Valid types: cifar, mnist, gaussexp, gmm, snn")
    end

    # Normalize by column
    X = X ./ sqrt.(sum(abs2.(X), dims=1))

    return X
end


"""
    Matrix_GMM(n, d)

Generate Gaussian Mixture Model matrix.

Creates a matrix with k=100 clusters, each with m=n/k samples.
Each cluster has a mean scaled by cluster index.
"""
function Matrix_GMM(n::Int, d::Int)
    k = 100
    m = n / k

    A = randn(n, d)

    for i in 1:min(k, d)
        I = (1 + Int((i-1)*m)):Int(i*m)
        A[I, i] .+= 10*i
    end

    return A'
end


"""
    Matrix_Gaussian_exp(n)

Generate matrix with Gaussian entries and exponentially decaying singular values.

Singular values:
- First 100: sv = 1
- Remaining: sv = 0.8^k (minimum 1e-5)
"""
function Matrix_Gaussian_exp(n::Int)
    m = 100
    sv = zeros(n)
    sv[1:m] .= 1.0
    sv[m+1:end] .= 0.8 .^ (1:(n-m))
    sv[sv .< 1e-5] .= 1e-5

    U, _ = qr(randn(n, n))
    V, _ = qr(randn(n, n))

    A = Matrix(U) * Diagonal(sv) * Matrix(V)'

    return A
end


"""
    Matrix_SNN(n)

Generate Sparse Neural Network matrix.

Sparse matrices (10% density) with singular value decay:
- First 100: sv = 10/k
- Remaining: sv = 1/k
"""
function Matrix_SNN(n::Int)
    m = 100
    sv = zeros(n)
    sv[1:m] = 10.0 ./ (1:m)
    sv[m+1:end] = 1.0 ./ (m+1:n)

    U = sprand(n, n, 0.1)
    V = sprand(n, n, 0.1)

    A = U * Diagonal(sv) * V'

    return Matrix(A)
end


#=============================================================================
# Additional Matrix Generators (from demo_utils)
=============================================================================#

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

end  # module TestUtils
