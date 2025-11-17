using LinearAlgebra
using SparseArrays

# Optional: MAT.jl for MNIST/CIFAR loading from .mat files
# Uncomment if you have MAT.jl installed and need to load data files:
# using MAT

"""
    make_mat(m, n, flag_type)

Generate test matrices for ID benchmarking.

Generates various types of test matrices from:
"Robust blockwise random pivoting: Fast and accurate adaptive
 interpolative decomposition"

# Arguments
- `m::Int`: Number of rows
- `n::Int`: Number of columns
- `flag_type::String`: Matrix type: "cifar", "mnist", "gaussexp", "gmm", "snn"

# Returns
- `X::Matrix`: Generated matrix (normalized by column)

# Data Requirements
- cifar: Requires exampledata/cifar10.mat (or ../cifar/cifar-10-batches-mat/)
- mnist: Requires exampledata/mnist_mat.mat
- Other types generate synthetic matrices (no data files needed)

# Examples
```julia
# Generate Gaussian with exponential decay
X = make_mat(500, 300, "gaussexp")

# Generate GMM matrix
X = make_mat(500, 300, "gmm")

# Generate sparse neural network matrix
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

        # CIFAR loading requires MAT.jl package
        # To use: uncomment "using MAT" at top of file and install MAT.jl
        error("CIFAR loading not implemented. Requires MAT.jl package and exampledata/cifar10.mat file.")

        # Commented out CIFAR loading code:
        # if isfile("exampledata/cifar10.mat")
        #     MM = matread("exampledata/cifar10.mat")
        #     A = MM["full_matrix"]'  # Transpose
        #     X = A[1:m, randperm(60000)[1:n]]
        # else
        #     error("CIFAR-10 data not found. Need exampledata/cifar10.mat")
        # end

    elseif flag_type == "mnist"
        if n > m
            error("columns must be less than or equal to rows for MNIST")
        end
        if m > 784
            error("too many rows! MNIST has 784 features (28x28)")
        end

        # MNIST loading requires MAT.jl package
        # To use: uncomment "using MAT" at top of file and install MAT.jl
        error("MNIST loading not implemented. Requires MAT.jl package and exampledata/mnist_mat.mat file.")

        # Commented out MNIST loading code:
        # if isfile("exampledata/mnist_mat.mat")
        #     MM = matread("exampledata/mnist_mat.mat")
        #     A = MM["mnist_mat"]'  # Transpose
        #     X = A[1:m, randperm(60000)[1:n]]
        # else
        #     error("MNIST data not found. Need exampledata/mnist_mat.mat")
        # end

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

    for i in 1:min(k, d)  # Only modify columns that exist
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
    # singular values decay fast
    m = 100
    sv = zeros(n)
    sv[1:m] .= 1.0
    sv[m+1:end] .= 0.8 .^ (1:(n-m))
    sv[sv .< 1e-5] .= 1e-5

    # Generate random orthogonal matrices
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

    # Create sparse random matrices with 10% density
    U = sprand(n, n, 0.1)
    V = sprand(n, n, 0.1)

    A = U * Diagonal(sv) * V'

    return Matrix(A)  # Convert to dense for compatibility
end
