"""
    LinearOperator

Matrix-free linear operator for randomized algorithms.

This type represents a linear operator that can be applied to vectors
without explicitly forming the matrix.

Mimics scipy.sparse.linalg.LinearOperator and the MATLAB LinearOperator class.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.2
Date: June 22, 2026
Assisted by: Claude Code (Anthropic)

# Construction
```julia
A = LinearOperator(matvec_fun, rmatvec_fun, m, n;
                   is_complex=false, dtype=Float64, matmat=true, matrix=nothing)
```

# Properties
- `m::Int` - Number of rows
- `n::Int` - Number of columns
- `is_complex::Bool` - Whether operator acts on complex vectors
- `dtype::Type` - Data type (Float64, Float32, ComplexF64, ComplexF32, etc.)
- `matmat::Bool` - Whether function handles support matrix input (default: true)
- `matrix` - Stored matrix data (nothing for matrix-free)

# Methods
- `matvec(A, x)` - Apply operator: A * x
- `rmatvec(A, x)` - Apply adjoint: A' * x
- `size(A)` - Return (m, n)

# Overloaded Operators
- `A * x` - Matrix-vector or matrix-matrix product
- `A'` - Adjoint operator
- `size(A)` - Get dimensions

# Performance Note
For best performance, ensure your matvec_fun and rmatvec_fun handle matrix
input (multiple columns) efficiently. For example:
```julia
matvec_fun = x -> H * x  # Good - supports matrices via BLAS3
```
This enables BLAS3 operations when matmat=true (default).

If your functions only work with column vectors, set matmat=false:
```julia
A = LinearOperator(my_vec_fun, my_rvec_fun, m, n; matmat=false)
```

# Example
```julia
# Create operator from Hilbert matrix
using LinearAlgebra
H = [1/(i+j-1) for i=1:100, j=1:100]
matvec = x -> H * x
rmatvec = x -> H' * x
A = LinearOperator(matvec, rmatvec, 100, 100)  # matmat=true by default

# Use with randomized algorithms
U, s, V = svd_sketch(A, 10)
```

See also: svd_sketch, qr_sketch, id_sketch
"""
struct LinearOperator{T, F1, F2, M}
    m::Int                # Number of rows
    n::Int                # Number of columns
    is_complex::Bool      # Whether operator is complex
    dtype::Type{T}        # Data type
    matmat::Bool          # Whether matvec/rmatvec support matrix input
    matrix::M             # Stored matrix data (nothing for matrix-free)
    matvec_fun::F1        # Function for forward multiplication
    rmatvec_fun::F2       # Function for adjoint multiplication

    function LinearOperator(matvec_fun::F1, rmatvec_fun::F2, m::Int, n::Int;
                           is_complex::Bool=false,
                           dtype::Type{T}=Float64,
                           matmat::Bool=true,
                           matrix::M=nothing) where {T, F1, F2, M}
        # Validate matrix dimensions if provided
        if !isnothing(matrix)
            mat_m, mat_n = size(matrix)
            if mat_m != m || mat_n != n
                error("Matrix size ($mat_m×$mat_n) does not match specified dimensions ($m×$n)")
            end
        end

        new{T, F1, F2, M}(m, n, is_complex, dtype, matmat, matrix,
                          matvec_fun, rmatvec_fun)
    end
end

"""
    matvec(A::LinearOperator, x)

Apply operator to vector or matrix: y = A * x

If x is a matrix and matmat=true, applies to all columns at once.
Otherwise, applies column-by-column.
"""
function matvec(obj::LinearOperator, x::AbstractVecOrMat)
    if size(x, 1) != obj.n
        error("Vector size $(size(x, 1)) does not match operator columns $(obj.n)")
    end

    if obj.matmat
        # Function handle supports matrix input
        return obj.matvec_fun(x)
    else
        # Function handle only supports vectors - apply column by column
        if ndims(x) == 1 || size(x, 2) == 1
            return obj.matvec_fun(x)
        else
            y = zeros(obj.dtype, obj.m, size(x, 2))
            for i in 1:size(x, 2)
                y[:, i] = obj.matvec_fun(x[:, i])
            end
            return y
        end
    end
end

"""
    rmatvec(A::LinearOperator, x)

Apply adjoint operator to vector or matrix: y = A' * x (Hermitian adjoint)

If x is a matrix and matmat=true, applies to all columns at once.
Otherwise, applies column-by-column.
"""
function rmatvec(obj::LinearOperator, x::AbstractVecOrMat)
    if size(x, 1) != obj.m
        error("Vector size $(size(x, 1)) does not match operator rows $(obj.m)")
    end

    if obj.matmat
        # Function handle supports matrix input
        return obj.rmatvec_fun(x)
    else
        # Function handle only supports vectors - apply column by column
        if ndims(x) == 1 || size(x, 2) == 1
            return obj.rmatvec_fun(x)
        else
            y = zeros(obj.dtype, obj.n, size(x, 2))
            for i in 1:size(x, 2)
                y[:, i] = obj.rmatvec_fun(x[:, i])
            end
            return y
        end
    end
end

# Overload * operator
"""
    *(A::LinearOperator, x)

Matrix-vector or matrix-matrix product. Calls matvec(A, x).
"""
function Base.:*(obj::LinearOperator, x::AbstractVecOrMat)
    if isa(x, LinearOperator)
        error("Multiplication of two LinearOperators not supported")
    end
    return matvec(obj, x)
end

# Overload adjoint operator (')
"""
    adjoint(A::LinearOperator)

Hermitian adjoint. Returns a new LinearOperator representing A'.
"""
function Base.adjoint(obj::LinearOperator{T, F1, F2, M}) where {T, F1, F2, M}
    # Swap forward and adjoint functions, swap dimensions
    transposed_matrix = isnothing(obj.matrix) ? nothing : adjoint(obj.matrix)

    return LinearOperator(obj.rmatvec_fun, obj.matvec_fun,
                         obj.n, obj.m;
                         is_complex=obj.is_complex,
                         dtype=obj.dtype,
                         matmat=obj.matmat,
                         matrix=transposed_matrix)
end

# Overload size function
"""
    size(A::LinearOperator)
    size(A::LinearOperator, dim)

Return operator dimensions.
"""
function Base.size(obj::LinearOperator)
    return (obj.m, obj.n)
end

function Base.size(obj::LinearOperator, dim::Int)
    if dim == 1
        return obj.m
    elseif dim == 2
        return obj.n
    else
        error("Dimension argument must be 1 or 2")
    end
end

# Overload eltype function
"""
    eltype(A::LinearOperator)

Return element type of the operator.
"""
function Base.eltype(obj::LinearOperator{T, F1, F2, M}) where {T, F1, F2, M}
    return obj.dtype
end

# Overload show for nice display
"""
    show(io::IO, A::LinearOperator)

Display operator information.
"""
function Base.show(io::IO, obj::LinearOperator)
    dtype_str = obj.is_complex ? "complex" : "real"
    matmat_str = obj.matmat ? "true" : "false (column-by-column)"
    matrix_str = isnothing(obj.matrix) ? "matrix-free" : "explicit [$(size(obj.matrix, 1))×$(size(obj.matrix, 2))]"

    println(io, "LinearOperator ($dtype_str) with properties:")
    println(io, "      size: [$(obj.m),$(obj.n)]")
    println(io, "     dtype: $(obj.dtype)")
    println(io, "    matmat: $matmat_str")
    print(io,   "    matrix: $matrix_str")
end

"""
    from_matrix(M::AbstractMatrix)

Create LinearOperator from explicit matrix.

This is mainly for testing - normally you would just use M directly.

# Example
```julia
M = randn(100, 50)
A = from_matrix(M)
```
"""
function from_matrix(M::AbstractMatrix{T}) where T
    m, n = size(M)
    is_complex = T <: Complex
    dtype = T

    matvec_fun = x -> M * x
    rmatvec_fun = x -> M' * x

    return LinearOperator(matvec_fun, rmatvec_fun, m, n;
                         is_complex=is_complex,
                         dtype=dtype,
                         matmat=true,
                         matrix=M)
end
