"""
Linear operator abstraction for Julia.

This module provides a uniform interface for linear operators, supporting both
explicit matrices and matrix-free function handles. Compatible with Julia's
AbstractMatrix interface and iterative solvers.
"""

using LinearAlgebra

"""
    LinearOperator{T}

Structure representing a linear operator with element type T.

# Type Parameters
- `T`: Element type (e.g., Float64, Float32, ComplexF64, ComplexF32)

# Fields
- `m::Int`: Number of rows
- `n::Int`: Number of columns
- `apply::Function`: Forward operation A*x
- `applyT::Function`: Adjoint operation A'*x
- `is_explicit::Bool`: True if backed by explicit matrix
- `matrix::Union{AbstractMatrix,Nothing}`: The matrix (if is_explicit=true)

# Note
For matrix-free operators, the type parameter T MUST be specified explicitly:
```julia
# Correct:
op = LinearOperator{Float64}(m, n, afun, atfun, false, nothing)
op = LinearOperator{ComplexF32}(m, n, afun, atfun, false, nothing)

# Incorrect (will error):
op = LinearOperator(m, n, afun, atfun)  # No type specified
```
"""
struct LinearOperator{T}
    m::Int
    n::Int
    apply::Function
    applyT::Function
    is_explicit::Bool
    matrix::Union{AbstractMatrix{T},Nothing}
end

# Constructor for explicit matrix operators
function LinearOperator(A::AbstractMatrix{T}) where T
    m, n = size(A)
    LinearOperator{T}(
        m, n,
        x -> A * x,
        y -> A' * y,
        true,
        A
    )
end

function Base.show(io::IO, op::LinearOperator)
    op_type = op.is_explicit ? "explicit" : "matrix-free"
    print(io, "LinearOperator($(op.m)x$(op.n), $(op_type))")
end

"""
    make_linop(A::AbstractMatrix)
    make_linop(::Type{T}, m::Int, n::Int, Afun::Function, ATfun::Function) where T

Create a linear operator structure from matrix or function handles.

# Arguments
- `A`: Explicit matrix (dtype inferred from matrix)
- `T`: Element type (REQUIRED for matrix-free operators)
        Examples: Float64, Float32, ComplexF64, ComplexF32
- `m`: Number of rows (required for matrix-free)
- `n`: Number of columns (required for matrix-free)
- `Afun`: Function for A*x (required for matrix-free)
- `ATfun`: Function for A'*x (required for matrix-free)

# Returns
- `LinearOperator{T}` structure

# Examples
```julia
# From explicit matrix (dtype inferred):
A = randn(100, 50)
op = make_linop(A)
y = op.apply(x)    # Same as A * x
z = op.applyT(y)   # Same as A' * y

# From function handles (dtype MUST be specified):
Afun = x -> my_forward_op(x)
ATfun = x -> my_adjoint_op(x)
op = make_linop(Float64, 100, 50, Afun, ATfun)      # Real double precision
op = make_linop(ComplexF64, 100, 50, Afun, ATfun)   # Complex double precision
op = make_linop(Float32, 100, 50, Afun, ATfun)      # Real single precision
```
"""
function make_linop(A::AbstractMatrix)
    LinearOperator(A)
end

function make_linop(::Type{T}, m::Int, n::Int, Afun::Function, ATfun::Function) where T
    if m <= 0 || n <= 0
        error("m and n must be positive integers")
    end
    LinearOperator{T}(m, n, Afun, ATfun, false, nothing)
end

# ============================================================================
# Julia Standard Interface - makes LinearOperator work like AbstractMatrix
# ============================================================================

"""
    size(op::LinearOperator)

Return the dimensions of the linear operator as a tuple (m, n).
"""
Base.size(op::LinearOperator) = (op.m, op.n)

"""
    size(op::LinearOperator, dim::Int)

Return the size of the linear operator along dimension `dim`.
"""
Base.size(op::LinearOperator, dim::Int) = size(op)[dim]

"""
    eltype(op::LinearOperator{T})

Return the element type of the linear operator.
"""
Base.eltype(::Type{LinearOperator{T}}) where T = T
Base.eltype(op::LinearOperator{T}) where T = T

"""
    *(op::LinearOperator, x::AbstractVector)

Matrix-vector multiplication: compute A*x using the forward operator.

This enables natural Julia syntax: `y = A * x`
"""
function Base.:*(op::LinearOperator, x::AbstractVector)
    if length(x) != op.n
        throw(DimensionMismatch("dimension mismatch: operator has $(op.n) columns, vector has $(length(x)) elements"))
    end
    return op.apply(x)
end

"""
    *(op::LinearOperator, X::AbstractMatrix)

Matrix-matrix multiplication: compute A*X using BLAS3 when explicit.

For explicit operators (backed by a matrix), uses direct BLAS3 GEMM.
For matrix-free operators, falls back to column-by-column matvec.
"""
function Base.:*(op::LinearOperator, X::AbstractMatrix)
    if size(X, 1) != op.n
        throw(DimensionMismatch("dimension mismatch"))
    end
    if op.is_explicit && op.matrix !== nothing
        # BLAS3: Direct matrix-matrix multiplication
        return op.matrix * X
    else
        # Fallback: column-by-column matvec (BLAS2)
        return hcat([op.apply(X[:, i]) for i in 1:size(X, 2)]...)
    end
end

"""
    adjoint(op::LinearOperator)

Return the adjoint (conjugate transpose) of the linear operator.

This enables the syntax: `y = A' * x` which calls the adjoint operator.
"""
function Base.adjoint(op::LinearOperator{T}) where T
    # Return a new LinearOperator with swapped dimensions and swapped apply/applyT
    LinearOperator{T}(
        op.n,          # Adjoint has swapped dimensions
        op.m,
        op.applyT,     # Forward of adjoint is applyT of original
        op.apply,      # Adjoint of adjoint is apply of original
        op.is_explicit,
        op.is_explicit ? adjoint(op.matrix) : nothing
    )
end

"""
    transpose(op::LinearOperator)

Return the transpose of the linear operator.

For real operators, this is the same as adjoint. For complex operators,
adjoint includes conjugation while transpose does not.
"""
function Base.transpose(op::LinearOperator{T}) where T
    if T <: Complex
        # For complex, transpose != adjoint
        # Need to create versions without conjugation
        # For simplicity, we assume users want adjoint for complex
        @warn "transpose of complex LinearOperator: using adjoint instead" maxlog=1
    end
    return adjoint(op)
end

"""
    mul!(y::AbstractVector, op::LinearOperator, x::AbstractVector)

In-place matrix-vector multiplication: compute `y .= A*x`.

This is the three-argument form required by many iterative solvers.
"""
function LinearAlgebra.mul!(y::AbstractVector, op::LinearOperator, x::AbstractVector)
    if length(x) != op.n
        throw(DimensionMismatch("dimension mismatch"))
    end
    if length(y) != op.m
        throw(DimensionMismatch("dimension mismatch"))
    end
    result = op.apply(x)
    copyto!(y, result)
    return y
end

"""
    mul!(y::AbstractVector, op::LinearOperator, x::AbstractVector, alpha::Number, beta::Number)

In-place matrix-vector multiplication: compute `y := alpha*A*x + beta*y`.

This is the five-argument form used by some iterative solvers for efficiency.
"""
function LinearAlgebra.mul!(y::AbstractVector, op::LinearOperator, x::AbstractVector,
                            alpha::Number, beta::Number)
    if length(x) != op.n || length(y) != op.m
        throw(DimensionMismatch("dimension mismatch"))
    end
    temp = op.apply(x)
    # y := alpha*A*x + beta*y
    if beta == 0
        @. y = alpha * temp
    else
        @. y = alpha * temp + beta * y
    end
    return y
end
