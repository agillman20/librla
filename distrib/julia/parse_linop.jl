"""
Parse input as linear operator structure.

This module provides a utility to normalize inputs (matrix or operator structure)
into a uniform LinearOperator structure.
"""

# Include make_linop.jl if not already loaded
if !@isdefined(LinearOperator)
    include("make_linop.jl")
end

"""
    parse_linop(A::Union{AbstractMatrix,LinearOperator})

Parse input as linear operator structure.

Converts either an explicit matrix or an existing LinearOperator
structure into a LinearOperator.

# Arguments
- `A`: Either a matrix or LinearOperator structure

# Returns
- `LinearOperator` structure

# Examples
```julia
# From matrix:
A = randn(100, 50)
op = parse_linop(A)

# From operator structure:
op_in = make_linop(100, 50, x -> A * x, x -> A' * x)
op = parse_linop(op_in)  # Returns same structure
```

This function is used internally by iterative solvers to support both
matrix and matrix-free inputs uniformly.
"""
function parse_linop(A::AbstractMatrix)
    make_linop(A)
end

function parse_linop(A::LinearOperator)
    # Already a LinearOperator, return it
    A
end
