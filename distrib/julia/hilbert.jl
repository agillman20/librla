"""
Hilbert matrix generator for Julia.

The Hilbert matrix is a classic ill-conditioned test matrix for numerical
algorithms. Entries are H[i,j] = 1/(i+j-1).

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

using LinearAlgebra
using Printf

"""
    hilbert(m, n=m)

Generate Hilbert matrix.

The Hilbert matrix is severely ill-conditioned, with entries
H[i,j] = 1/(i+j-1). Useful for testing numerical stability.

# Arguments
- `m::Int`: Number of rows
- `n::Int`: Number of columns (default: m)

# Returns
- `H::Matrix`: Hilbert matrix (m x n)
"""
function hilbert(m::Int, n::Int=m)
    if m < 1 || n < 1
        error("Dimensions must be positive, got $m x $n")
    end

    i = reshape(1:m, m, 1)
    j = reshape(1:n, 1, n)
    return 1.0 ./ (i .+ j .- 1)
end


# Test code (run when executed as script)
if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 60)
    println("Hilbert Matrix Tests")
    println("=" ^ 60)

    # Test 1: Small matrix
    println("\nTest 1: 5x5 Hilbert matrix")
    H = hilbert(5)
    println("Matrix:")
    display(H)
    println()
    @printf("Condition number: %.2e\n", cond(H))

    # Test 2: Verify entries
    println("\nTest 2: Verify H[i,j] = 1/(i+j-1)")
    H3 = hilbert(3)
    expected = [1.0 1/2 1/3; 1/2 1/3 1/4; 1/3 1/4 1/5]
    println("Matches expected: ", isapprox(H3, expected))

    # Test 3: Rectangular
    println("\nTest 3: Rectangular matrices")
    H_wide = hilbert(3, 5)
    H_tall = hilbert(5, 3)
    println("3x5 shape: ", size(H_wide))
    println("5x3 shape: ", size(H_tall))

    # Test 4: Condition number growth
    println("\nTest 4: Condition number vs size")
    for sz in [5, 10, 15, 20]
        local H = hilbert(sz)
        @printf("  n=%2d: cond(H) = %.2e\n", sz, cond(H))
    end

    println("\n" * "=" ^ 60)
    println("All tests completed!")
    println("=" ^ 60)
end
