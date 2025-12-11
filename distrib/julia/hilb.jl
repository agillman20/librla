"""
Hilbert matrix generator for Julia.

Generates rectangular Hilbert matrices for testing. Hilbert matrices are
ill-conditioned with rapidly-decaying singular values.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: BSD-3-Clause
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

"""
    hilb(m::Int, n::Int=m)

Generate rectangular Hilbert matrix.

The (i,j) entry of the Hilbert matrix is 1/(i+j-1).

# Arguments
- `m::Int`: Number of rows
- `n::Int`: Number of columns (defaults to m for square matrix)

# Returns
- mxn Hilbert matrix

# Examples
```julia
# Square Hilbert matrix
H = hilb(5)

# Rectangular Hilbert matrix
H = hilb(100, 50)
```

# Note
Hilbert matrices are notoriously ill-conditioned and have
rapidly-decaying singular values, making them useful for
testing iterative methods and rank-revealing algorithms.
"""
function hilb(m::Int, n::Int=m)
    # Create index arrays - need to be careful about broadcasting
    # We want H[j,i] = 1/(i+j-1) where j is row index, i is column index
    j = reshape(1:m, m, 1)  # Column vector for row indices
    i = reshape(1:n, 1, n)  # Row vector for column indices

    # Compute Hilbert matrix: H[j,i] = 1/(i+j-1)
    H = 1.0 ./ (i .+ j .- 1)

    return H
end
