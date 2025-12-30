"""
    torch_compat - PyTorch-compatible wrappers for librla

Thin wrappers providing torch.svd_lowrank and torch.pca_lowrank
compatible interfaces using librla's randomized algorithms.

# Usage
```julia
include("torch_compat.jl")
using .torch_compat

U, s, V = svd_lowrank(A, q=10, niter=2)
U, s, V = pca_lowrank(A, q=10, center=true, niter=2)
```

Note: q is the oversampled rank (sketch size), not the final rank.
User is responsible for choosing q = k + oversampling where k is
the target rank. Typical oversampling is 5-10.

Reference: Halko et al., "Finding structure with randomness" (2009)

# Author
Adrianna Gillman, Zydrunas Gimbutas

# SPDX-License-Identifier
TBD

# Version
0.1.0

# Date
TBD

# Assisted by
Claude Code (Anthropic)
"""
module torch_compat

include("librla.jl")
using .librla: svd_sketch

using LinearAlgebra
using Statistics

export svd_lowrank, pca_lowrank

"""
    svd_lowrank(A; q=6, niter=2, M=nothing)

PyTorch-compatible randomized low-rank SVD.

# Arguments
- `A`: Input matrix (m×n)
- `q`: Oversampled rank / sketch size (default: 6)
- `niter`: Power iterations (default: 2)
- `M`: Matrix to subtract before decomposition (default: nothing)

# Returns
- `U`: Left singular vectors (m×q)
- `s`: Singular values (1D vector of length q)
- `V`: Right singular vectors (n×q), NOT transposed

# Note
Unlike librla.svd_sketch which returns Vt (transposed), this function
matches PyTorch convention and returns V (not transposed).
"""
function svd_lowrank(A; q::Int=6, niter::Int=2, M=nothing)
    # Subtract M if provided
    if M !== nothing
        A = A - M
    end

    # Call librla.svd_sketch with rank mode
    # Julia librla returns Vt (transposed), need to transpose to match PyTorch
    U, s, Vt = svd_sketch(A, q; power_iter=niter, extra_samples=0)

    return U, s, Vt'
end

"""
    pca_lowrank(A; q=nothing, center=true, niter=2)

PyTorch-compatible randomized low-rank PCA.

# Arguments
- `A`: Input matrix (m×n) - m samples, n features
- `q`: Oversampled rank / sketch size (default: min(6, m, n))
- `center`: Subtract column means (default: true)
- `niter`: Power iterations (default: 2)

# Returns
- `U`: Left singular vectors (m×q)
- `s`: Singular values (1D vector of length q)
- `V`: Right singular vectors (n×q), NOT transposed

# Notes
The relation to PCA:
- V columns are principal directions
- A * V[:, 1:k] projects data to first k principal components
"""
function pca_lowrank(A; q::Union{Int,Nothing}=nothing, center::Bool=true, niter::Int=2)
    m, n = size(A)

    # Default q
    if q === nothing
        q = min(6, m, n)
    end

    # Center data if requested
    if center
        A = A .- mean(A, dims=1)
    end

    # Call librla.svd_sketch with rank mode
    U, s, Vt = svd_sketch(A, q; power_iter=niter, extra_samples=0)

    return U, s, Vt'
end

end # module
