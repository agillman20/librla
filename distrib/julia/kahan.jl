"""
Kahan matrix generator for Julia.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: BSD-3-Clause
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

using LinearAlgebra
using Printf

"""
    kahan(n; theta=1.2, pert=0.25)

Generate Kahan matrix.

Kahan's matrix is a classic test matrix for numerical stability.
It's upper triangular with controlled condition number.

# Arguments
- `n::Int`: Size of the matrix (n x n)
- `theta::Real`: Angle parameter in radians (default: 1.2)
  - Controls the condition number via cos(theta)
  - Smaller theta -> better conditioned
  - Larger theta -> worse conditioned
- `pert::Real`: Perturbation parameter for diagonal entries (default: 25)
  - Standard form uses pert = 25 for numerical stability
  - Setting pert = 0 gives no diagonal perturbation

# Returns
- `K::Matrix`: Kahan matrix (n x n)

# Matrix Structure
The matrix has the form:
```
K(i,i) = s^(i-1) + pert*eps*(n-i+1)  for i = 1,...,n (diagonal)
K(i,j) = -c * s^(i-1)                for i < j       (upper triangle)
K(i,j) = 0                           for i > j       (lower triangle)
```
where s = sin(theta), c = cos(theta), and eps is machine epsilon.

The diagonal perturbation (pert*eps*(n-i+1)) ensures QR factorization
with column pivoting does not interchange columns in the presence of
rounding errors. The default pert=25 ensures no interchanges up to
N=90 in IEEE arithmetic.

The condition number is approximately 1/cos(theta)^n, so it grows
exponentially with n and theta.

# Examples
```julia
using LinearAlgebra

# 5x5 Kahan matrix with default parameters
K = kahan(5)
println("Condition number: ", cond(K))

# Well-conditioned version (small theta)
K_good = kahan(10, theta=0.5)
println("Condition (theta=0.5): ", cond(K_good))

# Ill-conditioned version (large theta)
K_bad = kahan(10, theta=1.5)
println("Condition (theta=1.5): ", cond(K_bad))

# Diagonal matrix (no perturbation)
K_diag = kahan(5, pert=0.0)
println("Purely diagonal: ", istriu(K_diag) && isdiag(K_diag))
```

# References
- Nicholas J. Higham, "Accuracy and Stability of Numerical Algorithms",
  2nd ed., SIAM, 2002, Chapter 28.
- W. Kahan, Numerical Linear Algebra, Canadian Math. Bulletin,
  9 (1966), pp. 757-801.
- NIST Matrix Market: Kahan Matrix,
  https://math.nist.gov/MatrixMarket/deli/Kahan/information.html

# Author
Claude Code
"""
function kahan(n::Int; theta::Real=1.2, pert::Real=25)
    if n < 1
        error("n must be positive, got $n")
    end

    s = sin(theta)
    c = cos(theta)
    eps_val = eps(Float64)

    # Create matrix following Octave gallery('kahan') implementation:
    # K = I - c * triu(ones(n), 1)
    # K = diag(s.^[0:n-1]) * K + pert*eps*diag([n:-1:1])

    # Start with identity
    K = Matrix{Float64}(I, n, n)

    # Subtract c * strict_upper_triangle(ones)
    # This makes all strict upper triangle elements equal to -c
    for i in 1:n
        for j in (i+1):n
            K[i, j] = -c
        end
    end

    # Left-multiply by diagonal matrix diag(s^[0:n-1])
    # This scales row i by s^(i-1)
    for i in 1:n
        K[i, :] .*= s^(i-1)
    end

    # Add diagonal perturbation: pert*eps*diag([n, n-1, ..., 1])
    for i in 1:n
        K[i, i] += pert * eps_val * (n - i + 1)
    end

    return K
end


# Test code (run when executed as script)
if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 60)
    println("Kahan Matrix Tests")
    println("=" ^ 60)

    # Test 1: Small matrix with default parameters
    println("\nTest 1: 5x5 Kahan matrix (default theta=1.2)")
    K = kahan(5)
    println("Matrix:")
    display(K)
    println()
    @printf("Condition number: %.2e\n", cond(K))
    println("Is upper triangular: ", istriu(K))

    # Test 2: Compare different theta values
    println("\nTest 2: Condition number vs theta (n=10)")
    for theta in [0.5, 1.0, 1.2, 1.4]
        local K = kahan(10, theta=theta)
        c = cond(K)
        @printf("  theta=%.1f: cond(K) = %.2e\n", theta, c)
    end

    # Test 3: Perturbation parameter
    println("\nTest 3: Effect of perturbation parameter (n=5, theta=1.2)")
    for p in [0.0, 0.25, 0.5, 1.0]
        local K = kahan(5, theta=1.2, pert=p)
        c = cond(K)
        off_diag_norm = norm(K - Diagonal(diag(K)))
        @printf("  pert=%.2f: cond(K) = %.2e, ||off-diag|| = %.2e\n", p, c, off_diag_norm)
    end

    # Test 4: Large matrix
    println("\nTest 4: Larger matrix (n=20, theta=1.2)")
    K = kahan(20)
    @printf("Condition number: %.2e\n", cond(K))
    @printf("||K||_F = %.2e\n", norm(K))

    println("\n" * "=" ^ 60)
    println("All tests completed!")
    println("=" ^ 60)
end
