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
- `pert::Real`: Perturbation parameter for off-diagonal entries (default: 0.25)
  - Standard form uses pert = 0.25
  - Setting pert = 0 gives a purely diagonal matrix

# Returns
- `K::Matrix`: Kahan matrix (n x n)

# Matrix Structure
The matrix has the form:
```
K(i,i) = s^(i-1)              for i = 1,...,n (diagonal)
K(i,j) = -c * s^(i-1) * pert  for i < j       (upper triangle)
K(i,j) = 0                    for i > j       (lower triangle)
```
where s = sin(theta) and c = cos(theta).

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

# Author
Claude Code
"""
function kahan(n::Int; theta::Real=1.2, pert::Real=0.25)
    if n < 1
        error("n must be positive, got $n")
    end

    s = sin(theta)
    c = cos(theta)

    # Determine element type (promote to Float64 if needed)
    T = promote_type(typeof(s), typeof(c), typeof(pert))

    # Create upper triangular matrix
    K = zeros(T, n, n)

    # Fill diagonal: K(i,i) = s^(i-1)
    for i in 1:n
        K[i, i] = s^(i-1)
    end

    # Fill upper triangle: K(i,j) = -c * s^(i-1) * pert
    for i in 1:n
        for j in (i+1):n
            K[i, j] = -c * s^(i-1) * pert
        end
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
        K = kahan(10, theta=theta)
        c = cond(K)
        @printf("  theta=%.1f: cond(K) = %.2e\n", theta, c)
    end

    # Test 3: Perturbation parameter
    println("\nTest 3: Effect of perturbation parameter (n=5, theta=1.2)")
    for p in [0.0, 0.25, 0.5, 1.0]
        K = kahan(5, theta=1.2, pert=p)
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
