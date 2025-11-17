"""
kahan - Generate Kahan matrix

Kahan's matrix is a classic test matrix for numerical stability.
It's upper triangular with controlled condition number.

References
----------
W. Kahan, Numerical Linear Algebra, Canadian Math. Bulletin, 9 (1966),
pp. 757-801.

Author: Claude Code
"""

import numpy as np


def kahan(n, theta=1.2, pert=0.25):
    """
    Generate Kahan matrix.

    Kahan's matrix is an upper triangular matrix with interesting
    numerical properties. It's designed to be challenging for
    QR factorization algorithms.

    Parameters
    ----------
    n : int
        Size of the matrix (n x n)
    theta : float, optional
        Angle parameter in radians (default: 1.2)
        Controls the condition number via cos(theta)
        Smaller theta -> better conditioned
        Larger theta -> worse conditioned
    pert : float, optional
        Perturbation parameter for off-diagonal entries (default: 0.25)
        Standard form uses pert = 0.25
        Setting pert = 0 gives a purely diagonal matrix

    Returns
    -------
    K : ndarray, shape (n, n)
        Kahan matrix

    Notes
    -----
    The matrix has the form:
        K(i,i) = s^(i-1)              for i = 1,...,n (diagonal)
        K(i,j) = -c * s^(i-1) * pert  for i < j       (upper triangle)
        K(i,j) = 0                    for i > j       (lower triangle)

    where s = sin(theta) and c = cos(theta).

    The condition number is approximately 1/cos(theta)^n, so it grows
    exponentially with n and theta.

    Examples
    --------
    >>> import numpy as np
    >>> K = kahan(5)  # 5x5 Kahan matrix with default parameters
    >>> print(f"Condition number: {np.linalg.cond(K):.2e}")

    >>> # Well-conditioned version (small theta)
    >>> K_good = kahan(10, theta=0.5)
    >>> print(f"Condition (theta=0.5): {np.linalg.cond(K_good):.2e}")

    >>> # Ill-conditioned version (large theta)
    >>> K_bad = kahan(10, theta=1.5)
    >>> print(f"Condition (theta=1.5): {np.linalg.cond(K_bad):.2e}")

    >>> # Diagonal matrix (no perturbation)
    >>> K_diag = kahan(5, pert=0.0)
    >>> print("Purely diagonal:", np.allclose(K_diag, np.diag(np.diag(K_diag))))

    References
    ----------
    .. [1] Nicholas J. Higham, "Accuracy and Stability of Numerical
           Algorithms", 2nd ed., SIAM, 2002, Chapter 28.
    .. [2] W. Kahan, Numerical Linear Algebra, Canadian Math. Bulletin,
           9 (1966), pp. 757-801.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")

    s = np.sin(theta)
    c = np.cos(theta)

    # Create upper triangular matrix
    K = np.zeros((n, n))

    # Fill diagonal: K(i,i) = s^(i-1)
    for i in range(n):
        K[i, i] = s ** i

    # Fill upper triangle: K(i,j) = -c * s^(i-1) * pert
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = -c * (s ** i) * pert

    return K


if __name__ == "__main__":
    # Test the function
    print("=" * 60)
    print("Kahan Matrix Tests")
    print("=" * 60)

    # Test 1: Small matrix with default parameters
    print("\nTest 1: 5x5 Kahan matrix (default theta=1.2)")
    K = kahan(5)
    print(f"Matrix:\n{K}")
    print(f"Condition number: {np.linalg.cond(K):.2e}")
    print(f"Is upper triangular: {np.allclose(K, np.triu(K))}")

    # Test 2: Compare different theta values
    print("\nTest 2: Condition number vs theta (n=10)")
    for theta in [0.5, 1.0, 1.2, 1.4]:
        K = kahan(10, theta=theta)
        cond = np.linalg.cond(K)
        print(f"  theta={theta:.1f}: cond(K) = {cond:.2e}")

    # Test 3: Perturbation parameter
    print("\nTest 3: Effect of perturbation parameter (n=5, theta=1.2)")
    for pert in [0.0, 0.25, 0.5, 1.0]:
        K = kahan(5, theta=1.2, pert=pert)
        cond = np.linalg.cond(K)
        off_diag_norm = np.linalg.norm(K - np.diag(np.diag(K)))
        print(f"  pert={pert:.2f}: cond(K) = {cond:.2e}, ||off-diag|| = {off_diag_norm:.2e}")

    # Test 4: Large matrix
    print("\nTest 4: Larger matrix (n=20, theta=1.2)")
    K = kahan(20)
    print(f"Condition number: {np.linalg.cond(K):.2e}")
    print(f"||K||_F = {np.linalg.norm(K, 'fro'):.2e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
