"""
hilbert - Generate Hilbert matrix

The Hilbert matrix is a classic ill-conditioned test matrix for numerical
algorithms. Entries are H[i,j] = 1/(i+j-1).

References
----------
D. Hilbert, "Ein Beitrag zur Theorie des Legendre'schen Polynoms",
Acta Mathematica, 18 (1894), pp. 155-159.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

import numpy as np


def hilbert(m, n=None):
    """
    Generate Hilbert matrix.

    The Hilbert matrix is severely ill-conditioned, with entries
    H[i,j] = 1/(i+j-1). Useful for testing numerical stability.

    Parameters
    ----------
    m : int
        Number of rows
    n : int, optional
        Number of columns (default: m)

    Returns
    -------
    H : ndarray, shape (m, n)
        Hilbert matrix

    Notes
    -----
    The condition number grows exponentially with matrix size:
    cond(H) ~ O((1+sqrt(2))^(4n) / sqrt(n))

    The singular values decay rapidly, making this matrix ideal
    for testing rank-revealing and low-rank approximation algorithms.

    Examples
    --------
    >>> import numpy as np
    >>> H = hilbert(5)  # 5x5 Hilbert matrix
    >>> print(f"Condition number: {np.linalg.cond(H):.2e}")

    >>> # Rectangular matrix
    >>> H_rect = hilbert(100, 50)
    >>> print(f"Shape: {H_rect.shape}")

    References
    ----------
    .. [1] Nicholas J. Higham, "Accuracy and Stability of Numerical
           Algorithms", 2nd ed., SIAM, 2002, Chapter 28.
    """
    if n is None:
        n = m
    if m < 1 or n < 1:
        raise ValueError(f"Dimensions must be positive, got {m} x {n}")

    i = np.arange(1, m + 1)[:, None]
    j = np.arange(1, n + 1)[None, :]
    return 1.0 / (i + j - 1)


if __name__ == "__main__":
    # Test the function
    print("=" * 60)
    print("Hilbert Matrix Tests")
    print("=" * 60)

    # Test 1: Small matrix
    print("\nTest 1: 5x5 Hilbert matrix")
    H = hilbert(5)
    print(f"Matrix:\n{H}")
    print(f"Condition number: {np.linalg.cond(H):.2e}")

    # Test 2: Verify entries
    print("\nTest 2: Verify H[i,j] = 1/(i+j-1)")
    H3 = hilbert(3)
    expected = np.array([[1.0, 1/2, 1/3],
                         [1/2, 1/3, 1/4],
                         [1/3, 1/4, 1/5]])
    print(f"Matches expected: {np.allclose(H3, expected)}")

    # Test 3: Rectangular
    print("\nTest 3: Rectangular matrices")
    H_wide = hilbert(3, 5)
    H_tall = hilbert(5, 3)
    print(f"3x5 shape: {H_wide.shape}")
    print(f"5x3 shape: {H_tall.shape}")

    # Test 4: Condition number growth
    print("\nTest 4: Condition number vs size")
    for size in [5, 10, 15, 20]:
        H = hilbert(size)
        print(f"  n={size:2d}: cond(H) = {np.linalg.cond(H):.2e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
