"""
hilb - Generate Hilbert matrix

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

import numpy as np
from scipy import linalg

def hilb(n: int, m: int) -> np.ndarray:
    """
    Creates an n x m Hilbert matrix using NumPy/SciPy.

    Args:
        n (int): The order of the Hilbert matrix.
        m (int): other direction

    Returns:
        numpy.ndarray: The n x m Hilbert matrix.
    """

    i = np.arange(1, n + 1)[:, None]
    j = np.arange(1, m + 1)[None, :]
    return 1.0 / (i + j - 1)

    """
    # Optimized version, via scipy.linalg.hankel
    c = np.zeros(n)
    r = np.zeros(m)

    for i in range(n):
        c[i] = 1.0 / (i + 1)  # Adjust for 0-based indexing

    for i in range(m):
        r[i] = 1.0 / (i + n)  # Adjust for 0-based indexing

    return linalg.hankel(c,r)
    """

def _test():

    expected = np.array([[1.0, 0.5, 1/3],
                         [0.5, 1/3, 0.25],
                         [1/3, 0.25, 0.2]])
    assert np.allclose(hilb(3, 3), expected), "3x3 test failed"


    exp_rect = np.array([[1.0, 0.5, 1/3, 0.25],
                         [0.5, 1/3, 0.25, 0.2]])
    assert np.allclose(hilb(2, 4), exp_rect), "2x4 test failed"

    print("All tests passed!")

if __name__ == "__main__":
    _test()
