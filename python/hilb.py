import numpy as np
from scipy import linalg

def hilb(n:int,m:int):
    """
    Creates an n x m Hilbert matrix using NumPy/SciPy.

    Args:
        n (int): The order of the Hilbert matrix.
        m (int): other direction

    Returns:
        numpy.ndarray: The n x m Hilbert matrix.
    """

    """
    # Shamelessly taken from internet
    # Create an empty n x m array
    a = np.zeros((n, m))
    
    # Populate the matrix using the Hilbert formula
    for i in range(n):
        for j in range(m):
            a[i, j] = 1.0 / (i + j + 1)  # Adjust for 0-based indexing
    return a
    """

    # Optimized version, via scipy.linalg.hankel
    c = np.zeros(n)
    r = np.zeros(m)

    for i in range(n):
        c[i] = 1.0 / (i + 1)  # Adjust for 0-based indexing

    for i in range(m):
        r[i] = 1.0 / (i + n)  # Adjust for 0-based indexing

    return linalg.hankel(c,r)
