import numpy as np

def hilb(n,m):
    """
    Creates an n x m Hilbert matrix using NumPy.

    Args:
        n (int): The order of the Hilbert matrix.
        m (int): other direction

    Returns:
        numpy.ndarray: The n x m Hilbert matrix.
        
    Shamelessly taken from internet    
    """
    # Create an empty n x m array
    a = np.zeros((n, m))
    
    # Populate the matrix using the Hilbert formula
    for i in range(n):
        for j in range(m):
            a[i, j] = 1.0 / (i + j + 1)  # Adjust for 0-based indexing
    return a

