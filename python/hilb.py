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

    # Build first column and last row for the Hankel representation.
    c = 1.0 / (np.arange(n) + 1)  # Adjust for 0-based indexing
    r = 1.0 / (np.arange(m) + n)  # Adjust for 0-based indexing
    return linalg.hankel(c, r)


def _test():
    # 3x3 Hilbert matrix known analytically
    expected = np.array([[1.0, 0.5, 1/3],
                         [0.5, 1/3, 0.25],
                         [1/3, 0.25, 0.2]])
    assert np.allclose(hilb(3, 3), expected), "3x3 test failed"

    # Non-square case
    exp_rect = np.array([[1.0, 0.5, 1/3, 0.25],
                         [0.5, 1/3, 0.25, 0.2]])
    assert np.allclose(hilb(2, 4), exp_rect), "2x4 test failed"

    print("All tests passed!")

if __name__ == "__main__":
    _test()
