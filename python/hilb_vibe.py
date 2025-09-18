"""
hilb.py
-------

Construct an m-by-n Hilbert matrix.

The (i, j)-entry of a Hilbert matrix is

        H[i, j] = 1 / (i + j + 1)          # 0-based indexing

The implementation mirrors the MATLAB function ``hilb`` and offers three
construction methods:

* ``'vectorized'`` - default, uses NumPy broadcasting.
* ``'hankel'``    - builds the matrix via ``scipy.linalg.hankel``.
* ``'loops'``    - explicit double-for-loop (educational).

Example
-------
>>> from hilb import hilb
>>> hilb(5)                     # 5-by-5 Hilbert matrix
>>> hilb(3, 6)                  # 3-by-6 Hilbert matrix
>>> hilb(4, 4, method='hankel')
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Optional

try:
    # scipy is optional - only needed for the 'hankel' method
    from scipy.linalg import hankel
except Exception:  # pragma: no cover
    hankel = None  # type: ignore


Method = Literal["vectorized", "hankel", "loops"]


def hilb(
    m: int,
    n: Optional[int] = None,
    method: Method = "vectorized",
) -> np.ndarray:
    """
    Construct an m-by-n Hilbert matrix.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int, optional
        Number of columns. If omitted, a square ``mxm`` matrix is returned.
    method : {"vectorized", "hankel", "loops"}, optional
        Construction method. Default is ``'vectorized'``.

    Returns
    -------
    H : ndarray, shape (m, n)
        The Hilbert matrix with entries ``1/(i + j - 1)`` (1-based indexing).

    Raises
    ------
    ValueError
        If an unsupported ``method`` is supplied or ``scipy`` is not available
        for the ``'hankel'`` method.
    """
    # ------------------------------------------------------------------
    # Argument handling - make the call signature compatible with MATLAB
    # ------------------------------------------------------------------
    if n is None:          # only one size supplied -> square matrix
        n = m

    if m <= 0 or n <= 0:
        raise ValueError("Matrix dimensions must be positive integers.")

    # --------------------------------------------------------------
    # 1) Vectorized construction (default)
    # --------------------------------------------------------------
    if method == "vectorized":
        # i is a column vector (m, 1), j is a row vector (1, n)
        i = np.arange(1, m + 1)[:, np.newaxis]   # shape (m, 1)
        j = np.arange(1, n + 1)                  # shape (n,)
        # NumPy broadcasting produces an (m, n) array
        H = 1.0 / (i + j - 1)
        return H

    # --------------------------------------------------------------
    # 2) Using the built-in hankel function (via SciPy)
    # --------------------------------------------------------------
    if method == "hankel":
        if hankel is None:  # pragma: no cover
            raise ValueError(
                "SciPy is required for the 'hankel' method but could not be imported."
            )
        # First column: 1/(1:m)
        c = 1.0 / np.arange(1, m + 1)
        # Last row: 1/(m + (1:n) - 1) = 1/(m + 0:n-1)
        r = 1.0 / (m + np.arange(0, n))
        H = hankel(c, r)
        return H

    # --------------------------------------------------------------
    # 3) Explicit double-loop construction
    # --------------------------------------------------------------
    if method == "loops":
        H = np.empty((m, n), dtype=float)
        for col in range(n):
            for row in range(m):
                # Convert from 0-based Python indexing to 1-based formula
                H[row, col] = 1.0 / (row + col + 1)
        return H

    # --------------------------------------------------------------
    # If we reach here, the user supplied an unknown method.
    # --------------------------------------------------------------
    raise ValueError(
        f"Unknown method '{method}'. Choose from 'vectorized', 'hankel', or 'loops'."
    )


import argparse
import sys

# ----------------------------------------------------------------------
# Command-line interface for quick testing / demonstration
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an m-by-n Hilbert matrix."
    )
    parser.add_argument(
        "m",
        type=int,
        help="Number of rows of the Hilbert matrix.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Number of columns (defaults to a square matrix).",
    )
    parser.add_argument(
        "-mth",
        "--method",
        choices=["vectorized", "hankel", "loops"],
        default="vectorized",
        help="Construction method (default: vectorized).",
    )
    args = parser.parse_args()

    try:
        H = hilb(args.m, args.n, method=args.method)
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)

    # Print the matrix with a readable format
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(H)
