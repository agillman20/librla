#!/usr/bin/env python
# -*- coding: ascii -*-
"""
hilb.py - Construct Hilbert matrices (square or rectangular).

The (i, j)-entry of a Hilbert matrix is
    H[i, j] = 1 / (i + j - 1)      (1-based indexing)

The function mirrors the MATLAB version:
    hilb(m)                -> m x m square matrix
    hilb(m, n)             -> m x n rectangular matrix
    hilb(m, n, method)     -> choose construction method:
        'vectorized' - default, uses NumPy broadcasting.
        'hankel'    - builds the matrix via scipy.linalg.hankel.
        'loops'     - explicit double-for-loop (educational).

Example
-------
>>> from hilb import hilb
>>> hilb(5)                     # 5x5 Hilbert matrix
>>> hilb(3, 6)                  # 3x6 Hilbert matrix
>>> hilb(4, 4, method='hankel')
"""
from __future__ import annotations
import numpy as np
from typing import Literal, Optional

# ``hankel`` is optional – it is only needed when the user requests the
# 'hankel' method.  Import lazily so that the module works even if SciPy is
# not installed, unless that method is explicitly requested.
try:
    from scipy.linalg import hankel  # type: ignore
except Exception:  # pragma: no cover
    hankel = None  # type: ignore[assignment]


def hilb(
    m: int,
    n: Optional[int] = None,
    method: Literal["vectorized", "hankel", "loops"] = "vectorized",
) -> np.ndarray:
    """
    Return an ``m x n`` Hilbert matrix.

    Parameters
    ----------
    m : int
        Number of rows (must be >= 1).
    n : int, optional
        Number of columns. If omitted, a square ``m x m`` matrix is produced.
    method : {'vectorized', 'hankel', 'loops'}, default 'vectorized'
        Construction strategy.

    Returns
    -------
    np.ndarray
        The Hilbert matrix with shape ``(m, n)`` and dtype ``float64``.

    Raises
    ------
    ValueError
        If ``method`` is unknown or if ``n`` is non-positive.
    ImportError
        If the ``'hankel'`` method is requested but SciPy is not available.
    """
    if m <= 0:
        raise ValueError("Number of rows 'm' must be a positive integer.")
    if n is None:
        n = m
    if n <= 0:
        raise ValueError("Number of columns 'n' must be a positive integer.")

    method = method.lower()
    if method not in {"vectorized", "hankel", "loops"}:
        raise ValueError(
            f"Invalid method '{method}'. Choose from 'vectorized', 'hankel', 'loops'."
        )

    # -----------------------------------------------------------------
    # 1) Vectorized construction (default)
    # -----------------------------------------------------------------
    if method == "vectorized":
        # 1-based indices as column/row vectors, then broadcast.
        i = np.arange(1, m + 1).reshape(m, 1)   # shape (m,1)
        j = np.arange(1, n + 1).reshape(1, n)   # shape (1,n)
        H = 1.0 / (i + j - 1)
        return H

    # -----------------------------------------------------------------
    # 2) Using SciPy's hankel function
    # -----------------------------------------------------------------
    if method == "hankel":
        if hankel is None:  # pragma: no cover
            raise ImportError(
                "SciPy is required for method='hankel' but could not be imported."
            )
        # First column: 1/1, 1/2, ..., 1/m
        c = 1.0 / np.arange(1, m + 1)
        # Last row: 1/m, 1/(m+1), ..., 1/(m+n-1)
        r = 1.0 / np.arange(m, m + n)
        H = hankel(c, r)
        return H

    # -----------------------------------------------------------------
    # 3) Explicit double-loop construction
    # -----------------------------------------------------------------
    # (mostly for teaching; considerably slower than the vectorized version)
    H = np.empty((m, n), dtype=float)
    for ii in range(m):
        for jj in range(n):
            # Convert from 0-based Python indices to the 1-based formula.
            H[ii, jj] = 1.0 / ((ii + 1) + (jj + 1) - 1)
    return H


# ----------------------------------------------------------------------
# Simple sanity-check when the module is executed directly
# ----------------------------------------------------------------------
def _test() -> None:
    """Run a few basic checks against known values."""
    # 3x3 square Hilbert matrix (exact rational values)
    expected_sq = np.array(
        [[1.0, 0.5, 1 / 3],
         [0.5, 1 / 3, 0.25],
         [1 / 3, 0.25, 0.2]],
        dtype=float,
    )
    assert np.allclose(hilb(3), expected_sq), "square test (default) failed"
    assert np.allclose(hilb(3, 3), expected_sq), "square test (explicit n) failed"

    # 2x4 rectangular case
    expected_rect = np.array(
        [[1.0, 0.5, 1 / 3, 0.25],
         [0.5, 1 / 3, 0.25, 0.2]],
        dtype=float,
    )
    assert np.allclose(hilb(2, 4), expected_rect), "rectangular test failed"

    # 'hankel' method (if SciPy is available)
    if hankel is not None:
        assert np.allclose(
            hilb(3, 5, method="hankel"), hilb(3, 5, "vectorized")
        ), "hankel method mismatch"

    # 'loops' method – compare against the fast version
    assert np.allclose(
        hilb(4, 6, method="loops"), hilb(4, 6, "vectorized")
    ), "loops method mismatch"

    print("All tests passed!")


if __name__ == "__main__":
    _test()
