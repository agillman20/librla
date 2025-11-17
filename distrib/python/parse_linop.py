"""Parse input as linear operator structure.

This module provides a utility to normalize inputs (matrix or operator structure)
into a uniform LinearOperator structure. Compatible with scipy LinearOperators.
"""

import numpy as np
from typing import Union
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from make_linop import LinearOperator, make_linop


def parse_linop(A: Union[np.ndarray, LinearOperator, ScipyLinearOperator]) -> LinearOperator:
    """Parse input as linear operator structure.

    Converts either an explicit matrix, our custom LinearOperator,
    or a scipy LinearOperator into our LinearOperator structure.

    Args:
        A: Either a numpy array, LinearOperator, or scipy LinearOperator

    Returns:
        LinearOperator structure

    Examples:
        >>> # From matrix:
        >>> A = np.random.randn(100, 50)
        >>> op = parse_linop(A)

        >>> # From our LinearOperator:
        >>> op_in = make_linop(100, 50, lambda x: A @ x, lambda x: A.conj().T @ x)
        >>> op = parse_linop(op_in)  # Returns same structure

        >>> # From scipy LinearOperator:
        >>> from scipy.sparse.linalg import LinearOperator as ScipyLO
        >>> scipy_op = ScipyLO(shape=(100, 50), matvec=lambda x: A @ x)
        >>> op = parse_linop(scipy_op)  # Converts to our structure

    This function is used internally by iterative solvers to support both
    matrix and matrix-free inputs uniformly.
    """

    # Case 1: Already our custom LinearOperator (which is also a scipy one)
    if isinstance(A, LinearOperator):
        # Validate required attributes
        required = ['m', 'n', 'apply', 'applyT']
        for attr in required:
            if not hasattr(A, attr):
                raise ValueError(
                    f"LinearOperator must have attribute '{attr}'"
                )
        return A

    # Case 2: Scipy LinearOperator (but not our subclass)
    elif isinstance(A, ScipyLinearOperator):
        # Convert scipy LinearOperator to our LinearOperator
        m, n = A.shape

        # Check if rmatvec is available
        if hasattr(A, 'rmatvec') and A.rmatvec is not None:
            return make_linop(m, n, A.matvec, A.rmatvec)
        else:
            raise ValueError(
                "scipy LinearOperator must have rmatvec method for adjoint operation"
            )

    # Case 3: Numpy array - convert to LinearOperator
    elif isinstance(A, np.ndarray):
        return make_linop(A)

    else:
        raise TypeError(
            f"Input must be ndarray or LinearOperator, got {type(A).__name__}"
        )
