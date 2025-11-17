"""Linear operator abstraction for Python/NumPy.

This module provides a uniform interface for linear operators, supporting both
explicit matrices and matrix-free function handles. The implementation is
compatible with scipy.sparse.linalg.LinearOperator.
"""

import numpy as np
from typing import Callable, Union, Optional
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator


class LinearOperator(ScipyLinearOperator):
    """Linear operator structure compatible with scipy.

    This class inherits from scipy.sparse.linalg.LinearOperator and provides
    a unified API with MATLAB and Julia implementations.

    Attributes:
        m (int): Number of rows
        n (int): Number of columns
        apply (Callable): Forward operation A*x
        applyT (Callable): Adjoint operation A'*x (Hermitian adjoint)
        is_explicit (bool): True if backed by explicit matrix
        matrix (Optional[np.ndarray]): The matrix (if is_explicit=True)

    Scipy-compatible properties:
        shape (tuple): Matrix dimensions (m, n)
        dtype: Data type of the operator

    Examples:
        >>> # Works with your custom API
        >>> op = make_linop(A)
        >>> y = op.apply(x)
        >>>
        >>> # Also works with scipy API
        >>> y = op @ x  # or op.matvec(x)
        >>> z = op.T @ y  # or op.rmatvec(y)
        >>>
        >>> # Compatible with scipy solvers
        >>> from scipy.sparse.linalg import lsqr
        >>> x, istop, itn, r1norm = lsqr(op, b)[:4]
    """

    def __init__(self, m: int, n: int,
                 apply: Callable[[np.ndarray], np.ndarray],
                 applyT: Callable[[np.ndarray], np.ndarray],
                 is_explicit: bool = False,
                 matrix: Optional[np.ndarray] = None,
                 dtype=None):
        # Store custom attributes
        self.m = m
        self.n = n
        self._apply_func = apply
        self._applyT_func = applyT
        self.is_explicit = is_explicit
        self.matrix = matrix

        # Infer dtype if not provided
        if dtype is None:
            if is_explicit and matrix is not None:
                dtype = matrix.dtype
            else:
                raise ValueError(
                    "dtype must be specified for matrix-free LinearOperators. "
                    "For example: make_linop(m, n, afun, atfun, dtype=np.float64) or "
                    "dtype=np.complex128 for complex operators."
                )

        # Initialize scipy base class
        super().__init__(dtype=dtype, shape=(m, n))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Forward operation: A @ x"""
        return self._apply_func(x)

    def applyT(self, x: np.ndarray) -> np.ndarray:
        """Adjoint operation: A.H @ x (Hermitian adjoint)"""
        return self._applyT_func(x)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """Scipy interface: matrix-vector product"""
        return self._apply_func(x)

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        """Scipy interface: adjoint matrix-vector product"""
        return self._applyT_func(x)

    def _matmat(self, X: np.ndarray) -> np.ndarray:
        """Scipy interface: matrix-matrix product (BLAS3-rich).

        Computes A @ X where X is a matrix (n, k).

        For explicit operators (backed by a matrix), uses direct BLAS3 GEMM.
        For matrix-free operators, falls back to scipy's default column-by-column
        matvec implementation unless a custom batched operation is provided.
        """
        if self.is_explicit and self.matrix is not None:
            # BLAS3: Direct matrix-matrix multiplication
            return self.matrix @ X
        else:
            # Fallback to scipy's default (column-by-column matvec)
            return super()._matmat(X)

    def _rmatmat(self, Y: np.ndarray) -> np.ndarray:
        """Scipy interface: adjoint matrix-matrix product (BLAS3-rich).

        Computes A.H @ Y where Y is a matrix (m, k).

        For explicit operators (backed by a matrix), uses direct BLAS3 GEMM.
        For matrix-free operators, falls back to scipy's default column-by-column
        rmatvec implementation unless a custom batched operation is provided.
        """
        if self.is_explicit and self.matrix is not None:
            # BLAS3: Direct matrix-matrix multiplication (Hermitian adjoint)
            return self.matrix.conj().T @ Y
        else:
            # Fallback to scipy's default (column-by-column rmatvec)
            return super()._rmatmat(Y)

    def __repr__(self):
        op_type = "explicit" if self.is_explicit else "matrix-free"
        return f"LinearOperator({self.m}x{self.n}, {op_type})"


def make_linop(A: Union[np.ndarray, int],
               n: Optional[int] = None,
               Afun: Optional[Callable] = None,
               ATfun: Optional[Callable] = None,
               dtype=None) -> LinearOperator:
    """Create a linear operator structure from matrix or function handles.

    Usage:
        # From explicit matrix:
        op = make_linop(A)

        # From function handles (dtype REQUIRED for matrix-free):
        op = make_linop(m, n, Afun, ATfun, dtype=np.float64)

    Args:
        A: Either a matrix (numpy array) or m (number of rows)
        n: Number of columns (required if A is int)
        Afun: Function handle for A*x (required if A is int)
        ATfun: Function handle for A'*x (required if A is int)
        dtype: Data type (REQUIRED for matrix-free operators).
               Examples: np.float64, np.float32, np.complex128, np.complex64

    Returns:
        LinearOperator structure

    Examples:
        >>> # From explicit matrix (dtype inferred):
        >>> A = np.random.randn(100, 50)
        >>> op = make_linop(A)
        >>> y = op.apply(x)    # Same as A @ x
        >>> z = op.applyT(y)   # Same as A.conj().T @ y (Hermitian adjoint)

        >>> # From function handles (dtype MUST be specified):
        >>> Afun = lambda x: my_forward_op(x)
        >>> ATfun = lambda x: my_adjoint_op(x)
        >>> op = make_linop(100, 50, Afun, ATfun, dtype=np.float64)      # Real double
        >>> op = make_linop(100, 50, Afun, ATfun, dtype=np.complex128)   # Complex double
        >>> op = make_linop(100, 50, Afun, ATfun, dtype=np.float32)      # Real single
    """

    # Case 1: Matrix form
    if isinstance(A, np.ndarray):
        if n is not None or Afun is not None or ATfun is not None:
            raise ValueError("Extra arguments provided with matrix input")

        m, n = A.shape
        return LinearOperator(
            m=m,
            n=n,
            apply=lambda x: A @ x,
            applyT=lambda x: A.conj().T @ x,  # Hermitian adjoint for complex matrices
            is_explicit=True,
            matrix=A
        )

    # Case 2: Function handle form
    elif isinstance(A, int):
        if n is None or Afun is None or ATfun is None:
            raise ValueError("Must provide n, Afun, and ATfun when A is an integer")

        m = A
        if not isinstance(n, int) or m <= 0 or n <= 0:
            raise ValueError("m and n must be positive integers")
        if not callable(Afun) or not callable(ATfun):
            raise ValueError("Afun and ATfun must be callable")

        return LinearOperator(
            m=m,
            n=n,
            apply=Afun,
            applyT=ATfun,
            is_explicit=False,
            matrix=None,
            dtype=dtype
        )

    else:
        raise TypeError(
            f"First argument must be ndarray or int, got {type(A).__name__}"
        )
