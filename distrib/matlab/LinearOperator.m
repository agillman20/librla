classdef LinearOperator
% LinearOperator - Matrix-free linear operator for randomized algorithms
%
% This class represents a linear operator that can be applied to vectors
% without explicitly forming the matrix.
%
% Mimics scipy.sparse.linalg.LinearOperator.
%
% CONSTRUCTION:
%   A = LinearOperator(matvec_fun, rmatvec_fun, m, n)
%   A = LinearOperator(matvec_fun, rmatvec_fun, m, n, 'is_complex', true)
%   A = LinearOperator(matvec_fun, rmatvec_fun, m, n, 'dtype', 'single')
%   A = LinearOperator(matvec_fun, rmatvec_fun, m, n, 'matmat', false)
%
% PROPERTIES:
%   m          - Number of rows
%   n          - Number of columns
%   is_complex - Whether operator acts on complex vectors
%   dtype      - Data type string ('double', 'single', etc.)
%   matmat     - Whether function handles support matrix input (default: true)
%
% METHODS:
%   matvec(x)   - Apply operator: A * x
%   rmatvec(x)  - Apply adjoint: A' * x
%   size()      - Return [m, n]
%
% OVERLOADED OPERATORS:
%   A * x       - Matrix-vector or matrix-matrix product
%   A'          - Adjoint operator
%   size(A)     - Get dimensions
%
% PERFORMANCE NOTE:
%   For best performance, ensure your matvec_fun and rmatvec_fun handle matrix
%   input (multiple columns) efficiently. For example:
%     matvec_fun = @(x) H * x;  % Good - supports matrices via BLAS3
%   This enables BLAS3 operations when matmat=true (default).
%
%   If your functions only work with column vectors, set matmat=false:
%     A = LinearOperator(my_vec_fun, my_rvec_fun, m, n, 'matmat', false);
%
% EXAMPLE:
%   % Create operator from Hilbert matrix
%   H = hilbert(100);
%   matvec = @(x) H * x;
%   rmatvec = @(x) H' * x;
%   A = LinearOperator(matvec, rmatvec, 100, 100);  % matmat=true by default
%
%   % Use with randomized algorithms
%   [U, s, V] = librla.svd_sketch(A, 10);
%
% See also: svd_sketch, qr_sketch, id_sketch
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: BSD-3-Clause AND NIST-PD
% Version: 0.1.0
% Date: January 5, 2026
% Assisted by: Claude Code (Anthropic)

properties (SetAccess = private)
    m            % Number of rows
    n            % Number of columns
    is_complex   % Whether operator is complex
    dtype        % Data type string ('double', 'single', etc.)
    matmat       % Whether matvec/rmatvec support matrix input
    matrix       % Stored matrix data ([] for matrix-free)
end

properties (Access = private)
    matvec_fun   % Function handle for forward multiplication
    rmatvec_fun  % Function handle for adjoint multiplication
end

methods

function obj = LinearOperator(matvec_fun, rmatvec_fun, m, n, varargin)
% LINEAROPERATOR - Construct a matrix-free linear operator
%
% Inputs:
%   matvec_fun  - Function handle: y = matvec_fun(x)
%   rmatvec_fun - Function handle: y = rmatvec_fun(x)
%   m           - Number of rows
%   n           - Number of columns
%
% Optional name-value pairs:
%   'is_complex' - Boolean, default: false
%   'dtype'      - Data type string ('double', 'single'), default: 'double'
%   'matmat'     - Boolean, whether function handles support matrix input.
%                  Default: true. Set to false if your
%                  function handles only work with column vectors.
%   'matrix'     - Explicit matrix data (default: [], for matrix-free operators)

  if nargin < 4
      error('LinearOperator:NotEnoughInputs', ...
            'Need at least 4 arguments: matvec_fun, rmatvec_fun, m, n');
  end

  % Parse optional arguments
  p = inputParser;
  addParameter(p, 'is_complex', false, @islogical);
  addParameter(p, 'dtype', 'double', @ischar);
  addParameter(p, 'matmat', true, @islogical);
  addParameter(p, 'matrix', []);
  parse(p, varargin{:});

  % Validate matrix dimensions if provided
  if ~isempty(p.Results.matrix)
      [mat_m, mat_n] = size(p.Results.matrix);
      if mat_m ~= m || mat_n ~= n
          error('LinearOperator:SizeMismatch', ...
                'Matrix size [%d×%d] does not match specified dimensions [%d×%d]', ...
                mat_m, mat_n, m, n);
      end
  end

  % Set properties
  obj.matvec_fun = matvec_fun;
  obj.rmatvec_fun = rmatvec_fun;
  obj.m = m;
  obj.n = n;
  obj.is_complex = p.Results.is_complex;
  obj.dtype = p.Results.dtype;
  obj.matmat = p.Results.matmat;
  obj.matrix = p.Results.matrix;
end

function y = matvec(obj, x)
% MATVEC - Apply operator to vector or matrix
%
% y = matvec(A, x)
%   Computes y = A * x
%
% If x is a matrix and matmat=true, applies to all columns at once.
% Otherwise, applies column-by-column.

  if size(x, 1) ~= obj.n
      error('LinearOperator:SizeMismatch', ...
            'Vector size %d does not match operator columns %d', ...
            size(x, 1), obj.n);
  end

  if obj.matmat
      % Function handle supports matrix input
      y = obj.matvec_fun(x);
  else
      % Function handle only supports vectors - apply column by column
      if size(x, 2) == 1
          y = obj.matvec_fun(x);
      else
          y = zeros(obj.m, size(x, 2), obj.dtype);
          for i = 1:size(x, 2)
              y(:, i) = obj.matvec_fun(x(:, i));
          end
      end
  end
end

function y = rmatvec(obj, x)
% RMATVEC - Apply adjoint operator to vector or matrix
%
% y = rmatvec(A, x)
%   Computes y = A' * x (Hermitian adjoint)
%
% If x is a matrix and matmat=true, applies to all columns at once.
% Otherwise, applies column-by-column.

  if size(x, 1) ~= obj.m
      error('LinearOperator:SizeMismatch', ...
            'Vector size %d does not match operator rows %d', ...
            size(x, 1), obj.m);
  end

  if obj.matmat
      % Function handle supports matrix input
      y = obj.rmatvec_fun(x);
  else
      % Function handle only supports vectors - apply column by column
      if size(x, 2) == 1
          y = obj.rmatvec_fun(x);
      else
          y = zeros(obj.n, size(x, 2), obj.dtype);
          for i = 1:size(x, 2)
              y(:, i) = obj.rmatvec_fun(x(:, i));
          end
      end
  end
end

function y = mtimes(obj, x)
% MTIMES - Overload * operator
%
% y = A * x
%   Calls matvec(A, x)

  if isa(x, 'LinearOperator')
      error('LinearOperator:NotSupported', ...
            'Multiplication of two LinearOperators not supported');
  end
  y = obj.matvec(x);
end

function A_T = ctranspose(obj)
% CTRANSPOSE - Overload ' operator (Hermitian adjoint)
%
% A_T = A'
%   Returns a new LinearOperator representing A'

% Swap forward and adjoint functions, swap dimensions
  if isempty(obj.matrix)
      transposed_matrix = [];
  else
      transposed_matrix = obj.matrix';
  end

  A_T = LinearOperator(obj.rmatvec_fun, obj.matvec_fun, ...
                      obj.n, obj.m, ...
                      'is_complex', obj.is_complex, ...
                      'dtype', obj.dtype, ...
                      'matmat', obj.matmat, ...
                      'matrix', transposed_matrix);
end

function varargout = size(obj, dim)
% SIZE - Return operator dimensions
%
% [m, n] = size(A)
% s = size(A)
% m = size(A, 1)
% n = size(A, 2)

  if nargin == 1
      if nargout <= 1
          varargout{1} = [obj.m, obj.n];
      else
          varargout{1} = obj.m;
          varargout{2} = obj.n;
      end
  else
      if dim == 1
          varargout{1} = obj.m;
      elseif dim == 2
          varargout{1} = obj.n;
      else
          error('LinearOperator:InvalidDimension', ...
                'Dimension argument must be 1 or 2');
      end
  end
end

function disp(obj)
% DISP - Display operator information

  if obj.is_complex
      dtype_str = 'complex';
  else
      dtype_str = 'real';
  end

  if obj.matmat
      matmat_str = 'true';
  else
      matmat_str = 'false (column-by-column)';
  end

  if isempty(obj.matrix)
      matrix_str = 'matrix-free';
  else
      matrix_str = sprintf('explicit [%d×%d]', size(obj.matrix, 1), size(obj.matrix, 2));
  end

  fprintf('  LinearOperator (%s) with properties:\n\n', dtype_str);
  fprintf('        size: [%d,%d]\n', obj.m, obj.n);
  fprintf('       dtype: %s\n', obj.dtype);
  fprintf('      matmat: %s\n', matmat_str);
  fprintf('      matrix: %s\n', matrix_str);
end
end


methods (Static)
    
function A = from_matrix(M)
% FROM_MATRIX - Create LinearOperator from explicit matrix
%
% A = LinearOperator.from_matrix(M)
%   Creates a LinearOperator that wraps a dense or sparse matrix
%
% This is mainly for testing - normally you would just use M directly

  [m, n] = size(M);
  is_complex = ~isreal(M);
  dtype = class(M);

  matvec_fun = @(x) M * x;
  rmatvec_fun = @(x) M' * x;

  A = LinearOperator(matvec_fun, rmatvec_fun, m, n, ...
                    'is_complex', is_complex, ...
                    'dtype', dtype, ...
                    'matmat', true, ...
                    'matrix', M);
end

end

end
