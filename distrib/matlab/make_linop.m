function op = make_linop(A, varargin)
%MAKE_LINOP Create a linear operator structure from matrix or function handles
%
%  op = MAKE_LINOP(A) converts matrix A to operator structure
%  op = MAKE_LINOP(m, n, Afun, ATfun, dtype) creates operator from function handles
%
%  Input (matrix form):
%    A - explicit matrix (m x n)
%
%  Input (function handle form):
%    m     - number of rows
%    n     - number of columns
%    Afun  - function handle for A*x  (forward operation)
%    ATfun - function handle for A'*x (adjoint operation)
%    dtype - data type string (REQUIRED for matrix-free operators)
%            Examples: 'double', 'single' for real operators
%                      'double', 'single' for complex operators (determined at runtime)
%
%  Output:
%    op - structure with fields:
%      .m         - number of rows
%      .n         - number of columns
%      .apply     - function handle for A*x
%      .applyT    - function handle for A'*x
%      .is_explicit - true if backed by explicit matrix
%      .matrix    - the matrix (if is_explicit=true)
%      .dtype     - data type string ('double', 'single', etc.)
%
%  Examples:
%    % From explicit matrix (dtype inferred):
%    A = randn(100, 50);
%    op = make_linop(A);
%    y = op.apply(x);    % Same as A*x
%    z = op.applyT(y);   % Same as A'*y
%
%    % From function handles (dtype MUST be specified):
%    Afun = @(x) my_forward_op(x);
%    ATfun = @(x) my_adjoint_op(x);
%    op = make_linop(100, 50, Afun, ATfun, 'double');   % Real or complex double
%    op = make_linop(100, 50, Afun, ATfun, 'single');   % Real or complex single
%
%  Compatibility with MATLAB iterative solvers:
%    MATLAB has two function handle conventions:
%
%    1. Simple signature (symmetric solvers):
%       afun(x) returns A*x
%       Used by: minres, pcg, gmres, bicgstab, cgs, tfqmr, symmlq
%       Example: x = pcg(op.apply, b, tol, maxit);
%
%    2. Transpose flag signature (non-symmetric solvers):
%       afun(x, 'notransp') returns A*x
%       afun(x, 'transp')   returns A'*x
%       Used by: lsqr, qmr, bicg
%       Example: afun = linop_to_afun_transp(op);
%                x = lsqr(afun, b, tol, maxit);
%
%  See also: parse_linop, linop_to_afun, linop_to_afun_transp, pcg, lsqr

if nargin == 1
    % Matrix form: op = make_linop(A)
    if ~isnumeric(A)
        error('Single argument must be a numeric matrix');
    end

    [m, n] = size(A);
    op.m = m;
    op.n = n;
    % BLAS3-rich function handles: detect matrix input and use BLAS3
    op.apply = @(x) apply_explicit(A, x);
    op.applyT = @(x) applyT_explicit(A, x);
    op.is_explicit = true;
    op.matrix = A;
    op.dtype = class(A);  % Infer dtype from matrix

elseif nargin == 4 || nargin == 5
    % Function handle form: op = make_linop(m, n, Afun, ATfun, dtype)
    m = A;  % First argument is m
    n = varargin{1};
    Afun = varargin{2};
    ATfun = varargin{3};

    if nargin == 5
        dtype = varargin{4};
    else
        dtype = 'double';  % Default to double with warning
        warning('make_linop:missingDtype', ...
                'dtype not specified for matrix-free operator, defaulting to ''double''. For complex or single-precision operators, specify dtype explicitly.');
    end

    if ~isscalar(m) || ~isscalar(n)
        error('m and n must be scalar integers');
    end
    if ~isa(Afun, 'function_handle') || ~isa(ATfun, 'function_handle')
        error('Afun and ATfun must be function handles');
    end
    if ~ischar(dtype)
        error('dtype must be a string (''double'', ''single'', etc.)');
    end

    op.m = m;
    op.n = n;
    op.apply = Afun;
    op.applyT = ATfun;
    op.is_explicit = false;
    op.dtype = dtype;

else
    error('Usage: make_linop(A) or make_linop(m, n, Afun, ATfun, dtype)');
end

end

% ============================================================================
% Helper functions for BLAS3-rich operations
% ============================================================================

function y = apply_explicit(A, x)
%APPLY_EXPLICIT  BLAS3-rich forward operation for explicit matrices.
%
%   y = apply_explicit(A, x)
%
% Detects if x is a vector or matrix and uses appropriate BLAS level.
    if size(x, 2) == 1
        % Vector input - BLAS2
        y = A * x;
    else
        % Matrix input - BLAS3 (GEMM)
        y = A * x;
    end
end

function y = applyT_explicit(A, x)
%APPLYT_EXPLICIT  BLAS3-rich adjoint operation for explicit matrices.
%
%   y = applyT_explicit(A, x)
%
% Detects if x is a vector or matrix and uses appropriate BLAS level.
    if size(x, 2) == 1
        % Vector input - BLAS2
        y = A' * x;
    else
        % Matrix input - BLAS3 (GEMM)
        y = A' * x;
    end
end
