function op = parse_linop(A)
%PARSE_LINOP Parse input as linear operator structure
%
%  op = PARSE_LINOP(A) converts A to operator structure
%
%  Input:
%    A - can be:
%        1. Explicit matrix (m x n)
%        2. Operator structure from make_linop
%
%  Output:
%    op - structure with fields:
%      .m, .n, .apply, .applyT, .is_explicit, [.matrix]
%
%  This function is used internally by iterative solvers to
%  support both matrix and matrix-free inputs uniformly.
%
%  Examples:
%    % From matrix:
%    A = randn(100, 50);
%    op = parse_linop(A);
%
%    % From operator structure:
%    op_in = make_linop(100, 50, @(x) A*x, @(x) A'*x);
%    op = parse_linop(op_in);  % Returns same structure
%
%  See also: make_linop

if isstruct(A)
    % Already an operator structure, validate and return
    if ~isfield(A, 'm') || ~isfield(A, 'n') || ...
       ~isfield(A, 'apply') || ~isfield(A, 'applyT')
        error('Operator structure must have fields: m, n, apply, applyT');
    end
    op = A;

elseif isnumeric(A)
    % Convert matrix to operator structure
    op = make_linop(A);

else
    error('Input must be a matrix or operator structure');
end

end
