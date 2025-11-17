function afun = linop_to_afun(op)
%LINOP_TO_AFUN Convert LinearOperator to simple function handle for MATLAB
%
%  afun = LINOP_TO_AFUN(op) returns a function handle with simple signature
%  for MATLAB iterative solvers that work with symmetric/square matrices.
%
%  The returned function handle has signature: y = afun(x)
%  Returns: A*x (forward operation only)
%
%  MATLAB solvers using simple signature (afun(x)):
%    minres, pcg, gmres, bicgstab, cgs, tfqmr, symmlq
%
%  This is a convenience function - op.apply can also be used directly.
%
%  Example:
%    op = make_linop(A);
%    x = minres(linop_to_afun(op), b, tol, maxit);
%    % or simply:
%    x = pcg(op.apply, b, tol, maxit);
%
%  For solvers needing transpose (lsqr, qmr, bicg), use:
%    linop_to_afun_transp(op)
%
%  See also: make_linop, linop_to_afun_transp, minres, pcg, gmres

afun = op.apply;
end
