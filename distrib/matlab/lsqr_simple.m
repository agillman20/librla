function [x, flag, relres, iter] = lsqr_simple(A, b, tol, maxit)
%LSQR_SIMPLE Simple LSQR implementation (Octave compatible)
%
%  Solves the least squares problem:
%    min ||A*x - b||^2
%  using the LSQR algorithm (Golub-Kahan bidiagonalization).
%
%  [x, flag, relres, iter] = LSQR_SIMPLE(A, b, tol, maxit)
%
%  Input:
%    A     - matrix (m x n) OR operator structure from make_linop
%    b     - right-hand side (m x 1)
%    tol   - convergence tolerance (default 1e-6)
%    maxit - max iterations (default min(m,n))
%
%  Output:
%    x      - solution (n x 1)
%    flag   - 0 = converged, 1 = max iterations
%    relres - relative residual norm
%    iter   - iterations performed
%
%  Note: This implementation is compatible with both MATLAB and Octave.
%        Octave does not have a built-in lsqr function.
%
%  Reference: Paige & Saunders (1982), LSQR algorithm
%
%  See also: make_linop, parse_linop

if nargin < 3, tol = 1e-6; end
if nargin < 4, maxit = []; end

% Parse linear operator
op = parse_linop(A);
m = op.m;
n = op.n;
A = op.apply;      % Forward operator: A*x
AT = op.applyT;    % Adjoint operator: A'*y

if isempty(maxit), maxit = min(m, n); end

      % Initialize
      u = b;
      beta = norm(u);
      u = u / beta;
      v = AT(u);
      alpha = norm(v);
      v = v / alpha;

      % Initialize bidiagonal matrix and solution
      w = v;
      x = zeros(n, 1);
      phi_bar = beta;
      rho_bar = alpha;

      bnorm = norm(b);
      if bnorm == 0, bnorm = 1; end

      for iter = 1:maxit
          % Continue bidiagonalization
          u = A(v) - alpha*u;
          beta = norm(u);
          u = u / beta;

          v = AT(u) - beta*v;
          alpha = norm(v);
          v = v / alpha;

          % Update QR factorization of B_k (using Givens rotations)
          rho = sqrt(rho_bar^2 + beta^2);
          c = rho_bar / rho;
          s = beta / rho;
          theta = s * alpha;
          rho_bar = -c * alpha;
          phi = c * phi_bar;
          phi_bar = s * phi_bar;

          % Update solution
          x = x + (phi / rho) * w;
          w = v - (theta / rho) * w;

          % Check convergence
          resid_norm = abs(phi_bar);
          relres = resid_norm / bnorm;
          fprintf('  Iter %3d: %.2e\n', iter, relres);
          if relres < tol
              flag = 0;
              fprintf('  Converged at iteration %d\n', iter);
              return;
          end
      end

      flag = 1;  % Did not converge
      relres = abs(phi_bar) / bnorm;
      fprintf('\n  Warning: Did not converge (max iterations reached)\n');
end
