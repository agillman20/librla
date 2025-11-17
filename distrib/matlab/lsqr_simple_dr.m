% TEST4_LEAST_SQUARES
%
% LinearOperator framework validation test
%
% PURPOSE:
%   Verify that LinearOperator abstraction gives IDENTICAL results to
%   explicit matrices for LSQR algorithm
%
% LINEAROPERATOR FRAMEWORK:
%   Provides uniform interface for matrix operations:
%     - Explicit matrices (A)
%     - Function handles (A*x, A'*y)
%     - Implicit operators (e.g., FFT)
%
% METHODS TESTED:
%   1. LSQR with explicit matrix A
%   2. LSQR with LinearOperator from matrix (make_linop(A))
%   3. LSQR with matrix-free LinearOperator (function handles)
%
% PROBLEM:
%   Matrix: A is 2400 x 4800 Hilbert matrix (underdetermined)
%   Tolerance: 1e-10
%   Solution: Minimum norm least squares
%
% KEY INSIGHTS:
%   - LinearOperator provides abstraction for matrix-free methods
%   - Enables large-scale problems where forming A is impractical
%   - Must give bit-for-bit identical results (same Krylov sequence)
%   - Tests interface correctness and numerical reproducibility
%
% VALIDATION CRITERIA:
%   - All three methods must give IDENTICAL solutions
%   - Match error: ||x_method - x_explicit|| / ||x_explicit|| < 1e-14
%   - Same iteration counts
%   - Status: PASSED if match error < 1e-14
%
% EXPECTED RESULTS:
%   - All methods converge to same solution (machine precision match)
%   - Slight timing differences due to overhead
%   - Matrix-free slightly slower (function call overhead)
%

randn('seed',1)

m = 120*2*10;  % 2400 (rows of A)
k = 240*2*10;  % 4800 (cols of A)

% Create rectangular Hilbert matrix (ill-conditioned test case)
a = hilb(m,k);

fprintf('========================================================\n');
fprintf('Test 4: LinearOperator Framework Validation\n');
fprintf('========================================================\n');
fprintf('Matrix A: %d x %d (underdetermined)\n', m, k);
fprintf('Tolerance: 1e-10\n');
fprintf('Problem: min ||A*x - y|| (minimum norm solution)\n\n');

%% Test 1: Single right-hand side - LinearOperator validation

fprintf('------------------------------------------------------------\n');
fprintf('Test 1: Single RHS - LinearOperator Validation\n');
fprintf('------------------------------------------------------------\n');

x0 = randn(k,1);
y = a*x0 + rand(m,1)*1e-16;  % Add small noise

% Direct solve (ground truth - minimum norm solution)
tic;
x_direct = a\y;
t_direct = toc;
err_direct = norm(a*x_direct-y,'fro')/norm(y,'fro');
fprintf('Direct solve (backslash):\n');
fprintf('  Time: %.4f seconds\n', t_direct);
fprintf('  Relative error: %.4e\n\n', err_direct);

%% Method 1: LSQR with explicit matrix

fprintf('Method 1: LSQR with Explicit Matrix\n');
fprintf('  Algorithm: Golub-Kahan bidiagonalization\n');
fprintf('  Operations: A*v and A''*u per iteration\n');
fprintf('  Memory: O(k) - short recurrence\n\n');

tic;
[x_explicit, flag_explicit, relres_explicit, iter_explicit] = lsqr_simple(a, y, 1e-10, 100);
t_explicit = toc;

err_explicit = norm(a*x_explicit-y,'fro')/norm(y,'fro');
fprintf('LSQR (explicit matrix) results:\n');
fprintf('  Time: %.4f seconds\n', t_explicit);
fprintf('  Iterations: %d\n', iter_explicit);
fprintf('  Relative error: %.4e\n', err_explicit);
fprintf('  Flag: %d (0=converged)\n', flag_explicit);

%% Method 2: LSQR with LinearOperator from matrix

fprintf('\nMethod 2: LSQR with LinearOperator from Matrix\n');
fprintf('  Algorithm: Same as Method 1\n');
fprintf('  Input: LinearOperator structure (from matrix)\n');
fprintf('  Purpose: Verify LinearOperator interface\n\n');

% Create LinearOperator from matrix
op_from_matrix = make_linop(a);

tic;
[x_operator, flag_operator, relres_operator, iter_operator] = lsqr_simple(op_from_matrix, y, 1e-10, 100);
t_operator = toc;

err_operator = norm(a*x_operator-y,'fro')/norm(y,'fro');
match_explicit = norm(x_explicit - x_operator) / norm(x_explicit);

fprintf('LSQR (LinearOperator) results:\n');
fprintf('  Time: %.4f seconds\n', t_operator);
fprintf('  Iterations: %d\n', iter_operator);
fprintf('  Relative error: %.4e\n', err_operator);
fprintf('  Flag: %d (0=converged)\n', flag_operator);
fprintf('  Match with Method 1: %.4e (should be ~0)\n', match_explicit);

%% Method 3: LSQR with matrix-free LinearOperator

fprintf('\nMethod 3: LSQR with Matrix-Free LinearOperator\n');
fprintf('  Algorithm: Same as Method 1\n');
fprintf('  Input: LinearOperator from function handles\n');
fprintf('  Purpose: Verify matrix-free operations\n\n');

% Create matrix-free LinearOperator
A_forward = @(x) a * x;
A_adjoint = @(y) a' * y;
op_matfree = make_linop(m, k, A_forward, A_adjoint, class(a));

tic;
[x_matfree, flag_matfree, relres_matfree, iter_matfree] = lsqr_simple(op_matfree, y, 1e-10, 100);
t_matfree = toc;

err_matfree = norm(a*x_matfree-y,'fro')/norm(y,'fro');
match_matfree = norm(x_explicit - x_matfree) / norm(x_explicit);

fprintf('LSQR (matrix-free) results:\n');
fprintf('  Time: %.4f seconds\n', t_matfree);
fprintf('  Iterations: %d\n', iter_matfree);
fprintf('  Relative error: %.4e\n', err_matfree);
fprintf('  Flag: %d (0=converged)\n', flag_matfree);
fprintf('  Match with Method 1: %.4e (should be ~0)\n', match_matfree);

%% Comparison

fprintf('\n------------------------------------------------------------\n');
fprintf('Single RHS Comparison Summary\n');
fprintf('------------------------------------------------------------\n');
fprintf('%-30s %10s %12s %15s\n', 'Method', 'Time (s)', 'Iters', 'Match Error');
fprintf('%-30s %10.4f %12d %15s\n', 'LSQR (explicit)', t_explicit, iter_explicit, '-');
fprintf('%-30s %10.4f %12d %15.4e\n', 'LSQR (LinearOperator)', t_operator, iter_operator, match_explicit);
fprintf('%-30s %10.4f %12d %15.4e\n', 'LSQR (matrix-free)', t_matfree, iter_matfree, match_matfree);

if match_explicit < 1e-14 && match_matfree < 1e-14
    fprintf('\n[OK] VALIDATION PASSED: All three methods give IDENTICAL results\n');
else
    fprintf('\n[FAIL] VALIDATION FAILED: Methods do not match\n');
end

