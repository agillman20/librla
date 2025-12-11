%==========================================================================
% test4_linop_random.m - LinearOperator test with random matrix
%
%   Tests librla.id_sketch with LinearOperators on a medium-size random matrix:
%   1. Dense matrix (baseline)
%   2. Explicit LinearOperator (matrix wrapper)
%   3. Matrix-free LinearOperator (function handles - rank mode only)
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: BSD-3-Clause
% Version: 1.0.0
% Date: TBD
% Assisted by: Claude Code (Anthropic)
%==========================================================================

function test4_linop_random()
    fprintf('======================================================================\n');
    fprintf('TEST 4: LinearOperators - Random Matrix\n');
    fprintf('======================================================================\n');

    % Create medium-size low-rank random matrix
    rng(42);  % Set seed for reproducibility
    m = 500;
    n = 300;
    true_rank = 30;
    fprintf('\nMatrix size: %d x %d\n', m, n);
    fprintf('Matrix type: Low-rank random (rank ~%d)\n', true_rank);

    % Create low-rank matrix: A = U * V' + noise
    U = randn(m, true_rank);
    V = randn(n, true_rank);
    A = U * V' + 1e-10 * randn(m, n);
    normA = norm(A, 'fro');

    % Target rank
    k_target = 20;
    fprintf('Target rank: %d\n', k_target);
    fprintf('======================================================================\n');

    % =========================================================================
    % Test 1: Dense Matrix (Baseline)
    % =========================================================================
    fprintf('\n1. Dense Matrix (baseline)\n');
    fprintf('----------------------------------------------------------------------\n');

    tic;
    [k1, piv1, T1] = librla.id_sketch(A, k_target);
    t1 = toc;

    % Compute error
    A_skel1 = A(:, piv1(k1+1:end));
    A_basis1 = A(:, piv1(1:k1));
    if ~isempty(T1)
        err1 = norm(A_skel1 - A_basis1 * T1, 'fro') / normA;
        maxT1 = max(abs(T1(:)));
    else
        err1 = 0.0;
        maxT1 = 0.0;
    end

    fprintf('  Rank:      k = %d\n', k1);
    fprintf('  Error:     ||A_skel - A_basis @ T|| / ||A|| = %.3e\n', err1);
    fprintf('  Max |T|:   %.3e\n', maxT1);
    fprintf('  Time:      %.4f s\n', t1);

    % =========================================================================
    % Test 2: Explicit LinearOperator (Matrix Wrapper)
    % =========================================================================
    fprintf('\n2. Explicit LinearOperator (matrix wrapper)\n');
    fprintf('----------------------------------------------------------------------\n');

    A_linop_explicit = LinearOperator.from_matrix(A);
    fprintf('  Operator: %d x %d\n', A_linop_explicit.m, A_linop_explicit.n);
    fprintf('  Matrix-free: %d\n', isempty(A_linop_explicit.matrix));

    tic;
    [k2, piv2, T2] = librla.id_sketch(A_linop_explicit, k_target);
    t2 = toc;

    % Compute error using explicit matrix access
    A_skel2 = A(:, piv2(k2+1:end));
    A_basis2 = A(:, piv2(1:k2));
    if ~isempty(T2)
        err2 = norm(A_skel2 - A_basis2 * T2, 'fro') / normA;
        maxT2 = max(abs(T2(:)));
    else
        err2 = 0.0;
        maxT2 = 0.0;
    end

    fprintf('  Rank:      k = %d\n', k2);
    fprintf('  Error:     ||A_skel - A_basis @ T|| / ||A|| = %.3e\n', err2);
    fprintf('  Max |T|:   %.3e\n', maxT2);
    fprintf('  Time:      %.4f s\n', t2);

    % Verify explicit matches dense (rank and error should be similar)
    if k1 == k2 && abs(err1 - err2) < 1e-12
        fprintf('  [OK] Explicit LinearOperator produces same rank and error as dense!\n');
    else
        fprintf('  [NOTE] k_dense=%d, k_linop=%d, err_diff=%.3e\n', k1, k2, abs(err1-err2));
        fprintf('  (Pivots may differ due to randomness, but results should be similar)\n');
    end

    % =========================================================================
    % Test 3: Matrix-Free LinearOperator (Function Handles - Rank Mode Only)
    % =========================================================================
    fprintf('\n3. Matrix-free LinearOperator (function handles)\n');
    fprintf('----------------------------------------------------------------------\n');

    % Create matrix-free operator with function handles
    Afun = @(x) A * x;        % Forward operation: y = A*x
    ATfun = @(x) A' * x;      % Adjoint operation: y = A'*x

    A_linop_mf = LinearOperator(Afun, ATfun, m, n);
    fprintf('  Operator: %d x %d\n', A_linop_mf.m, A_linop_mf.n);
    fprintf('  Matrix-free: %d\n', isempty(A_linop_mf.matrix));
    fprintf('  Mode: Rank mode only (rtol >= 1)\n');

    tic;
    [k3, piv3, T3] = librla.id_sketch(A_linop_mf, k_target);
    t3 = toc;

    % Compute error using explicit matrix (for validation)
    A_skel3 = A(:, piv3(k3+1:end));
    A_basis3 = A(:, piv3(1:k3));
    if ~isempty(T3)
        err3 = norm(A_skel3 - A_basis3 * T3, 'fro') / normA;
        maxT3 = max(abs(T3(:)));
    else
        err3 = 0.0;
        maxT3 = 0.0;
    end

    fprintf('  Rank:      k = %d\n', k3);
    fprintf('  Error:     ||A_skel - A_basis @ T|| / ||A|| = %.3e\n', err3);
    fprintf('  Max |T|:   %.3e\n', maxT3);
    fprintf('  Time:      %.4f s\n', t3);

    if k3 == k_target
        fprintf('  [OK] Matrix-free returns target rank k=%d\n', k_target);
    else
        fprintf('  [WARNING] Expected k=%d, got k=%d\n', k_target, k3);
    end

    % =========================================================================
    % Summary
    % =========================================================================
    fprintf('\n======================================================================\n');
    fprintf('SUMMARY\n');
    fprintf('======================================================================\n');
    fprintf('  Method              Rank    Error        Max|T|       Time\n');
    fprintf('----------------------------------------------------------------------\n');
    fprintf('  Dense (baseline)    %4d    %.3e    %.3e    %.4fs\n', k1, err1, maxT1, t1);
    fprintf('  Explicit LinOp      %4d    %.3e    %.3e    %.4fs\n', k2, err2, maxT2, t2);
    fprintf('  Matrix-free LinOp   %4d    %.3e    %.3e    %.4fs\n', k3, err3, maxT3, t3);
    fprintf('======================================================================\n');

    % Validate
    success = true;

    if err1 > 1.0 || err2 > 1.0 || err3 > 1.0
        fprintf('\n[FAIL] Error > 1.0 detected!\n');
        success = false;
    end

    if k1 ~= k2 || abs(err1 - err2) > 1e-10
        fprintf('\n[FAIL] Explicit LinearOperator should match dense! k1=%d, k2=%d, err_diff=%.3e\n', ...
                k1, k2, abs(err1-err2));
        success = false;
    end

    if k3 ~= k_target
        fprintf('\n[FAIL] Matrix-free should return rank k=%d!\n', k_target);
        success = false;
    end

    if success
        fprintf('\n[PASS] All LinearOperator tests passed!\n');
    end
end
