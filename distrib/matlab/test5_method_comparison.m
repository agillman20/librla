function test5_method_comparison()
% test5_method_comparison.m - Compare all three T computation methods
%
% Tests all three T computation methods on a full-rank random matrix:
% 1. method='fast' - Triangular solve (fastest, may have error > 1.0)
% 2. method='svd' - SVD-based pseudoinverse (stable)
% 3. method='lstsq' - Least-squares from original A (most accurate)
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: BSD-3-Clause
% Version: 1.0.0
% Date: TBD
% Assisted by: Claude Code (Anthropic)

    fprintf('======================================================================\n');
    fprintf('TEST 5: T Computation Method Comparison\n');
    fprintf('======================================================================\n');

    % Create full-rank random matrix
    rng(42);
    m = 400;
    n = 300;
    fprintf('\nMatrix size: %d x %d\n', m, n);
    fprintf('Matrix type: Full-rank random (all %d columns independent)\n', n);

    % Create full-rank matrix
    A = randn(m, n);
    normA = norm(A, 'fro');

    % Target rank (low compared to matrix rank)
    k_target = 20;
    fprintf('Target rank: %d (%.1f%% of columns)\n', k_target, 100*k_target/n);
    fprintf('======================================================================\n');

    % =========================================================================
    % Test 1: method='fast' (fastest, may have error > 1.0)
    % =========================================================================
    fprintf('\n1. method=''fast'' (triangular solve)\n');
    fprintf('----------------------------------------------------------------------\n');

    tic;
    [k1, piv1, T1] = librla.id_sketch(A, k_target, 'method', 'fast');
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
    if err1 > 1.0
        fprintf('  [NOTE] Error > 1.0 is expected for full-rank matrices with method=''fast''\n');
    end

    % =========================================================================
    % Test 2: method='svd' (stable for ill-conditioned)
    % =========================================================================
    fprintf('\n2. method=''svd'' (SVD-based pseudoinverse)\n');
    fprintf('----------------------------------------------------------------------\n');

    tic;
    [k2, piv2, T2] = librla.id_sketch(A, k_target, 'method', 'svd');
    t2 = toc;

    % Compute error
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

    % =========================================================================
    % Test 3: method='lstsq' (most accurate, slowest)
    % =========================================================================
    fprintf('\n3. method=''lstsq'' (least-squares from original A)\n');
    fprintf('----------------------------------------------------------------------\n');

    tic;
    [k3, piv3, T3] = librla.id_sketch(A, k_target, 'method', 'lstsq');
    t3 = toc;

    % Compute error
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
    if err3 < 1.0
        fprintf('  [OK] method=''lstsq'' guarantees error < 1.0\n');
    end

    % =========================================================================
    % Summary
    % =========================================================================
    fprintf('\n======================================================================\n');
    fprintf('SUMMARY\n');
    fprintf('======================================================================\n');
    fprintf('  Method     Rank    Error        Max|T|       Time      Notes\n');
    fprintf('----------------------------------------------------------------------\n');
    fprintf('  fast       %4d    %.3e    %.3e    %.4fs   Fastest\n', k1, err1, maxT1, t1);
    fprintf('  svd        %4d    %.3e    %.3e    %.4fs   Stable\n', k2, err2, maxT2, t2);
    fprintf('  lstsq      %4d    %.3e    %.3e    %.4fs   Most accurate\n', k3, err3, maxT3, t3);
    fprintf('======================================================================\n');

    % Validate
    success = true;

    if err3 > 1.0
        fprintf('\n[FAIL] method=''lstsq'' should guarantee error < 1.0!\n');
        success = false;
    end

    if k1 ~= k_target || k2 ~= k_target || k3 ~= k_target
        fprintf('\n[FAIL] All methods should return rank k=%d!\n', k_target);
        success = false;
    end

    if success
        fprintf('\n[PASS] All method tests passed!\n');
    end
end
