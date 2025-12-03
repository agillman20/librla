%==========================================================================
% test7_power - Test power iteration in svd_sketch (librla version)
%
%   Tests power iteration in the svd_sketch function.
%   Power iteration applies (A^H A)^n to improve sketch quality by amplifying
%   the dominant subspace.
%
%   This version uses librla instead of libid.
%
%   Usage:
%       test7_power           % structured matrix only
%       test7_power('random') % both structured and random
%
%   Tests:
%       Test 1: Power iteration in svd_sketch
%           - Tests extra_samples: 24, 18, 12, 6, 3
%           - Tests power_iter: 0-6
%           - Measures reconstruction error and singular value accuracy
%           - Can use structured or random matrix
%
%   Author : Power iteration tests (librla version)
%   SPDX-License-Identifier : TBD
%==========================================================================

function test7_power(varargin)
    % Parse input
    test_random = false;
    if nargin > 0 && strcmpi(varargin{1}, 'random')
        test_random = true;
    end

    fprintf('======================================================================\n');
    fprintf('POWER ITERATION IN SVD_SKETCH TESTS (librla)\n');
    fprintf('======================================================================\n');

    rng(42); % For reproducibility

    % Test 1: Power iteration in svd_sketch
    if test_random
        % Run with structured matrix first
        test_svd_sketch_power_iter(false);
        % Then run with random matrix
        test_svd_sketch_power_iter(true);
    else
        % Default: structured matrix only
        test_svd_sketch_power_iter(false);
    end

    fprintf('\n======================================================================\n');
    fprintf('ALL TESTS PASSED [PASS]\n');
    fprintf('======================================================================\n');
end


function test_svd_sketch_power_iter(use_random_matrix)
    % Test 1: Power iteration in svd_sketch
    fprintf('\n======================================================================\n');
    fprintf('TEST 1: Power iteration in svd_sketch\n');
    fprintf('======================================================================\n');

    % Test matrix configuration (same as test6_power)
    m = 500;
    n = 300;
    k = 50;

    if use_random_matrix
        fprintf('\nMatrix type: RANDOM (no prescribed singular values)\n');
        % Simple random matrix
        A = randn(m, n);

        % Compute SVD to get true singular values
        [~, s_vec, ~] = svd(A, 'econ');
        s = diag(s_vec);
    else
        fprintf('\nMatrix type: STRUCTURED (prescribed singular values)\n');
        % Create matrix with decaying spectrum
        U_full = orth(randn(m, m));
        V_full = orth(randn(n, n));
        s = [logspace(0, -2, k), logspace(-2, -10, n-k)]';
        % Use first n columns of U to match dimensions
        U = U_full(:, 1:n);
        V = V_full;
        A = U * diag(s) * V';
    end

    % Compute detailed matrix properties
    cond_number = s(1) / s(end);
    spectral_gap_k = s(k) / s(k+1);
    decay_rate_k = s(1) / s(k);

    fprintf('\nMatrix Properties:\n');
    fprintf('  Dimensions:       %dx%d\n', m, n);
    fprintf('  Target rank:      %d (first %d singular values)\n', k, k);
    fprintf('  Condition number: %.2e\n', cond_number);
    fprintf('  Spectral gap at k=%d: %.2fx (s[%d]/s[%d])\n', k, spectral_gap_k, k, k+1);
    fprintf('  Decay rate (s[1]/s[%d]): %.2fx\n', k, decay_rate_k);
    fprintf('\nSingular value distribution:\n');
    fprintf('  s[1]    = %.6e (largest)\n', s(1));
    fprintf('  s[%3d]  = %.6e (target cutoff)\n', k, s(k));
    fprintf('  s[%3d]  = %.6e (first neglected)\n', k+1, s(k+1));
    fprintf('  s[%3d]  = %.6e (smallest)\n', n, s(n));

    % Test different extra_samples and power_iter values
    extra_samples_list = [24, 18, 12, 6, 3];
    power_iter_list = 0:6;

    % Store results for summary
    errors = zeros(length(extra_samples_list), length(power_iter_list));
    sval_errors = zeros(length(extra_samples_list), length(power_iter_list));

    for idx = 1:length(extra_samples_list)
        extra_samples = extra_samples_list(idx);

        fprintf('\n======================================================================\n');
        fprintf('extra_samples = %d\n', extra_samples);
        fprintf('======================================================================\n');

        for jdx = 1:length(power_iter_list)
            power_iter = power_iter_list(jdx);
            fprintf('\n--- power_iter = %d ---\n', power_iter);

            % Run svd_sketch
            tic;
            [U_sketch, s_sketch, V_sketch] = librla.svd_sketch(A, k, ...
                'power_iter', power_iter, 'extra_samples', extra_samples);
            t_total = toc;

            % Compute reconstruction error
            A_approx = U_sketch * diag(s_sketch) * V_sketch';
            err = norm(A - A_approx, 'fro') / norm(A, 'fro');
            errors(idx, jdx) = err;

            % Singular value accuracy
            s_ref = s(1:k);
            sval_err = norm(s_sketch - s_ref) / norm(s_ref);
            sval_errors(idx, jdx) = sval_err;

            fprintf('  Rank:         k = %d\n', length(s_sketch));
            fprintf('  Error:        %.12e\n', err);
            fprintf('  SVal error:   %.12e\n', sval_err);
            fprintf('  Time:         %.6fs\n', t_total);

            % Sanity check
            assert(length(s_sketch) == k, sprintf('Expected rank %d, got %d', k, length(s_sketch)));
        end
    end

    % Print matrix info before summary
    fprintf('\n======================================================================\n');
    if use_random_matrix
        fprintf('Matrix type: RANDOM\n');
        fprintf('Singular values: from SVD of randn(m,n)\n');
    else
        fprintf('Matrix type: STRUCTURED\n');
        fprintf('Singular values: logspace(0,-2,k) + logspace(-2,-10,n-k)\n');
    end
    fprintf('Matrix: %dx%d, target rank: %d\n', m, n, k);
    fprintf('Condition number: %.2e, Spectral gap: %.2fx\n', cond_number, spectral_gap_k);

    % Print summary table for reconstruction error
    fprintf('\n======================================================================\n');
    fprintf('SUMMARY: Reconstruction error (Frobenius norm)\n');
    fprintf('======================================================================\n');
    fprintf('extra_samples |');
    for power_iter = power_iter_list
        fprintf('  iter=%d  |', power_iter);
    end
    fprintf('\n');
    fprintf('--------------+');
    for jdx = 1:length(power_iter_list)
        fprintf('----------+');
    end
    fprintf('\n');
    for idx = 1:length(extra_samples_list)
        fprintf('%13d |', extra_samples_list(idx));
        for jdx = 1:length(power_iter_list)
            fprintf(' %.2e |', errors(idx, jdx));
        end
        fprintf('\n');
    end

    % Print summary table for singular value accuracy
    fprintf('\n======================================================================\n');
    fprintf('SUMMARY: Singular value error (relative 2-norm)\n');
    fprintf('======================================================================\n');
    fprintf('extra_samples |');
    for power_iter = power_iter_list
        fprintf('  iter=%d  |', power_iter);
    end
    fprintf('\n');
    fprintf('--------------+');
    for jdx = 1:length(power_iter_list)
        fprintf('----------+');
    end
    fprintf('\n');
    for idx = 1:length(extra_samples_list)
        fprintf('%13d |', extra_samples_list(idx));
        for jdx = 1:length(power_iter_list)
            fprintf(' %.2e |', sval_errors(idx, jdx));
        end
        fprintf('\n');
    end

    fprintf('\n[PASS] Test 1 complete\n');
end
