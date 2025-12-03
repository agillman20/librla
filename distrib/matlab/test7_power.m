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
%       test7_power
%
%   Tests:
%       Test 1: Power iteration in svd_sketch
%           - Tests extra_samples: 24, 18, 12, 6, 3
%           - Tests power_iter: 0-6
%           - Measures reconstruction error and singular value accuracy
%
%   Author : Power iteration tests (librla version)
%   SPDX-License-Identifier : TBD
%==========================================================================

function test7_power()
    fprintf('======================================================================\n');
    fprintf('POWER ITERATION IN SVD_SKETCH TESTS (librla)\n');
    fprintf('======================================================================\n');

    rng(42); % For reproducibility

    % Test 1: Power iteration in svd_sketch
    test_svd_sketch_power_iter();

    fprintf('\n======================================================================\n');
    fprintf('ALL TESTS PASSED [PASS]\n');
    fprintf('======================================================================\n');
end


function test_svd_sketch_power_iter()
    % Test 1: Power iteration in svd_sketch
    fprintf('\n======================================================================\n');
    fprintf('TEST 1: Power iteration in svd_sketch\n');
    fprintf('======================================================================\n');

    % Test matrix with prescribed singular values
    m = 350;
    n = 200;
    k = 40;

    U_full = orth(randn(m, m));
    V_full = orth(randn(n, n));
    s = logspace(0, -6, n)';
    % Use first n columns of U to match dimensions
    U = U_full(:, 1:n);
    V = V_full;
    A = U * diag(s) * V';

    fprintf('\nMatrix: %dx%d, target rank: %d\n', m, n, k);
    fprintf('True singular values: s[1]=%.12e, s[%d]=%.12e\n', s(1), k+1, s(k+1));

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
