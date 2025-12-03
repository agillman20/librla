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
    fprintf('True singular values: s[0]=%.12e, s[%d]=%.12e\n', s(1), k+1, s(k+1));

    % Test with different power_iter values (extended to 6)
    for power_iter = 0:6
        fprintf('\n--- power_iter = %d ---\n', power_iter);

        % Run svd_sketch
        tic;
        [U_sketch, s_sketch, V_sketch] = librla.svd_sketch(A, k, 'power_iter', power_iter);
        t_total = toc;

        % Compute reconstruction error
        A_approx = U_sketch * diag(s_sketch) * V_sketch';
        err = norm(A - A_approx, 'fro') / norm(A, 'fro');

        % Singular value accuracy
        s_ref = s(1:k);
        sval_err = norm(s_sketch - s_ref) / norm(s_ref);

        fprintf('  Rank:         k = %d\n', length(s_sketch));
        fprintf('  Error:        %.12e\n', err);
        fprintf('  SVal error:   %.12e\n', sval_err);
        fprintf('  Time:         %.6fs\n', t_total);

        % Sanity check
        assert(length(s_sketch) == k, sprintf('Expected rank %d, got %d', k, length(s_sketch)));
        assert(err < 0.1, sprintf('Error %.3e too large', err));
    end

    fprintf('\n[PASS] Test 1 complete\n');
end
