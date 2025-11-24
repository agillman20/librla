%==========================================================================
% test6_power - Test power iteration for range estimation (librla version)
%
%   Tests simple power iteration for range estimation in sketching algorithms.
%   Power iteration applies (A^H A)^n to improve sketch quality by amplifying
%   the dominant subspace.
%
%   This version uses librla instead of libid.
%
%   Usage:
%       test6_power           % structured matrix only
%       test6_power('random') % both structured and random
%
%   Tests:
%       Test 1: Range estimation quality
%           - Measures subspace angles to dominant subspace
%           - Tests extra_samples: 24, 18, 12, 6, 3
%           - Tests iterations: 0-6
%           - Can use structured or random matrix
%
%       Test 3: SVD sketch integration
%           - Tests power_iter: 0-6
%           - Measures reconstruction error and singular value accuracy
%
%   Author : Power iteration range estimator tests (librla version)
%   SPDX-License-Identifier : TBD
%==========================================================================

function test6_power(varargin)
    % Parse input
    test_random = false;
    if nargin > 0 && strcmpi(varargin{1}, 'random')
        test_random = true;
    end

    fprintf('======================================================================\n');
    fprintf('POWER ITERATION RANGE ESTIMATION TESTS (librla)\n');
    fprintf('======================================================================\n');

    rng(42); % For reproducibility

    % Test 1: Range estimation quality
    if test_random
        % Run with structured matrix first
        test_range_estimation_quality(false);
        % Then run with random matrix
        test_range_estimation_quality(true);
    else
        % Default: structured matrix only
        test_range_estimation_quality(false);
    end

    % Test 3: SVD sketch integration
    test_svd_sketch_integration();

    fprintf('\n======================================================================\n');
    fprintf('ALL TESTS PASSED [PASS]\n');
    fprintf('======================================================================\n');
end


function test_range_estimation_quality(use_random_matrix)
    % Test 1: Range estimation quality with power iteration
    fprintf('\n======================================================================\n');
    fprintf('TEST 1: Range Estimation Quality\n');
    fprintf('======================================================================\n');

    % Test matrix configuration
    m = 500;
    n = 300;
    k = 50; % True rank we want to capture

    if use_random_matrix
        fprintf('\nMatrix type: RANDOM (no prescribed singular values)\n');
        % Simple random matrix
        A = randn(m, n);

        % Compute SVD to get true dominant subspace
        [~, s_vec, V] = svd(A, 'econ');
        s = diag(s_vec);
        V_true = V(:, 1:k);

        % Note: Random matrices have clustered singular values with minimal
        % spectral gap, so power iteration converges much more slowly than
        % for structured matrices with prescribed decay.
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

        % True dominant subspace (right singular vectors)
        V_true = V(:, 1:k);
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
    fprintf('  Decay rate (s[0]/s[%d]): %.2fx\n', k, decay_rate_k);
    fprintf('\nSingular value distribution:\n');
    fprintf('  s[0]    = %.12e (largest)\n', s(1));
    fprintf('  s[%3d]  = %.12e (target cutoff)\n', k, s(k));
    fprintf('  s[%3d]  = %.12e (first neglected)\n', k+1, s(k+1));
    fprintf('  s[%3d] = %.12e (smallest)\n', n, s(n));

    % Test different extra_samples values
    extra_samples_list = [24, 18, 12, 6, 3];
    for idx = 1:length(extra_samples_list)
        extra_samples = extra_samples_list(idx);
        block_size = k + extra_samples;

        fprintf('\n======================================================================\n');
        fprintf('extra_samples = %d (block_size = %d)\n', extra_samples, block_size);
        fprintf('======================================================================\n');

        % Test different iteration counts (0-6)
        for num_iters = 0:6
            fprintf('\n--- num_iters = %d ---\n', num_iters);

            % Generate same random test matrix
            rng(42);
            if isreal(A)
                X_init = 2 * rand(n, block_size) - 1;
            else
                X_init = 2 * rand(n, block_size) - 1 + 1i * (2 * rand(n, block_size) - 1);
            end

            % Power iteration (manual implementation since librla.power_iteration is private)
            tic;
            X_power = X_init;
            % Orthogonalize X_init when num_iters=0
            if num_iters == 0
                [X_power, ~, ~] = qr(X_power, 0);
            end
            for iter = 1:num_iters
                X_power = A' * (A * X_power);
                [X_power, ~, ~] = qr(X_power, 0);
            end
            t_power = toc;
            angle_power = subspace_angle(V_true, X_power);

            % Orthogonality check
            orth_power = norm(X_power' * X_power - eye(size(X_power, 2)), 'fro');

            % Compute alignment with dominant singular vectors
            M_power = V_true' * X_power;
            svals_power = svd(M_power);
            capture_quality_power = mean(svals_power);

            fprintf('Power iteration:\n');
            fprintf('  Subspace angle:   %18.12fdeg\n', angle_power);
            fprintf('  Capture quality:  %.12f (mean singular value)\n', capture_quality_power);
            fprintf('  Orthogonality:    ||Q^H Q - I||_F = %.12e\n', orth_power);
            fprintf('  Time:             %.6fs\n', t_power);
            fprintf('  Basis size:       %d\n', size(X_power, 2));
        end
    end

    fprintf('\n[PASS] Test 1 complete\n');
end


function test_svd_sketch_integration()
    % Test 3: Power iteration integration with svd_sketch pipeline
    fprintf('\n======================================================================\n');
    fprintf('TEST 3: Integration with svd_sketch\n');
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

    fprintf('\n[PASS] Test 3 complete\n');
end


function angle = subspace_angle(Q1, Q2)
    % Compute maximum principal angle between subspaces
    if size(Q1, 2) == 0 || size(Q2, 2) == 0
        angle = 90.0;
        return;
    end

    % Compute singular values of Q1^H @ Q2
    M = Q1' * Q2;
    s = svd(M);

    % Principal angles: theta_i = arccos(s_i)
    % Maximum angle (worst alignment)
    theta_max = acos(min(max(s(end), 0), 1));
    angle = rad2deg(theta_max);
end
