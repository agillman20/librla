%==========================================================================
% demo04_power.m - Power Iteration for Improved Accuracy
%
% This demo shows how power iteration improves sketching accuracy.
%
% Power iteration applies (A'*A)^q to the random sketch, which amplifies
% the dominant singular components. This is especially helpful when:
%   - The spectral gap is small
%   - High accuracy is needed
%   - The matrix has slowly decaying singular values
%
% The demo tests a 2D grid of parameters:
%   - extra_samples: How much oversampling (more = better accuracy)
%   - power_iter: Number of power iterations (more = better accuracy)
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: NIST-PD
% Assisted by: Claude Code (Anthropic)
%
% Trade-off: More iterations/samples = better accuracy but more computation.
%
% Try changing the CONFIGURATION parameters below to experiment!
%==========================================================================

function demo04_power()

    %======================================================================
    % CONFIGURATION - Modify these to experiment
    %======================================================================

    MATRIX_SIZE = [500, 300];   % [rows, columns]
    TARGET_RANK = 50;           % Number of singular values to compute
    RANDOM_SEED = 42;           % For reproducibility

    % Power iteration settings - test grid of values
    EXTRA_SAMPLES_LIST = [24, 18, 12, 6, 3];  % Oversampling values to test
    POWER_ITER_LIST = [0, 1, 2, 3, 4];  % Power iteration counts to test

    % Matrix type: 'structured' (clear spectral gap) or 'random' (no gap)
    MATRIX_TYPE = 'structured';

    %======================================================================
    % Demo code below
    %======================================================================

    if ~isempty(RANDOM_SEED)
        rng(RANDOM_SEED);
    end

    m = MATRIX_SIZE(1);
    n = MATRIX_SIZE(2);
    k = TARGET_RANK;

    demo_utils.print_header('Demo 04: Power Iteration');
    fprintf('\nMatrix: %d x %d\n', m, n);
    fprintf('Target rank: %d\n', k);

    %----------------------------------------------------------------------
    % Create test matrix
    %----------------------------------------------------------------------
    if strcmp(MATRIX_TYPE, 'structured')
        fprintf('Matrix type: STRUCTURED\n');
        fprintf('Singular values: logspace(0,-2,k) + logspace(-2,-10,n-k)\n');
        % Create matrix with decaying spectrum and clear gap at rank k
        [U_full, ~] = qr(randn(m, m), 0);
        [V_full, ~] = qr(randn(n, n), 0);
        s_true = [logspace(0, -2, k), logspace(-2, -10, n-k)]';
        U = U_full(:, 1:n);
        A = U * diag(s_true) * V_full';
    else
        fprintf('Matrix type: RANDOM (no spectral gap)\n');
        A = randn(m, n);
        s_true = svd(A);
    end

    % Matrix properties
    cnd = s_true(1) / s_true(end);
    gap = s_true(k) / s_true(k+1);

    fprintf('\nSpectral properties:\n');
    fprintf('   s(1)     = %.6e (largest)\n', s_true(1));
    fprintf('   s(%d)   = %.6e (at target rank)\n', k, s_true(k));
    fprintf('   s(%d)   = %.6e (first neglected)\n', k+1, s_true(k+1));
    fprintf('   s(%d) = %.6e (smallest)\n', n, s_true(n));
    fprintf('   Condition number: %.2e\n', cnd);
    fprintf('   Spectral gap at k=%d: %.1fx\n', k, gap);

    %----------------------------------------------------------------------
    % Test grid of extra_samples and power_iter values
    %----------------------------------------------------------------------
    demo_utils.print_subheader('Testing Parameter Grid');
    fprintf('   power_iter=0 means no power iteration (baseline)\n');
    fprintf('   Each power iteration costs 2 extra matrix-vector products\n');
    fprintf('   extra_samples controls oversampling (block_size = k + extra_samples)\n');

    % Store results in 2D arrays
    num_extra = length(EXTRA_SAMPLES_LIST);
    num_power = length(POWER_ITER_LIST);
    errors = zeros(num_extra, num_power);
    sval_errors = zeros(num_extra, num_power);
    s_ref = s_true(1:k);

    for idx = 1:num_extra
        extra_samples = EXTRA_SAMPLES_LIST(idx);
        block_size = k + extra_samples;
        fprintf('\n--- extra_samples = %d (block_size = %d) ---\n', extra_samples, block_size);

        for jdx = 1:num_power
            power_iter = POWER_ITER_LIST(jdx);

            tic;
            [U, s, V] = librla.svd_sketch(A, k, 'power_iter', power_iter, ...
                                          'extra_samples', extra_samples);
            elapsed = toc;

            % Reconstruction error
            A_approx = U * diag(s) * V';
            recon_err = norm(A - A_approx, 'fro') / norm(A, 'fro');
            errors(idx, jdx) = recon_err;

            % Singular value accuracy
            sval_err = norm(s - s_ref) / norm(s_ref);
            sval_errors(idx, jdx) = sval_err;

            fprintf('   power_iter=%d: err=%.2e, sval_err=%.2e, time=%.4fs\n', ...
                    power_iter, recon_err, sval_err, elapsed);
        end
    end

    %----------------------------------------------------------------------
    % Summary tables
    %----------------------------------------------------------------------
    demo_utils.print_subheader('Summary: Reconstruction Error');

    % Header row
    header = 'extra_samples |';
    for jdx = 1:num_power
        header = [header, sprintf('  iter=%d  |', POWER_ITER_LIST(jdx))];
    end
    fprintf('%s\n', header);
    fprintf('%s+%s\n', repmat('-', 1, 14), repmat([repmat('-', 1, 10), '+'], 1, num_power));

    % Data rows
    for idx = 1:num_extra
        row = sprintf('%13d |', EXTRA_SAMPLES_LIST(idx));
        for jdx = 1:num_power
            row = [row, sprintf(' %.2e |', errors(idx, jdx))];
        end
        fprintf('%s\n', row);
    end

    demo_utils.print_subheader('Summary: Singular Value Error');

    % Header row
    header = 'extra_samples |';
    for jdx = 1:num_power
        header = [header, sprintf('  iter=%d  |', POWER_ITER_LIST(jdx))];
    end
    fprintf('%s\n', header);
    fprintf('%s+%s\n', repmat('-', 1, 14), repmat([repmat('-', 1, 10), '+'], 1, num_power));

    % Data rows
    for idx = 1:num_extra
        row = sprintf('%13d |', EXTRA_SAMPLES_LIST(idx));
        for jdx = 1:num_power
            row = [row, sprintf(' %.2e |', sval_errors(idx, jdx))];
        end
        fprintf('%s\n', row);
    end

    fprintf('\nNotes:\n');
    fprintf('  - Power iteration amplifies dominant singular components\n');
    fprintf('  - Extra samples (oversampling) improves subspace capture\n');

end
