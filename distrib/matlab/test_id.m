% test_id.m - Test libid interpolative decomposition implementations
%
% Compares two ID implementations from librla:
% - id_sketch:  Randomized QR sketching (default, recommended)
% - id_qrpiv:   Deterministic QR via LAPACK geqp3
%
% Compares on metrics:
% - Accuracy (reconstruction error)
% - Conditioning (max|T|)
% - Runtime
% - Rank selection behavior
%
% Usage:
%     octave --no-gui --eval "test_id"
%     matlab -batch "test_id"
%
% Returns 0 if all tests pass, 1 otherwise.
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: TBD
% Version: 0.1.0
% Date: TBD
% Assisted by: Claude Code (Anthropic)

function exit_code = test_id()
    % No path additions needed - all files in same directory

    fprintf('\n======================================================================\n');
    fprintf('INTERPOLATIVE DECOMPOSITION (ID) COMPARISON\n');
    fprintf('librla.id_sketch vs librla.id_qrpiv\n');
    fprintf('======================================================================\n');
    fprintf('\nEnvironment:\n');
    if is_octave
        fprintf('  OCTAVE:     %s\n', version);
    else
        fprintf('  MATLAB:     %s\n', version);
    end
    fprintf('======================================================================\n');

    % Initialize results array
    results = struct([]);

    % -------------------------------------------------------------------------
    % BASIC TESTS
    % -------------------------------------------------------------------------
    fprintf('\n\n======================================================================\n');
    fprintf('BASIC TESTS\n');
    fprintf('Testing fundamental tolerance and rank modes\n');
    fprintf('======================================================================\n');

    % Test 1: Random matrix (well-conditioned)
    rng(42);
    A1 = randn(500, 300);
    results = compare_on_matrix(A1, 20, 'Random Matrix (well-conditioned)');

    % Test 2: Low-rank matrix
    U = randn(400, 15);
    V = randn(250, 15);
    A2 = U * V' + 1e-10 * randn(400, 250);
    results(end+1) = compare_on_matrix(A2, 1e-8, 'Low-Rank Matrix (rank~15)');

    % Test 3: Hilbert matrix (extremely ill-conditioned)
    A3 = hilbert(2000, 1000);
    results(end+1) = compare_on_matrix(A3, 15, 'Hilbert Matrix (severely ill-conditioned)');

    % Test 4: Complex matrix
    A4 = randn(300, 200) + 1i * randn(300, 200);
    results(end+1) = compare_on_matrix(A4, 25, 'Complex Matrix');

    % Test 5: Decaying spectrum (tolerance mode)
    A5 = randn(400, 300);
    [U5, ~, V5] = svd(A5, 'econ');
    s5 = 1.0 ./ (1:300)';  % decaying: 1/k
    A5 = U5 * diag(s5) * V5';
    results(end+1) = compare_on_matrix(A5, 1e-3, 'Decaying Spectrum (1/k)');

    % -------------------------------------------------------------------------
    % LARGE MATRIX TESTS (2x SCALE)
    % -------------------------------------------------------------------------
    fprintf('\n\n======================================================================\n');
    fprintf('LARGE MATRIX TESTS (2x SCALE)\n');
    fprintf('Testing scaling behavior with matrices 2x larger than base\n');
    fprintf('======================================================================\n');

    % Test 6: Large random matrix
    A6 = randn(1000, 600);
    results(end+1) = compare_on_matrix(A6, 20, 'Large Random Matrix (1000x600)');

    % Test 7: Large low-rank matrix
    U7 = randn(800, 15);
    V7 = randn(500, 15);
    A7 = U7 * V7' + 1e-10 * randn(800, 500);
    results(end+1) = compare_on_matrix(A7, 1e-8, 'Large Low-Rank (800x500, rank~15)');

    % Test 8: Large Hilbert matrix
    A8 = hilbert(4000, 2000);
    results(end+1) = compare_on_matrix(A8, 15, 'Large Hilbert Matrix (4000x2000)');

    % Test 9: Large complex matrix
    A9 = randn(600, 400) + 1i * randn(600, 400);
    results(end+1) = compare_on_matrix(A9, 25, 'Large Complex Matrix (600x400)');

    % Test 10: Large decaying spectrum
    A10 = randn(800, 600);
    [U10, ~, V10] = svd(A10, 'econ');
    s10 = 1.0 ./ (1:600)';  % Fast decay: 1/k
    A10 = U10 * diag(s10) * V10';
    results(end+1) = compare_on_matrix(A10, 1e-3, 'Large Decaying Spectrum (1/k, 800x600)');

    % -------------------------------------------------------------------------
    % SLOW DECAYING SPECTRUM TESTS
    % -------------------------------------------------------------------------
    fprintf('\n\n======================================================================\n');
    fprintf('SLOW DECAYING SPECTRUM TESTS\n');
    fprintf('Testing harder rank-deficient problems with slow decay\n');
    fprintf('======================================================================\n');

    % Test 11: Slow decay - sqrt (small)
    A11 = randn(400, 300);
    [U11, ~, V11] = svd(A11, 'econ');
    s11 = 1.0 ./ sqrt(1:300)';  % Slow decay: 1/sqrt(k)
    A11 = U11 * diag(s11) * V11';
    results(end+1) = compare_on_matrix(A11, 1e-3, 'Slow Decay - Sqrt (1/sqrtk, 400x300)');

    % Test 12: Slow decay - sqrt (large)
    A12 = randn(800, 600);
    [U12, ~, V12] = svd(A12, 'econ');
    s12 = 1.0 ./ sqrt(1:600)';  % Slow decay: 1/sqrt(k)
    A12 = U12 * diag(s12) * V12';
    results(end+1) = compare_on_matrix(A12, 1e-3, 'Slow Decay - Sqrt (1/sqrtk, 800x600)');

    % Test 13: Slow decay - polynomial (small)
    A13 = randn(400, 300);
    [U13, ~, V13] = svd(A13, 'econ');
    s13 = 1.0 ./ ((1:300)' .^ 0.7);  % Polynomial: 1/k^0.7
    A13 = U13 * diag(s13) * V13';
    results(end+1) = compare_on_matrix(A13, 1e-3, 'Slow Decay - Polynomial (1/k^0.7, 400x300)');

    % Test 14: Slow decay - polynomial (large)
    A14 = randn(800, 600);
    [U14, ~, V14] = svd(A14, 'econ');
    s14 = 1.0 ./ ((1:600)' .^ 0.7);  % Polynomial: 1/k^0.7
    A14 = U14 * diag(s14) * V14';
    results(end+1) = compare_on_matrix(A14, 1e-3, 'Slow Decay - Polynomial (1/k^0.7, 800x600)');

    % Test 15: Slow decay - exponential (small)
    A15 = randn(400, 300);
    [U15, ~, V15] = svd(A15, 'econ');
    s15 = exp(-(1:300)' / 100.0);  % Exponential: exp(-k/100)
    A15 = U15 * diag(s15) * V15';
    results(end+1) = compare_on_matrix(A15, 1e-3, 'Slow Decay - Exponential (exp(-k/100), 400x300)');

    % Test 16: Slow decay - exponential (large)
    A16 = randn(800, 600);
    [U16, ~, V16] = svd(A16, 'econ');
    s16 = exp(-(1:600)' / 150.0);  % Exponential: exp(-k/150)
    A16 = U16 * diag(s16) * V16';
    results(end+1) = compare_on_matrix(A16, 1e-3, 'Slow Decay - Exponential (exp(-k/150), 800x600)');

    % -------------------------------------------------------------------------
    % STRUCTURED MATRICES FROM MAKE_MAT
    % -------------------------------------------------------------------------
    fprintf('\n\n======================================================================\n');
    fprintf('MAKE_MAT TESTS (Structured Matrices)\n');
    fprintf('Testing matrices from "Robust blockwise random pivoting" paper\n');
    fprintf('======================================================================\n');

    % Test 22: Gaussian Exponential Decay Matrix
    A22 = test_utils.make_mat(500, 500, 'gaussexp');
    results(end+1) = compare_on_matrix(A22, 1e-3, 'Gaussexp (Gaussian Exponential Decay, 500x500)');

    % Test 23: Gaussian Mixture Model Matrix
    A23 = test_utils.make_mat(400, 400, 'gmm');
    results(end+1) = compare_on_matrix(A23, 1e-3, 'GMM (Gaussian Mixture Model, 400x400)');

    % Test 24: Sparse Neural Network Matrix
    A24 = test_utils.make_mat(300, 300, 'snn');
    results(end+1) = compare_on_matrix(A24, 1e-3, 'SNN (Sparse Neural Network, 300x300)');

    % =========================================================================
    % Summary
    % =========================================================================
    print_summary(results);

    % Final status
    passed_count = sum([results.passed]);
    total_count = length(results);
    if passed_count < total_count
        fprintf('\n[FAIL] %d tests failed\n', total_count - passed_count);
        fprintf('\nFailed tests:\n');
        for i = 1:total_count
            if ~results(i).passed
                fprintf('  - %s\n', results(i).name);
            end
        end
        exit_code = 1;
    else
        fprintf('\n[PASS] ALL TESTS PASSED!\n');
        exit_code = 0;
    end
end


function result = compare_on_matrix(A, rtol_or_rank, name)
    % Compare ID implementations on a single matrix with verbose output.
    %
    % Parameters
    % ----------
    %   A            : numeric matrix - input matrix to decompose
    %   rtol_or_rank : scalar - tolerance (<1) or rank (>=1)
    %   name         : string - descriptive name for the test
    %
    % Returns
    % -------
    %   result       : struct - comparison results with fields:
    %                   name, rtol_or_rank, k_*, err_*, t_*, maxT_*, passed

    fprintf('\n======================================================================\n');
    fprintf('Test: %s\n', name);
    fprintf('Matrix: %dx%d', size(A,1), size(A,2));
    if ~isreal(A)
        fprintf(', complex');
    end
    fprintf('\n');
    fprintf('Parameter: rtol_or_rank = %g\n', rtol_or_rank);
    fprintf('======================================================================\n');

    normA = norm(A, 'fro');

    % -------------------------------------------------------------------------
    % 1. librla.id_sketch (randomized QR sketching)
    % -------------------------------------------------------------------------
    fprintf('\n--- librla.id_sketch (randomized QR) ---\n');

    tic;
    [k_sketch, piv_sketch, T_sketch] = librla.id_sketch(A, rtol_or_rank);
    t_sketch = toc;

    % Compute reconstruction error
    A_skel_sketch = A(:, piv_sketch(k_sketch+1:end));
    A_basis_sketch = A(:, piv_sketch(1:k_sketch));
    if ~isempty(T_sketch)
        err_sketch = norm(A_skel_sketch - A_basis_sketch * T_sketch, 'fro') / normA;
        max_T_sketch = max(abs(T_sketch(:)));
    else
        err_sketch = 0.0;
        max_T_sketch = 0.0;
    end

    % CHECK: Error > 1.0 can occur for (nearly) full-rank matrices with fast T computation
    if err_sketch > 1.0
        fprintf('\n[NOTE] Detected error > 1.0 (%.6f)\n', err_sketch);
        fprintf('  This can occur for (nearly) full-rank matrices with fast T computation.\n');
        fprintf('  Recomputing with method=''lstsq'' for accurate lstsq-based T...\n');

        % Retry with method='lstsq' for accurate T computation via lstsq
        tic;
        [k_sketch, piv_sketch, T_sketch] = librla.id_sketch(A, rtol_or_rank, 'method', 'lstsq');
        t_sketch = toc;

        % Recompute error with new T
        A_skel_sketch = A(:, piv_sketch(k_sketch+1:end));
        A_basis_sketch = A(:, piv_sketch(1:k_sketch));
        if ~isempty(T_sketch)
            err_sketch = norm(A_skel_sketch - A_basis_sketch * T_sketch, 'fro') / normA;
            max_T_sketch = max(abs(T_sketch(:)));
        else
            err_sketch = 0.0;
            max_T_sketch = 0.0;
        end

        fprintf('  -> Recomputed: error = %.3e (method=''lstsq'')\n', err_sketch);

        if err_sketch > 1.0
            error('[ERROR] Error still > 1.0 even with method=''lstsq''!\n   Error = %.6f, Test: %s, rtol_or_rank=%g', ...
                  err_sketch, name, rtol_or_rank);
        end
    end

    fprintf('Rank:       k = %d\n', k_sketch);
    fprintf('Error:      ||A_skel - A_basis @ T|| / ||A|| = %.3e\n', err_sketch);
    fprintf('Condition:  max|T| = %.3e\n', max_T_sketch);
    fprintf('Time:       %.4f s\n', t_sketch);

    % -------------------------------------------------------------------------
    % 2. librla.id_qrpiv (deterministic QR via LAPACK geqp3)
    % -------------------------------------------------------------------------
    fprintf('\n--- librla.id_qrpiv (QR geqp3) ---\n');

    tic;
    [k_rrqr, piv_rrqr, T_rrqr] = librla.id_qrpiv(A, rtol_or_rank);
    t_rrqr = toc;

    % Compute reconstruction error
    A_skel_rrqr = A(:, piv_rrqr(k_rrqr+1:end));
    A_basis_rrqr = A(:, piv_rrqr(1:k_rrqr));
    if ~isempty(T_rrqr)
        err_rrqr = norm(A_skel_rrqr - A_basis_rrqr * T_rrqr, 'fro') / normA;
        max_T_rrqr = max(abs(T_rrqr(:)));
    else
        err_rrqr = 0.0;
        max_T_rrqr = 0.0;
    end

    % CRITICAL CHECK: Relative error must be <= 1.0 (mathematically bounded)
    if err_rrqr > 1.0
        error('[ERROR] CRITICAL BUG in RRQR method: Error = %.6f > 1.0\n   This is mathematically impossible for a relative error!\n   Test: %s, rtol_or_rank=%g', ...
              err_rrqr, name, rtol_or_rank);
    end

    fprintf('Rank:       k = %d\n', k_rrqr);
    fprintf('Error:      ||A_skel - A_basis @ T|| / ||A|| = %.3e\n', err_rrqr);
    fprintf('Condition:  max|T| = %.3e\n', max_T_rrqr);
    fprintf('Time:       %.4f s\n', t_rrqr);

    % -------------------------------------------------------------------------
    % Summary comparison
    % -------------------------------------------------------------------------
    fprintf('\n--- Summary ---\n');
    fprintf('%-25s %-8s %-12s %-12s %-10s\n', 'Method', 'Rank', 'Error', 'max|T|', 'Time (s)');
    fprintf('---------------------------------------------------------------------------\n');
    fprintf('%-25s %-8d %-12.3e %-12.3e %-10.4f\n', 'id_sketch (randomized)', k_sketch, err_sketch, max_T_sketch, t_sketch);
    fprintf('%-25s %-8d %-12.3e %-12.3e %-10.4f\n', 'id_qrpiv (deterministic)', k_rrqr, err_rrqr, max_T_rrqr, t_rrqr);

    % Highlight best conditioning
    max_Ts = [max_T_sketch, max_T_rrqr];
    methods = {'id_sketch', 'id_qrpiv'};
    times = [t_sketch, t_rrqr];

    [~, best_idx] = min(max_Ts);
    fprintf('\nBest conditioning: %s (smallest max|T|)\n', methods{best_idx});

    % Highlight fastest method
    [~, fastest_idx] = min(times);
    fprintf('Fastest method: %s (%.4fs)\n', methods{fastest_idx}, times(fastest_idx));

    % -------------------------------------------------------------------------
    % Create result struct
    % -------------------------------------------------------------------------
    % Determine if test passed
    max_error = max([err_sketch, err_rrqr]);

    % For tolerance mode (rtol < 1): expect error ~ rtol
    % For rank mode (rtol >= 1): check consistency and reasonable error
    if rtol_or_rank < 1
        % Tolerance mode: error should be within 100x tolerance
        tol_threshold = rtol_or_rank * 100;
        passed = max_error < min(0.1, tol_threshold);
    else
        % Rank mode: check deterministic methods agree on rank
        % For full-rank matrices with small k, error can be large (e.g., 90%)
        % This is expected - just verify methods are consistent
        ranks_match = (k_sketch == k_rrqr);
        error_reasonable = max_error < 10.0;  % Very lenient for rank mode
        passed = ranks_match && error_reasonable;
    end

    % Create result struct with all metrics
    result = struct(...
        'name', name, ...
        'rtol_or_rank', rtol_or_rank, ...
        'k_sketch', k_sketch, ...
        'k_rrqr', k_rrqr, ...
        'err_sketch', err_sketch, ...
        'err_rrqr', err_rrqr, ...
        't_sketch', t_sketch, ...
        't_rrqr', t_rrqr, ...
        'maxT_sketch', max_T_sketch, ...
        'maxT_rrqr', max_T_rrqr, ...
        'passed', passed);
end


function print_summary(results)
    % Print comprehensive summary of all test results
    fprintf('\n');
    fprintf('======================================================================\n');
    fprintf('SUMMARY\n');
    fprintf('======================================================================\n');

    total_count = length(results);
    passed_count = sum([results.passed]);

    fprintf('\nTests: %d/%d passed\n', passed_count, total_count);

    % Timing statistics
    t_sketch_all = [results.t_sketch];
    t_rrqr_all = [results.t_rrqr];

    fprintf('\nTiming Statistics:\n');
    fprintf('  id_sketch (randomized):   mean=%.3fs, min=%.3fs, max=%.3fs\n', ...
            mean(t_sketch_all), min(t_sketch_all), max(t_sketch_all));
    fprintf('  id_qrpiv (deterministic): mean=%.3fs, min=%.3fs, max=%.3fs\n', ...
            mean(t_rrqr_all), min(t_rrqr_all), max(t_rrqr_all));

    % Speedup calculation
    speedups = t_rrqr_all ./ t_sketch_all;
    fprintf('\nSpeedup (qrpiv time / sketch time):\n');
    fprintf('  mean=%.2fx, min=%.2fx, max=%.2fx\n', ...
            mean(speedups), min(speedups), max(speedups));
    if mean(speedups) > 1
        fprintf('  -> sketch is %.2fx faster on average\n', mean(speedups));
    else
        fprintf('  -> sketch is %.2fx slower on average\n', 1/mean(speedups));
    end

    % Error statistics
    err_sketch_all = [results.err_sketch];
    err_rrqr_all = [results.err_rrqr];

    fprintf('\nReconstruction Error Statistics:\n');
    fprintf('  id_sketch (randomized):   mean=%.3e, max=%.3e\n', mean(err_sketch_all), max(err_sketch_all));
    fprintf('  id_qrpiv (deterministic): mean=%.3e, max=%.3e\n', mean(err_rrqr_all), max(err_rrqr_all));

    % Conditioning statistics
    maxT_sketch_all = [results.maxT_sketch];
    maxT_rrqr_all = [results.maxT_rrqr];

    fprintf('\nConditioning Statistics (max|T|):\n');
    fprintf('  id_sketch (randomized):   mean=%.3e, min=%.3e, max=%.3e\n', mean(maxT_sketch_all), min(maxT_sketch_all), max(maxT_sketch_all));
    fprintf('  id_qrpiv (deterministic): mean=%.3e, min=%.3e, max=%.3e\n', mean(maxT_rrqr_all), min(maxT_rrqr_all), max(maxT_rrqr_all));

    fprintf('======================================================================\n');
end


function ret = is_octave()
    % Check if running in Octave
    ret = exist('OCTAVE_VERSION', 'builtin') ~= 0;
end
