% test_svd.m - Test librla SVD implementations
%
% Compares svd_sketch (randomized) vs svd (deterministic):
% - Accuracy (reconstruction error)
% - Singular value accuracy
% - Orthonormality of U and V
% - Runtime
%
% Usage:
%     octave --no-gui --eval "test_svd"
%     matlab -batch "test_svd"
%
% Returns 0 if all tests pass, 1 otherwise.
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: NIST-PD
% Version: 1.0.0
% Date: January 5, 2026
% Assisted by: Claude Code (Anthropic)

function exit_code = test_svd()

    fprintf('\n======================================================================\n');
    fprintf('SVD COMPARISON\n');
    fprintf('svd_sketch (randomized) vs svd (deterministic)\n');
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
    results = run_test_case(A1, 20, 'Random Matrix (well-conditioned)');

    % Test 2: Low-rank matrix
    U = randn(400, 15);
    V = randn(250, 15);
    A2 = U * V' + 1e-10 * randn(400, 250);
    results(end+1) = run_test_case(A2, 1e-8, 'Low-Rank Matrix (rank~15)');

    % Test 3: Hilbert matrix (extremely ill-conditioned)
    A3 = demo_utils.hilbert(2000, 1000);
    results(end+1) = run_test_case(A3, 15, 'Hilbert Matrix (severely ill-conditioned)');

    % Test 4: Complex matrix
    A4 = randn(300, 200) + 1i * randn(300, 200);
    results(end+1) = run_test_case(A4, 25, 'Complex Matrix');

    % Test 5: Decaying spectrum (tolerance mode)
    A5 = randn(400, 300);
    [U5, ~, V5] = svd(A5, 'econ');
    s5 = 1.0 ./ (1:300)';
    A5 = U5 * diag(s5) * V5';
    results(end+1) = run_test_case(A5, 1e-3, 'Decaying Spectrum (1/k)');

    % -------------------------------------------------------------------------
    % LARGE MATRIX TESTS (2x SCALE)
    % -------------------------------------------------------------------------
    fprintf('\n\n======================================================================\n');
    fprintf('LARGE MATRIX TESTS (2x SCALE)\n');
    fprintf('Testing scaling behavior with matrices 2x larger than base\n');
    fprintf('======================================================================\n');

    % Test 6: Large random matrix
    A6 = randn(1000, 600);
    results(end+1) = run_test_case(A6, 20, 'Large Random Matrix (1000x600)');

    % Test 7: Large low-rank matrix
    U7 = randn(800, 15);
    V7 = randn(500, 15);
    A7 = U7 * V7' + 1e-10 * randn(800, 500);
    results(end+1) = run_test_case(A7, 1e-8, 'Large Low-Rank (800x500, rank~15)');

    % Test 8: Large Hilbert matrix
    A8 = demo_utils.hilbert(4000, 2000);
    results(end+1) = run_test_case(A8, 15, 'Large Hilbert Matrix (4000x2000)');

    % Test 9: Large complex matrix
    A9 = randn(600, 400) + 1i * randn(600, 400);
    results(end+1) = run_test_case(A9, 25, 'Large Complex Matrix (600x400)');

    % Test 10: Large decaying spectrum
    A10 = randn(800, 600);
    [U10, ~, V10] = svd(A10, 'econ');
    s10 = 1.0 ./ (1:600)';
    A10 = U10 * diag(s10) * V10';
    results(end+1) = run_test_case(A10, 1e-3, 'Large Decaying Spectrum (1/k, 800x600)');

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
    s11 = 1.0 ./ sqrt(1:300)';
    A11 = U11 * diag(s11) * V11';
    results(end+1) = run_test_case(A11, 1e-3, 'Slow Decay - Sqrt (1/sqrtk, 400x300)');

    % Test 12: Slow decay - sqrt (large)
    A12 = randn(800, 600);
    [U12, ~, V12] = svd(A12, 'econ');
    s12 = 1.0 ./ sqrt(1:600)';
    A12 = U12 * diag(s12) * V12';
    results(end+1) = run_test_case(A12, 1e-3, 'Slow Decay - Sqrt (1/sqrtk, 800x600)');

    % Test 13: Slow decay - polynomial (small)
    A13 = randn(400, 300);
    [U13, ~, V13] = svd(A13, 'econ');
    s13 = 1.0 ./ ((1:300)' .^ 0.7);
    A13 = U13 * diag(s13) * V13';
    results(end+1) = run_test_case(A13, 1e-3, 'Slow Decay - Polynomial (1/k^0.7, 400x300)');

    % Test 14: Slow decay - polynomial (large)
    A14 = randn(800, 600);
    [U14, ~, V14] = svd(A14, 'econ');
    s14 = 1.0 ./ ((1:600)' .^ 0.7);
    A14 = U14 * diag(s14) * V14';
    results(end+1) = run_test_case(A14, 1e-3, 'Slow Decay - Polynomial (1/k^0.7, 800x600)');

    % Test 15: Slow decay - exponential (small)
    A15 = randn(400, 300);
    [U15, ~, V15] = svd(A15, 'econ');
    s15 = exp(-(1:300)' / 100.0);
    A15 = U15 * diag(s15) * V15';
    results(end+1) = run_test_case(A15, 1e-3, 'Slow Decay - Exponential (exp(-k/100), 400x300)');

    % Test 16: Slow decay - exponential (large)
    A16 = randn(800, 600);
    [U16, ~, V16] = svd(A16, 'econ');
    s16 = exp(-(1:600)' / 150.0);
    A16 = U16 * diag(s16) * V16';
    results(end+1) = run_test_case(A16, 1e-3, 'Slow Decay - Exponential (exp(-k/150), 800x600)');

    % -------------------------------------------------------------------------
    % STRUCTURED MATRICES FROM MAKE_MAT
    % -------------------------------------------------------------------------
    fprintf('\n\n======================================================================\n');
    fprintf('MAKE_MAT TESTS (Structured Matrices)\n');
    fprintf('Testing matrices from "Robust blockwise random pivoting" paper\n');
    fprintf('======================================================================\n');

    % Test 22: Gaussian Exponential Decay Matrix
    A22 = test_utils.make_mat(500, 500, 'gaussexp');
    results(end+1) = run_test_case(A22, 1e-3, 'Gaussexp (Gaussian Exponential Decay, 500x500)');

    % Test 23: Gaussian Mixture Model Matrix
    A23 = test_utils.make_mat(400, 400, 'gmm');
    results(end+1) = run_test_case(A23, 1e-3, 'GMM (Gaussian Mixture Model, 400x400)');

    % Test 24: Sparse Neural Network Matrix
    A24 = test_utils.make_mat(300, 300, 'snn');
    results(end+1) = run_test_case(A24, 1e-3, 'SNN (Sparse Neural Network, 300x300)');

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


function result = run_test_case(A, rtol_or_rank, name)
    % Compare SVD implementations on a single matrix.

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
    % 1. svd_sketch (randomized)
    % -------------------------------------------------------------------------
    fprintf('\n--- svd_sketch (randomized) ---\n');

    tic;
    [U_sketch, s_sketch, V_sketch] = librla.svd_sketch(A, rtol_or_rank);
    t_sketch = toc;

    k_sketch = length(s_sketch);

    % Reconstruction error (V is NOT transposed in MATLAB convention)
    A_recon_sketch = U_sketch * diag(s_sketch) * V_sketch';
    err_sketch = norm(A - A_recon_sketch, 'fro') / normA;

    % Orthonormality checks
    orth_U_sketch = norm(U_sketch' * U_sketch - eye(k_sketch), 'fro');
    orth_V_sketch = norm(V_sketch' * V_sketch - eye(k_sketch), 'fro');

    fprintf('Rank:       k = %d\n', k_sketch);
    fprintf('Error:      ||A - U @ S @ V''|| / ||A|| = %.3e\n', err_sketch);
    fprintf('Orth U:     ||U''U - I|| = %.3e\n', orth_U_sketch);
    fprintf('Orth V:     ||V''V - I|| = %.3e\n', orth_V_sketch);
    fprintf('Time:       %.4f s\n', t_sketch);

    % -------------------------------------------------------------------------
    % 2. svd (deterministic, truncated)
    % -------------------------------------------------------------------------
    fprintf('\n--- svd (deterministic) ---\n');

    tic;
    [U_ref_full, S_ref_full, V_ref_full] = svd(A, 'econ');
    t_ref = toc;

    s_ref_full = diag(S_ref_full);

    % Determine reference rank (same as sketch for fair comparison)
    if rtol_or_rank >= 1
        k_ref = floor(rtol_or_rank);
    else
        k_ref = k_sketch;
    end

    % Truncate to target rank
    U_ref = U_ref_full(:, 1:k_ref);
    s_ref = s_ref_full(1:k_ref);
    V_ref = V_ref_full(:, 1:k_ref);

    % Reconstruction error
    A_recon_ref = U_ref * diag(s_ref) * V_ref';
    err_ref = norm(A - A_recon_ref, 'fro') / normA;

    fprintf('Rank:       k = %d\n', k_ref);
    fprintf('Error:      ||A - U @ S @ V''|| / ||A|| = %.3e\n', err_ref);
    fprintf('Time:       %.4f s (full SVD)\n', t_ref);

    % -------------------------------------------------------------------------
    % Singular value accuracy
    % -------------------------------------------------------------------------
    k_cmp = min(k_sketch, length(s_ref_full));
    if k_cmp > 0
        sval_err = norm(s_sketch(1:k_cmp) - s_ref_full(1:k_cmp)) / norm(s_ref_full(1:k_cmp));
    else
        sval_err = 0.0;
    end

    fprintf('\nSingular value accuracy: ||s_sketch - s_ref|| / ||s_ref|| = %.3e\n', sval_err);

    % -------------------------------------------------------------------------
    % Summary comparison
    % -------------------------------------------------------------------------
    fprintf('\n--- Summary ---\n');
    fprintf('%-28s %-8s %-12s %-12s %-10s\n', 'Method', 'Rank', 'Recon Err', 'SVal Err', 'Time (s)');
    fprintf('---------------------------------------------------------------------------\n');
    fprintf('%-28s %-8d %-12.3e %-12.3e %-10.4f\n', 'svd_sketch (randomized)', k_sketch, err_sketch, sval_err, t_sketch);
    fprintf('%-28s %-8d %-12.3e %-12s %-10.4f\n', 'svd (deterministic)', k_ref, err_ref, '(ref)', t_ref);

    % Highlight fastest method
    methods = {'svd_sketch', 'svd'};
    times = [t_sketch, t_ref];
    [~, fastest_idx] = min(times);
    fprintf('\nFastest method: %s (%.4fs)\n', methods{fastest_idx}, times(fastest_idx));

    % Speedup
    if t_sketch > 0
        speedup = t_ref / t_sketch;
        if speedup > 1
            fprintf('Speedup: svd_sketch is %.1fx faster\n', speedup);
        else
            fprintf('Speedup: svd is %.1fx faster\n', 1/speedup);
        end
    end

    % -------------------------------------------------------------------------
    % Determine if test passed
    % -------------------------------------------------------------------------
    if rtol_or_rank < 1
        tol_threshold = rtol_or_rank * 100;
        passed = err_sketch < min(0.1, tol_threshold) && orth_U_sketch < 1e-10 && orth_V_sketch < 1e-10;
    else
        % Rank mode: sketch error should be within 4x of optimal
        if err_ref == 0
            error_ratio_ok = true;
        else
            error_ratio_ok = (err_sketch / max(err_ref, 1e-15)) < 4.0;
        end
        passed = error_ratio_ok && sval_err < 0.5 && orth_U_sketch < 1e-10 && orth_V_sketch < 1e-10;
    end

    % Create result struct
    result = struct(...
        'name', name, ...
        'rtol_or_rank', rtol_or_rank, ...
        'k_sketch', k_sketch, ...
        'k_ref', k_ref, ...
        'err_sketch', err_sketch, ...
        'err_ref', err_ref, ...
        'sval_err', sval_err, ...
        'orth_U_sketch', orth_U_sketch, ...
        'orth_V_sketch', orth_V_sketch, ...
        't_sketch', t_sketch, ...
        't_ref', t_ref, ...
        'passed', passed);
end


function print_summary(results)
    fprintf('\n');
    fprintf('================================================================================\n');
    fprintf('TEST SUMMARY - %d tests completed\n', length(results));
    fprintf('================================================================================\n');

    passed_count = sum([results.passed]);
    total_count = length(results);
    pass_rate = 100.0 * passed_count / total_count;

    fprintf('\nPass Rate: %d/%d (%.1f%%)\n', passed_count, total_count, pass_rate);

    if passed_count == total_count
        fprintf('[PASS] All tests PASSED\n');
    else
        fprintf('[WARNING] Some tests FAILED\n');
        for i = 1:total_count
            if ~results(i).passed
                fprintf('  [FAIL] %s\n', results(i).name);
            end
        end
    end

    % Performance summary
    fprintf('\n================================================================================\n');
    fprintf('Performance Summary\n');
    fprintf('================================================================================\n');

    avg_time_sketch = mean([results.t_sketch]);
    avg_time_ref = mean([results.t_ref]);

    fprintf('\n%-28s %-12s %-15s\n', 'Method', 'Avg Time', 'vs SVD');
    fprintf('--------------------------------------------------------------------------------\n');
    fprintf('%-28s %8.4fs    %6.1fx\n', 'svd_sketch (randomized)', avg_time_sketch, avg_time_ref/avg_time_sketch);
    fprintf('%-28s %8.4fs    %6.1fx -\n', 'svd (deterministic)', avg_time_ref, 1.0);

    % Accuracy summary
    fprintf('\nReconstruction Error Summary:\n');
    fprintf('--------------------------------------------------------------------------------\n');
    avg_err_sketch = mean([results.err_sketch]);
    max_err_sketch = max([results.err_sketch]);
    fprintf('  svd_sketch:    mean=%.3e, max=%.3e\n', avg_err_sketch, max_err_sketch);

    % Singular value accuracy
    fprintf('\nSingular Value Accuracy (vs reference):\n');
    fprintf('--------------------------------------------------------------------------------\n');
    avg_sval_err = mean([results.sval_err]);
    max_sval_err = max([results.sval_err]);
    fprintf('  svd_sketch:    mean=%.3e, max=%.3e\n', avg_sval_err, max_sval_err);

    % Orthonormality summary
    fprintf('\nOrthonormality Summary:\n');
    fprintf('--------------------------------------------------------------------------------\n');
    max_orth_U = max([results.orth_U_sketch]);
    max_orth_V = max([results.orth_V_sketch]);
    fprintf('  max ||U''U - I||:    %.3e\n', max_orth_U);
    fprintf('  max ||V''V - I||:    %.3e\n', max_orth_V);

    fprintf('\n================================================================================\n');
end


function ret = is_octave()
    ret = exist('OCTAVE_VERSION', 'builtin') ~= 0;
end
