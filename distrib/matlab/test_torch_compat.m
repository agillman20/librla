% test_torch_compat.m - Test PyTorch-compatible wrappers for librla
%
% Tests svd_lowrank and pca_lowrank functions:
% - Reconstruction error
% - Singular value accuracy
% - Orthonormality of U and V
% - M parameter (subtraction matrix)
% - center parameter for PCA
%
% Usage:
%     octave --no-gui --eval "test_torch_compat"
%     matlab -batch "test_torch_compat"
%
% Returns 0 if all tests pass, 1 otherwise.
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: TBD
% Version: 0.1.0
% Date: TBD
% Assisted by: Claude Code (Anthropic)

function exit_code = test_torch_compat()

    fprintf('\n======================================================================\n');
    fprintf('TORCH_COMPAT TESTS\n');
    fprintf('Testing PyTorch-compatible wrappers: svd_lowrank, pca_lowrank\n');
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
    % SVD_LOWRANK TESTS
    % -------------------------------------------------------------------------
    fprintf('\n\n======================================================================\n');
    fprintf('SVD_LOWRANK TESTS\n');
    fprintf('Testing PyTorch-compatible randomized SVD\n');
    fprintf('======================================================================\n');

    rng(42);

    % Test 1: Random matrix (well-conditioned)
    A1 = randn(500, 300);
    results = run_svd_lowrank_test(A1, 20, 'svd_lowrank: Random Matrix', []);

    % Test 2: Low-rank matrix
    U = randn(400, 15);
    V = randn(250, 15);
    A2 = U * V' + 1e-10 * randn(400, 250);
    results(end+1) = run_svd_lowrank_test(A2, 20, 'svd_lowrank: Low-Rank Matrix (rank~15)', []);

    % Test 3: Hilbert matrix (ill-conditioned)
    A3 = hilbert(500, 300);
    results(end+1) = run_svd_lowrank_test(A3, 15, 'svd_lowrank: Hilbert Matrix', []);

    % Test 4: Complex matrix
    A4 = randn(300, 200) + 1i * randn(300, 200);
    results(end+1) = run_svd_lowrank_test(A4, 25, 'svd_lowrank: Complex Matrix', []);

    % Test 5: With M parameter (subtraction)
    A5 = randn(400, 300);
    M5 = randn(400, 300) * 0.1;
    results(end+1) = run_svd_lowrank_test(A5, 20, 'svd_lowrank: With M parameter', M5);

    % Test 6: Tall matrix
    A6 = randn(1000, 100);
    results(end+1) = run_svd_lowrank_test(A6, 30, 'svd_lowrank: Tall Matrix (1000x100)', []);

    % Test 7: Wide matrix
    A7 = randn(100, 1000);
    results(end+1) = run_svd_lowrank_test(A7, 30, 'svd_lowrank: Wide Matrix (100x1000)', []);

    % -------------------------------------------------------------------------
    % PCA_LOWRANK TESTS
    % -------------------------------------------------------------------------
    fprintf('\n\n======================================================================\n');
    fprintf('PCA_LOWRANK TESTS\n');
    fprintf('Testing PyTorch-compatible randomized PCA\n');
    fprintf('======================================================================\n');

    % Test 8: Random data with centering
    A8 = randn(500, 100) + 5.0;  % Add offset to test centering
    results(end+1) = run_pca_lowrank_test(A8, 20, 'pca_lowrank: Centered Random Data', true);

    % Test 9: Without centering
    A9 = randn(500, 100);
    results(end+1) = run_pca_lowrank_test(A9, 20, 'pca_lowrank: Uncentered Random Data', false);

    % Test 10: Low-rank data with centering
    U10 = randn(400, 10);
    V10 = randn(80, 10);
    A10 = U10 * V10' + 3.0 + 1e-8 * randn(400, 80);
    results(end+1) = run_pca_lowrank_test(A10, 15, 'pca_lowrank: Low-Rank Centered', true);

    % Test 11: Default q parameter
    A11 = randn(100, 50);
    results(end+1) = run_pca_lowrank_test(A11, 6, 'pca_lowrank: Default q=6', true);

    % =========================================================================
    % PRINT SUMMARY
    % =========================================================================
    fprintf('\n\n================================================================================\n');
    fprintf('TEST SUMMARY - %d tests completed\n', length(results));
    fprintf('================================================================================\n');

    % Overall pass/fail
    passed_tests = sum([results.passed]);
    total_tests = length(results);
    pass_rate = 100.0 * passed_tests / total_tests;

    fprintf('\n');
    fprintf('Pass Rate: %d/%d (%.1f%%)\n', passed_tests, total_tests, pass_rate);

    if passed_tests == total_tests
        fprintf('[PASS] All tests PASSED\n');
    else
        fprintf('[WARNING] Some tests FAILED\n');
        for i = 1:total_tests
            if ~results(i).passed
                fprintf('  [FAIL] %s\n', results(i).name);
            end
        end
    end

    % Performance summary
    fprintf('\n');
    fprintf('================================================================================\n');
    fprintf('Performance Summary\n');
    fprintf('================================================================================\n');

    avg_time_compat = mean([results.t_compat]);
    avg_time_ref = mean([results.t_ref]);

    fprintf('\n');
    fprintf('%-28s %-12s %-15s\n', 'Method', 'Avg Time', 'vs Reference');
    fprintf('%s\n', repmat('-', 1, 80));
    fprintf('%-28s %8.4fs    %6.1fx \n', 'torch_compat', avg_time_compat, avg_time_ref/avg_time_compat);
    fprintf('%-28s %8.4fs    %6.1fx -\n', 'svd (reference)', avg_time_ref, 1.0);

    % Accuracy summary
    fprintf('\n');
    fprintf('Reconstruction Error Summary:\n');
    fprintf('%s\n', repmat('-', 1, 80));

    avg_recon_err = mean([results.recon_err]);
    max_recon_err = max([results.recon_err]);

    fprintf('  torch_compat:  mean=%.3e, max=%.3e\n', avg_recon_err, max_recon_err);

    % Singular value accuracy summary
    fprintf('\n');
    fprintf('Singular Value Accuracy (vs reference):\n');
    fprintf('%s\n', repmat('-', 1, 80));

    avg_sval_err = mean([results.sval_err]);
    max_sval_err = max([results.sval_err]);

    fprintf('  torch_compat:  mean=%.3e, max=%.3e\n', avg_sval_err, max_sval_err);

    % Orthonormality summary
    fprintf('\n');
    fprintf('Orthonormality Summary:\n');
    fprintf('%s\n', repmat('-', 1, 80));

    max_orth_U = max([results.orth_U]);
    max_orth_V = max([results.orth_V]);

    fprintf('  max ||U''U - I||: %.3e\n', max_orth_U);
    fprintf('  max ||V''V - I||: %.3e\n', max_orth_V);

    fprintf('\n');
    fprintf('================================================================================\n');

    % Return exit code based on pass/fail
    if all([results.passed])
        exit_code = 0;
    else
        exit_code = 1;
    end

end


function result = run_svd_lowrank_test(A, q, name, M)
% Run svd_lowrank test on a single matrix

    fprintf('\n======================================================================\n');
    fprintf('Test: %s\n', name);
    fprintf('Matrix: %dx%d', size(A, 1), size(A, 2));
    if ~isreal(A)
        fprintf(', complex\n');
    else
        fprintf('\n');
    end
    fprintf('Parameter: q = %d\n', q);
    if ~isempty(M)
        fprintf('Using M parameter (subtraction matrix)\n');
    end
    fprintf('======================================================================\n');

    % Matrix to actually decompose
    if ~isempty(M)
        A_eff = A - M;
    else
        A_eff = A;
    end
    normA = norm(A_eff, 'fro');

    % -------------------------------------------------------------------------
    % svd_lowrank (torch_compat)
    % -------------------------------------------------------------------------
    fprintf('\n--- svd_lowrank (torch_compat) ---\n');

    tic;
    [U, s, V] = torch_compat.svd_lowrank(A, q, 2, M);
    t_compat = toc;

    k = length(s);

    % Reconstruction error (V is not transposed in torch_compat)
    A_recon = U * diag(s) * V';
    recon_err = norm(A_eff - A_recon, 'fro') / normA;

    % Orthonormality checks
    orth_U = norm(U' * U - eye(k), 'fro');
    orth_V = norm(V' * V - eye(k), 'fro');

    fprintf('Rank:       k = %d\n', k);
    fprintf('Recon Err:  ||A - U @ S @ V''|| / ||A|| = %.3e\n', recon_err);
    fprintf('Orth U:     ||U''U - I|| = %.3e\n', orth_U);
    fprintf('Orth V:     ||V''V - I|| = %.3e\n', orth_V);
    fprintf('Time:       %.4f s\n', t_compat);

    % -------------------------------------------------------------------------
    % Reference (svd truncated)
    % -------------------------------------------------------------------------
    fprintf('\n--- Reference (svd truncated) ---\n');

    tic;
    [U_ref, S_ref, V_ref] = svd(A_eff, 'econ');
    t_ref = toc;

    % Truncate to same rank
    U_ref = U_ref(:, 1:k);
    s_ref = diag(S_ref);
    s_ref = s_ref(1:k);
    V_ref = V_ref(:, 1:k);

    % Reconstruction error
    A_recon_ref = U_ref * diag(s_ref) * V_ref';
    recon_err_ref = norm(A_eff - A_recon_ref, 'fro') / normA;

    fprintf('Rank:       k = %d\n', k);
    fprintf('Recon Err:  ||A - U @ S @ V''|| / ||A|| = %.3e\n', recon_err_ref);
    fprintf('Time:       %.4f s (full SVD)\n', t_ref);

    % Singular value accuracy
    if norm(s_ref) > 0
        sval_err = norm(s - s_ref) / norm(s_ref);
    else
        sval_err = 0.0;
    end

    fprintf('\nSingular value accuracy: ||s - s_ref|| / ||s_ref|| = %.3e\n', sval_err);

    % -------------------------------------------------------------------------
    % Summary
    % -------------------------------------------------------------------------
    fprintf('\n--- Summary ---\n');
    fprintf('%-28s %-8s %-12s %-12s %-10s\n', 'Method', 'Rank', 'Recon Err', 'SVal Err', 'Time (s)');
    fprintf('%s\n', repmat('-', 1, 75));
    fprintf('%-28s %-8d %-12.3e %-12.3e %-10.4f\n', 'svd_lowrank (torch_compat)', k, recon_err, sval_err, t_compat);
    fprintf('%-28s %-8d %-12.3e %-12s %-10.4f\n', 'svd (reference)', k, recon_err_ref, '(ref)', t_ref);

    % -------------------------------------------------------------------------
    % Determine if test passed
    % -------------------------------------------------------------------------
    if recon_err_ref == 0
        error_ratio_ok = true;
    else
        error_ratio_ok = (recon_err / max(recon_err_ref, 1e-15) < 4.0);
    end
    passed = error_ratio_ok && sval_err < 0.5 && orth_U < 1e-10 && orth_V < 1e-10;

    result = struct('name', name, 'q', q, 'k', k, ...
                    'recon_err', recon_err, 'sval_err', sval_err, ...
                    'orth_U', orth_U, 'orth_V', orth_V, ...
                    't_compat', t_compat, 't_ref', t_ref, ...
                    'passed', passed);
end


function result = run_pca_lowrank_test(A, q, name, center)
% Run pca_lowrank test on a single matrix

    fprintf('\n======================================================================\n');
    fprintf('Test: %s\n', name);
    fprintf('Matrix: %dx%d\n', size(A, 1), size(A, 2));
    fprintf('Parameter: q = %d, center = %d\n', q, center);
    fprintf('======================================================================\n');

    % Centered matrix for comparison
    if center
        A_centered = A - mean(A, 1);
    else
        A_centered = A;
    end
    normA = norm(A_centered, 'fro');

    % -------------------------------------------------------------------------
    % pca_lowrank (torch_compat)
    % -------------------------------------------------------------------------
    fprintf('\n--- pca_lowrank (torch_compat) ---\n');

    tic;
    [U, s, V] = torch_compat.pca_lowrank(A, q, center, 2);
    t_compat = toc;

    k = length(s);

    % Reconstruction error (V is not transposed in torch_compat)
    A_recon = U * diag(s) * V';
    recon_err = norm(A_centered - A_recon, 'fro') / normA;

    % Orthonormality checks
    orth_U = norm(U' * U - eye(k), 'fro');
    orth_V = norm(V' * V - eye(k), 'fro');

    fprintf('Rank:       k = %d\n', k);
    fprintf('Recon Err:  ||A_c - U @ S @ V''|| / ||A_c|| = %.3e\n', recon_err);
    fprintf('Orth U:     ||U''U - I|| = %.3e\n', orth_U);
    fprintf('Orth V:     ||V''V - I|| = %.3e\n', orth_V);
    fprintf('Time:       %.4f s\n', t_compat);

    % -------------------------------------------------------------------------
    % Reference (svd on centered data)
    % -------------------------------------------------------------------------
    fprintf('\n--- Reference (svd on centered data) ---\n');

    tic;
    [U_ref, S_ref, V_ref] = svd(A_centered, 'econ');
    t_ref = toc;

    % Truncate to same rank
    U_ref = U_ref(:, 1:k);
    s_ref = diag(S_ref);
    s_ref = s_ref(1:k);
    V_ref = V_ref(:, 1:k);

    % Reconstruction error
    A_recon_ref = U_ref * diag(s_ref) * V_ref';
    recon_err_ref = norm(A_centered - A_recon_ref, 'fro') / normA;

    fprintf('Rank:       k = %d\n', k);
    fprintf('Recon Err:  ||A_c - U @ S @ V''|| / ||A_c|| = %.3e\n', recon_err_ref);
    fprintf('Time:       %.4f s (full SVD)\n', t_ref);

    % Singular value accuracy
    if norm(s_ref) > 0
        sval_err = norm(s - s_ref) / norm(s_ref);
    else
        sval_err = 0.0;
    end

    fprintf('\nSingular value accuracy: ||s - s_ref|| / ||s_ref|| = %.3e\n', sval_err);

    % -------------------------------------------------------------------------
    % Summary
    % -------------------------------------------------------------------------
    fprintf('\n--- Summary ---\n');
    fprintf('%-28s %-8s %-12s %-12s %-10s\n', 'Method', 'Rank', 'Recon Err', 'SVal Err', 'Time (s)');
    fprintf('%s\n', repmat('-', 1, 75));
    fprintf('%-28s %-8d %-12.3e %-12.3e %-10.4f\n', 'pca_lowrank (torch_compat)', k, recon_err, sval_err, t_compat);
    fprintf('%-28s %-8d %-12.3e %-12s %-10.4f\n', 'svd (reference)', k, recon_err_ref, '(ref)', t_ref);

    % -------------------------------------------------------------------------
    % Determine if test passed
    % -------------------------------------------------------------------------
    if recon_err_ref == 0
        error_ratio_ok = true;
    else
        error_ratio_ok = (recon_err / max(recon_err_ref, 1e-15) < 4.0);
    end
    passed = error_ratio_ok && sval_err < 0.5 && orth_U < 1e-10 && orth_V < 1e-10;

    result = struct('name', name, 'q', q, 'k', k, ...
                    'recon_err', recon_err, 'sval_err', sval_err, ...
                    'orth_U', orth_U, 'orth_V', orth_V, ...
                    't_compat', t_compat, 't_ref', t_ref, ...
                    'passed', passed);
end


function tf = is_octave()
    tf = exist('OCTAVE_VERSION', 'builtin') ~= 0;
end
