%==========================================================================
% test2_svd_hilbert.m - Simple test with medium-size Hilbert matrix for SVD
%
%   Tests SVD algorithms (svd_sketch vs standard SVD) on an ill-conditioned
%   Hilbert matrix.
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: TBD
% Version: 1.0.0
% Date: TBD
% Assisted by: Claude Code (Anthropic)
%==========================================================================

function test2_svd_hilbert()
    fprintf('======================================================================\n');
    fprintf('TEST 2: Medium Hilbert Matrix - SVD\n');
    fprintf('======================================================================\n');

    % Create medium-size Hilbert matrix (severely ill-conditioned)
    m = 300;
    n = 200;
    fprintf('\nMatrix size: %d x %d\n', m, n);
    fprintf('Matrix type: Hilbert (severely ill-conditioned)\n');

    A = hilb(m, n);
    normA = norm(A, 'fro');

    % Target rank
    k_target = 15;
    fprintf('Target rank: %d\n', k_target);
    fprintf('======================================================================\n');

    % -------------------------------------------------------------------------
    % Reference: Full SVD for singular value comparison
    % -------------------------------------------------------------------------
    [~, s_ref_mat, ~] = svd(A, 0);  % Economy SVD
    s_ref = diag(s_ref_mat);

    % -------------------------------------------------------------------------
    % Method 1: svd_sketch (randomized)
    % -------------------------------------------------------------------------
    fprintf('\n1. librla.svd_sketch (randomized SVD via sketching)\n');
    fprintf('----------------------------------------------------------------------\n');

    tic;
    [U1, s1, V1] = librla.svd_sketch(A, k_target);
    t1 = toc;

    k1 = length(s1);

    % Reconstruction error (V' needed per MATLAB svd convention)
    A1_recon = U1 * diag(s1) * V1';
    err1 = norm(A - A1_recon, 'fro') / normA;

    % Singular value accuracy
    s1_ref = s_ref(1:k1);
    sval_err1 = norm(s1 - s1_ref) / norm(s1_ref);

    fprintf('  Rank:      k = %d\n', k1);
    fprintf('  Error:     ||A - U @ S @ V''|| / ||A|| = %.3e\n', err1);
    fprintf('  SVal Err:  ||s - s_ref|| / ||s_ref|| = %.3e\n', sval_err1);
    fprintf('  Time:      %.4f s\n', t1);

    % -------------------------------------------------------------------------
    % Method 2: svd (LAPACK, truncated)
    % -------------------------------------------------------------------------
    fprintf('\n2. svd (LAPACK, deterministic, truncated)\n');
    fprintf('----------------------------------------------------------------------\n');

    tic;
    [U2, s2_mat, V2] = svd(A, 0);
    s2_full = diag(s2_mat);
    t2 = toc;

    % Truncate to target rank
    k2 = k_target;
    U2_k = U2(:, 1:k2);
    s2 = s2_full(1:k2);
    V2_k = V2(:, 1:k2);

    % Reconstruction error
    A2_recon = U2_k * diag(s2) * V2_k';
    err2 = norm(A - A2_recon, 'fro') / normA;

    % Singular value accuracy
    sval_err2 = norm(s2 - s_ref(1:k2)) / norm(s_ref(1:k2));

    fprintf('  Rank:      k = %d\n', k2);
    fprintf('  Error:     ||A - U @ S @ V''|| / ||A|| = %.3e\n', err2);
    fprintf('  SVal Err:  ||s - s_ref|| / ||s_ref|| = %.3e\n', sval_err2);
    fprintf('  Time:      %.4f s\n', t2);

    % -------------------------------------------------------------------------
    % Summary
    % -------------------------------------------------------------------------
    fprintf('\n======================================================================\n');
    fprintf('SUMMARY\n');
    fprintf('======================================================================\n');
    fprintf('  Method         Rank    Recon Error   SVal Error    Time\n');
    fprintf('----------------------------------------------------------------------\n');
    fprintf('  svd_sketch     %4d    %.3e    %.3e    %.4fs\n', k1, err1, sval_err1, t1);
    fprintf('  svd (LAPACK)   %4d    %.3e    %.3e    %.4fs\n', k2, err2, sval_err2, t2);
    fprintf('======================================================================\n');

    % Validate
    if err1 > 1.0 || err2 > 1.0
        fprintf('\n[FAIL] Reconstruction error > 1.0 detected!\n');
        return
    end

    if sval_err1 > 1e-6 || sval_err2 > 1e-10
        fprintf('\n[FAIL] Singular value error too large!\n');
        return
    end

    fprintf('\n[PASS] Test completed successfully!\n');
end
