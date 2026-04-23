%==========================================================================
% demo02_svd.m - Truncated SVD via Randomized Sketching
%
% This demo shows how to compute truncated SVD using librla:
%   - svd_sketch: Randomized truncated SVD
%   - qr_sketch:  Truncated QR factorization
%
% Both functions use randomized sketching for efficiency on large matrices.
%
% The SVD factorizes A as: A = U * diag(s) * V'
% The QR factorizes A as: A(:, p) ≈ Q * R
%
% Try changing the CONFIGURATION parameters below to experiment!
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: NIST-PD
% Version: 1.0.1
% Date: April 22, 2026
% Assisted by: Claude Code (Anthropic)
%==========================================================================

function demo02_svd()
% % add next level up directory to search path
    addpath(genpath('..'));

    %======================================================================
    % CONFIGURATION - Modify these to experiment
    %======================================================================

    MATRIX_SIZE = [1000, 2000];   % [rows, columns]
    TARGET_RANK = 30;           % Number of singular values to compute
    RANDOM_SEED = 42;           % For reproducibility

    %======================================================================
    % Demo code below
    %======================================================================

    if ~isempty(RANDOM_SEED)
        rng(RANDOM_SEED);
    end

    m = MATRIX_SIZE(1);
    n = MATRIX_SIZE(2);
    k = TARGET_RANK;

    demo_utils.print_header('Demo 02: Truncated SVD and QR');
    fprintf('\nMatrix: %d x %d Hilbert matrix\n', m, n);
    fprintf('Target rank: %d\n', k);

    % Generate test matrix
    A = demo_utils.hilbert(m, n);
    normA = norm(A, 'fro');

    % Compute reference SVD for comparison
    s_true = svd(A);
    fprintf('\nTrue singular values:\n');
    fprintf('   s(1) = %.6e\n', s_true(1));
    fprintf('   s(%d) = %.6e\n', k, s_true(k));
    fprintf('   s(%d) = %.6e\n', k+1, s_true(k+1));
    fprintf('   s(%d) = %.6e\n', min(m,n), s_true(min(m,n)));

    %----------------------------------------------------------------------
    % Method 1: svd_sketch
    %----------------------------------------------------------------------
    demo_utils.print_subheader('1. svd_sketch (truncated SVD)');
    fprintf('   Returns U, s, V where A = U * diag(s) * V''\n');

    tic;
    [U, s, V] = librla.svd_sketch(A, k);
    elapsed = toc;

    err = demo_utils.svd_error(A, U, s, V);
    k_out = length(s);

    fprintf('   Rank:      %d\n', k_out);
    fprintf('   Error:     %.6e\n', err);
    fprintf('   Time:      %.4f s\n', elapsed);

    % Compare singular values
    s_err = norm(s - s_true(1:k_out)) / norm(s_true(1:k_out));
    fprintf('   SVal err:  %.6e (relative)\n', s_err);

    % Check orthogonality
    orth_U = norm(U' * U - eye(k_out), 'fro');
    orth_V = norm(V' * V - eye(k_out), 'fro');
    fprintf('   ||U''U-I||: %.2e\n', orth_U);
    fprintf('   ||V''V-I||: %.2e\n', orth_V);

    %----------------------------------------------------------------------
    % Method 2: qr_sketch
    %----------------------------------------------------------------------
    demo_utils.print_subheader('2. qr_sketch (truncated QR)');
    fprintf('   Returns Q, R, piv where A(:, piv) = Q * R\n');

    tic;
    [Q, R, piv] = librla.qr_sketch(A, k);
    elapsed2 = toc;

    % Reconstruct: A(:, piv) = Q * R
    A_qr = zeros(size(A));
    A_qr(:, piv) = Q * R;
    err2 = norm(A - A_qr, 'fro') / normA;
    k2 = size(Q, 2);

    fprintf('   Rank:      %d\n', k2);
    fprintf('   Error:     %.6e\n', err2);
    fprintf('   Time:      %.4f s\n', elapsed2);

    % Check orthogonality
    orth_Q = norm(Q' * Q - eye(k2), 'fro');
    fprintf('   ||Q''Q-I||: %.2e\n', orth_Q);

    %----------------------------------------------------------------------
    % Summary
    %----------------------------------------------------------------------
    demo_utils.print_subheader('Summary');
    fprintf('   %-14s %6s %12s %10s\n', 'Method', 'Rank', 'Error', 'Time');
    fprintf('   %s %s %s %s\n', repmat('-',1,14), repmat('-',1,6), repmat('-',1,12), repmat('-',1,10));
    fprintf('   %-14s %6d %12.3e %9.4fs\n', 'svd_sketch', k_out, err, elapsed);
    fprintf('   %-14s %6d %12.3e %9.4fs\n', 'qr_sketch', k2, err2, elapsed2);

    fprintf('\nNotes:\n');
    fprintf('  - svd_sketch gives U, s, V for best rank-k approximation\n');
    fprintf('  - qr_sketch gives Q, R factorization with column pivoting\n');

end
