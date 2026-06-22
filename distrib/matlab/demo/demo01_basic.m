%==========================================================================
% demo01_basic.m - Introduction to Interpolative Decomposition
%
% This demo introduces the two core ID algorithms:
%   - id_sketch: Randomized ID using QR sketching (fast, approximate)
%   - id_qrpiv:  Deterministic ID using column-pivoted QR (exact, slower)
%
% The ID factorizes a matrix A as:
%   A(:, piv(k+1:end)) = A(:, piv(1:k)) * T
%
% where piv(1:k) selects the "skeleton" columns and T is the interpolation matrix.
%
% Try changing the CONFIGURATION parameters below to experiment!
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: MIT
% Version: 1.0.2
% Date: June 22, 2026
% Assisted by: Claude Code (Anthropic)
%==========================================================================

function demo01_basic()
% % add next level up directory to search path
    addpath(genpath('..'));

    %======================================================================
    % CONFIGURATION - Modify these to experiment
    %======================================================================

    MATRIX_SIZE = [1000, 2000];   % [rows, columns]
    TARGET_RANK = 15;           % Number of skeleton columns to select
    RANDOM_SEED = 42;           % For reproducibility (set to [] for random)

    % Matrix type: 'hilbert' or 'kahan'
    MATRIX_TYPE = 'hilbert';
    KAHAN_THETA = 1.2;          % Kahan matrix parameter (only used if MATRIX_TYPE='kahan')

    %======================================================================
    % Demo code below
    %======================================================================

    if ~isempty(RANDOM_SEED)
        rng(RANDOM_SEED);
    end

    m = MATRIX_SIZE(1);
    n = MATRIX_SIZE(2);
    k = TARGET_RANK;

    demo_utils.print_header('Demo 01: Basic Interpolative Decomposition');
    fprintf('\nMatrix: %d x %d\n', m, n);
    fprintf('Target rank: %d\n', k);

    % Generate test matrix
    if strcmp(MATRIX_TYPE, 'hilbert')
        fprintf('Matrix type: Hilbert (ill-conditioned)\n');
        A = demo_utils.hilbert(m, n);
    else  % kahan
        fprintf('Matrix type: Kahan (theta=%.1f)\n', KAHAN_THETA);
        A = demo_utils.kahan(m, n, KAHAN_THETA);
    end

    normA = norm(A, 'fro');
    fprintf('||A||_F = %.3e\n', normA);

    %----------------------------------------------------------------------
    % Method 1: id_sketch (randomized)
    %----------------------------------------------------------------------
    demo_utils.print_subheader('1. id_sketch (randomized)');
    fprintf('   Uses random projections + QR. Fast but approximate.\n');

    tic;
    [k1, piv1, T1] = librla.id_sketch(A, k);
    elapsed1 = toc;

    err1 = demo_utils.id_error(A, k1, piv1, T1);
    if ~isempty(T1)
        maxT1 = max(abs(T1(:)));
    else
        maxT1 = 0.0;
    end

    fprintf('   Rank:     %d\n', k1);
    fprintf('   Error:    %.3e\n', err1);
    fprintf('   Max |T|:  %.3e\n', maxT1);
    fprintf('   Time:     %.4f s\n', elapsed1);

    %----------------------------------------------------------------------
    % Method 2: id_qrpiv (deterministic)
    %----------------------------------------------------------------------
    demo_utils.print_subheader('2. id_qrpiv (deterministic)');
    fprintf('   Uses LAPACK column-pivoted QR. More accurate but slower.\n');

    tic;
    [k2, piv2, T2] = librla.id_qrpiv(A, k);
    elapsed2 = toc;

    err2 = demo_utils.id_error(A, k2, piv2, T2);
    if ~isempty(T2)
        maxT2 = max(abs(T2(:)));
    else
        maxT2 = 0.0;
    end

    fprintf('   Rank:     %d\n', k2);
    fprintf('   Error:    %.3e\n', err2);
    fprintf('   Max |T|:  %.3e\n', maxT2);
    fprintf('   Time:     %.4f s\n', elapsed2);

    %----------------------------------------------------------------------
    % Summary
    %----------------------------------------------------------------------
    demo_utils.print_subheader('Summary');
    fprintf('   %-12s %6s %12s %12s %10s\n', 'Method', 'Rank', 'Error', 'Max|T|', 'Time');
    fprintf('   %s %s %s %s %s\n', repmat('-',1,12), repmat('-',1,6), repmat('-',1,12), repmat('-',1,12), repmat('-',1,10));
    fprintf('   %-12s %6d %12.3e %12.3e %9.4fs\n', 'id_sketch', k1, err1, maxT1, elapsed1);
    fprintf('   %-12s %6d %12.3e %12.3e %9.4fs\n', 'id_qrpiv', k2, err2, maxT2, elapsed2);

    % Validate
    if err1 < 1.0 && err2 < 1.0
        fprintf('\n   [PASS] Both methods produced valid decompositions.\n');
    else
        fprintf('\n   [FAIL] Error > 1.0 detected!\n');
    end

end
