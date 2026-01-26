%==========================================================================
% demo03_linop.m - Matrix-Free Computation with LinearOperator
%
% This demo shows how to use librla with LinearOperators for matrix-free
% computation. This is essential for large-scale problems where the matrix
% doesn't fit in memory.
%
% Three modes are demonstrated:
%   1. Dense matrix (baseline)
%   2. Explicit LinearOperator (matrix stored, accessed via matvec)
%   3. Matrix-free LinearOperator (only matvec/rmatvec functions provided)
%
% Note: Matrix-free mode only supports rank mode (rtol >= 1).
%
% Try changing the CONFIGURATION parameters below to experiment!
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: NIST-PD
% Assisted by: Claude Code (Anthropic)
%==========================================================================

function demo03_linop()

    %======================================================================
    % CONFIGURATION - Modify these to experiment
    %======================================================================

    MATRIX_SIZE = [300, 200];   % [rows, columns]
    TARGET_RANK = 15;           % Must be >= 1 for matrix-free mode
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

    demo_utils.print_header('Demo 03: Matrix-Free Computation');
    fprintf('\nMatrix: %d x %d Hilbert matrix\n', m, n);
    fprintf('Target rank: %d\n', k);

    % Generate test matrix (we'll wrap it in LinearOperators)
    A = demo_utils.hilbert(m, n);
    normA = norm(A, 'fro');

    %======================================================================
    % Part 1: ID with different input types
    %======================================================================
    demo_utils.print_subheader('Part 1: id_sketch with LinearOperator');

    %----------------------------------------------------------------------
    % Test 1: Dense matrix (baseline)
    %----------------------------------------------------------------------
    fprintf('\n   1a. Dense matrix (baseline)\n');

    tic;
    [k1, piv1, T1] = librla.id_sketch(A, k);
    elapsed1 = toc;

    err1 = demo_utils.id_error(A, k1, piv1, T1);
    fprintf('       Rank: %d, Error: %.3e, Time: %.4fs\n', k1, err1, elapsed1);

    %----------------------------------------------------------------------
    % Test 2: Explicit LinearOperator
    %----------------------------------------------------------------------
    fprintf('\n   1b. Explicit LinearOperator (matrix wrapper)\n');
    fprintf('       Matrix is stored; LinearOperator wraps it.\n');

    matvec_explicit = @(x) A * x;
    rmatvec_explicit = @(x) A' * x;
    A_explicit = LinearOperator(matvec_explicit, rmatvec_explicit, m, n);

    tic;
    [k2, piv2, T2] = librla.id_sketch(A_explicit, k);
    elapsed2 = toc;

    err2 = demo_utils.id_error(A, k2, piv2, T2);
    fprintf('       Rank: %d, Error: %.3e, Time: %.4fs\n', k2, err2, elapsed2);

    % Verify same result as dense
    if k1 == k2 && abs(err1 - err2) < 1e-12
        fprintf('       [OK] Same result as dense matrix\n');
    end

    %----------------------------------------------------------------------
    % Test 3: Matrix-free LinearOperator
    %----------------------------------------------------------------------
    fprintf('\n   1c. Matrix-free LinearOperator\n');
    fprintf('       Only matvec/rmatvec functions provided.\n');
    fprintf('       Requires rank mode (rtol >= 1).\n');

    % Define matvec functions (in real applications, these would compute
    % matrix-vector products without storing the full matrix)
    my_matvec = @(x) A * x;
    my_rmatvec = @(x) A' * x;
    A_matfree = LinearOperator(my_matvec, my_rmatvec, m, n);

    tic;
    [k3, piv3, T3] = librla.id_sketch(A_matfree, k);
    elapsed3 = toc;

    err3 = demo_utils.id_error(A, k3, piv3, T3);
    fprintf('       Rank: %d, Error: %.3e, Time: %.4fs\n', k3, err3, elapsed3);

    %======================================================================
    % Part 2: SVD with LinearOperator
    %======================================================================
    demo_utils.print_subheader('Part 2: svd_sketch with LinearOperator');

    % Dense baseline
    fprintf('\n   2a. Dense matrix\n');
    tic;
    [U1, s1, V1] = librla.svd_sketch(A, k);
    elapsed_svd1 = toc;

    A_approx1 = U1 * diag(s1) * V1';
    err_svd1 = norm(A - A_approx1, 'fro') / normA;
    fprintf('       Rank: %d, Error: %.3e, Time: %.4fs\n', length(s1), err_svd1, elapsed_svd1);

    % Matrix-free
    fprintf('\n   2b. Matrix-free LinearOperator\n');
    tic;
    [U2, s2, V2] = librla.svd_sketch(A_matfree, k);
    elapsed_svd2 = toc;

    A_approx2 = U2 * diag(s2) * V2';
    err_svd2 = norm(A - A_approx2, 'fro') / normA;
    fprintf('       Rank: %d, Error: %.3e, Time: %.4fs\n', length(s2), err_svd2, elapsed_svd2);

    %======================================================================
    % Summary
    %======================================================================
    demo_utils.print_subheader('Summary: id_sketch');
    fprintf('   %-28s %6s %12s %10s\n', 'Input Type', 'Rank', 'Error', 'Time');
    fprintf('   %s %s %s %s\n', repmat('-',1,28), repmat('-',1,6), repmat('-',1,12), repmat('-',1,10));
    fprintf('   %-28s %6d %12.3e %9.4fs\n', 'Dense matrix', k1, err1, elapsed1);
    fprintf('   %-28s %6d %12.3e %9.4fs\n', 'Explicit LinearOperator', k2, err2, elapsed2);
    fprintf('   %-28s %6d %12.3e %9.4fs\n', 'Matrix-free LinearOperator', k3, err3, elapsed3);

    fprintf('\nNotes:\n');
    fprintf('  - LinearOperator allows matrix-free computation\n');
    fprintf('  - Essential for large-scale problems (matrix doesn''t fit in memory)\n');
    fprintf('  - Matrix-free mode requires rank mode (rtol >= 1)\n');
    fprintf('  - Define only matvec(x) = A * x and rmatvec(x) = A'' * x\n');

end
