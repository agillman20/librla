%==========================================================================
% compare_svd - Compare SVD implementations
%
%   Comprehensive comparison of two SVD implementations:
%   - libid.svd_sketch:          Randomized SVD via sketching
%   - svd (LAPACK):              Deterministic full SVD (truncated)
%
%   Compares on metrics:
%   - Accuracy (reconstruction error)
%   - Singular value accuracy
%   - Runtime
%
%   Author: Port from compare_svd.py
%==========================================================================

function compare_svd()
    % No path additions needed - all files in same directory

    % Use fast SVD driver (gesdd is faster than default gesvd)
    if is_octave
        svd_driver('gesdd');
        fprintf('Using fast SVD driver: gesdd\n');
    end

    fprintf('======================================================================\n');
    fprintf('SVD COMPARISON\n');
    fprintf('libid.svd_sketch vs svd (LAPACK)\n');
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
    % Test 1: Random matrix (well-conditioned)
    % -------------------------------------------------------------------------
    rng(42);  % Set seed for reproducibility
    A1 = randn(400, 250);
    results = compare_on_matrix(A1, 1e-8, 'Random Matrix (well-conditioned)');

    % -------------------------------------------------------------------------
    % Test 2: Low-rank matrix
    % -------------------------------------------------------------------------
    U = randn(400, 15);
    V = randn(250, 15);
    A2 = U * V' + 1e-10 * randn(400, 250);
    results(end+1) = compare_on_matrix(A2, 1e-8, 'Low-Rank Matrix (rank~15)');

    % -------------------------------------------------------------------------
    % Test 3: Hilbert matrix (severely ill-conditioned)
    % -------------------------------------------------------------------------
    A3 = libid.hilb(2000, 1000);
    results(end+1) = compare_on_matrix(A3, 15, 'Hilbert Matrix (severely ill-conditioned)');

    % -------------------------------------------------------------------------
    % Test 4: Complex matrix
    % -------------------------------------------------------------------------
    A4 = randn(300, 200) + 1i*randn(300, 200);
    results(end+1) = compare_on_matrix(A4, 1e-8, 'Complex Matrix');

    % -------------------------------------------------------------------------
    % Test 5: Power-law decay (slow decay)
    % -------------------------------------------------------------------------
    rank = 50;
    s_decay = 1.0 ./ sqrt(1:rank)';  % s_k ~ 1/sqrt(k)
    [U5, ~] = qr(randn(300, rank), 0);
    [V5, ~] = qr(randn(200, rank), 0);
    A5 = U5 * diag(s_decay) * V5';
    results(end+1) = compare_on_matrix(A5, 1e-6, 'Power-Law Decay (slow)');

    % -------------------------------------------------------------------------
    % Test 6: Rank mode test (fixed rank=20)
    % -------------------------------------------------------------------------
    A6 = randn(300, 200);
    results(end+1) = compare_on_matrix(A6, 20, 'Rank Mode Test (k=20)');

    % -------------------------------------------------------------------------
    % Test 7: Large low-rank matrix
    % -------------------------------------------------------------------------
    U7 = randn(800, 15);
    V7 = randn(500, 15);
    A7 = U7 * V7' + 1e-10 * randn(800, 500);
    results(end+1) = compare_on_matrix(A7, 1e-8, 'Large Low-Rank Matrix (800x500, rank~15)');

    % -------------------------------------------------------------------------
    % Test 8: Large Hilbert matrix
    % -------------------------------------------------------------------------
    A8 = libid.hilb(4000, 2000);
    results(end+1) = compare_on_matrix(A8, 15, 'Large Hilbert Matrix (4000x2000)');

    % -------------------------------------------------------------------------
    % Test 9: Large complex matrix
    % -------------------------------------------------------------------------
    A9 = randn(600, 400) + 1i*randn(600, 400);
    results(end+1) = compare_on_matrix(A9, 1e-8, 'Large Complex Matrix');

    % -------------------------------------------------------------------------
    % Test 10-12: Structured matrices
    % -------------------------------------------------------------------------
    for mat_type = {'gmm', 'gaussexp', 'snn'}
        A_struct = make_mat(400, 250, mat_type{1});
        results(end+1) = compare_on_matrix(A_struct, 1e-8, sprintf('Structured: %s', mat_type{1}));
    end

    % -------------------------------------------------------------------------
    % Test 13-14: Wide matrices
    % -------------------------------------------------------------------------
    A13 = randn(200, 500);
    results(end+1) = compare_on_matrix(A13, 1e-8, 'Wide Random Matrix (200x500)');

    U14 = randn(200, 15);
    V14 = randn(500, 15);
    A14 = U14 * V14' + 1e-10 * randn(200, 500);
    results(end+1) = compare_on_matrix(A14, 1e-8, 'Wide Low-Rank Matrix (200x500, rank~15)');

    % -------------------------------------------------------------------------
    % Test 15: XL Random matrix
    % -------------------------------------------------------------------------
    A15 = randn(1200, 800);
    results(end+1) = compare_on_matrix(A15, 1e-8, 'XL Random Matrix (1200x800)');

    % -------------------------------------------------------------------------
    % Test 16: XL Low-rank matrix
    % -------------------------------------------------------------------------
    U16 = randn(1600, 15);
    V16 = randn(1000, 15);
    A16 = U16 * V16' + 1e-10 * randn(1600, 1000);
    results(end+1) = compare_on_matrix(A16, 1e-8, 'XL Low-Rank Matrix (1600x1000, rank~15)');

    % -------------------------------------------------------------------------
    % Test 17: XL Hilbert matrix
    % -------------------------------------------------------------------------
    A17 = libid.hilb(8000, 4000);
    results(end+1) = compare_on_matrix(A17, 15, 'XL Hilbert Matrix (8000x4000)');

    % -------------------------------------------------------------------------
    % Test 18: XL Complex matrix
    % -------------------------------------------------------------------------
    A18 = randn(1200, 800) + 1i*randn(1200, 800);
    results(end+1) = compare_on_matrix(A18, 1e-8, 'XL Complex Matrix (1200x800)');

    % =========================================================================
    % Power Iteration Tests (Rank Mode)
    % =========================================================================
    fprintf('\n======================================================================\n');
    fprintf('POWER ITERATION TESTS (Rank Mode)\n');
    fprintf('======================================================================\n');

    % Test matrix for power iteration tests
    A_power = randn(400, 300);
    target_rank = 30;

    % Test 19: Power iteration = 0 (no power iteration)
    results(end+1) = compare_on_matrix(A_power, target_rank, ...
                                       sprintf('Rank Mode k=%d, power=0', target_rank), 0);

    % Test 20: Power iteration = 1
    results(end+1) = compare_on_matrix(A_power, target_rank, ...
                                       sprintf('Rank Mode k=%d, power=1', target_rank), 1);

    % Test 21: Power iteration = 2
    results(end+1) = compare_on_matrix(A_power, target_rank, ...
                                       sprintf('Rank Mode k=%d, power=2', target_rank), 2);

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
    else
        fprintf('\n[PASS] ALL TESTS PASSED!\n');
    end
end


function result = compare_on_matrix(A, rtol_or_rank, name, flagPower)
    % Compare SVD implementations on a single matrix.
    %
    % Parameters:
    %   A              : Input matrix (real or complex)
    %   rtol_or_rank   : Tolerance (< 1) or target rank (>= 1)
    %   name           : Descriptive test name
    %   flagPower      : Power iteration count (optional, default=0)
    %
    % Returns:
    %   result         : Struct with comparison metrics

    if nargin < 4
        flagPower = 0;
    end

    fprintf('\nTesting: %s\n', name);
    [m, n] = size(A);
    if flagPower > 0
        fprintf('  Matrix shape: (%d, %d), rtol_or_rank=%g, flagPower=%d\n', m, n, rtol_or_rank, flagPower);
    else
        fprintf('  Matrix shape: (%d, %d), rtol_or_rank=%g\n', m, n, rtol_or_rank);
    end

    is_rank_mode = (rtol_or_rank >= 1);
    normA = norm(A, 'fro');

    % Reference: Full SVD for singular value comparison
    [~, s_ref_mat, ~] = svd(A, 0);  % Economy SVD
    s_ref = diag(s_ref_mat);  % Extract singular values as vector

    % -------------------------------------------------------------------------
    % Method 1: svd_sketch (randomized)
    % -------------------------------------------------------------------------
    tic;
    [U1, s1, V1] = libid.svd_sketch(A, rtol_or_rank, 42, flagPower);
    t_sketch = toc;
    k_sketch = length(s1);

    % Reconstruction error (V' needed per MATLAB svd convention)
    A1_recon = U1 * diag(s1) * V1';
    err_sketch = norm(A - A1_recon, 'fro') / normA;

    % Singular value accuracy
    s1_ref = s_ref(1:k_sketch);
    sval_err_sketch = norm(s1 - s1_ref) / norm(s1_ref);

    % -------------------------------------------------------------------------
    % Method 2: svd (LAPACK, deterministic)
    % -------------------------------------------------------------------------
    tic;
    [U2, s2_diag, V2] = svd(A, 0);
    s2 = diag(s2_diag);
    t_lapack = toc;

    % Truncate based on tolerance or rank
    if is_rank_mode
        k2 = min(floor(rtol_or_rank), length(s2));
    else
        k2 = sum(s2 >= rtol_or_rank * s2(1));
        if k2 == 0, k2 = 1; end  % At least one singular value
    end
    k_lapack = k2;

    % Reconstruct with truncation
    U2_k = U2(:, 1:k2);
    s2_k = s2(1:k2);
    V2_k = V2(:, 1:k2);
    A2_recon = U2_k * diag(s2_k) * V2_k';
    err_lapack = norm(A - A2_recon, 'fro') / normA;

    % Singular value accuracy
    sval_err_lapack = norm(s2_k - s_ref(1:k2)) / norm(s_ref(1:k2));

    % -------------------------------------------------------------------------
    % Display results
    % -------------------------------------------------------------------------
    fprintf('  Ranks:  sketch=%d, lapack=%d\n', k_sketch, k_lapack);
    fprintf('  Errors: sketch=%.3e, lapack=%.3e\n', err_sketch, err_lapack);
    fprintf('  Times:  sketch=%.3fs, lapack=%.3fs\n', t_sketch, t_lapack);
    fprintf('  SVal:   sketch=%.3e, lapack=%.3e\n', sval_err_sketch, sval_err_lapack);

    % -------------------------------------------------------------------------
    % Validation
    % -------------------------------------------------------------------------
    passed = true;

    % Check reconstruction errors are reasonable
    if err_sketch > 1.0 || err_lapack > 1.0
        fprintf('  [ERROR] Reconstruction error > 1.0 (larger than input norm)\n');
        passed = false;
    end

    % Check sketch is close to lapack
    rel_err = abs(err_sketch - err_lapack) / max(err_lapack, 1e-15);
    if rel_err > 0.1  % 10% tolerance
        fprintf('  [WARNING] sketch error differs from lapack by %.1f%%\n', rel_err * 100);
    end

    % -------------------------------------------------------------------------
    % Build result struct
    % -------------------------------------------------------------------------
    result = struct(...
        'name', name, ...
        'rtol_or_rank', rtol_or_rank, ...
        'k_sketch', k_sketch, ...
        'k_lapack', k_lapack, ...
        'err_sketch', err_sketch, ...
        'err_lapack', err_lapack, ...
        't_sketch', t_sketch, ...
        't_lapack', t_lapack, ...
        'sval_err_sketch', sval_err_sketch, ...
        'sval_err_lapack', sval_err_lapack, ...
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
    t_lapack_all = [results.t_lapack];

    fprintf('\nTiming Statistics:\n');
    fprintf('  svd_sketch:  mean=%.3fs, min=%.3fs, max=%.3fs\n', ...
            mean(t_sketch_all), min(t_sketch_all), max(t_sketch_all));
    fprintf('  lapack.svd:  mean=%.3fs, min=%.3fs, max=%.3fs\n', ...
            mean(t_lapack_all), min(t_lapack_all), max(t_lapack_all));

    % Error statistics
    err_sketch_all = [results.err_sketch];
    err_lapack_all = [results.err_lapack];

    fprintf('\nReconstruction Error Statistics:\n');
    fprintf('  svd_sketch:  mean=%.3e, max=%.3e\n', mean(err_sketch_all), max(err_sketch_all));
    fprintf('  lapack.svd:   mean=%.3e, max=%.3e\n', mean(err_lapack_all), max(err_lapack_all));

    % Singular value error statistics
    sval_err_sketch_all = [results.sval_err_sketch];
    sval_err_lapack_all = [results.sval_err_lapack];

    fprintf('\nSingular Value Error Statistics:\n');
    fprintf('  svd_sketch:  mean=%.3e, max=%.3e\n', mean(sval_err_sketch_all), max(sval_err_sketch_all));
    fprintf('  lapack.svd:   mean=%.3e, max=%.3e\n', mean(sval_err_lapack_all), max(sval_err_lapack_all));

    fprintf('======================================================================\n');
end


function ret = is_octave()
    % Check if running in Octave
    ret = exist('OCTAVE_VERSION', 'builtin') ~= 0;
end
