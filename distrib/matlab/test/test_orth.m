% test_orth.m - Test librla orthonormal basis computation
%
% Tests orth_sketch (randomized orthonormal basis for column space):
% - Column space accuracy (how well Q spans A's column space)
% - Orthonormality of Q
% - diagR values (sorted column norms, conditioning indicator)
% - Runtime
%
% Usage:
%     octave --no-gui --eval "test_orth"
%     matlab -batch "test_orth"
%
% Returns 0 if all tests pass, 1 otherwise.
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: MIT
% Version: 1.0.1
% Date: April 22, 2026
% Assisted by: Claude Code (Anthropic)

function exit_code = test_orth()
% % add next level up directory to search path
addpath(genpath('..'));

fprintf('\n======================================================================\n');
fprintf('ORTH_SKETCH TESTS\n');
fprintf('Testing orthonormal basis computation via randomized sketching\n');
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
% Test orth_sketch on a single matrix.

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
% orth_sketch (randomized)
% -------------------------------------------------------------------------
fprintf('\n--- orth_sketch (randomized) ---\n');

tic;
[Q, flag, diagR] = librla.orth_sketch(A, rtol_or_rank);
t_sketch = toc;

k = size(Q, 2);

% Column space error: how well Q spans A's column space
QQtA = Q * (Q' * A);
span_err = norm(A - QQtA, 'fro') / normA;

% Orthonormality check
if k > 0
    orth_err = norm(Q' * Q - eye(k), 'fro');
else
    orth_err = 0.0;
end

% diagR ratio (conditioning indicator)
abs_diagR = abs(diagR);
if ~isempty(abs_diagR) && abs_diagR(1) ~= 0
    diagR_ratio = abs_diagR(end) / abs_diagR(1);
else
    diagR_ratio = 0.0;
end

fprintf('Rank:       k = %d\n', k);
if flag == 0
    fprintf('Flag:       %d (success)\n', flag);
else
    fprintf('Flag:       %d (early termination)\n', flag);
end
fprintf('Span Err:   ||A - Q*Q''*A|| / ||A|| = %.3e\n', span_err);
fprintf('Orth Err:   ||Q''Q - I|| = %.3e\n', orth_err);
if ~isempty(abs_diagR)
    fprintf('|diagR(1)|: %.3e\n', abs_diagR(1));
    fprintf('|diagR(end)|:%.3e\n', abs_diagR(end));
else
    fprintf('|diagR(1)|: N/A\n');
    fprintf('|diagR(end)|:N/A\n');
end
fprintf('diagR ratio: %.3e\n', diagR_ratio);
fprintf('Time:       %.4f s\n', t_sketch);

% -------------------------------------------------------------------------
% Compare with full SVD (reference)
% -------------------------------------------------------------------------
fprintf('\n--- Reference (full SVD truncated) ---\n');

tic;
[U_ref, ~, ~] = svd(A, 'econ');
t_ref = toc;

% Truncate to same rank
if k > 0
    U_k = U_ref(:, 1:k);
    UUtA = U_k * (U_k' * A);
    span_err_ref = norm(A - UUtA, 'fro') / normA;
else
    span_err_ref = 1.0;
end

fprintf('Rank:       k = %d\n', k);
fprintf('Span Err:   ||A - U_k*U_k''*A|| / ||A|| = %.3e\n', span_err_ref);
fprintf('Time:       %.4f s (full SVD)\n', t_ref);

% -------------------------------------------------------------------------
% Summary comparison
% -------------------------------------------------------------------------
fprintf('\n--- Summary ---\n');
fprintf('%-28s %-8s %-12s %-12s %-10s\n', 'Method', 'Rank', 'Span Err', 'Orth Err', 'Time (s)');
fprintf('---------------------------------------------------------------------------\n');
fprintf('%-28s %-8d %-12.3e %-12.3e %-10.4f\n', 'orth_sketch (randomized)', k, span_err, orth_err, t_sketch);
fprintf('%-28s %-8d %-12.3e %-12s %-10.4f\n', 'SVD (optimal)', k, span_err_ref, '(ref)', t_ref);

% Quality comparison
if span_err_ref > 0
    quality_ratio = span_err / span_err_ref;
    fprintf('\nQuality ratio: orth_sketch error / optimal error = %.2fx\n', quality_ratio);
end

% Speedup
if t_sketch > 0
    speedup = t_ref / t_sketch;
    if speedup > 1
        fprintf('Speedup: orth_sketch is %.1fx faster than full SVD\n', speedup);
    else
        fprintf('Speedup: full SVD is %.1fx faster\n', 1/speedup);
    end
end

% -------------------------------------------------------------------------
% Determine if test passed
% -------------------------------------------------------------------------
% Orthonormality should be near machine precision (or 0 if k=0)
% Early termination (flag=1) is OK - it's expected for some matrices
% Key criterion: span error should be close to optimal (SVD reference)

if k == 0
    orth_ok = true;
else
    orth_ok = orth_err < 1e-10;
end

% Quality threshold: randomized methods typically achieve within 8x of optimal
% (slightly relaxed to account for randomness in ill-conditioned cases)
quality_threshold = 8.0;

if rtol_or_rank < 1
    % Tolerance mode: span error should be within threshold of optimal
    if span_err_ref == 0
        quality_ok = span_err < 1e-10;
    else
        quality_ok = (span_err / max(span_err_ref, 1e-15)) < quality_threshold;
    end
    passed = quality_ok && orth_ok;
else
    % Rank mode: span error should be within threshold of optimal
    if span_err_ref == 0
        passed = orth_ok;
    else
        passed = ((span_err / max(span_err_ref, 1e-15)) < quality_threshold) && orth_ok;
    end
end

% Create result struct
result = struct(...
    'name', name, ...
    'rtol_or_rank', rtol_or_rank, ...
    'k', k, ...
    'span_err', span_err, ...
    'orth_err', orth_err, ...
    'diagR_ratio', diagR_ratio, ...
    'flag', flag, ...
    't_sketch', t_sketch, ...
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

avg_time = mean([results.t_sketch]);
min_time = min([results.t_sketch]);
max_time = max([results.t_sketch]);

fprintf('\north_sketch timing: mean=%.4fs, min=%.4fs, max=%.4fs\n', avg_time, min_time, max_time);

% Accuracy summary
fprintf('\nColumn Space Error Summary (||A - Q*Q''*A|| / ||A||):\n');
fprintf('--------------------------------------------------------------------------------\n');
avg_span_err = mean([results.span_err]);
max_span_err = max([results.span_err]);
fprintf('  orth_sketch:    mean=%.3e, max=%.3e\n', avg_span_err, max_span_err);

% Orthonormality summary
fprintf('\nOrthonormality Summary (||Q''Q - I||):\n');
fprintf('--------------------------------------------------------------------------------\n');
max_orth_err = max([results.orth_err]);
fprintf('  orth_sketch:    max=%.3e\n', max_orth_err);

% Flag summary
early_term_count = sum([results.flag] == 1);
fprintf('\nEarly termination flags: %d/%d\n', early_term_count, total_count);

fprintf('\n================================================================================\n');
end


function ret = is_octave()
ret = exist('OCTAVE_VERSION', 'builtin') ~= 0;
end
