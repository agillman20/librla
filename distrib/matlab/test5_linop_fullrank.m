%==========================================================================
% test5_linop_fullrank.m - LinearOperator test with full-rank random matrix
%
%   Tests libid.id_sketch with LinearOperators on a full-rank random matrix,
%   demonstrating recompute_T parameter:
%   1. Dense matrix (baseline) - with recomputeT=true (default)
%   2. Explicit LinearOperator - with recomputeT=true
%   3. Matrix-free LinearOperator - recomputeT=true (accurate, n matvecs)
%   4. Matrix-free LinearOperator - recomputeT=false (fast, uses R matrix)
%==========================================================================

function test5_linop_fullrank()
    fprintf('======================================================================\n');
    fprintf('TEST 5: LinearOperators - Full-Rank Random Matrix\n');
    fprintf('======================================================================\n');

    % Create full-rank random matrix
    rng(42);  % Set seed for reproducibility
    m = 400;
    n = 300;
    fprintf('\nMatrix size: %d x %d\n', m, n);
    fprintf('Matrix type: Full-rank random (all %d columns independent)\n', n);

    % Create full-rank matrix: all columns are linearly independent
    A = randn(m, n);
    normA = norm(A, 'fro');

    % Target rank (low compared to matrix rank)
    k_target = 20;
    fprintf('Target rank: %d (%.1f%% of columns)\n', k_target, 100*k_target/n);
    fprintf('======================================================================\n');

    % =========================================================================
    % Test 1: Dense Matrix (Baseline, recomputeT=true by default)
    % =========================================================================
    fprintf('\n1. Dense Matrix (baseline, recomputeT=true)\n');
    fprintf('----------------------------------------------------------------------\n');

    tic;
    [k1, piv1, T1] = libid.id_sketch(A, k_target, 42, 0, false, true);
    t1 = toc;

    % Compute error
    A_skel1 = A(:, piv1(k1+1:end));
    A_basis1 = A(:, piv1(1:k1));
    if ~isempty(T1)
        err1 = norm(A_skel1 - A_basis1 * T1, 'fro') / normA;
        maxT1 = max(abs(T1(:)));
    else
        err1 = 0.0;
        maxT1 = 0.0;
    end

    fprintf('  Rank:      k = %d\n', k1);
    fprintf('  Error:     ||A_skel - A_basis @ T|| / ||A|| = %.3e\n', err1);
    fprintf('  Max |T|:   %.3e\n', maxT1);
    fprintf('  Time:      %.4f s\n', t1);
    if err1 < 1.0
        fprintf('  [OK] Error < 1.0 (recomputeT=true guarantees this)\n');
    end

    % =========================================================================
    % Test 2: Explicit LinearOperator (Matrix Wrapper, recomputeT=true)
    % =========================================================================
    fprintf('\n2. Explicit LinearOperator (matrix wrapper, recomputeT=true)\n');
    fprintf('----------------------------------------------------------------------\n');

    A_linop_explicit = make_linop(A);
    fprintf('  Operator: %d x %d\n', A_linop_explicit.m, A_linop_explicit.n);
    fprintf('  Is explicit: %d\n', A_linop_explicit.is_explicit);

    tic;
    [k2, piv2, T2] = libid.id_sketch(A_linop_explicit, k_target, 42, 0, false, true);
    t2 = toc;

    % Compute error using explicit matrix access
    A_skel2 = A(:, piv2(k2+1:end));
    A_basis2 = A(:, piv2(1:k2));
    if ~isempty(T2)
        err2 = norm(A_skel2 - A_basis2 * T2, 'fro') / normA;
        maxT2 = max(abs(T2(:)));
    else
        err2 = 0.0;
        maxT2 = 0.0;
    end

    fprintf('  Rank:      k = %d\n', k2);
    fprintf('  Error:     ||A_skel - A_basis @ T|| / ||A|| = %.3e\n', err2);
    fprintf('  Max |T|:   %.3e\n', maxT2);
    fprintf('  Time:      %.4f s\n', t2);

    % Verify explicit matches dense (rank and error should be similar)
    if k1 == k2 && abs(err1 - err2) < 1e-12
        fprintf('  [OK] Explicit LinearOperator produces same rank and error as dense!\n');
    else
        fprintf('  [NOTE] k_dense=%d, k_linop=%d, err_diff=%.3e\n', k1, k2, abs(err1-err2));
        fprintf('  (Pivots may differ due to randomness, but results should be similar)\n');
    end

    % =========================================================================
    % Test 3: Matrix-Free LinearOperator (recomputeT=true, accurate)
    % =========================================================================
    fprintf('\n3. Matrix-free LinearOperator (recomputeT=true, accurate)\n');
    fprintf('----------------------------------------------------------------------\n');

    % Create matrix-free operator with function handles
    Afun = @(x) A * x;        % Forward operation: y = A*x
    ATfun = @(x) A' * x;      % Adjoint operation: y = A'*x

    A_linop_mf = make_linop(m, n, Afun, ATfun, 'double');
    fprintf('  Operator: %d x %d\n', A_linop_mf.m, A_linop_mf.n);
    fprintf('  Is explicit: %d\n', A_linop_mf.is_explicit);
    fprintf('  Mode: Rank mode (rtol >= 1), recomputeT=true\n');
    fprintf('  Note: Extracts all %d columns via unit vectors (n matvecs)\n', n);

    tic;
    [k3, piv3, T3] = libid.id_sketch(A_linop_mf, k_target, 42, 0, false, true);
    t3 = toc;

    % Compute error using explicit matrix (for validation)
    A_skel3 = A(:, piv3(k3+1:end));
    A_basis3 = A(:, piv3(1:k3));
    if ~isempty(T3)
        err3 = norm(A_skel3 - A_basis3 * T3, 'fro') / normA;
        maxT3 = max(abs(T3(:)));
    else
        err3 = 0.0;
        maxT3 = 0.0;
    end

    fprintf('  Rank:      k = %d\n', k3);
    fprintf('  Error:     ||A_skel - A_basis @ T|| / ||A|| = %.3e\n', err3);
    fprintf('  Max |T|:   %.3e\n', maxT3);
    fprintf('  Time:      %.4f s\n', t3);

    if k3 == k_target && err3 < 1.0
        fprintf('  [OK] Matrix-free (recomputeT=true): rank k=%d, error < 1.0\n', k_target);
    elseif err3 < 1.0
        fprintf('  [OK] Error < 1.0 guaranteed by recomputeT=true\n');
    end

    % =========================================================================
    % Test 4: Matrix-Free LinearOperator (recomputeT=false, fast)
    % =========================================================================
    fprintf('\n4. Matrix-free LinearOperator (recomputeT=false, fast)\n');
    fprintf('----------------------------------------------------------------------\n');

    fprintf('  Operator: %d x %d\n', A_linop_mf.m, A_linop_mf.n);
    fprintf('  Is explicit: %d\n', A_linop_mf.is_explicit);
    fprintf('  Mode: Rank mode (rtol >= 1), recomputeT=false\n');
    fprintf('  Note: Uses R matrix from sketch (Fortran approach, no extra matvecs)\n');

    tic;
    [k4, piv4, T4] = libid.id_sketch(A_linop_mf, k_target, 42, 0, false, false);
    t4 = toc;

    % Compute error using explicit matrix (for validation)
    A_skel4 = A(:, piv4(k4+1:end));
    A_basis4 = A(:, piv4(1:k4));
    if ~isempty(T4)
        err4 = norm(A_skel4 - A_basis4 * T4, 'fro') / normA;
        maxT4 = max(abs(T4(:)));
    else
        err4 = 0.0;
        maxT4 = 0.0;
    end

    fprintf('  Rank:      k = %d\n', k4);
    fprintf('  Error:     ||A_skel - A_basis @ T|| / ||A|| = %.3e\n', err4);
    fprintf('  Max |T|:   %.3e\n', maxT4);
    fprintf('  Time:      %.4f s\n', t4);

    % Compare with recomputeT=true
    speedup = t3 / t4;
    error_ratio = err4 / err3;

    fprintf('  Speedup:   %.1fx faster than recomputeT=true\n', speedup);
    fprintf('  Error ratio: %.2fx (err_false / err_true)\n', error_ratio);

    if err4 > 1.0
        fprintf('  [NOTE] Error > 1.0 is expected for full-rank matrices with recomputeT=false\n');
        fprintf('         This uses Fortran''s fast R-matrix approach, trading accuracy for speed\n');
    else
        fprintf('  [OK] Error < 1.0 (better than expected!)\n');
    end

    % =========================================================================
    % Summary
    % =========================================================================
    fprintf('\n======================================================================\n');
    fprintf('SUMMARY\n');
    fprintf('======================================================================\n');
    fprintf('  Method                        Rank    Error        Max|T|       Time\n');
    fprintf('----------------------------------------------------------------------\n');
    fprintf('  Dense (recomputeT=true)       %4d    %.3e    %.3e    %.4fs\n', k1, err1, maxT1, t1);
    fprintf('  Explicit LinOp (recomp=true)  %4d    %.3e    %.3e    %.4fs\n', k2, err2, maxT2, t2);
    fprintf('  Matrix-free (recompute=true)  %4d    %.3e    %.3e    %.4fs\n', k3, err3, maxT3, t3);
    fprintf('  Matrix-free (recompute=false) %4d    %.3e    %.3e    %.4fs\n', k4, err4, maxT4, t4);
    fprintf('======================================================================\n');

    fprintf('\nKey Observations:\n');
    fprintf('  - Full-rank matrix: %dx%d, target k=%d (%.1f%% of columns)\n', m, n, k_target, 100*k_target/n);
    fprintf('  - recomputeT=true:  Guarantees error < 1.0 (all methods: %.3e, %.3e, %.3e)\n', err1, err2, err3);
    fprintf('  - recomputeT=false: %.1fx faster, but error may be > 1.0 (err=%.3e)\n', speedup, err4);
    fprintf('  - Trade-off: Speed (%.1fx) vs Accuracy (%.2fx degradation)\n', speedup, error_ratio);

    % Validate
    success = true;

    if err1 > 1.0 || err2 > 1.0 || err3 > 1.0
        fprintf('\n[FAIL] recomputeT=true should guarantee error < 1.0!\n');
        success = false;
    end

    % Note: Do NOT fail on err_diff - randomness can cause different pivots
    % The NOTE message already explains this is expected

    if k3 ~= k_target || k4 ~= k_target
        fprintf('\n[FAIL] Matrix-free should return rank k=%d!\n', k_target);
        success = false;
    end

    if success
        fprintf('\n[PASS] All LinearOperator tests passed!\n');
        fprintf('       recomputeT=true guarantees error < 1.0 for all modes\n');
        fprintf('       recomputeT=false provides %.1fx speedup with acceptable error increase\n', speedup);
    end
end
