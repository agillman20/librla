%==========================================================================
% demo05_methods.m - T Matrix Computation Methods
%
% This demo compares the three methods for computing the interpolation
% matrix T in the ID factorization: A(:, piv(k+1:end)) = A(:, piv(1:k)) * T
%
% Methods:
%   - 'fast':   Triangular solve (fastest, may have large T entries)
%   - 'svd':    SVD-based pseudoinverse
%   - 'lstsq':  Least squares from original A (most accurate, slowest)
%
% The choice of method affects:
%   - Speed (fast < svd < lstsq)
%   - Stability (fast may produce large T, svd/lstsq are stable)
%   - Accuracy (lstsq gives best reconstruction)
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: TBD
% Assisted by: Claude Code (Anthropic)
%
% Try changing the CONFIGURATION parameters below to experiment!
%==========================================================================

function demo05_methods()

    %======================================================================
    % CONFIGURATION - Modify these to experiment
    %======================================================================

    MATRIX_SIZE = [400, 300];   % [rows, columns]
    TARGET_RANK = 20;           % Number of skeleton columns
    RANDOM_SEED = 42;           % For reproducibility

    % Matrix type: 'lowrank', 'fullrank', or 'hilbert'
    MATRIX_TYPE = 'fullrank';

    %======================================================================
    % Demo code below
    %======================================================================

    if ~isempty(RANDOM_SEED)
        rng(RANDOM_SEED);
    end

    m = MATRIX_SIZE(1);
    n = MATRIX_SIZE(2);
    k = TARGET_RANK;

    demo_utils.print_header('Demo 05: T Matrix Computation Methods');
    fprintf('\nMatrix: %d x %d\n', m, n);
    fprintf('Target rank: %d\n', k);

    %----------------------------------------------------------------------
    % Create test matrix
    %----------------------------------------------------------------------
    if strcmp(MATRIX_TYPE, 'lowrank')
        fprintf('Matrix type: LOW-RANK (true rank = 30)\n');
        [A, ~] = demo_utils.lowrank(m, n, 30, 'exponential', 100.0);
    elseif strcmp(MATRIX_TYPE, 'fullrank')
        fprintf('Matrix type: FULL-RANK RANDOM\n');
        fprintf('(All columns are linearly independent)\n');
        A = demo_utils.random_matrix(m, n, []);
    else  % hilbert
        fprintf('Matrix type: HILBERT (ill-conditioned)\n');
        A = demo_utils.hilbert(m, n);
    end

    normA = norm(A, 'fro');
    s = svd(A);
    cnd = s(1) / s(end);
    fprintf('Condition number: %.2e\n', cnd);

    %----------------------------------------------------------------------
    % Test all three methods
    %----------------------------------------------------------------------
    methods = {'fast', 'svd', 'lstsq'};
    results = {};

    for i = 1:length(methods)
        method = methods{i};
        demo_utils.print_subheader(sprintf('Method: ''%s''', method));

        if strcmp(method, 'fast')
            fprintf('   Triangular solve on R factor. Fastest but may be unstable.\n');
        elseif strcmp(method, 'svd')
            fprintf('   SVD-based pseudoinverse. Stable for ill-conditioned R.\n');
        else
            fprintf('   Least squares from original A. Most accurate, slowest.\n');
        end

        tic;
        [k_out, piv, T] = librla.id_sketch(A, k, 'method', method);
        elapsed = toc;

        err = demo_utils.id_error(A, k_out, piv, T);
        if ~isempty(T)
            maxT = max(abs(T(:)));
        else
            maxT = 0.0;
        end

        results{i}.method = method;
        results{i}.k = k_out;
        results{i}.error = err;
        results{i}.maxT = maxT;
        results{i}.time = elapsed;

        fprintf('   Rank:     %d\n', k_out);
        fprintf('   Error:    %.6e\n', err);
        fprintf('   Max |T|:  %.3e\n', maxT);
        fprintf('   Time:     %.4f s\n', elapsed);

        % Warn about large T entries
        if maxT > 10.0
            fprintf('   [NOTE] Max|T| > 10 indicates potential instability\n');
        end
        if err > 1.0
            fprintf('   [NOTE] Error > 1.0: approximation worse than zero matrix\n');
        end
    end

    %----------------------------------------------------------------------
    % Summary
    %----------------------------------------------------------------------
    demo_utils.print_subheader('Summary');
    fprintf('   %-8s %6s %14s %12s %10s\n', 'Method', 'Rank', 'Error', 'Max|T|', 'Time');
    fprintf('   %s %s %s %s %s\n', repmat('-',1,8), repmat('-',1,6), repmat('-',1,14), repmat('-',1,12), repmat('-',1,10));
    for i = 1:length(results)
        r = results{i};
        fprintf('   %-8s %6d %14.6e %12.3e %9.4fs\n', r.method, r.k, r.error, r.maxT, r.time);
    end

    % Analysis
    fprintf('\nAnalysis:\n');

    fast_err = results{1}.error;
    lstsq_err = results{3}.error;

    if fast_err > 1.0 && lstsq_err < 1.0
        fprintf('  - ''fast'' failed (error > 1) but ''lstsq'' succeeded\n');
        fprintf('  - This happens with full-rank matrices: skeleton columns\n');
        fprintf('    cannot exactly represent other columns\n');
        fprintf('  - Use method=''lstsq'' for best least-squares approximation\n');
    elseif results{1}.maxT > 100 * results{3}.maxT
        fprintf('  - ''fast'' produced much larger T entries than ''lstsq''\n');
        fprintf('  - This indicates numerical instability in triangular solve\n');
        fprintf('  - Consider using method=''svd'' or ''lstsq'' for stability\n');
    else
        fprintf('  - All methods performed similarly\n');
        fprintf('  - ''fast'' is recommended for speed\n');
    end

    fprintf('\nRecommendations:\n');
    fprintf('  - Use ''fast'' (default) for low-rank matrices\n');
    fprintf('  - Use ''svd'' when R factor is ill-conditioned\n');
    fprintf('  - Use ''lstsq'' when best accuracy is needed\n');
    fprintf('  - Use ''lstsq'' for full-rank matrices (guarantees error < 1)\n');

end
