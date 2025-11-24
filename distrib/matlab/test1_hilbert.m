%==========================================================================
% test1_hilbert.m - Simple test with medium-size Hilbert matrix
%
%   Tests basic ID algorithms (id_sketch, id_rrqr) on an ill-conditioned
%   Hilbert matrix.
%==========================================================================

function test1_hilbert()
    fprintf('======================================================================\n');
    fprintf('TEST 1: Medium Hilbert Matrix\n');
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
    % Method 1: id_sketch (randomized)
    % -------------------------------------------------------------------------
    fprintf('\n1. librla.id_sketch (randomized QR sketching)\n');
    fprintf('----------------------------------------------------------------------\n');

    tic;
    [k1, piv1, T1] = librla.id_sketch(A, k_target);
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

    % -------------------------------------------------------------------------
    % Method 2: id_rrqr (deterministic QR with column pivoting)
    % -------------------------------------------------------------------------
    fprintf('\n2. librla.id_qrpiv (deterministic QR with column pivoting via LAPACK)\n');
    fprintf('----------------------------------------------------------------------\n');

    tic;
    [k2, piv2, T2] = librla.id_qrpiv(A, k_target);
    t2 = toc;

    % Compute error
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

    % -------------------------------------------------------------------------
    % Summary
    % -------------------------------------------------------------------------
    fprintf('\n======================================================================\n');
    fprintf('SUMMARY\n');
    fprintf('======================================================================\n');
    fprintf('  Method         Rank    Error        Max|T|       Time\n');
    fprintf('----------------------------------------------------------------------\n');
    fprintf('  id_sketch      %4d    %.3e    %.3e    %.4fs\n', k1, err1, maxT1, t1);
    fprintf('  id_rrqr        %4d    %.3e    %.3e    %.4fs\n', k2, err2, maxT2, t2);
    fprintf('======================================================================\n');

    % Validate
    if err1 > 1.0 || err2 > 1.0
        fprintf('\n[FAIL] Error > 1.0 detected!\n');
        return
    end

    fprintf('\n[PASS] Test completed successfully!\n');
end
