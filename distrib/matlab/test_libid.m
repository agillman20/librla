%==========================================================================
% test_libid - Test and demonstrate libid interpolative decomposition
%
%   Quick test script to validate libid.id_sketch and libid_rrqr.id_rrqr
%   implementations. Runs several test cases and prints results.
%
%   Usage:
%       test_libid          % Run all tests
%
%   Author : Your Name
%   SPDX-License-Identifier : TBD
%==========================================================================

function test_libid()
    fprintf('=================================================================\n');
    fprintf('Testing libid Interpolative Decomposition\n');
    fprintf('=================================================================\n');
    fprintf('\n');

    % Test 1: Random matrix
    fprintf('Test 1: Random Matrix (500x300, rank=20)\n');
    fprintf('-----------------------------------------------------------------\n');
    A1 = randn(500, 300);
    run_test(A1, 20, 'Random 500x300');

    % Test 2: Low-rank matrix
    fprintf('\nTest 2: Low-Rank Matrix (400x250, true rank~15)\n');
    fprintf('-----------------------------------------------------------------\n');
    U = randn(400, 15);
    V = randn(250, 15);
    A2 = U * V' + 1e-10 * randn(400, 250);
    run_test(A2, 1e-8, 'Low-Rank 400x250');

    % Test 3: Hilbert matrix (ill-conditioned)
    fprintf('\nTest 3: Hilbert Matrix (200x100, ill-conditioned)\n');
    fprintf('-----------------------------------------------------------------\n');
    A3 = libid.hilb(200, 100);
    run_test(A3, 15, 'Hilbert 200x100');

    % Test 4: Complex matrix
    fprintf('\nTest 4: Complex Matrix (300x200, rank=25)\n');
    fprintf('-----------------------------------------------------------------\n');
    A4 = randn(300, 200) + 1i*randn(300, 200);
    run_test(A4, 25, 'Complex 300x200');

    fprintf('\n=================================================================\n');
    fprintf('All tests completed!\n');
    fprintf('=================================================================\n');
end

function run_test(A, rtol, name)
    % Helper function to run both methods and compare results

    [m, n] = size(A);
    normA = norm(A, 'fro');

    % Test libid randomized version
    tic;
    [k_libid, piv_libid, T_libid] = libid.id_sketch(A, rtol);
    t_libid = toc;

    % Compute error
    A_skel_libid = A(:, piv_libid(k_libid+1:end));
    A_basis_libid = A(:, piv_libid(1:k_libid));
    if ~isempty(T_libid)
        err_libid = norm(A_skel_libid - A_basis_libid * T_libid, 'fro') / normA;
        max_T_libid = max(abs(T_libid(:)));
    else
        err_libid = 0;
        max_T_libid = 0;
    end

    % Test RRQR deterministic version
    tic;
    [k_rrqr, piv_rrqr, T_rrqr] = libid_rrqr.id_rrqr(A, rtol);
    t_rrqr = toc;

    % Compute error
    A_skel_rrqr = A(:, piv_rrqr(k_rrqr+1:end));
    A_basis_rrqr = A(:, piv_rrqr(1:k_rrqr));
    if ~isempty(T_rrqr)
        err_rrqr = norm(A_skel_rrqr - A_basis_rrqr * T_rrqr, 'fro') / normA;
        max_T_rrqr = max(abs(T_rrqr(:)));
    else
        err_rrqr = 0;
        max_T_rrqr = 0;
    end

    % Print results
    fprintf('Matrix: %s (%dx%d', name, m, n);
    if ~isreal(A)
        fprintf(', complex');
    end
    fprintf(')\n');
    fprintf('Target: rtol = %g\n\n', rtol);

    fprintf('%-20s %-10s %-15s %-15s %-10s\n', 'Method', 'Rank', 'Error', 'max|T|', 'Time (s)');
    fprintf('%-20s %-10s %-15s %-15s %-10s\n', repmat('-',1,20), repmat('-',1,10), repmat('-',1,15), repmat('-',1,15), repmat('-',1,10));
    fprintf('%-20s %-10d %-15.3e %-15.3e %-10.4f\n', 'libid (randomized)', k_libid, err_libid, max_T_libid, t_libid);
    fprintf('%-20s %-10d %-15.3e %-15.3e %-10.4f\n', 'RRQR (deterministic)', k_rrqr, err_rrqr, max_T_rrqr, t_rrqr);

    % Compare ranks
    if k_libid == k_rrqr
        fprintf('\n[OK] Ranks match (k=%d)\n', k_libid);
    else
        fprintf('\n[WARNING] Different ranks: libid=%d, RRQR=%d (difference=%d)\n', k_libid, k_rrqr, abs(k_libid - k_rrqr));
    end

    % Compare speeds
    if t_libid < t_rrqr
        fprintf('[OK] libid %.2fx faster than RRQR\n', t_rrqr/t_libid);
    else
        fprintf('[WARNING] RRQR %.2fx faster than libid\n', t_libid/t_rrqr);
    end
end
