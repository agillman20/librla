% Run the four tests when the file is executed directly.
rng(0);                         % reproducible random numbers

% --------------------------------------------------------------
% Test 1: full-rank random matrix
% --------------------------------------------------------------
m = 8; n = 5;
A = randn(m,n);
[Q,R,p,k] = rrqr.rrqr_lapack(A, 1e-12);

A_perm = A(:,p);
A_recon = Q*R;
err = norm(A_perm - A_recon);
orth_err = norm(Q'*Q - eye(k));

fprintf('Test 1: full-rank random matrix\n');
fprintf('  Numerical rank k = %d\n', k);
fprintf('  Reconstruction error ||A(:,p) - Q*R|| = %.2e\n', err);
fprintf('  Orthogonality error   ||Q''*Q - I||   = %.2e\n\n', orth_err);

% --------------------------------------------------------------
% Test 2: rank-deficient matrix
% --------------------------------------------------------------
rank_true = 3;
U = randn(m, rank_true);
V = randn(rank_true, n);
A2 = U*V;                       % rank <= 3
[Q2,R2,p2,k2] = rrqr.rrqr_lapack(A2, 1e-12);

A2_perm = A2(:,p2);
A2_recon = Q2*R2;
err2 = norm(A2_perm - A2_recon);
orth_err2 = norm(Q2'*Q2 - eye(k2));

fprintf('Test 2: rank-deficient matrix\n');
fprintf('  Expected rank <= %d, detected rank k = %d\n', rank_true, k2);
fprintf('  Reconstruction error ||A2(:,p) - Q2*R2|| = %.2e\n', err2);
fprintf('  Orthogonality error   ||Q2''*Q2 - I||   = %.2e\n\n', orth_err2);

% --------------------------------------------------------------
% Test 3: Hilbert matrix (large, ill-conditioned) - LAPACK driver
% --------------------------------------------------------------
try
    hilb = @(m,n) hilb(m,n);   % placeholder - will be overwritten
    A_hilb = hilb(4000,2000);
catch
    % Fallback using MATLAB's built-in hilb (requires Symbolic Toolbox)
    % If not available, use a smaller test.
    if exist('hilb','file')
        A_hilb = hilb(4000,2000);
    else
        warning('Hilbert generator not found - using a smaller matrix.');
        A_hilb = hilb(500,250);
    end
end

fprintf('Test 3: Hilbert matrix (LAPACK driver)\n');
fprintf('shape(A): %d x %d\n', size(A_hilb,1), size(A_hilb,2));
tic;
[Q,R,perm,rank_est] = rrqr.rrqr_lapack(A_hilb, 1e-12);
t = toc;
fprintf('rrqr (LAPACK) elapsed time: %.4f s\n', t);
fprintf('Permutation vector (1-based):\n');
% disp(perm(:).');
fprintf('Estimated rank: %d\n', rank_est);

recon_err = norm(A_hilb(:,perm) - Q*R,'fro');
ortho_err = norm(Q'*Q - eye(size(Q,2)),'fro');
fprintf('Reconstruction error  ||A(:,perm) - Q*R||_F = %.2e\n', recon_err);
fprintf('Orthogonality error   ||Q''*Q - I||_F    = %.2e\n\n', ortho_err);

% --------------------------------------------------------------
% Test 4: Hilbert matrix - native implementation
% --------------------------------------------------------------
fprintf('Test 4: Hilbert matrix (native implementation)\n');
tic;
[Qn,Rn,permn,rank_est_n] = rrqr.rrqr_native(A_hilb, 1e-12);
t = toc;
fprintf('rrqr_native elapsed time: %.4f s\n', t);
fprintf('Permutation vector (1-based):\n');
% disp(permn(:).');
fprintf('Estimated rank: %d\n', rank_est_n);

recon_err_n = norm(A_hilb(:,permn) - Qn*Rn,'fro');
ortho_err_n = norm(Qn'*Qn - eye(size(Qn,2)),'fro');
fprintf('Reconstruction error  ||A(:,permn) - Q*R||_F = %.2e\n', recon_err_n);
fprintf('Orthogonality error   ||Q''*Q - I||_F    = %.2e\n\n', ortho_err_n);

% --------------------------------------------------------------
% Compare against MATLAB's high-level QR (also LAPACK based)
% --------------------------------------------------------------
fprintf('MATLAB built-in qr (economic, pivoting)\n');
tic;
[Qs,Rs,perms] = qr(A_hilb,0);
t = toc;
fprintf('qr elapsed time: %.4f s\n', t);
fprintf('Permutation vector (1-based):\n');
% disp(perms(:).');

recon_err_m = norm(A_hilb(:,perms) - Qs*Rs,'fro');
ortho_err_m = norm(Qs'*Qs - eye(size(Qs,2)),'fro');
fprintf('Reconstruction error  ||A(:,perm) - Q*R||_F = %.2e\n', recon_err_m);
fprintf('Orthogonality error   ||Q''*Q - I||_F    = %.2e\n', ortho_err_m);
