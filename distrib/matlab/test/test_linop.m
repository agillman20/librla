% test_linop.m - LinearOperator, wide-matrix, and method='svd' regression tests
%
% Covers code paths not exercised by test_id/test_svd/test_qr/test_orth:
%   - Explicit LinearOperator (LinearOperator.from_matrix — .matrix attached)
%   - Matrix-free LinearOperator (matvec/rmatvec closures only)
%   - Wide matrices (m < n), which trigger svd_sketch's transpose branch
%   - id_sketch / id_qrpiv with method='svd' and method='lstsq'
%
% Usage:
%     octave --no-gui --eval "test_linop"
%     matlab -batch "test_linop"
%
% Returns 0 if all tests pass, 1 otherwise.
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: MIT
% Version: 1.2.1
% Date: July 30, 2026
% Assisted by: Claude Code (Anthropic)

function exit_code = test_linop()
  addpath(genpath('..'));

  fprintf('\n======================================================================\n');
  fprintf('LINEAROPERATOR / WIDE-MATRIX / METHOD REGRESSION TESTS\n');
  fprintf('======================================================================\n');

  rng(17);
  errors = {};

  shapes = {
    200, 100,  8, 'tall real  200x100', false;
    100, 200,  8, 'wide real  100x200', false;
    150,  80,  6, 'tall cplx  150x80',  true;
     80, 150,  6, 'wide cplx   80x150', true;
  };

  for i = 1:size(shapes, 1)
    m   = shapes{i, 1};
    n   = shapes{i, 2};
    k   = shapes{i, 3};
    lbl = shapes{i, 4};
    is_cplx = shapes{i, 5};

    if is_cplx
      M = randn(m, n) + 1i * randn(m, n);
    else
      M = randn(m, n);
    end

    fprintf('\n--- %s ---\n', lbl);
    errors = check_orth(M, k, lbl, errors);
    errors = check_qr  (M, k, lbl, errors);
    errors = check_svd (M, k, lbl, errors);
    if m >= n
      errors = check_id(M, k, lbl, errors);
    end
  end

  errors = check_power_iter(errors);
  errors = check_overrank_id(errors);
  errors = check_zero_tol_rank(errors);
  errors = check_matfree_tol(errors);
  errors = check_sparse_backed(errors);

  fprintf('\n======================================================================\n');
  if ~isempty(errors)
    fprintf('[FAIL] %d check(s) failed:\n', length(errors));
    for i = 1:length(errors)
      fprintf('   - %s\n', errors{i});
    end
    fprintf('======================================================================\n');
    exit_code = 1;
  else
    fprintf('[PASS] All LinearOperator / wide / method regression checks passed.\n');
    fprintf('======================================================================\n');
    exit_code = 0;
  end
end


function A_lo = wrap_matfree(M)
  [m, n] = size(M);
  is_cplx = ~isreal(M);
  A_lo = LinearOperator(@(x) M * x, @(x) M' * x, m, n, ...
                        'is_complex', is_cplx, 'dtype', class(M));
end


function errors = check_orth(M, k, lbl, errors)
  % Rank mode returns the oversampled basis: rank + extra_samples (12)
  % buffer columns, capped by the number of rows
  k_exp = min(k + 12, size(M, 1));
  variants = {'dense', 'explicit', 'matfree'};
  As = {M, LinearOperator.from_matrix(M), wrap_matfree(M)};
  for vi = 1:3
    [Q, flag, ~] = librla.orth_sketch(As{vi}, k);
    if size(Q, 2) > 0
      ortho = norm(Q' * Q - eye(size(Q, 2)), 'fro');
    else
      ortho = 0.0;
    end
    ok = (flag == 0) && (size(Q, 2) == k_exp) && (ortho < 1e-10);
    status = iif(ok, 'PASS', 'FAIL');
    fprintf('  [%s] orth_sketch %-26s %-9s flag=%d ortho=%.1e k=%d\n', ...
            status, lbl, variants{vi}, flag, ortho, size(Q, 2));
    if ~ok
      errors{end+1} = sprintf('orth_sketch %s %s', lbl, variants{vi});
    end
  end
end


function errors = check_qr(M, k, lbl, errors)
  normM = norm(M, 'fro');
  variants = {'dense', 'explicit', 'matfree'};
  As = {M, LinearOperator.from_matrix(M), wrap_matfree(M)};
  for vi = 1:3
    [Q, R, p] = librla.qr_sketch(As{vi}, k);
    err = norm(M(:, p) - Q * R, 'fro') / normM;
    ortho = norm(Q' * Q - eye(size(Q, 2)), 'fro');
    ok = (size(Q, 2) == k) && (ortho < 1e-10);
    status = iif(ok, 'PASS', 'FAIL');
    fprintf('  [%s] qr_sketch   %-26s %-9s err=%.2e ortho=%.1e k=%d\n', ...
            status, lbl, variants{vi}, err, ortho, size(Q, 2));
    if ~ok
      errors{end+1} = sprintf('qr_sketch %s %s', lbl, variants{vi});
    end
  end
end


function errors = check_svd(M, k, lbl, errors)
  normM = norm(M, 'fro');
  s_true = svd(M);
  if k < length(s_true)
    err_opt = norm(s_true((k+1):end)) / normM;
  else
    err_opt = 0.0;
  end
  tol = max(4.0 * err_opt, 1e-10);
  variants = {'dense', 'explicit', 'matfree'};
  As = {M, LinearOperator.from_matrix(M), wrap_matfree(M)};
  for vi = 1:3
    [U, s, V] = librla.svd_sketch(As{vi}, k);
    err = norm(M - U * diag(s) * V', 'fro') / normM;
    ortho_U = norm(U' * U - eye(size(U, 2)), 'fro');
    ortho_V = norm(V' * V - eye(size(V, 2)), 'fro');
    ok = (err < tol) && (ortho_U < 1e-10) && (ortho_V < 1e-10) && (length(s) == k);
    status = iif(ok, 'PASS', 'FAIL');
    fprintf('  [%s] svd_sketch  %-26s %-9s err=%.2e opt=%.2e orthU=%.1e k=%d\n', ...
            status, lbl, variants{vi}, err, err_opt, ortho_U, length(s));
    if ~ok
      errors{end+1} = sprintf('svd_sketch %s %s', lbl, variants{vi});
    end
  end
end


function errors = check_id(M, k, lbl, errors)
  normM = norm(M, 'fro');
  [~, n] = size(M);
  fns = {'id_sketch', 'id_qrpiv'};
  methods = {'fast', 'svd', 'lstsq'};
  for fi = 1:2
    for mi = 1:3
      if strcmp(fns{fi}, 'id_sketch')
        [kk, piv, T] = librla.id_sketch(M, k, 'method', methods{mi});
      else
        [kk, piv, T] = librla.id_qrpiv(M, k, 'method', methods{mi});
      end
      if ~isempty(T) && kk < n
        A_skel  = M(:, piv((kk+1):end));
        A_basis = M(:, piv(1:kk));
        err = norm(A_skel - A_basis * T, 'fro') / normM;
      else
        err = 0.0;
      end
      ok = (kk == k) && (err < 1.5) && isequal(size(T), [k, n - k]);
      status = iif(ok, 'PASS', 'FAIL');
      fprintf('  [%s] %-10s method=%-6s %-20s err=%.2e k=%d\n', ...
              status, fns{fi}, methods{mi}, lbl, err, kk);
      if ~ok
        errors{end+1} = sprintf('%s method=%s %s', fns{fi}, methods{mi}, lbl);
      end
    end
  end
end


function errors = check_power_iter(errors)
  % Regression: rank mode with power_iter >= 1 where block_size = rank +
  % extra_samples exceeds n. The intermediate sketch inside power_iteration
  % has n rows, so block_size > n must not overrun its QR Q-factor (this was
  % a BoundsError in the Julia port before the min(rows, cols) cap was added).
  fprintf('\n--- power_iter, rank + extra_samples > n ---\n');
  m = 50; n = 10; k = 6;            % block_size = k + 12 = 18 > n = 10
  M = randn(m, k) * randn(k, n);    % exact rank k
  normM = norm(M, 'fro');
  variants = {'dense', 'explicit', 'matfree'};
  As = {M, LinearOperator.from_matrix(M), wrap_matfree(M)};
  for vi = 1:3
    ok = false; err = NaN; msg = '';
    try
      [Q0, ~, ~]   = librla.orth_sketch(As{vi}, k, 'power_iter', 2);
      [U, s, V]    = librla.svd_sketch (As{vi}, k, 'power_iter', 2);
      err = norm(M - U * diag(s) * V', 'fro') / normM;
      [Q, ~, ~]    = librla.qr_sketch  (As{vi}, k, 'power_iter', 2);
      [kk, ~, ~]   = librla.id_sketch  (As{vi}, k, 'power_iter', 2);
      % orth_sketch returns the oversampled basis (rank + extra_samples,
      % capped at n by the power-iteration orthonormalization), so only
      % require >= k columns; svd/qr/id truncate to exactly k.
      ok = (size(Q0, 2) >= k) && (length(s) == k) && (err < 1e-8) && ...
           (size(Q, 2) == k) && (kk == k);
      if ~ok; msg = 'wrong shape/err'; end
    catch e
      msg = e.message;
    end
    status = iif(ok, 'PASS', 'FAIL');
    fprintf('  [%s] power_iter=2 orth/svd/qr/id rank=%d (n=%d) %-9s err=%.1e\n', ...
            status, k, n, variants{vi}, err);
    if ~ok
      errors{end+1} = sprintf('power_iter largeblock %s (%s)', variants{vi}, msg);
    end
  end
end


function errors = check_zero_tol_rank(errors)
  % Regression: in tolerance mode a zero matrix has rank 0. svd_sketch must
  % agree with qr_sketch / id_sketch (rank_from_svals needs the same
  % s(1) <= 0 guard that rank_from_diag has).
  fprintf('\n--- zero matrix rank in tolerance mode ---\n');
  A = zeros(100, 60);
  [~, s, ~] = librla.svd_sketch(A, 1e-6);
  [Q, ~, ~] = librla.qr_sketch(A, 1e-6);
  [k, ~, ~] = librla.id_sketch(A, 1e-6);
  ksvd = numel(s); kqr = size(Q, 2); kid = k;
  ok = (ksvd == 0) && (kqr == 0) && (kid == 0);
  status = iif(ok, 'PASS', 'FAIL');
  fprintf('  [%s] zero 100x60 rtol=1e-6: svd k=%d, qr k=%d, id k=%d\n', status, ksvd, kqr, kid);
  if ~ok
    errors{end+1} = sprintf('zero-tol-rank svd=%d qr=%d id=%d', ksvd, kqr, kid);
  end
end


function errors = check_overrank_id(errors)
  % Regression: ID in rank mode with the requested rank exceeding the true
  % rank (singular R11). Every T-method must stay finite, return shape
  % (k, n-k), and still reconstruct A -- in particular the all-zero matrix,
  % where R11 is exactly singular (fast gave a warning + NaN before the guard).
  fprintf('\n--- ID over-rank (requested rank > true rank) ---\n');
  % fast/lstsq legitimately warn on rank-deficient input in MATLAB; we assert
  % the results are finite and reconstruct, not that no warning is raised.
  ws = warning('off', 'all');
  restore_warn = onCleanup(@() warning(ws));
  labels = {'zeros', 'rank3'};
  As = {zeros(40,20), randn(40,3)*randn(3,20)};
  rs = {5, 8};
  fns = {@librla.id_sketch, @librla.id_qrpiv}; fnames = {'id_sketch','id_qrpiv '};
  meths = {'fast','svd','lstsq'};
  for ci = 1:numel(labels)
    A = As{ci}; r = rs{ci}; [m,n] = size(A); normA = max(norm(A,'fro'), 1.0);
    for fi = 1:2
      for mi = 1:3
        ok = false; err = NaN;
        try
          [k, piv, T] = fns{fi}(A, r, 'method', meths{mi});
          Arec = zeros(m, n);
          Arec(:, piv(1:k))     = A(:, piv(1:k));
          Arec(:, piv(k+1:end)) = A(:, piv(1:k)) * T;
          err = norm(A - Arec, 'fro') / normA;
          ok = isequal(size(T), [k, n-k]) && all(isfinite(T(:))) && (err < 1e-8);
        catch
          ok = false;
        end
        status = iif(ok, 'PASS', 'FAIL');
        fprintf('  [%s] %-6s %-9s %-5s rec_err=%.1e\n', status, labels{ci}, fnames{fi}, meths{mi}, err);
        if ~ok
          errors{end+1} = sprintf('overrank-id %s %s %s', labels{ci}, fnames{fi}, meths{mi});
        end
      end
    end
  end
end


function errors = check_matfree_tol(errors)
% Regression: matrix-free LinearOperators in tolerance mode (previously
% errored). The deterministic fallback now materializes the operator via
% get_matrix, one matvec per column.
  fprintf('\n--- matrix-free LinearOperator in tolerance mode ---\n');
  rtol = 1e-6;
  k_true = 8;
  dims = [200, 100; 100, 200];   % tall and wide
  for di = 1:2
    m = dims(di, 1); n = dims(di, 2);
    M = randn(m, k_true) * randn(k_true, n);
    shape_lbl = sprintf('%s %dx%d', iif(m >= n, 'tall', 'wide'), m, n);
    ok = false; err = NaN; msg = '';
    try
      [U, s, V] = librla.svd_sketch(wrap_matfree(M), rtol);
      err = norm(M - U * diag(s) * V', 'fro') / norm(M, 'fro');
      ok = (numel(s) == k_true) && (err < 10 * rtol);
      if ~ok, msg = 'wrong rank/err'; end
    catch e
      msg = e.identifier;
    end
    fprintf('  [%s] svd_sketch matfree %s rtol=%.0e err=%.1e %s\n', ...
            iif(ok, 'PASS', 'FAIL'), shape_lbl, rtol, err, msg);
    if ~ok
      errors{end+1} = sprintf('matfree-tol svd %s (%s)', shape_lbl, msg);
    end
  end

  % Paths that fully materialize the operator: deterministic ID and the
  % full-rank fallback of qr_sketch
  M = randn(120, 5) * randn(5, 60);
  [k, ~, ~] = librla.id_qrpiv(wrap_matfree(M), rtol);
  ok = (k == 5);
  fprintf('  [%s] id_qrpiv matfree tol k=%d (true 5)\n', iif(ok, 'PASS', 'FAIL'), k);
  if ~ok, errors{end+1} = sprintf('matfree-tol id_qrpiv k=%d', k); end

  [k, ~, T] = librla.id_sketch(wrap_matfree(M), rtol, 'method', 'lstsq');
  ok = (k == 5) && all(isfinite(T(:)));
  fprintf('  [%s] id_sketch lstsq matfree tol k=%d\n', iif(ok, 'PASS', 'FAIL'), k);
  if ~ok, errors{end+1} = sprintf('matfree-tol id_sketch lstsq k=%d', k); end

  Mf = randn(40, 30);   % full rank: forces the dense fallback
  [Q, ~, ~] = librla.qr_sketch(wrap_matfree(Mf), 1e-14);
  ok = isequal(size(Q), [40, 30]);
  fprintf('  [%s] qr_sketch matfree full-rank fallback shape=%dx%d\n', ...
          iif(ok, 'PASS', 'FAIL'), size(Q, 1), size(Q, 2));
  if ~ok, errors{end+1} = 'matfree-tol qr fallback'; end
end


function errors = check_sparse_backed(errors)
% Regression: sparse-backed operators (from_matrix of a sparse matrix) and
% raw sparse matrices. get_matrix materializes them densely, so the dense
% LAPACK paths (id_qrpiv, lstsq, full QR/SVD fallback) work, sparse qr's
% fill-reducing (non-rank-revealing) ordering is never used, and outputs
% are dense-typed.
  fprintf('\n--- sparse-backed operators / raw sparse matrices ---\n');
  % Genuinely rank-5 sparse matrix: product of two sparse factors
  S = sprandn(120, 5, 0.4) * sprandn(5, 60, 0.4);
  D = full(S);
  A = LinearOperator.from_matrix(S);

  labels = {'id_qrpiv tol       ', 'id_qrpiv rank      ', ...
            'id_sketch lstsq tol', 'raw sparse id_qrpiv'};
  calls  = {@() librla.id_qrpiv(A, 1e-6), @() librla.id_qrpiv(A, 5.0), ...
            @() librla.id_sketch(A, 1e-6, 'method', 'lstsq'), ...
            @() librla.id_qrpiv(S, 1e-6)};
  for ci = 1:numel(calls)
    ok = false; msg = '';
    try
      [k, piv, T] = calls{ci}();
      ok = (k == 5) && ~issparse(k) && ~issparse(piv) && ~issparse(T);
      msg = sprintf('k=%d dense-typed=%d', full(k), ...
                    ~issparse(k) && ~issparse(piv) && ~issparse(T));
    catch e
      msg = e.identifier;
    end
    fprintf('  [%s] %s %s\n', iif(ok, 'PASS', 'FAIL'), labels{ci}, msg);
    if ~ok
      errors{end+1} = sprintf('sparse-backed %s (%s)', strtrim(labels{ci}), msg);
    end
  end

  [U, s, V] = librla.svd_sketch(A, 1e-6);
  err = norm(D - U * diag(s) * V', 'fro') / norm(D, 'fro');
  ok = (numel(s) == 5) && (err < 1e-5);
  fprintf('  [%s] svd_sketch sparse-backed tol k=%d err=%.1e\n', ...
          iif(ok, 'PASS', 'FAIL'), numel(s), err);
  if ~ok, errors{end+1} = sprintf('sparse-backed svd k=%d', numel(s)); end

  % Full-rank sparse: forces the dense QR/SVD fallback
  Sf = sparse(randn(40, 30));
  Af = LinearOperator.from_matrix(Sf);
  ok = false; msg = '';
  try
    [Q, ~, ~] = librla.qr_sketch(Af, 1e-14);
    [~, s, ~] = librla.svd_sketch(Af, 1e-14);
    ok = isequal(size(Q), [40, 30]) && (numel(s) == 30);
    msg = sprintf('qr=%dx%d svd k=%d', size(Q, 1), size(Q, 2), numel(s));
  catch e
    msg = e.identifier;
  end
  fprintf('  [%s] full-rank sparse fallback %s\n', iif(ok, 'PASS', 'FAIL'), msg);
  if ~ok, errors{end+1} = sprintf('sparse-backed fallback (%s)', msg); end

  % Graded-spectrum sparse: tolerance-mode rank selection must match the
  % dense answer (sparse qr's COLAMD ordering used to inflate k silently)
  G = sparse(100, 50);
  for i = 1:24
    G = G + 10^(-(i-1)*0.5) * (sprandn(100, 1, 0.3) * sprandn(1, 50, 0.3));
  end
  [kd, ~, ~] = librla.id_qrpiv(full(G), 1e-6);
  [ks, ~, ~] = librla.id_qrpiv(G, 1e-6);
  ok = isequal(full(kd), full(ks));
  fprintf('  [%s] graded sparse id_qrpiv k: dense=%d sparse=%d (must match)\n', ...
          iif(ok, 'PASS', 'FAIL'), full(kd), full(ks));
  if ~ok
    errors{end+1} = sprintf('sparse-backed graded k dense=%d sparse=%d', ...
                            full(kd), full(ks));
  end
end


function out = iif(cond, a, b)
  if cond; out = a; else; out = b; end
end
