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
% Version: 1.0.1
% Date: April 22, 2026
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
  variants = {'dense', 'explicit', 'matfree'};
  As = {M, LinearOperator.from_matrix(M), wrap_matfree(M)};
  for vi = 1:3
    [Q, flag, ~] = librla.orth_sketch(As{vi}, k);
    if size(Q, 2) > 0
      ortho = norm(Q' * Q - eye(size(Q, 2)), 'fro');
    else
      ortho = 0.0;
    end
    ok = (flag == 0) && (size(Q, 2) == k) && (ortho < 1e-10);
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


function out = iif(cond, a, b)
  if cond; out = a; else; out = b; end
end
