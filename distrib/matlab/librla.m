classdef librla
% LIBRLA - Randomized linear algebra routines for MATLAB/Octave
%
% Randomized algorithms for low-rank matrix approximations:
%   librla.orth_sketch  - Approximate orthonormal basis for column space
%   librla.qr_sketch    - Truncated QR factorization with column pivoting
%   librla.svd_sketch   - Truncated singular value decomposition (SVD)
%   librla.id_sketch    - Interpolative decomposition (ID)
%
% Deterministic:
%   librla.id_qrpiv     - Interpolative decomposition via QR with pivoting
%
% USAGE:
%   [Q, flag, diagR] = librla.orth_sketch(A, rtol_or_rank);
%   [Q, R, p] = librla.qr_sketch(A, rtol_or_rank);
%   [U, s, V] = librla.svd_sketch(A, rtol_or_rank);
%   [k, piv, T] = librla.id_sketch(A, rtol_or_rank);
%   % tolerance mode: rtol < 1, rank mode: rtol >= 1
%
% MATRIX-FREE OPERATORS:
%   Use the LinearOperator class for matrix-free operators:
%     A = LinearOperator(matvec_fun, rmatvec_fun, m, n);
%     [U, s, V] = librla.svd_sketch(A, rank);  % rank mode only: rtol >= 1
%
% Author: Adrianna Gillman, Zydrunas Gimbutas
% SPDX-License-Identifier: NIST-PD
% Version: 1.0.1
% Date: April 22, 2026
% Assisted by: Claude Code (Anthropic)

methods (Static)

function [Q, flag, diagR] = orth_sketch(A, rtol, varargin)
% ORTH_SKETCH - Approximate orthonormal basis for column space using randomized sketching
%
% Syntax:
%   Q = librla.orth_sketch(A, rtol)
%   [Q, flag] = librla.orth_sketch(A, rtol)
%   [Q, flag, diagR] = librla.orth_sketch(A, rtol)
%   [___] = librla.orth_sketch(A, rtol, Name, Value)
%
% Description:
%   This function uses random test matrix multiplication (A*Omega where Omega has
%   i.i.d. uniform[-1,1] entries) followed by QR factorization to approximate
%   the range of A. The approach is particularly efficient for matrices with
%   rapidly decaying singular values.
%
%   The algorithm has two modes:
%     - Tolerance mode (rtol < 1): Adaptively grows the sketch size until the
%       smallest column norm falls below rtol times the largest norm
%     - Rank mode (rtol >= 1): Performs a single sketch and returns the
%       requested number of columns (rtol interpreted as target rank)
%
% Input Arguments:
%   A          - Input matrix (m x n) or LinearOperator
%   rtol       - Relative tolerance (< 1) or target rank (>= 1)
%   block_size - Initial number of random test vectors (default: 42)
%   power_iter - Number of power iterations to improve accuracy (default: 0)
%                Setting power_iter=1 or 2 can improve results for
%                matrices with slowly decaying singular values
%   rng        - Random number generator state or seed (default: [] uses current state)
%                Can be a seed (integer) or rng state struct from rng()
%
% Output Arguments:
%   Q     - Orthonormal matrix (m x k) spanning approximate range of A
%   flag  - Exit status:
%           0: Success, Q contains valid orthonormal basis
%           1: Early termination (tolerance mode only). Occurs when:
%              (a) rtol < machine epsilon (tolerance too tight), or
%              (b) sketch size grew to min(m,n) without meeting tolerance,
%                  indicating matrix is effectively full-rank at this tolerance
%              When flag=1, Q is empty (m x 0).
%   diagR - Diagonal elements from pivoted QR factorization, representing
%           column norms of the sketched matrix (sorted in decreasing order)
%
% Note:
%   Higher-level functions (qr_sketch, svd_sketch, id_sketch) automatically
%   fall back to deterministic (full) QR or SVD when orth_sketch terminates
%   early, so users of those functions do not need to handle flag=1 explicitly.

% Parse optional parameters
  p = inputParser;
  addParameter(p, 'block_size', 42);
  addParameter(p, 'power_iter', 0);
  addParameter(p, 'rng', []);
  parse(p, varargin{:});

  block_size = p.Results.block_size;
  power_iter = p.Results.power_iter;
  rng_param = p.Results.rng;

  [m, n] = size(A);
  is_complex_op = librla.is_complex_type(A);
  dtype_str = librla.get_dtype_string(A);

  % Rank mode (rtol >= 1): single sketch with rank filtering
  if rtol >= 1
      kmax = floor(rtol);
      x = librla.uniform_omega(n, block_size, is_complex_op, dtype_str, rng_param);
      x = librla.power_iteration(A, x, power_iter);
      y = librla.matvec(A, x);
      [Q, R, ~] = qr(y, 0);

      % Use requested rank directly (capped at available columns)
      diagR = abs(diag(R));
      rank = min(kmax, size(Q, 2));

      Q = Q(:, 1:rank);
      flag = 0;
      return;
  end

  % Tolerance mode (rtol < 1): geometric growth with tolerance checking
  if rtol < eps(dtype_str)
      Q = zeros(m, 0, dtype_str);
      flag = 1;
      diagR = zeros(0, 1, dtype_str);
      return;
  end

  if block_size >= min(m, n)
      Q = zeros(m, 0, dtype_str);
      flag = 1;
      diagR = zeros(0, 1, dtype_str);
      return;
  end

  % Main loop with geometric growth
  while true
      x = librla.uniform_omega(n, block_size, is_complex_op, dtype_str, rng_param);
      x = librla.power_iteration(A, x, power_iter);
      y = librla.matvec(A, x);
      [Q, R, ~] = qr(y, 0);

      % Check tolerance
      diagR = abs(diag(R));
      if isempty(diagR) || diagR(1) == 0
          d = 0.0;
      else
          d = diagR(end) / diagR(1);
      end
      if d <= rtol
          flag = 0;
          return;
      end

      % Grow block size
      block_size = min(block_size * 4, min(m, n));
      if block_size >= min(m, n)
          Q = zeros(m, 0, dtype_str);
          flag = 1;
          diagR = zeros(0, 1, dtype_str);
          return;
      end
  end
end

function [Q, R, p] = qr_sketch(A, rtol, varargin)
% QR_SKETCH - Compute truncated QR factorization with column pivoting via randomized sketching
%
% Syntax:
%   [Q, R, p] = librla.qr_sketch(A, rtol)
%   [Q, R, p] = librla.qr_sketch(A, rtol, Name, Value)
%
% Description:
%   The algorithm sketches an orthonormal basis for the column space of A,
%   projects A onto this basis, computes the QR of the smaller projected matrix,
%   and then expands back to the original space. If the matrix is effectively
%   full rank a deterministic QR is performed.
%
%   This is much faster than full QR for matrices where the target rank k is
%   much smaller than min(m,n).
%
% Input Arguments:
%   A             - Input matrix (m x n) or LinearOperator
%   rtol          - Relative tolerance (< 1) or target rank (>= 1)
%                   - Tolerance mode: keep columns with norm >= rtol * max_norm
%                   - Rank mode: return k leading columns
%   block_size    - Sketch size for tolerance mode (default: 42)
%   power_iter    - Number of power iterations for accuracy (default: 0)
%   extra_samples - Oversampling for rank mode (default: 12)
%                   Rank mode uses block_size = rank + extra_samples
%   rng           - Random number generator state or seed (default: [] uses current state)
%
% Output Arguments:
%   Q - Orthonormal matrix (m x k), k <= min(m,n)
%   R - Upper triangular matrix (k x n)
%   p - Column permutation vector (1-based indexing), length n
%       The decomposition satisfies A(:, p) ≈ Q*R

% Parse optional parameters
  p_parser = inputParser;
  addParameter(p_parser, 'block_size', 42);
  addParameter(p_parser, 'power_iter', 0);
  addParameter(p_parser, 'extra_samples', 12);
  addParameter(p_parser, 'rng', []);
  parse(p_parser, varargin{:});

  block_size = p_parser.Results.block_size;
  power_iter = p_parser.Results.power_iter;
  extra_samples = p_parser.Results.extra_samples;
  rng_param = p_parser.Results.rng;

  [m, n] = size(A);
  is_matrix_free = isa(A, 'LinearOperator');
  dtype_str = librla.get_dtype_string(A);

  % Rank mode vs tolerance mode
  rank_mode = false;
  if rtol >= 1
      rank_mode = true;
      kmax = floor(rtol);
      block_size = kmax + extra_samples;
  elseif is_matrix_free
      error('Matrix-free operators only supported in rank mode (rtol >= 1)');
  end

  % Compute sketch: in rank mode, request all oversampled columns
  % for better accuracy (truncate to kmax after QR)
  if rank_mode
      orth_rtol = block_size;
  else
      orth_rtol = rtol;
  end
  [Qs, flag, ~] = librla.orth_sketch(A, orth_rtol, ...
                  'block_size', block_size, ...
                  'power_iter', power_iter, ...
                  'rng', rng_param);

  k = size(Qs, 2);
  if flag ~= 0
      k = min(m, n);
  end

  % Fallback to full QR if needed
  needs_fallback = (flag ~= 0 || k >= min(m, n));
  if needs_fallback && rank_mode
      needs_fallback = false;
  end

  if needs_fallback
      A = librla.get_matrix(A);
      [Q, R, p] = qr(A, 0);

      % Determine rank
      if rank_mode
          rank = min(kmax, size(Q, 2));
      else
          rank = librla.rank_from_diag(diag(R), rtol);
      end

      Q = Q(:, 1:rank);
      R = R(1:rank, :);
      return;
  end

  % Project and compute QR
  B = librla.matmat_left(Qs, A);
  [Qproj, R, p] = qr(B, 0);
  Q = Qs * Qproj;

  % Determine rank
  if rank_mode
      rank = min(kmax, size(Q, 2));
  else
      rank = librla.rank_from_diag(diag(R), rtol);
  end

  Q = Q(:, 1:rank);
  R = R(1:rank, :);
end

function [U, s, V] = svd_sketch(A, rtol, varargin)
% SVD_SKETCH - Compute truncated singular value decomposition (SVD) via randomized sketching
%
% Syntax:
%   [U, s, V] = librla.svd_sketch(A, rtol)
%   [U, s, V] = librla.svd_sketch(A, rtol, Name, Value)
%
% Description:
%   The algorithm sketches an orthonormal basis for the column space of A,
%   projects A onto this basis, computes the SVD of the smaller projected matrix,
%   and then expands back to the original space. If the matrix is effectively
%   full rank a deterministic SVD is performed.
%
%   This is much faster than full SVD for matrices where the target rank k is
%   much smaller than min(m,n).
%
% Input Arguments:
%   A             - Input matrix (m x n) or LinearOperator
%   rtol          - Relative tolerance (< 1) or target rank (>= 1)
%                   - Tolerance mode: keep singular values >= rtol * s(1)
%                   - Rank mode: return k leading singular triplets
%   block_size    - Sketch size for tolerance mode (default: 42)
%   power_iter    - Number of power iterations for accuracy (default: 0)
%   extra_samples - Oversampling for rank mode (default: 12)
%   rng           - Random number generator state or seed (default: [] uses current state)
%
% Output Arguments:
%   U - Left singular vectors (m x k), orthonormal columns
%   s - Singular values (length k), sorted descending
%   V - Right singular vectors (n x k), orthonormal columns
%       The decomposition satisfies A ≈ U*diag(s)*V'

% Parse optional parameters
  p = inputParser;
  addParameter(p, 'block_size', 42);
  addParameter(p, 'power_iter', 0);
  addParameter(p, 'extra_samples', 12);
  addParameter(p, 'rng', []);
  parse(p, varargin{:});

  block_size = p.Results.block_size;
  power_iter = p.Results.power_iter;
  extra_samples = p.Results.extra_samples;
  rng_param = p.Results.rng;

  [m, n] = size(A);
  is_matrix_free = isa(A, 'LinearOperator');
  dtype_str = librla.get_dtype_string(A);

  % Handle wide matrices via transpose
  if m < n
      [V, s, U] = librla.svd_sketch(A', rtol, 'block_size', block_size, 'power_iter', power_iter, 'extra_samples', extra_samples, 'rng', rng_param);
      return;
  end

  % Rank mode vs tolerance mode
  rank_mode = false;
  if rtol >= 1
      rank_mode = true;
      kmax = floor(rtol);
      block_size = kmax + extra_samples;
  elseif is_matrix_free
      error('Matrix-free operators only supported in rank mode (rtol >= 1)');
  end

  % Compute sketch: in rank mode, request all oversampled columns
  % to get more accurate singular values (truncate to kmax after SVD)
  if rank_mode
      orth_rtol = block_size;
  else
      orth_rtol = rtol;
  end
  [Qs, flag, ~] = librla.orth_sketch(A, orth_rtol, ...
                  'block_size', block_size, 'power_iter', power_iter, 'rng', rng_param);

  k = size(Qs, 2);
  if flag ~= 0
      k = min(m, n);
  end

  % Fallback to full SVD if needed
  needs_fallback = (flag ~= 0 || k >= min(m, n));
  if needs_fallback && rank_mode
      needs_fallback = false;
  end

  if needs_fallback
      A_mat = librla.get_matrix(A);
      [U, S, V] = svd(A_mat, 'econ');
      s = diag(S);

      % Determine rank
      if rank_mode
          rank = min(kmax, length(s));
      else
          rank = librla.rank_from_svals(s, rtol);
      end

      U = U(:, 1:rank);
      V = V(:, 1:rank);
      s = s(1:rank);
      return;
  end

  % Project and compute SVD
  Aproj = librla.matmat_left(Qs, A);
  [Uproj, S, V] = svd(Aproj, 'econ');
  s = diag(S);
  U = Qs * Uproj;

  % Determine rank
  if rank_mode
      rank = min(kmax, length(s));
  else
      rank = librla.rank_from_svals(s, rtol);
  end

  U = U(:, 1:rank);
  V = V(:, 1:rank);
  s = s(1:rank);
end

function [k, piv, T] = id_sketch(A, rtol, varargin)
% ID_SKETCH - Compute interpolative decomposition via randomized sketching
%
% Syntax:
%   [k, piv, T] = librla.id_sketch(A, rtol)
%   [k, piv, T] = librla.id_sketch(A, rtol, Name, Value)
%
% Description:
%   An ID represents a matrix A by selecting k of its columns and expressing
%   the remaining columns as linear combinations of the selected ones:
%
%     A(:, piv(k+1:end)) ≈ A(:, piv(1:k)) * T
%
%   where piv is a column permutation and T is a k x (n-k) interpolation matrix.
%   The selected columns (skeleton) capture the essential features of A, while
%   T provides the coefficients to reconstruct the other columns.
%
%   This function uses qr_sketch() to identify the column permutation.
%
% Input Arguments:
%   A             - Input matrix (m x n) or LinearOperator
%   rtol          - Relative tolerance (< 1) or target rank (>= 1)
%   block_size    - Sketch size for tolerance mode (default: 42)
%   power_iter    - Number of power iterations for accuracy (default: 0)
%   extra_samples - Oversampling for rank mode (default: 12)
%   method        - Method for computing T matrix (default: 'fast')
%            'fast'  - Triangular solve R11 \ R12 (fastest)
%            'svd'   - SVD-based pseudoinverse
%            'lstsq' - Least-squares from original A (most accurate, slowest)
%   rng           - Random number generator state or seed (default: [] uses current state)
%
% Output Arguments:
%   k   - Rank of the approximation (number of skeleton columns)
%   piv - Column permutation (1-based indexing), length n
%         piv(1:k) are indices of skeleton columns
%         piv(k+1:end) are indices of interpolated columns
%   T   - Interpolation matrix (k x (n-k))
%         The approximation is A(:, piv(k+1:end)) ≈ A(:, piv(1:k)) * T

% Parse optional parameters
  p = inputParser;
  addParameter(p, 'block_size', 42);
  addParameter(p, 'power_iter', 0);
  addParameter(p, 'extra_samples', 12);
  addParameter(p, 'method', 'fast', @(x) ismember(x, {'fast', 'svd', 'lstsq'}));
  addParameter(p, 'rng', []);
  parse(p, varargin{:});

  block_size = p.Results.block_size;
  power_iter = p.Results.power_iter;
  extra_samples = p.Results.extra_samples;
  method = p.Results.method;
  rng_param = p.Results.rng;

  % Get QR factorization
  [~, R, piv] = librla.qr_sketch(A, rtol, 'block_size', block_size, ...
                'power_iter', power_iter, 'extra_samples', extra_samples, 'rng', rng_param);

  k = size(R, 1);

  % Compute rtol for SVD filtering
  [m, n] = size(A);
  if rtol >= 1
      % Rank mode: minimal filtering (only exact zeros)
      rtol_for_svd = 0;
  else
      % Tolerance mode: use the provided tolerance
      rtol_for_svd = rtol;
  end

  % Dispatch to shared helper functions
  if strcmp(method, 'lstsq')
      T = librla.compute_T_lstsq(A, R, piv, k);
  elseif strcmp(method, 'svd')
      T = librla.compute_T_svd(R, k, rtol_for_svd);
  elseif strcmp(method, 'fast')
      T = librla.compute_T_fast(R, k);
  end
end

function [k, piv, T] = id_qrpiv(A, rtol, varargin)
% ID_QRPIV - Interpolative decomposition via deterministic QR with column pivoting
%
% Syntax:
%   [k, piv, T] = librla.id_qrpiv(A, rtol)
%   [k, piv, T] = librla.id_qrpiv(A, rtol, Name, Value)
%
% Description:
%   An ID represents a matrix A by selecting k of its columns and expressing
%   the remaining columns as linear combinations of the selected ones:
%
%     A(:, piv(k+1:end)) ≈ A(:, piv(1:k)) * T
%
%   where piv is a column permutation and T is a k x (n-k) interpolation matrix.
%   The selected columns (skeleton) capture the essential features of A, while
%   T provides the coefficients to reconstruct the other columns.
%
%   This function provides a deterministic alternative to id_sketch by
%   computing the interpolative decomposition using only QR with
%   column pivoting (LAPACK geqp3), without any randomized
%   sketching. It preserves LinearOperator support and uses the same T
%   matrix computation logic as id_sketch.
%
% Input Arguments:
%   A      - Input matrix or LinearOperator
%   rtol   - Tolerance (< 1) or rank (>= 1)
%   method - T computation method: 'fast', 'svd', 'lstsq' (default: 'fast')
%            'fast'  - Triangular solve R11 \ R12 (fastest)
%            'svd'   - SVD-based pseudoinverse
%            'lstsq' - Least-squares from original A (most accurate, slowest)
%
% Output Arguments:
%   k   - Rank
%   piv - Column permutation (1-based)
%   T   - Interpolation matrix, size (k, n-k)
%         The approximation is A(:, piv(k+1:end)) ≈ A(:, piv(1:k)) * T

% Parse optional parameters
  p = inputParser;
  addParameter(p, 'method', 'fast', @(x) ismember(x, {'fast', 'svd', 'lstsq'}));
  parse(p, varargin{:});

  method = p.Results.method;

  is_linop = isa(A, 'LinearOperator');
  is_matrix_free = is_linop && isempty(A.matrix);

  [m, n] = size(A);

  % Determine rank mode vs tolerance mode
  rank_mode = false;
  if rtol >= 1
      rank_mode = true;
      kmax = floor(rtol);
  end

  % Compute full QR with pivoting (deterministic)
  A_mat = librla.get_matrix(A);
  [Q, R, piv] = qr(A_mat, 0);  % Economy size QR with pivoting

  % Determine rank
  if rank_mode
      rank = min(kmax, min(m, n));
      rtol_for_svd = 0;  % Minimal filtering in rank mode
  else
      rank = librla.rank_from_diag(diag(R), rtol);
      rtol_for_svd = rtol;
  end

  k = rank;

  % Handle edge cases
  if k == 0
      T = zeros(0, n, class(R));
      return;
  end

  if k == n
      T = zeros(k, 0, class(R));
      return;
  end

  % Dispatch to shared helper functions
  if strcmp(method, 'lstsq')
      T = librla.compute_T_lstsq(A, R, piv, k);
  elseif strcmp(method, 'svd')
      T = librla.compute_T_svd(R, k, rtol_for_svd);
  elseif strcmp(method, 'fast')
      T = librla.compute_T_fast(R, k);
  end
end

end % methods (Static)


methods (Static, Access = private)

function x = power_iteration(A, x, power_iter)
% POWER_ITERATION - Apply (A'*A)^n and orthogonalize
  for i = 1:power_iter
      x = librla.rmatvec(A, librla.matvec(A, x));
      [x, ~, ~] = qr(x, 0);
  end
end

function omega = uniform_omega(n, block_size, is_complex, dtype_str, rng_param)
% UNIFORM_OMEGA - Generate uniform[-1,1] test matrix
  % Set rng state if provided
  if ~isempty(rng_param)
      rng(rng_param);
  end
  if is_complex
      omega = 2 * rand(n, block_size, dtype_str) - 1 + 1i * (2 * rand(n, block_size, dtype_str) - 1);
  else
      omega = 2 * rand(n, block_size, dtype_str) - 1;
  end
end

function omega = gaussian_omega(n, block_size, is_complex, dtype_str, rng_param)
% GAUSSIAN_OMEGA - Generate Gaussian test matrix
  % Set rng state if provided
  if ~isempty(rng_param)
      rng(rng_param);
  end
  if is_complex
      omega = randn(n, block_size, dtype_str) + 1i * randn(n, block_size, dtype_str);
  else
      omega = randn(n, block_size, dtype_str);
  end
end

function y = matvec(A, x)
% MATVEC - Apply operator A to vector/matrix x
  if isa(A, 'LinearOperator')
      y = A.matvec(x);
  else
      y = A * x;
  end
end

function y = rmatvec(A, x)
% RMATVEC - Apply adjoint A' to vector/matrix x
  if isa(A, 'LinearOperator')
      y = A.rmatvec(x);
  else
      y = A' * x;
  end
end

function B = matmat_left(Q, A)
% MATMAT_LEFT - Compute Q' * A
  if isa(A, 'LinearOperator')
      % Use adjoint: Q' * A = (A' * Q)'
      B = librla.rmatvec(A, Q)';
  else
      B = Q' * A;
  end
end

function rank = rank_from_svals(s, rtol)
% RANK_FROM_SVALS - Determine numerical rank from singular values
  if isempty(s)
      rank = 0;
  else
      rank = sum(s >= rtol * s(1));
  end
end

function rank = rank_from_diag(diag_vals, rtol)
% RANK_FROM_DIAG - Determine numerical rank from diagonal elements (e.g., from QR)
  diag_abs = abs(diag_vals);
  if isempty(diag_abs) || diag_abs(1) <= 0
      rank = 0;
  else
      rank = sum(diag_abs >= rtol * diag_abs(1));
  end
end

function result = is_complex_type(A)
% IS_COMPLEX_TYPE - Check if A represents complex data
  if isa(A, 'LinearOperator')
      result = A.is_complex;
  else
      result = ~isreal(A);
  end
end

function dtype_str = get_dtype_string(A)
% GET_DTYPE_STRING - Get dtype string for eps() and zeros() calls
  if isa(A, 'LinearOperator')
      dtype_str = A.dtype;
  else
      dtype_str = class(A);
  end
end

function A_mat = get_matrix(A)
% GET_MATRIX - Extract explicit matrix from LinearOperator or return matrix
  if isa(A, 'LinearOperator')
      if ~isempty(A.matrix)
          A_mat = A.matrix;
      else
          error('Cannot extract explicit matrix from matrix-free LinearOperator');
      end
  else
      A_mat = A;
  end
end

function T = compute_T_lstsq(A, R, piv, k)
% Compute T using least-squares from original A columns
  [m, n] = size(A);

  if k == 0 || k >= n
      T = zeros(k, n - k, class(R));
      return;
  end

  is_linop = isa(A, 'LinearOperator');
  is_matrix_free = is_linop && isempty(A.matrix);

  if is_matrix_free
      % Extract skeleton columns via unit vectors
      skeleton_cols = zeros(m, k, class(R));
      for j = 1:k
          e_j = zeros(n, 1, class(R));
          e_j(piv(j)) = 1.0;
          skeleton_cols(:, j) = librla.matvec(A, e_j);
      end

      remaining_cols = zeros(m, n - k, class(R));
      for j = 1:(n - k)
          e_j = zeros(n, 1, class(R));
          e_j(piv(k + j)) = 1.0;
          remaining_cols(:, j) = librla.matvec(A, e_j);
      end

      T = skeleton_cols \ remaining_cols;
  else
      A_mat = librla.get_matrix(A);
      cols = piv(1:k);
      remaining = piv((k+1):end);
      T = A_mat(:, cols) \ A_mat(:, remaining);
  end
end

function T = compute_T_svd(R, k, rtol_for_svd)
% Compute T using SVD-based pseudoinverse of R11
%
% Parameters:
%   R            - R factor from QR decomposition
%   k            - Rank
%   rtol_for_svd - Tolerance for filtering small singular values
%                  (should be actual tolerance, not rank)

  [~, n] = size(R);

  if k == 0
      T = zeros(k, n - k, class(R));
      return;
  end

  R11 = R(1:k, 1:k);
  R12 = R(1:k, (k+1):end);

  % MATLAB's svd returns V (not V'); R11^+ = V * Σ^-1 * U'.
  [U, S, V] = svd(R11, 'econ');
  s = diag(S);

  % Filter small singular values
  keep = s >= rtol_for_svd * max(s);
  if ~any(keep)
      T = zeros(size(R12), class(R));
  else
      inv_s = 1 ./ s(keep);
      T = V(:, keep) * diag(inv_s) * (U(:, keep)' * R12);
  end
end

function T = compute_T_fast(R, k)
% Compute T using fast triangular solve
  [~, n] = size(R);

  if k == 0
      T = zeros(k, n - k, class(R));
      return;
  end

  R11 = R(1:k, 1:k);
  R12 = R(1:k, (k+1):end);
  T = R11 \ R12;
end

end % methods (Static, Access = private)

end
