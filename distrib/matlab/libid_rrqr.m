%==========================================================================
% libid_rrqr - Interpolative Decomposition using Deterministic RRQR
%
%   This module provides ID implementation using MATLAB's built-in QR
%   with column pivoting, which internally calls LAPACK's dgeqp3/zgeqp3
%   (strong rank-revealing QR):
%
%       * id_rrqr    - interpolative decomposition (ID) using build-in QR
%
%   This serves as a comparison baseline to libid's randomized sketching
%   approach, representing what deterministic methods achieve.
%
%   Key differences from libid.id_sketch:
%   - Deterministic (same results every run)
%   - Examines all columns sequentially
%   - No early stopping capability (must compute full QR)
%   - Generally slower on large matrices
%   - No randomization overhead
%
%   Author : Your Name
%   SPDX-License-Identifier : TBD
%==========================================================================

classdef libid_rrqr
    %LIBID_RRQR  Interpolative decomposition using deterministic RRQR
    %
    %   This class provides a static method for computing interpolative
    %   decompositions using MATLAB's built-in QR factorization with
    %   column pivoting (LAPACK geqp3).
    %
    %   See also: qr, libid.id_sketch

    methods (Static)

        function [k, piv, T] = id_rrqr(A, rtol, kmax)
        % ID_RRQR  Interpolative decomposition using deterministic RRQR.
        %
        %   [k,piv,T] = libid_rrqr.id_rrqr(A, rtol)
        %   [k,piv,T] = libid_rrqr.id_rrqr(A, rtol, kmax)
        %
        % Description
        % -----------
        %   Computes an interpolative decomposition using MATLAB's built-in
        %   QR with column pivoting. This is equivalent to:
        %   1. Compute full pivoted QR: A(:,piv) = Q * R
        %   2. Determine rank k from diagonal of R
        %   3. Partition R = [R11 R12; 0 R22]
        %   4. Solve T = R11 \ R12
        %
        %   Result: A(:,piv(k+1:end)) ~= A(:,piv(1:k)) * T
        %
        % Parameters
        % ----------
        %   A     : numeric matrix, shape (m,n) - real or complex
        %   rtol  : positive scalar - tolerance for rank determination
        %           If rtol < 1: relative tolerance (k where R(k+1,k+1) < rtol*R(1,1))
        %           If rtol >= 1: interpreted as maximum rank
        %   kmax  : integer, optional - maximum rank (default: min(m,n))
        %
        % Returns
        % -------
        %   k     : integer - determined rank
        %   piv   : integer vector, shape (n,) - column permutation (1-based)
        %           A(:,piv) = A(:,piv(1:k)) * [I; T]
        %   T     : numeric matrix, shape (k, n-k) - interpolation matrix
        %
        % Notes
        % -----
        %   * This is a deterministic algorithm (same results every run)
        %   * Uses MATLAB's built-in qr(...,0) which calls LAPACK geqp3
        %   * No early stopping - always computes full QR factorization
        %   * Generally slower than randomized methods on large matrices
        %   * More accurate than randomized methods due to exhaustive search
        %
        % Example
        % -------
        %   % Low-rank matrix
        %   A = libid.hilb(400,250);
        %   [k,piv,T] = libid_rrqr.id_rrqr(A, 1e-10);
        %
        %   % Verify reconstruction
        %   A_skel = A(:,piv(k+1:end));
        %   A_basis = A(:,piv(1:k));
        %   err = norm(A_skel - A_basis*T, 'fro') / norm(A, 'fro');
        %   fprintf('Relative error, RRQR: %.3e\n', err);
        %
        %   % Compare with libid randomized version
        %   [k2,piv2,T2] = libid.id_sketch(A, 1e-10);
        %   A_skel2 = A(:,piv2(k2+1:end));
        %   A_basis2 = A(:,piv2(1:k2));
        %   err = norm(A_skel2 - A_basis2*T2, 'fro') / norm(A, 'fro');
        %   fprintf('Relative error, libid: %.3e\n', err);
        %
        %   fprintf('Ranks: RRQR=%d, libid=%d\n', k, k2);
        %
        % Performance
        % -----------
        %   Time complexity: O(m*n*min(m,n)) - full QR factorization
        %   Best use cases:
        %   - Small to medium matrices (< 1000x1000)
        %   - When determinism is required
        %   - As baseline for comparing randomized methods
        %
        %   Not recommended for:
        %   - Large matrices (> 2000x2000)
        %   - Low-rank matrices where early stopping would help
        %
        % See also
        % --------
        %   libid.id_sketch    : Randomized ID (usually faster)
        %   qr                 : MATLAB's built-in QR factorization
        % -----------------------------------------------------------------

            % Parse inputs
            if nargin < 3
                kmax = min(size(A));
            end

            [m, n] = size(A);

            % Compute full pivoted QR factorization
            % [Q,R,piv] = qr(A,0) computes economy QR with column pivoting
            % A(:,piv) = Q * R where R is upper triangular
            [Q, R, piv] = qr(A, 0);

            % Determine rank from diagonal of R
            if rtol >= 1
                % Rank mode: rtol specifies maximum rank directly
                k = min([floor(rtol), kmax, min(m,n)]);
            else
                % Tolerance mode: find rank where diagonal drops below threshold
                diagR = abs(diag(R));
                if isempty(diagR) || diagR(1) == 0
                    k = 0;
                else
                    % Find first index where R(k+1,k+1) < rtol * R(1,1)
                    threshold = rtol * diagR(1);
                    k = find(diagR < threshold, 1, 'first') - 1;
                    if isempty(k)
                        k = length(diagR);
                    end
                    k = min(k, kmax);
                end
            end

            % Handle edge cases
            if k == 0 || k >= n
                % Full rank or zero rank - no interpolation needed
                T = zeros(k, n-k);
                return;
            end

            % Partition R = [R11  R12]
            %               [  0  R22]
            % We only need the first k rows
            R11 = R(1:k, 1:k);
            R12 = R(1:k, k+1:end);

            % Solve for interpolation matrix: R11 * T = R12
            % Use upper-triangular solve (efficient and stable)
            if rcond(R11) < eps(class(A))
                % R11 is near-singular, use SVD-based solve
                warning('libid_rrqr:IllConditioned', ...
                    'R11 is ill-conditioned (rcond=%.2e). Using SVD.', rcond(R11));
                % Use SVD-based solve with singular value truncation
                [U,S,V] = svd(R11, 'econ');
                s = diag(S);
                rtol_svd = max(size(R11)) * eps(class(R11));
                keep = s >= rtol_svd * max(s);
                if ~any(keep)
                    T = zeros(size(R12), class(R12));
                else
                    inv_s = 1 ./ s(keep);
                    % Right-associative: V * (inv_s * (U' * R12))
                    T = V(:,keep) * (diag(inv_s) * (U(:,keep)' * R12));
                end
            else
                % Standard triangular solve
                T = R11 \ R12;
            end

            % Note: piv is already in MATLAB's 1-based indexing
            % (unlike Python which uses 0-based)
        end

    end
end
