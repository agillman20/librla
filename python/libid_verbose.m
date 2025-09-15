% ----------
% This module implements several randomized routines that approximate the
% column space, QR factorization, singular-value decomposition (SVD), and
% interpolative decomposition (ID) of a matrix A.
%
% User-callable methods
% ---------------------
%   orth_sketch        - Build an orthonormal basis for the column space.
%   rrqr_randomized    - Rank-revealing QR using a randomized basis.
%   rrsvd_randomized   - Truncated SVD using a randomized basis.
%   rrid_randomized    - Interpolative decomposition using randomized QR.
%
% Author: Your Name
% SPDX-License-Identifier: TBD


classdef libid
    % libid
    %
    % Collection of static methods that perform
    % randomized matrix factorizations.
    %
    % See also: rrid_randomized, rrqr_randomized, rrsvd_randomized,
    %           orth_sketch

    methods (Static)

        function [k, Q] = orth_sketch(A, rtol, block_size, flag_power)
        % ORTH_SKETCH Compute an orthonormal basis for the column space of A using random sketching.
        %
        % Calling sequence (options shown)
        % ---------------------------------
        %   [k,Q]          = libid.orth_sketch(A,rtol);
        %   [k,Q]          = libid.orth_sketch(A,rtol,block_size);
        %   [k,Q]          = libid.orth_sketch(A,rtol,block_size,flag_power);
        %
        % Description
        % -----------
        % The algorithm draws a random matrix, optionally improves it with
        % power iteration, and then performs a QR factorization with
        % column pivoting.  The process repeats with a larger sketch until
        % the smallest diagonal element of R, relative to the column
        % norms, falls below `rtol`.
        %
        % Parameters
        % ----------
        % A          : double matrix
        %   Input matrix.
        % rtol       : double
        %   Relative tolerance that determines when to stop sketching.
        % block_size : int, optional (default = 42)
        %   Initial number of random vectors.
        % flag_power : int, optional (default = 0)
        %   Number of power-iteration steps.
        %
        % Returns
        % -------
        % k          : int
        %   Number of basis vectors found (may equal min(m,n)).
        % Q          : double matrix
        %   Orthonormal basis matrix Q of size (m, k). If the rank
        %   equals min(m, n) an empty array with shape (m, 0) is returned.
        %
        % Notes
        % -----
        % 1. If the initial block already covers the whole space,
	%    the function returns early.
        % 2. The random matrix X has entries in [-1,1].
        % 3. Power iteration is performed by the private method _power_iteration.
        %
        % Example
        % -------
        %   A = libid.hilb(4000,2000);
        %   tic, [k,Q] = libid.orth_sketch(A,1e-8); toc
        %   k, size(Q)
        %
        %   -------------------------------------------------
        %   Code flow
        %   -------------------------------------------------
        %   1. Determine matrix dimensions.
        %   2. Check early-exit condition.
        %   3. Loop:
        %        a) Generate random test matrix X.
        %        b) Apply power iteration (if flag_power > 0).
        %        c) Form Y = A*X and compute its QR factorization.
        %        d) Estimate residual and compare with rtol.
        %        e) Increase block size if needed.
        %   4. Return block size and orthonormal basis Q.
        %   -------------------------------------------------
        %
        % See also: rrqr_randomized, rrsvd_randomized, rrid_randomized

            if nargin < 4, flag_power = 0; end
            if nargin < 3, block_size = 42; end

            [m, n] = size(A);

            % If the initial block already covers the whole space, return early.
            if block_size >= min(m,n)
                k = min(m,n);
                Q = [];
                return
            end

            while true
                % Random matrix with entries in [-1,1]
                X = 2*rand(n, block_size) - 1;
                X = libid._power_iteration(A, X, flag_power);

                Y = A * X;
                [Qtmp,R,~] = qr(Y,0);   % economy QR

                % Use the last diagonal entry of R as a proxy for the residual.
                rdiag = diag(R);
                residual = max(abs(rdiag(end))) / max(vecnorm(Y,2,1));

                if residual <= rtol
                    k = block_size;
                    Q = Qtmp;
                    return
                end

                % If residual is too large, increase the block size.
                block_size = min(block_size*4, min(m,n));

                if block_size >= min(m,n)
                    k = min(m,n);
                    Q = [];
                    return
                end
            end
        end


        function [Qk, Rk, p] = rrqr_randomized(A, rtol, block_size, flag_power)
        % RRQR_RANDOMIZED Rank-revealing QR factorization using a randomized basis.
        %
        % Calling sequence (options shown)
        % ---------------------------------
        %   [Q,R,p]        = libid.rrqr_randomized(A,rtol);
        %   [Q,R,p]        = libid.rrqr_randomized(A,rtol,block_size);
        %   [Q,R,p]        = libid.rrqr_randomized(A,rtol,block_size,flag_power);
        %
        % Description
        % -----------
        % Compute a rank-revealing QR factorization of A by first building a
        % randomized orthonormal basis for the column space.  If the matrix
        % is effectively full rank, a deterministic QR is performed.
        %
        % Parameters
        % ----------
        % A          : double matrix
        %   Input matrix.
        % rtol       : double
        %   Relative tolerance for rank determination.
        % block_size : int, optional (default = 42)
        %   Initial number of random vectors.
        % flag_power : int, optional (default = 0)
        %   Number of power-iteration steps.
        %
        % Returns
        % -------
        % Qk         : double matrix
        %   Leading k columns of the orthogonal factor.
        % Rk         : double matrix
        %   Leading k rows of the upper-triangular factor.
        % p          : int vector
        %   Pivot permutation vector.
        %
        % Notes
        % -----
        % 1. The rank k is chosen as the number of rows of R whose 2-norm
        %    exceeds ``rtol * ||A||`` (or ``||A_proj||`` for the projected case).
        % 2. The private method ``_power_iteration`` is used internally.
        %
        % Example
        % -------
        %   A = libid.hilb(4000,2000);
        %   tic, [Q,R,p] = libid.rrqr_randomized(A,1e-15); toc
        %   rel_err = norm(Q*R - A(:,p),'fro')/norm(A,'fro')
        %   tic, [Q,R,p] = qr(A,'econ'); toc
        %   rel_err = norm(Q*R - A(:,p),'fro')/norm(A,'fro')
        %
        %   -------------------------------------------------
        %   Code flow
        %   -------------------------------------------------
        %   1. Call orth_sketch to obtain basis Q_basis.
        %   2. If full rank, compute deterministic QR of A.
        %   3. Otherwise project A onto the basis and QR the small matrix.
        %   4. Determine numerical rank k from R.
        %   5. Return truncated factors and pivot vector.
        %   -------------------------------------------------
        %
        % See also: orth_sketch, rrsvd_randomized, rrid_randomized

            if nargin < 4, flag_power = 0; end
            if nargin < 3, block_size = 42; end

            [m, n] = size(A);
            [k, Q_basis] = libid.orth_sketch(A, rtol, block_size, flag_power);

            if k >= min(m,n)
                [Q,R,p] = qr(A,0);
                k = sum(vecnorm(R,2,2) >= rtol*norm(A,'fro'));
                Qk = Q(:,1:k);
                Rk = R(1:k,:);
                return
            end

            % Project A onto the basis and factor the small matrix.
            A_proj = Q_basis' * A;
            [Q_proj,R,p] = qr(A_proj,0);
            Qk = Q_basis * Q_proj;
            k = sum(vecnorm(R,2,2) >= rtol*norm(A_proj,'fro'));
            Qk = Qk(:,1:k);
            Rk = R(1:k,:);
        end


        function [Uk, sk, Vk] = rrsvd_randomized(A, rtol, block_size, flag_power)
        % RRSVD_RANDOMIZED Truncated singular-value decomposition using a randomized basis.
        %
        % Calling sequence (options shown)
        % ---------------------------------
        %   [U,s,Vt]       = libid.rrsvd_randomized(A,rtol);
        %   [U,s,Vt]       = libid.rrsvd_randomized(A,rtol,block_size);
        %   [U,s,Vt]       = libid.rrsvd_randomized(A,rtol,block_size,flag_power);
        %
        % Description
        % -----------
        % Compute a truncated SVD of A by first constructing a randomized
        % orthonormal basis for the column space.  If A is effectively full
        % rank, the full deterministic SVD is performed.
        %
        % Parameters
        % ----------
        % A          : double matrix
        %   Input matrix.
        % rtol       : double
        %   Relative tolerance for truncation.
        % block_size : int, optional (default = 42)
        %   Initial number of random vectors.
        % flag_power : int, optional (default = 0)
        %   Number of power-iteration steps.
        %
        % Returns
        % -------
        % Uk         : double matrix
        %   Leading k left singular vectors.
        % sk         : double vector
        %   Leading k singular values.
        % Vk         : double matrix
        %   Leading k right singular vectors (rows of V^H).
        %
        % Notes
        % -----
        % 1. The rank k is the number of singular values greater than
        %    ``rtol * ||A||`` (or ``||A_proj||`` for the projected case).
        %
        % Example
        % -------
        %   A = libid.hilb(4000,2000);
        %   tic, [U,s,Vt] = libid.rrsvd_randomized(A,1e-15); toc
        %   rel_err = norm(U*diag(s)*Vt - A,'fro')/norm(A,'fro')
        %   tic, [U,S,V] = svd(A,'econ'); toc
        %   rel_err = norm(U*S*V' - A,'fro')/norm(A,'fro')
        %
        %   -------------------------------------------------
        %   Code flow
        %   -------------------------------------------------
        %   1. Obtain basis Q_basis via orth_sketch.
        %   2. If full rank, call deterministic svd.
        %   3. Otherwise form A_proj = Q_basis' * A.
        %   4. Compute svd of the small matrix.
        %   5. Lift left singular vectors back: U = Q_basis * U_proj.
        %   6. Truncate to k based on rtol.
        %   -------------------------------------------------
        %
        % See also: orth_sketch, rrqr_randomized, rrid_randomized

            if nargin < 4, flag_power = 0; end
            if nargin < 3, block_size = 42; end

            [m, n] = size(A);
            [k, Q_basis] = libid.orth_sketch(A, rtol, block_size, flag_power);

            if k >= min(m,n)
                [U,S,V] = svd(A, 'econ');
                k = sum(abs(diag(S)) >= rtol*norm(A,'fro'));
                Uk = U(:,1:k);
                sk = diag(S(1:k,1:k));
                Vk = V(:,1:k)';
                return
            end

            A_proj = Q_basis' * A;
            [U_proj,S_proj,V_proj] = svd(A_proj, 'econ');
            k = sum(abs(diag(S_proj)) >= rtol*norm(A_proj,'fro'));

            Uk = Q_basis * U_proj(:,1:k);
            sk = diag(S_proj(1:k,1:k));
            Vk = V_proj(:,1:k)';
        end


        function [k, p, T] = rrid_randomized(A, rtol, block_size, flag_power)
        % RRID_RANDOMIZED Interpolative decomposition using a randomized QR factorization.
        %
        % Calling sequence (options shown)
        % ---------------------------------
        %   [k,p,T]        = libid.rrid_randomized(A,rtol);
        %   [k,p,T]        = libid.rrid_randomized(A,rtol,block_size);
        %   [k,p,T]        = libid.rrid_randomized(A,rtol,block_size,flag_power);
        %
        % Description
        % -----------
        % Form an interpolative decomposition (ID) of A by first computing a
        % randomized rank-revealing QR and then solving a triangular system to
        % obtain the interpolation matrix.
        %
        % Parameters
        % ----------
        % A          : double matrix
        %   Input matrix.
        % rtol       : double
        %   Relative tolerance for rank determination.
        % block_size : int, optional (default = 42)
        %   Initial number of random vectors.
        % flag_power : int, optional (default = 0)
        %   Number of power-iteration steps.
        %
        % Returns
        % -------
        % k          : int
        %   Numerical rank (size of R11).
        % p          : int vector
        %   Pivot permutation vector.
        % T          : double matrix
        %   Interpolation matrix such that A(:,p) approx A(:,p(1:k))*T.
        %
        % Notes
        % -----
        % 1. The method uses the private ``_power_iteration`` routine indirectly
        %    through ``rrqr_randomized``.
        %
        % Example
        % -------
        %   A = libid.hilb(4000,2000);
        %   tic, [k,p,T] = libid.rrid_randomized(A,1e-8); toc
        %   rel_err = norm(A(:,p((k+1):end)) - A(:,p(1:k))*T, 'fro')
        %
        %   -------------------------------------------------
        %   Code flow
        %   -------------------------------------------------
        %   1. Call rrqr_randomized to obtain Q, R, and pivot vector p.
        %   2. Extract R11 (upper-triangular leading block) and R12.
        %   3. Solve R11 * X = R12 for the interpolation matrix T.
        %   4. Return rank k, pivot vector, and T.
        %   -------------------------------------------------
        %
        % See also: rrqr_randomized, rrsvd_randomized, orth_sketch

            if nargin < 4, flag_power = 0; end
            if nargin < 3, block_size = 42; end

            [Q,R,p] = libid.rrqr_randomized(A, rtol, block_size, flag_power);
            k = size(R,1);

            % Solve R11 * X = R12 for X, where R = [R11 R12].
            R11 = triu(R(1:k,1:k));
            R12 = R(1:k,k+1:end);
            T = R11 \ R12;
        end

	function a = hilb(m,n)
	    if( nargin == 1 ) n = m; end
	    a = zeros(m,n);
	    i = [1:n];
	    j = [1:m]';
	    a = 1./(bsxfun(@plus,i,j)-1);
	end


        function test()
        % TEST Simple sanity checks for the public API.
        %
        % Notes
        % -----
        %  The test uses a fixed random seed for reproducibility.

        rng(0);  % Fixed seed

        % ------------------------------------------------
        % Small test matrix.
        % ------------------------------------------------
        m = 4000; n = 2000;
        A = libid.hilb(m, n);

        % ------------------------------------------------
        % Test orth_sketch
        % ------------------------------------------------
        [k_range, Q_range] = libid.orth_sketch(A, 1e-12);
        orth_err = norm(Q_range' * Q_range - eye(k_range), 'fro');
        fprintf('orth_sketch: k=%d, basis shape=%s\n', k_range, mat2str(size(Q_range)));
        fprintf('orth_sketch: orthonormality error=%e\n', orth_err);

        % ------------------------------------------------
        % Test rrqr_randomized
        % ------------------------------------------------
        [Q_rrqr, R_rrqr, piv] = libid.rrqr_randomized(A, 1e-12);
        A_perm = A(:, piv);
        recon_err = norm(Q_rrqr * R_rrqr - A_perm, 'fro') / norm(A_perm, 'fro');
        fprintf('rrqr_randomized: reconstruction relative error=%e\n', recon_err);

        % ------------------------------------------------
        % Test rrsvd_randomized
        % ------------------------------------------------
        [U_rrsvd, s_rrsvd, Vt_rrsvd] = libid.rrsvd_randomized(A, 1e-12);
        A_svd = U_rrsvd * diag(s_rrsvd) * Vt_rrsvd;
        svd_err = norm(A_svd - A, 'fro') / norm(A, 'fro');
        fprintf('rrsvd_randomized: reconstruction relative error=%e\n', svd_err);

        % ------------------------------------------------
        % Test rrid_randomized
        % ------------------------------------------------
        [k_id, piv_id, T_id] = libid.rrid_randomized(A, 1e-12);
        A_id_approx = A(:, piv_id(1:k_id)) * T_id;
        id_err = norm(A(:, piv_id(k_id+1:end)) - A_id_approx, 'fro') / norm(A, 'fro');
        fprintf('rrid_randomized: interpolation relative error=%e\n', id_err);
        end
	
    end

    
    methods (Static, Access = private)

        function X = _power_iteration(A, X, power)
        % _POWER_ITERATION Apply power iteration to improve the quality of the sketch.
        %
        % Calling sequence (options shown)
        % ---------------------------------
        %   X = libid._power_iteration(A,X);
        %   X = libid._power_iteration(A,X,power);
        %
        % Description
        % -----------
        % Multiply the test matrix X by A and A' repeatedly to amplify the
        % dominant singular directions.  After each iteration a QR factorization
        % re-orthogonalizes X.
        %
        % Parameters
        % ----------
        % A     : double matrix
        %   Input matrix.
        % X     : double matrix
        %   Random test matrix.
        % power : int, optional (default = 0)
        %   Number of power-iteration steps.
        %
        % Returns
        % -------
        % X     : double matrix
        %   Updated test matrix after power iteration.
        %
        % Notes
        % -----
        % This routine is used internally by ``orth_sketch``.
        %
        % Example
        % -------
        %   X = hilb(4000,2000);
        %   X = libid._power_iteration(A, X, 2);
        %   k, size(Q)
        %
        % See also: orth_sketch

            if nargin < 3, power = 0; end

            for ii = 1:power
                X = A' * (A * X);
                [X,~,~] = qr(X,0);
            end
        end

    end
end


