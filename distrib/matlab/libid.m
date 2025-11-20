%==========================================================================
% libid - Randomized linear-algebra routines (Octave compatible)
%
%   A compact toolbox that implements several randomized algorithms for
%   low-rank matrix factorizations:
%
%       * orth_sketch  - orthonormal basis for the column space
%       * qr_sketch    - rank-revealing QR using a random sketch
%       * svd_sketch   - truncated SVD via a random sketch
%       * id_sketch    - interpolative decomposition (ID)
%       * hilb         - rectangular Hilbert matrix generator
%       * safe_max_abs - helper that returns max(|X|) safely
%
%   Author : Your Name
%   SPDX-License-Identifier : TBD
%==========================================================================

classdef libid
    %LIBID  Collection of static methods that perform randomized
    %       matrix factorizations.
    %
    %   See also: orth_sketch, qr_sketch, svd_sketch, id_sketch, hilb

    %======================================================================
    % PUBLIC STATIC METHODS
    %======================================================================
    methods (Static)

        %------------------------------------------------------------------
        % 1. Gaussian test matrix
        %------------------------------------------------------------------
        function Omega = gaussian_omega(A,n,blockSize)
        % GAUSSIAN_OMEGA  Generate a Gaussian test matrix (real or complex).
        %
        %   Omega = libid.gaussian_omega(A,n,blockSize)
        %
        % Description
        % -----------
        %   Returns an (n x blockSize) matrix with i.i.d. standard-normal entries.
        %   The numeric class of the output matches that of A; if A is complex
        %   the real and imaginary parts are generated independently.
        %
        % Parameters
        % ----------
        %   A         : numeric matrix - only its class/complex flag are inspected.
        %   n         : positive integer - number of rows of the test matrix.
        %   blockSize : positive integer - number of columns of the test matrix.
        %
        % Returns
        % -------
        %   Omega     : numeric matrix of size (n,blockSize) with the same class as A.
        %
        % Notes
        % -----
        %   * No RNG argument - the global random stream is used.
        %
        % Example
        % -------
        %   A = libid.hilb(5);
        %   G = libid.gaussian_omega(A,5,3);
        % -----------------------------------------------------------------
            if libid.is_complex_data(A)                   % complex case
                realPart = randn(n,blockSize);
                imagPart = randn(n,blockSize);
                Omega = complex(realPart, imagPart);
            else                                          % real case
                Omega = randn(n,blockSize);
            end
        end

        %------------------------------------------------------------------
        % 2. Uniform[-1,1] test matrix
        %------------------------------------------------------------------
        function Omega = uniform_omega(A,n,blockSize)
        % UNIFORM_OMEGA  Generate a uniform[-1,1] test matrix (real or complex).
        %
        %   Omega = libid.uniform_omega(A,n,blockSize)
        %
        % Description
        % -----------
        %   Returns an (n x blockSize) matrix whose entries are drawn uniformly
        %   from the interval [-1,1].  The output matches the class of A; for a
        %   complex A the real and imaginary parts are generated independently.
        %
        % Parameters
        % ----------
        %   A         : numeric matrix.
        %   n         : positive integer - number of rows.
        %   blockSize : positive integer - number of columns.
        %
        % Returns
        % -------
        %   Omega     : numeric matrix of size (n,blockSize) with the same class as A.
        %
        % Notes
        % -----
        %   * No RNG argument - the global random stream is used.
        %
        % Example
        % -------
        %   A = libid.hilb(5);
        %   U = libid.uniform_omega(A,5,3);
        % -----------------------------------------------------------------
            if libid.is_complex_data(A)                   % complex case
                realPart = 2*rand(n,blockSize) - 1;
                imagPart = 2*rand(n,blockSize) - 1;
                Omega = complex(realPart, imagPart);
            else                                          % real case
                Omega = 2*rand(n,blockSize) - 1;
            end
        end

        %------------------------------------------------------------------
        % 3. Power iteration (optional)
        %------------------------------------------------------------------
        function X = power_iteration(A,X,flagPower)
        % POWER_ITERATION  Apply (A'*A)^flagPower to X, re-orthogonalizing each step.
        %
        %   X = libid.power_iteration(A,X)                % no iteration
        %   X = libid.power_iteration(A,X,flagPower)      % apply flagPower steps
        %
        % Description
        % -----------
        %   Multiplies the test matrix X by A and A' repeatedly to amplify the
        %   dominant singular directions.  After each multiplication the columns
        %   of X are orthogonalized via an economical QR (`qr(X,0)`).
        %
        % Parameters
        % ----------
        %   A          : numeric matrix.
        %   X          : numeric matrix - test matrix to be improved.
        %   flagPower  : non-negative integer - number of power-iteration steps
        %                (default = 0).
        %
        % Returns
        % -------
        %   X          : updated test matrix after power iteration.
        %
        % Notes
        % -----
        %   * The function uses the global random stream; no RNG argument is needed.
        %
        % Example
        % -------
        %   A = libid.hilb(100,50);
        %   X = libid.uniform_omega(A,50,10);
        %   X = libid.power_iteration(A,X,2);
        % -----------------------------------------------------------------
            if nargin < 3, flagPower = 0; end
            for ii = 1:flagPower
                X = libid.rmatvec(A, libid.matvec(A, X));
                [X,~] = qr(X,0);                     % economy QR
            end
        end

        %------------------------------------------------------------------
        % 4. Orthogonal sketch
        %------------------------------------------------------------------
        function [Q,flag] = orth_sketch(A,rtol,blockSize,flagPower,skipTolCheck)
        % ORTH_SKETCH  Orthonormal basis Q for the column space of A via a random sketch.
        %
        %   [Q,flag] = libid.orth_sketch(A,rtol)                     % defaults:
        %                blockSize=42, flagPower=0, skipTolCheck=false
        %   [Q,flag] = libid.orth_sketch(A,rtol,blockSize)          % specify block size
        %   [Q,flag] = libid.orth_sketch(A,rtol,blockSize,flagPower) % also specify power iteration
        %   [Q,flag] = libid.orth_sketch(A,rtol,blockSize,flagPower,skipTolCheck) % skip tolerance check
        %
        % Description
        % -----------
        %   The algorithm draws a random matrix, optionally improves it with
        %   power iteration, and then performs a QR factorization (economy size).
        %   The process repeats with a larger sketch until the smallest diagonal
        %   element of R, relative to the column norms, falls below `rtol`.
        %   If skipTolCheck is true, performs a single sketch without tolerance checking.
        %
        % Parameters
        % ----------
        %   A          : numeric matrix (real or complex).
        %   rtol       : positive scalar - relative tolerance for the stopping test.
        %   blockSize  : positive integer (default = 42) - initial number of random vectors.
        %   flagPower  : non-negative integer (default = 0) - power-iteration steps.
        %   skipTolCheck : logical (default = false) - if true, skip tolerance check
        %                  and geometric growth. Computes a single sketch of size blockSize,
        %                  then filters Q to remove vectors outside the operator's range
        %                  (based on diagonal of R vs column norms of Y).
        %                  Used for matrix-free operators in rank mode.
        %
        % Returns
        % -------
        %   Q          : numeric matrix (m x k) - orthonormal basis.
        %                If the routine exits early because the whole space is covered,
        %                Q is returned as an empty array of size (m,0).
        %   flag       : integer exit flag:
        %                0 - normal termination (tolerance satisfied);
        %                1 - early exit because the sketch already spans the full space.
        %
        % Notes
        % -----
        %   * The random test matrix has entries in [-1,1] (uniform).
        %   * Residual estimate:
        %        d = max(|diag(R(end))|) / max(colnorm(Y)),
        %        where Y = A*X and [Q,R] = qr(Y,0).
        %   * Block size grows geometrically (x4) until the matrix dimension limit is hit.
        %   * When skipTolCheck is true, performs single sketch without geometric growth.
        %
        % Example
        % -------
        %   A = libid.hilb(500,300);
        %   [Q,flag] = libid.orth_sketch(A,1e-12);
        %   fprintf('k = %d, flag = %d\n',size(Q,2),flag);
        % -----------------------------------------------------------------
            if nargin < 5, skipTolCheck = false; end
            if nargin < 4, flagPower = 0; end
            if nargin < 3, blockSize = 42; end

            [m,n] = libid.get_size(A);
            dtype_str = libid.get_dtype_string(A);

            % For matrix-free operators in rank mode, just compute a single sketch
            % without tolerance checking or geometric growth
            if skipTolCheck
                X = libid.uniform_omega(A, n, blockSize);
                X = libid.power_iteration(A, X, flagPower);
                Y = libid.matvec(A, X);
                [Q, R, ~] = qr(Y, 0);  % Pivoted QR (discard permutation)

                % Filter Q using same criterion as tolerance loop:
                % Keep columns where |R(i,i)| / max(||y_j||) > rtol
                col_norms = vecnorm(Y, 2, 1);
                max_col_norm = max(col_norms);
                if max_col_norm == 0
                    max_col_norm = 1.0;
                end
                diagR = abs(diag(R));

                % Use machine epsilon as tolerance
                rtol_eps = max(m,n) * eps(dtype_str);

                % Filter: keep column i if |R(i,i)| / max_col_norm > rtol_eps
                if ~isempty(diagR) && max_col_norm > 0
                    d_ratios = diagR / max_col_norm;
                    keep = d_ratios > rtol_eps;
                    rank = sum(keep);
                else
                    rank = 0;
                end

                % Truncate to actual rank
                Q = Q(:, 1:rank);
                flag = 0;
                return;
            end

            % Guard against tolerance smaller than machine precision (normal mode only)
            if rtol < eps(dtype_str)
                Q = zeros(m,0,dtype_str);
                flag = 1;
                return;
            end

            % Guard against initial block already covering whole space (normal mode only)
            if blockSize >= min(m,n)
                Q = zeros(m,0,dtype_str);
                flag = 1;
                return;
            end

            while true
                % a) random test matrix (uniform entries)
                X = libid.uniform_omega(A,n,blockSize);
                % b) optional power iteration
                X = libid.power_iteration(A,X,flagPower);
                % c) sketch the column space
                Y = libid.matvec(A, X);                  % (m x blockSize)
                [Qtmp,R,~] = qr(Y,0);                    % economy QR with pivoting
                % d) residual estimate
                d = max(abs(diag(R(end)))) / max(vecnorm(Y,2,1));
                if d <= rtol
                    Q = Qtmp;
                    flag = 0;
                    return;
                end
                % e) enlarge block size (geometric growth)
                blockSize = min(blockSize*4, min(m,n));
                if blockSize >= min(m,n)
                    Q = zeros(m,0,dtype_str);
                    flag = 1;
                    return;
                end
            end
        end

        %------------------------------------------------------------------
        % 5. Rank-revealing QR via a random sketch
        %------------------------------------------------------------------
        function [Qk,Rk,piv] = qr_sketch(A,rtol,blockSize,flagPower,extraSamples)
        % QR_SKETCH  Rank-revealing QR factorization using a random sketch.
        %
        %   [Q,R,p] = libid.qr_sketch(A,rtol)                     % defaults:
        %                blockSize=42, flagPower=0, extraSamples=12
        %   [Q,R,p] = libid.qr_sketch(A,rtol,blockSize)          % specify block size
        %   [Q,R,p] = libid.qr_sketch(A,rtol,blockSize,flagPower) % also specify power iteration
        %   [Q,R,p] = libid.qr_sketch(A,rtol,blockSize,flagPower,extraSamples) % also specify extra samples
        %
        % Description
        % -----------
        %   Computes a rank-revealing QR factorization of A by first building a
        %   random orthonormal basis for the column space.  If the matrix is
        %   effectively full rank a deterministic QR is performed.
        %
        % Parameters
        % ----------
        %   A            : numeric matrix.
        %   rtol         : positive scalar - relative tolerance for rank decision.
        %   blockSize    : positive integer (default = 42). Interpretation depends on mode:
        %                  * Tolerance mode (rtol < 1): Starting value for geometric growth.
        %                  * Rank mode (rtol >= 1): Ignored. Total sketch size is always
        %                    kmax + extraSamples. Control via extraSamples parameter.
        %   flagPower    : non-negative integer (default = 0).
        %   extraSamples : positive integer (default = 12) - extra samples for
        %                  oversampling in rank mode (rtol >= 1).
        %
        % Returns
        % -------
        %   Qk         : numeric matrix (m x k) - leading k columns of the orthogonal factor.
        %   Rk         : numeric matrix (k x n) - leading k rows of the upper-triangular factor.
        %   piv        : integer vector - column permutation such that
        %                A(:,piv) ~= Qk*Rk.
        %
        % Notes
        % -----
        %   * If the sketch already spans the whole space (or the early-exit flag
        %     from `orth_sketch` is set) a deterministic QR is performed.
        %   * The numerical rank k is the number of rows of R whose 2-norm exceeds
        %     `rtol*||A||_F` (or `||A_proj||_F` for the projected case).
        %
        % Example
        % -------
        %   A = libid.hilb(500,300);
        %   [Q,R,p] = libid.qr_sketch(A,1e-12);
        %   fprintf('rank = %d\n',size(R,1));
        % -----------------------------------------------------------------
            if nargin < 5, extraSamples = 12; end
            if nargin < 4, flagPower = 0; end
            if nargin < 3, blockSize = 42; end

            [m,n] = libid.get_size(A);

	    dtype_str = libid.get_dtype_string(A);
	    is_matrix_free = libid.is_matrix_free(A);

	    if( rtol >= 1 ),
		flag_kmax = 1;
		kmax = floor(rtol);

		% In rank mode, blockSize is always kmax + extraSamples
		% User controls via extraSamples parameter
		blockSize = kmax + extraSamples;

		rtol = max(m,n)*eps(dtype_str);
	    else
		flag_kmax = 0;
		% Matrix-free LinearOperator in tolerance mode not supported
		if is_matrix_free
		    error('Matrix-free LinearOperator only supported in rank mode (rtol >= 1)');
		end
	    end

            % In rank mode, skip tolerance check (user specifies exact rank)
            skipTol = flag_kmax;

            % 1) cheap orthogonal sketch
            [Q_basis, flag] = libid.orth_sketch(A,rtol,blockSize,flagPower,skipTol);
            k_eff = size(Q_basis,2);

            % 2) fallback to deterministic QR if sketch is full-rank or early exit
            needs_fallback = (flag ~= 0 || k_eff >= min(m,n));

            % In rank mode: skip fallback (user requested specific rank, use sketch as-is)
            if needs_fallback && flag_kmax
                needs_fallback = false;
            end

            if needs_fallback
                A_mat = libid.get_matrix(A);
                [Q,R,piv] = qr(A_mat,0);
                diagR = abs(diag(R));
                % Determine rank using diagonal elements (standard RRQR criterion)
                if ~isempty(diagR) && diagR(1) > 0
                    rank = sum(diagR >= rtol*diagR(1));
                else
                    rank = 0;
                end
		if( flag_kmax ), rank = min(kmax,rank); end
                Qk = Q(:,1:rank);
                Rk = R(1:rank,:);
                return;
            end

            % 3) Project onto the sketch space and factor the thin matrix
            B = libid.project_onto_basis(Q_basis, A);  % (k_eff x n)
            [Qproj,R,piv] = qr(B,0);                % economy QR with pivoting
            Qk = Q_basis*Qproj;                     % lift back to original space
            diagR = abs(diag(R));
            % Determine rank using diagonal elements (standard RRQR criterion)
            if ~isempty(diagR) && diagR(1) > 0
                rank = sum(diagR >= rtol*diagR(1));
            else
                rank = 0;
            end
	    if( flag_kmax ), rank = min(kmax,rank); end
            Qk = Qk(:,1:rank);
            Rk = R(1:rank,:);
        end

        %------------------------------------------------------------------
        % 6. Truncated SVD via a random sketch
        %------------------------------------------------------------------
        function [U,s,V] = svd_sketch(A,rtol,blockSize,flagPower,extraSamples)
        % SVD_SKETCH  Truncated singular-value decomposition via a random sketch.
        %
        %   [U,s,V] = libid.svd_sketch(A,rtol)                     % defaults:
        %                blockSize=42, flagPower=0, extraSamples=12
        %   [U,s,V] = libid.svd_sketch(A,rtol,blockSize,flagPower) % also specify block size & power
        %   [U,s,V] = libid.svd_sketch(A,rtol,blockSize,flagPower,extraSamples) % also specify extra samples
        %
        % Description
        % -----------
        %   Computes a truncated SVD of A by first constructing a random orthonormal
        %   basis for the column space.  If A is effectively full rank, a deterministic
        %   SVD is performed.
        %
        % Parameters
        % ----------
        %   A            : numeric matrix.
        %   rtol         : positive scalar - tolerance for singular values.
        %   blockSize    : positive integer (default = 42). Interpretation depends on mode:
        %                  * Tolerance mode (rtol < 1): Starting value for geometric growth.
        %                  * Rank mode (rtol >= 1): Ignored. Total sketch size is always
        %                    kmax + extraSamples. Control via extraSamples parameter.
        %   flagPower    : non-negative integer (default = 0).
        %   extraSamples : positive integer (default = 12) - extra samples for
        %                  oversampling in rank mode (rtol >= 1).
        %
        % Returns
        % -------
        %   U          : m-by-k matrix - leading left singular vectors.
        %   s          : k-by-1 vector - leading singular values.
        %   V          : n-by-k matrix - leading right singular vectors.
        %                Follows MATLAB convention: use V' for reconstruction.
        %                Reconstruction: A ~ U * diag(s) * V'
        %                Note: This differs from scipy which returns Vh directly.
        %
        % Notes
        % -----
        %   * Numerical rank k = sum(s >= rtol*s(1)).
        %   * Variable naming: V returned matches MATLAB's svd() convention where
        %     reconstruction requires conjugate transpose: A ~ U*S*V'
        %
        % Example
        % -------
        %   A = libid.hilb(200,150);
        %   [U,s,V] = libid.svd_sketch(A,1e-12);
        %   fprintf('rank = %d\n',numel(s));
        % -----------------------------------------------------------------
            if nargin < 5, extraSamples = 12; end
            if nargin < 4, flagPower = 0; end
            if nargin < 3, blockSize = 42; end

            [m,n] = libid.get_size(A);

	    % Transpose optimization for wide matrices (m < n)
	    if( m < n )
		if libid.is_linop(A)
		    % Create transposed LinearOperator with swapped matvec/rmatvec
		    A_T = make_linop(n, m, ...
			@(x) libid.rmatvec(A, x), ...  % matvec uses A's rmatvec
			@(x) libid.matvec(A, x));      % rmatvec uses A's matvec
		    % Mark as matrix-free if original was matrix-free
		    if libid.is_matrix_free(A)
			A_T.matrix = [];
		    end
		    % Note: For explicit LinearOperators, we don't copy the transposed matrix
		    % to save memory. The matvec/rmatvec functions handle transposition.
		else
		    A_T = A';
		end
		[V,s,U] = libid.svd_sketch(A_T,rtol,blockSize,flagPower,extraSamples);
		return
	    end

	    dtype_str = libid.get_dtype_string(A);
	    is_matrix_free = libid.is_matrix_free(A);

	    if( rtol >= 1 ),
		flag_kmax = 1;
		kmax = floor(rtol);

		% In rank mode, blockSize is always kmax + extraSamples
		% User controls via extraSamples parameter
		blockSize = kmax + extraSamples;

		rtol = max(m,n)*eps(dtype_str);
	    else
		flag_kmax = 0;
	    end

            % In rank mode, skip tolerance check (user specifies exact rank)
            skipTol = flag_kmax;

            % 1) orthogonal sketch
            [Q_basis, flag] = libid.orth_sketch(A,rtol,blockSize,flagPower,skipTol);
            k_eff = size(Q_basis,2);

            % 2) deterministic fallback
            needs_fallback = (flag ~= 0 || k_eff >= min(m,n));

            % In rank mode: skip fallback (user requested specific rank, use sketch as-is)
            if needs_fallback && flag_kmax
                needs_fallback = false;
            end

            if needs_fallback
                A_mat = libid.get_matrix(A);
                [Ufull,Sfull,Vfull] = svd(A_mat,0);   % economy SVD
                rank = libid.rank_from_svals(diag(Sfull),rtol);
		if( flag_kmax ), rank = min(kmax,rank); end
                U = Ufull(:,1:rank);
                s = diag(Sfull(1:rank,1:rank));
                V = Vfull(:,1:rank);
                return;
            end

            % 3) Project onto the sketch space and compute SVD there
            Aproj = libid.project_onto_basis(Q_basis, A);  % (k_eff x n)
            [Uproj,Sproj,Vproj] = svd(Aproj,0);    % economy SVD
            rank = libid.rank_from_svals(diag(Sproj),rtol);
	    if( flag_kmax ), rank = min(kmax,rank); end
            U = Q_basis * Uproj(:,1:rank);
            s = diag(Sproj(1:rank,1:rank));
            V = Vproj(:,1:rank);
        end

        %------------------------------------------------------------------
        % 7. Interpolative decomposition (ID)
        %------------------------------------------------------------------
        function [k,piv,T] = id_sketch(A,rtol,blockSize,flagPower,useSVD,recomputeT)
        % ID_SKETCH  Interpolative decomposition using randomized QR.
        %
        %   [k,piv,T] = libid.id_sketch(A,rtol)                              % defaults:
        %               blockSize=42, flagPower=0, useSVD=false, recomputeT=false
        %   [k,piv,T] = libid.id_sketch(A,rtol,blockSize,flagPower,useSVD,recomputeT)
        %
        % Description
        % -----------
        %   Forms an ID of A by first computing a randomized rank-revealing QR and
        %   then solving a triangular system (or an SVD-based solve) to obtain the
        %   interpolation matrix.
        %
        % Parameters
        % ----------
        %   A          : numeric matrix or LinearOperator.
        %   rtol       : positive scalar - tolerance for rank decision.
        %   blockSize  : positive integer (default = 42).
        %   flagPower  : non-negative integer (default = 0).
        %   useSVD     : logical (default = false) - compute T via SVD of R11.
        %   recomputeT : logical (default = false) - recompute T from original A.
        %                - false (default): Compute T from R matrix (Fortran's approach).
        %                  May give error > 1.0 on full-rank matrices. Use when speed is
        %                  critical and higher error is acceptable.
        %
        %                - true: Recompute T via least squares on original A.
        %                  Ensures error < 1.0 (mathematically guaranteed).
        %
        %                  For explicit matrices: Direct column indexing.
        %                  For matrix-free operators: Extracts all n columns via unit vectors
        %                  (n matvecs). Slower but guarantees accuracy.
        %
        % Returns
        % -------
        %   k          : integer - numerical rank (size of R11).
        %   piv        : integer vector - column permutation of the original matrix.
        %   T          : k-by-(n-k) interpolation matrix such that
        %                A(:,piv(k+1:end)) ~= A(:,piv(1:k))*T.
        %
        % Notes
        % -----
        %   * Uses single-stage QR sketch pipeline with column pivoting.
        %   * Supports LinearOperators (both explicit and matrix-free).
        %   * If `useSVD` is true the triangular solve is replaced by a thin SVD
        %     of R11 (more stable for ill-conditioned R11).
        %
        % Example
        % -------
        %   A = libid.hilb(400,300);
        %   [k,p,T] = libid.id_sketch(A,1e-10);
        %   err = norm(A(:,p(k+1:end)) - A(:,p(1:k))*T,'fro')/norm(A,'fro');
        %
        %   % Fast mode for matrix-free operators (expert use)
        %   [k,p,T] = libid.id_sketch(A_linop,50,42,0,false,false);
        %
        % -----------------------------------------------------------------
            if nargin < 6, recomputeT = false; end
            if nargin < 5, useSVD = false; end
            if nargin < 4, flagPower = 0; end
            if nargin < 3, blockSize = 42; end

            is_linop = libid.is_linop(A);
            is_matrix_free = libid.is_matrix_free(A);
            [m,n] = libid.get_size(A);

            % Standard path: QR with column pivoting on sketched A
            [~,R,piv] = libid.qr_sketch(A,rtol,blockSize,flagPower);

            k = size(R,1);
            [~, n] = libid.get_size(A);

            % Recompute T from original A for better accuracy
            if recomputeT && k > 0 && k < n
                if is_matrix_free
                    % Matrix-free path: Extract columns via unit vectors
                    % This is slower (n matvecs) but guarantees error < 1.0
                    dtype_str = libid.get_dtype_string(A);

                    skeleton_cols = zeros(m, k, dtype_str);
                    for j = 1:k
                        e_j = zeros(n, 1, dtype_str);
                        e_j(piv(j)) = 1.0;
                        skeleton_cols(:, j) = libid.matvec(A, e_j);
                    end

                    remaining_cols = zeros(m, n - k, dtype_str);
                    for j = 1:(n - k)
                        e_j = zeros(n, 1, dtype_str);
                        e_j(piv(k + j)) = 1.0;
                        remaining_cols(:, j) = libid.matvec(A, e_j);
                    end

                    % Solve: skeleton_cols * T ~ remaining_cols
                    T = skeleton_cols \ remaining_cols;
                    return;
                else
                    % Explicit matrix path: Direct column indexing
                    % Extract the underlying matrix
                    A_mat = libid.get_matrix(A);

                    % Recompute T via least squares on original A
                    % This ensures: A(:, piv(k+1:end)) ~ A(:, piv(1:k)) * T
                    cols = piv(1:k);
                    remaining = piv(k+1:end);
                    T = A_mat(:, cols) \ A_mat(:, remaining);
                    return;
                end
            end

            % Fall back to using T from R matrix (may have error > 1.0 for sketches)
            R11 = R(1:k,1:k);
            R12 = R(1:k,k+1:end);

            if useSVD
                [U,S,V] = svd(R11,0);
		s = diag(S);
                keep = s >= rtol*max(s);
		if( rtol >= 1 )
		    keep = 1:(min(rtol, size(S,1)));
		end
                if ~any(keep)
		    T = zeros(size(R12), class(R12));
                else
                    T = V(:,keep) * (S(keep,keep) \ (U(:,keep)' * R12));
                end
            else
                % Upper-triangular solve R11 * T = R12
                T = R11 \ R12;
            end
        end

        %------------------------------------------------------------------
        % 8. Hilbert matrix generator (public)
        %------------------------------------------------------------------
        function H = hilb(m,n)
        % HILB  Generate an m-by-n Hilbert matrix.
        %
        %   H = libid.hilb(m)        returns a square m-by-m Hilbert matrix.
        %   H = libid.hilb(m,n)      returns the rectangular version.
        %
        % Parameters
        % ----------
        %   m          : positive integer - number of rows.
        %   n          : positive integer - number of columns (default = m).
        %
        % Returns
        % -------
        %   H          : double matrix of size (m,n) with entries 1/(i+j-1).
        %
        % Notes
        % -----
        %   * The Hilbert matrix is extremely ill-conditioned; it is useful for
        %     testing numerical algorithms.
        %
        % Example
        % -------
        %   H = libid.hilb(4);
        %   disp(H);
        % -----------------------------------------------------------------
            if nargin < 2, n = m; end
            i = (1:m).';                     % column vector
            j = 1:n;                         % row vector
            H = 1./(i + j - 1);
        end

    end   % end of public methods block

    %======================================================================
    % PRIVATE STATIC HELPERS (not part of the public API)
    %======================================================================
    methods (Static, Access = private)

        %------------------------------------------------------------------
        % Helper: matrix-vector multiplication (handles both matrix and LinearOperator)
        %------------------------------------------------------------------
        function y = matvec(A, x)
        % MATVEC  Compute y = A * x for matrix or LinearOperator.
        %
        %   y = libid.matvec(A, x)
        %
        % Supports both vector and matrix inputs (BLAS3-rich).
            if libid.is_linop(A)
                % LinearOperator - use apply function handle
                % (handles BLAS3 internally if explicit)
                y = A.apply(x);
            else
                % Regular matrix - BLAS2 (vector) or BLAS3 (matrix)
                y = A * x;
            end
        end

        %------------------------------------------------------------------
        % Helper: adjoint matrix-vector multiplication
        %------------------------------------------------------------------
        function y = rmatvec(A, x)
        % RMATVEC  Compute y = A' * x for matrix or LinearOperator.
        %
        %   y = libid.rmatvec(A, x)
        %
        % Supports both vector and matrix inputs (BLAS3-rich).
            if libid.is_linop(A)
                % LinearOperator - use applyT function handle
                % (handles BLAS3 internally if explicit)
                y = A.applyT(x);
            else
                % Regular matrix - BLAS2 (vector) or BLAS3 (matrix)
                y = A' * x;
            end
        end

        %------------------------------------------------------------------
        % Helper: compute Q' * A efficiently for LinearOperators
        %------------------------------------------------------------------
        function B = project_onto_basis(Q, A)
        % PROJECT_ONTO_BASIS  Compute B = Q' * A efficiently.
        %
        %   B = libid.project_onto_basis(Q, A)
        %
        % For LinearOperators, uses adjoint matmat (BLAS3): B = (A' * Q)'
        % For regular matrices, uses standard multiplication: B = Q' * A
            if libid.is_linop(A)
                % Use adjoint matmat (BLAS3): Q' * A = (A' * Q)'
                % Q is a matrix (m, k), so A.applyT(Q) uses BLAS3 if explicit
                B = libid.rmatvec(A, Q)';
            else
                % Regular matrix multiplication (BLAS3)
                B = Q' * A;
            end
        end

        %------------------------------------------------------------------
        % Helper: check if A is a LinearOperator struct
        %------------------------------------------------------------------
        function result = is_linop(A)
        % IS_LINOP  Check if A is a LinearOperator structure.
        %
        %   result = libid.is_linop(A)
        %
        % Returns
        % -------
        %   result : logical - true if A is a LinearOperator struct.
            result = isstruct(A) && isfield(A, 'apply') && isfield(A, 'applyT') && ...
                     isfield(A, 'is_explicit') && isfield(A, 'm') && isfield(A, 'n');
        end

        %------------------------------------------------------------------
        % Helper: extract matrix from A (for fallback operations)
        %------------------------------------------------------------------
        function mat = get_matrix(A)
        % GET_MATRIX  Extract explicit matrix from A.
        %
        %   mat = libid.get_matrix(A)
        %
        % For regular matrices, returns A as-is.
        % For explicit LinearOperators, returns the underlying matrix.
        % For matrix-free LinearOperators, throws an error.
            if libid.is_linop(A)
                if A.is_explicit && isfield(A, 'matrix')
                    mat = A.matrix;
                else
                    error('libid:matrix_free_fallback', ...
                          'Cannot perform fallback QR/SVD on matrix-free LinearOperator');
                end
            else
                mat = A;
            end
        end

        %------------------------------------------------------------------
        % Helper: get dtype string from A (matrix or LinearOperator)
        %------------------------------------------------------------------
        function dtype_str = get_dtype_string(A)
        % GET_DTYPE_STRING  Get the dtype string for eps() and zeros() calls.
        %
        %   dtype_str = libid.get_dtype_string(A)
        %
        % Returns
        % -------
        %   dtype_str : char - 'double', 'single', etc.
            if libid.is_linop(A)
                % LinearOperator case - use stored dtype field
                if isfield(A, 'dtype')
                    dtype_str = A.dtype;
                else
                    % Fallback for legacy LinearOperators without dtype field
                    if A.is_explicit && isfield(A, 'matrix')
                        dtype_str = class(A.matrix);
                    else
                        error('LinearOperator missing dtype field. For matrix-free operators, dtype must be specified during creation.');
                    end
                end
            else
                % Regular matrix
                dtype_str = class(A);
            end
        end

        %------------------------------------------------------------------
        % Helper: check if A represents complex data
        %------------------------------------------------------------------
        function result = is_complex_data(A)
        % IS_COMPLEX_DATA  Check if A represents complex data.
        %
        %   result = libid.is_complex_data(A)
        %
        % Returns
        % -------
        %   result : logical - true if A represents complex data.
            if libid.is_linop(A)
                % LinearOperator case
                if A.is_explicit && isfield(A, 'matrix')
                    % Explicit operator - check underlying matrix
                    result = ~isreal(A.matrix);
                else
                    % Matrix-free operator - default to real
                    result = false;
                end
            else
                % Regular matrix
                result = ~isreal(A);
            end
        end

        %------------------------------------------------------------------
        % Helper: get size of matrix or LinearOperator
        %------------------------------------------------------------------
        function [m, n] = get_size(A)
        % GET_SIZE  Get dimensions of matrix or LinearOperator.
        %
        %   [m, n] = libid.get_size(A)
        %
        % Returns
        % -------
        %   m, n : dimensions of the linear operator
            if libid.is_linop(A)
                m = A.m;
                n = A.n;
            else
                [m, n] = size(A);
            end
        end

        %------------------------------------------------------------------
        % Helper: check if A is a matrix-free LinearOperator
        %------------------------------------------------------------------
        function result = is_matrix_free(A)
        % IS_MATRIX_FREE  Check if A is a matrix-free LinearOperator.
        %
        %   result = libid.is_matrix_free(A)
        %
        % Returns
        % -------
        %   result : logical - true if A is a LinearOperator not backed by explicit matrix.
            result = libid.is_linop(A) && isfield(A, 'is_explicit') && ~A.is_explicit;
        end

        %------------------------------------------------------------------
        % Helper: numerical rank from singular values
        %------------------------------------------------------------------
        function r = rank_from_svals(s,rtol)
        % RANK_FROM_SVALS  Return the numerical rank given singular values.
        %
        %   r = libid.rank_from_svals(s,rtol)
        %
        % Parameters
        % ----------
        %   s    : vector of singular values (non-negative, descending).
        %   rtol : relative tolerance.
        %
        % Returns
        % -------
        %   r    : integer rank.
            if isempty(s)
                r = 0;
            else
                r = sum(s >= rtol*s(1));
            end
        end

        %------------------------------------------------------------------
        % Helper: safe max-absolute value (used only in test harness)
        %------------------------------------------------------------------
        function m = safe_max_abs(X)
        % SAFE_MAX_ABS  Return max(|X|) or 0 for an empty array.
        %
        %   m = libid.safe_max_abs(X)
            if isempty(X)
                m = 0;
            else
                m = max(abs(X(:)));
            end
        end

    end   % end of private methods block
end   % classdef libid

