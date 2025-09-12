% Description
% ----------
% This class implements several randomized linear‑algebra routines that
% approximate the rank, QR factorization, singular‑value decomposition
% (SVD), and interpolative decomposition (ID) of a matrix A.
%
% User‑callable methods
% ---------------------
%   range_randomized   - Build an orthonormal basis for the column space.
%   rrqr_randomized    - Rank‑revealing QR using a randomized basis.
%   rrsvd_randomized   - Truncated SVD using a randomized basis.
%   rrid_randomized    - Interpolative decomposition using randomized QR.
%   image_randomized   - Basis for the row space via transpose.
%
% Author: Your Name
% SPDX-License-Identifier: TBD

classdef RandomizedLinearAlgebra
    % RandomizedLinearAlgebra
    %
    % Collection of static methods that perform randomized matrix factorizations.
    %
    % See also: range_randomized, rrqr_randomized, rrsvd_randomized,
    %           rrid_randomized, image_randomized

    methods (Static)

        function [k, Q] = range_randomized(A, rtol, block_size, flag_power)
        % RANGE_RANDOMIZED Compute an orthonormal basis for the column space of A using random sampling.
        %
        % Description
        % -----------
        % Build an orthonormal basis for the column space of A using random
        % sampling and optional power iterations.  The routine stops when
        % the relative residual falls below ``rtol``.
        %
        % Parameters
        % ----------
        % A          : double matrix
        %   Input matrix.
        % rtol       : double
        %   Relative tolerance that determines when to stop sampling.
        % block_size : int, optional (default = 42)
        %   Initial number of random vectors.
        % flag_power : int, optional (default = 0)
        %   Number of power‑iteration steps.
        %
        % Returns
        % -------
        % k          : int
        %   Number of basis vectors found (may equal min(m,n)).
        % Q          : double matrix
        %   Orthonormal basis matrix with size (m,k).  If k == 0 the array is empty.
        %
        % Notes
        % -----
        % 1. If the initial block already covers the whole space, the function
        %    returns early.
        % 2. The random matrix X has entries in [-1,1].
        % 3. Power iteration is performed by the private method powerIteration.
        %
        % Example
        % -------
        %   A = randn(100,50);
        %   [k,Q] = RandomizedLinearAlgebra.range_randomized(A,1e-6);
        %
        %   -------------------------------------------------
        %   Code flow
        %   -------------------------------------------------
        %   1. Determine matrix dimensions.
        %   2. Check early‑exit condition.
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
                X = RandomizedLinearAlgebra.powerIteration(A, X, flag_power);

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
        % RRQR_RANDOMIZED Rank‑revealing QR factorization using a randomized basis.
        %
        % Description
        % -----------
        % Compute a rank‑revealing QR factorization of A by first building a
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
        %   Number of power‑iteration steps.
        %
        % Returns
        % -------
        % Qk         : double matrix
        %   Leading k columns of the orthogonal factor.
        % Rk         : double matrix
        %   Leading k rows of the upper‑triangular factor.
        % p          : int vector
        %   Pivot permutation vector.
        %
        % Notes
        % -----
        % 1. The rank k is chosen as the number of rows of R whose 2‑norm
        %    exceeds ``rtol * ||A||`` (or ``||A_proj||`` for the projected case).
        % 2. The private method ``powerIteration`` is used internally.
        %
        % Example
        % -------
        %   A = randn(200,80);
        %   [Q,R,p] = RandomizedLinearAlgebra.rrqr_randomized(A,1e-8);
        %
        %   -------------------------------------------------
        %   Code flow
        %   -------------------------------------------------
        %   1. Call range_randomized to obtain basis Q_basis.
        %   2. If full rank, compute deterministic QR of A.
        %   3. Otherwise project A onto the basis and QR the small matrix.
        %   4. Determine numerical rank k from R.
        %   5. Return truncated factors and pivot vector.
        %   -------------------------------------------------
        %
        % See also: range_randomized, rrsvd_randomized, rrid_randomized

            if nargin < 4, flag_power = 0; end
            if nargin < 3, block_size = 42; end

            [m, n] = size(A);
            [k, Q_basis] = RandomizedLinearAlgebra.range_randomized(A, rtol, block_size, flag_power);

            if k >= min(m,n)
                [Q,R,E] = qr(A,0);
                p = E;
                k = sum(vecnorm(R,2,2) >= rtol*norm(A,'fro'));
                Qk = Q(:,1:k);
                Rk = R(1:k,:);
                return
            end

            % Project A onto the basis and factor the small matrix.
            A_proj = Q_basis' * A;
            [Q_proj,R,E] = qr(A_proj,0);
            p = E;
            Qk = Q_basis * Q_proj;
            k = sum(vecnorm(R,2,2) >= rtol*norm(A_proj,'fro'));
            Qk = Qk(:,1:k);
            Rk = R(1:k,:);
        end


        function [Uk, sk, Vk] = rrsvd_randomized(A, rtol, block_size, flag_power)
        % RRSVD_RANDOMIZED Truncated singular‑value decomposition using a randomized basis.
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
        %   Number of power‑iteration steps.
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
        %   A = randn(150,120);
        %   [U,S,V] = RandomizedLinearAlgebra.rrsvd_randomized(A,1e-7);
        %
        %   -------------------------------------------------
        %   Code flow
        %   -------------------------------------------------
        %   1. Obtain basis Q_basis via range_randomized.
        %   2. If full rank, call deterministic svd.
        %   3. Otherwise form A_proj = Q_basis' * A.
        %   4. Compute svd of the small matrix.
        %   5. Lift left singular vectors back: U = Q_basis * U_proj.
        %   6. Truncate to k based on rtol.
        %   -------------------------------------------------
        %
        % See also: range_randomized, rrqr_randomized, rrid_randomized

            if nargin < 4, flag_power = 0; end
            if nargin < 3, block_size = 42; end

            [m, n] = size(A);
            [k, Q_basis] = RandomizedLinearAlgebra.range_randomized(A, rtol, block_size, flag_power);

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


        function [k, p, proj] = rrid_randomized(A, rtol, block_size, flag_power)
        % RRID_RANDOMIZED Interpolative decomposition using a randomized QR factorization.
        %
        % Description
        % -----------
        % Form an interpolative decomposition (ID) of A by first computing a
        % randomized rank‑revealing QR and then solving a triangular system to
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
        %   Number of power‑iteration steps.
        %
        % Returns
        % -------
        % k          : int
        %   Numerical rank (size of R11).
        % p          : int vector
        %   Pivot permutation vector.
        % proj       : double matrix
        %   Interpolation matrix such that A(:,p) ≈ A(:,p(1:k))*proj.
        %
        % Notes
        % -----
        % 1. The method uses the private ``powerIteration`` routine indirectly
        %    through ``rrqr_randomized``.
        %
        % Example
        % -------
        %   A = randn(80,200);
        %   [k,p,proj] = RandomizedLinearAlgebra.rrid_randomized(A,1e-6);
        %
        %   -------------------------------------------------
        %   Code flow
        %   -------------------------------------------------
        %   1. Call rrqr_randomized to obtain Q, R, and pivot vector p.
        %   2. Extract R11 (upper‑triangular leading block) and R12.
        %   3. Solve R11 * X = R12 for the interpolation matrix X.
        %   4. Return rank k, pivot vector, and X.
        %   -------------------------------------------------
        %
        % See also: rrqr_randomized, rrsvd_randomized, range_randomized

            if nargin < 4, flag_power = 0; end
            if nargin < 3, block_size = 42; end

            [Q,R,p] = RandomizedLinearAlgebra.rrqr_randomized(A, rtol, block_size, flag_power);
            k = size(R,1);

            % Solve R11 * X = R12 for X, where R = [R11 R12].
            R11 = triu(R(1:k,1:k));
            R12 = R(1:k,k+1:end);
            proj = R11 \ R12;
        end


        function [k, Q] = image_randomized(A, rtol, block_size, flag_power)
        % IMAGE_RANDOMIZED Compute a basis for the row space of A by applying range_randomized to the transpose.
        %
        % Description
        % -----------
        % A thin wrapper that calls ``range_randomized`` on ``A.'`` to obtain
        % a basis for the row space.
        %
        % Parameters
        % ----------
        % A          : double matrix
        %   Input matrix.
        % rtol       : double
        %   Relative tolerance for stopping criterion.
        % block_size : int, optional (default = 42)
        %   Initial number of random vectors.
        % flag_power : int, optional (default = 0)
        %   Number of power‑iteration steps.
        %
        % Returns
        % -------
        % k          : int
        %   Number of basis vectors found.
        % Q          : double matrix
        %   Orthonormal basis for the row space (size n‑by‑k).
        %
        % Example
        % -------
        %   A = randn(60,120);
        %   [k,Q] = RandomizedLinearAlgebra.image_randomized(A,1e-5);
        %
        % See also: range_randomized

            if nargin < 4, flag_power = 0; end
            if nargin < 3, block_size = 42; end

            [k, Q] = RandomizedLinearAlgebra.range_randomized(A.', rtol, block_size, flag_power);
        end

    end

    methods (Static, Access = private)

        function X = powerIteration(A, X, power)
        % POWERITERATION Apply power iteration to improve the quality of the sampling matrix.
        %
        % Description
        % -----------
        % Multiply the test matrix X by A and A' repeatedly to amplify the
        % dominant singular directions.  After each iteration a QR factorization
        % re‑orthogonalizes X.
        %
        % Parameters
        % ----------
        % A     : double matrix
        %   Input matrix.
        % X     : double matrix
        %   Random test matrix.
        % power : int, optional (default = 0)
        %   Number of power‑iteration steps.
        %
        % Returns
        % -------
        % X     : double matrix
        %   Updated test matrix after power iteration.
        %
        % Notes
        % -----
        % This routine is used internally by ``range_randomized``.
        %
        % Example
        % -------
        %   X = randn(100,20);
        %   X = RandomizedLinearAlgebra.powerIteration(A, X, 2);
        %
        % See also: range_randomized

            if nargin < 3, power = 0; end

            for ii = 1:power
                X = A.' * (A * X);
                [X,~,~] = qr(X,0);
            end
        end

    end
end

