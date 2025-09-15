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
    % Collection of static methods for randomized matrix factorizations.

    methods (Static)

        function [k, Q] = orth_sketch(A, rtol, block_size, flag_power)
            if nargin < 4, flag_power = 0; end
            if nargin < 3, block_size = 42; end

            [m, n] = size(A);

            if block_size >= min(m,n)
                k = min(m,n);
                Q = [];
                return
            end

            while true
                X = 2*rand(n, block_size) - 1;
                X = libid._power_iteration(A, X, flag_power);

                Y = A * X;
                [Qtmp,R,~] = qr(Y,0);

                rdiag = diag(R);
                residual = max(abs(rdiag(end))) / max(vecnorm(Y,2,1));

                if residual <= rtol
                    k = block_size;
                    Q = Qtmp;
                    return
                end

                block_size = min(block_size*4, min(m,n));

                if block_size >= min(m,n)
                    k = min(m,n);
                    Q = [];
                    return
                end
            end
        end


        function [Qk, Rk, p] = rrqr_randomized(A, rtol, block_size, flag_power)
            if nargin < 4, flag_power = 0; end
            if nargin < 3, block_size = 42; end

            [m, n] = size(A);
            [k, Q_basis] = libid.orth_sketch(A, rtol, block_size, flag_power);

            if k >= min(m,n)
		% full rank, deterministic QR
                [Q,R,p] = qr(A,0);
                k = sum(vecnorm(R,2,2) >= rtol*norm(A,'fro'));
                Qk = Q(:,1:k);
                Rk = R(1:k,:);
                return
            end

            % Project onto the basis and factor the reduced matrix
            A_proj = Q_basis' * A;
            [Q_proj,R,p] = qr(A_proj,0);
            Qk = Q_basis * Q_proj;
            k = sum(vecnorm(R,2,2) >= rtol*norm(A_proj,'fro'));
            Qk = Qk(:,1:k);
            Rk = R(1:k,:);
        end


        function [Uk, sk, Vk] = rrsvd_randomized(A, rtol, block_size, flag_power)
            if nargin < 4, flag_power = 0; end
            if nargin < 3, block_size = 42; end

            [m, n] = size(A);
            [k, Q_basis] = libid.orth_sketch(A, rtol, block_size, flag_power);

            if k >= min(m,n)
		% full rank, deterministic SVD
                [U,S,V] = svd(A, 'econ');
                k = sum(abs(diag(S)) >= rtol*norm(A,'fro'));
                Uk = U(:,1:k);
                sk = diag(S(1:k,1:k));
                Vk = V(:,1:k)';
                return
            end

            % Project onto the basis and compute SVD of the reduced matrix
            A_proj = Q_basis' * A;
            [U_proj,S_proj,V_proj] = svd(A_proj, 'econ');
            k = sum(abs(diag(S_proj)) >= rtol*norm(A_proj,'fro'));

            Uk = Q_basis * U_proj(:,1:k);
            sk = diag(S_proj(1:k,1:k));
            Vk = V_proj(:,1:k)';
        end


        function [k, p, T] = rrid_randomized(A, rtol, block_size, flag_power)
            if nargin < 4, flag_power = 0; end
            if nargin < 3, block_size = 42; end

            [Q,R,p] = libid.rrqr_randomized(A, rtol, block_size, flag_power);
            k = size(R,1);

            R11 = triu(R(1:k,1:k));
            R12 = R(1:k,k+1:end);
            T = R11 \ R12;                     % interpolation matrix
        end


        function a = hilb(m,n)
            if nargin == 1, n = m; end
            i = 1:n;
            j = (1:m)';
            a = 1./(bsxfun(@plus,i,j)-1);
        end


        function test()
            rng(0);  % reproducible

            m = 4000; n = 2000;
            A = libid.hilb(m, n);

            % orth_sketch
            [k_range, Q_range] = libid.orth_sketch(A, 1e-12);
            orth_err = norm(Q_range' * Q_range - eye(k_range), 'fro');
            fprintf('orth_sketch: k=%d, basis=%s, orth_err=%e\n', ...
                    k_range, mat2str(size(Q_range)), orth_err);

            % rrqr_randomized
            [Q_rrqr, R_rrqr, piv] = libid.rrqr_randomized(A, 1e-12);
            A_perm = A(:, piv);
            recon_err = norm(Q_rrqr * R_rrqr - A_perm, 'fro') / norm(A_perm, 'fro');
            fprintf('rrqr_randomized: recon_err=%e\n', recon_err);

            % rrsvd_randomized
            [U_rrsvd, s_rrsvd, Vt_rrsvd] = libid.rrsvd_randomized(A, 1e-12);
            A_svd = U_rrsvd * diag(s_rrsvd) * Vt_rrsvd;
            svd_err = norm(A_svd - A, 'fro') / norm(A, 'fro');
            fprintf('rrsvd_randomized: svd_err=%e\n', svd_err);

            % rrid_randomized
            [k_id, piv_id, T_id] = libid.rrid_randomized(A, 1e-12);
            A_id_approx = A(:, piv_id(1:k_id)) * T_id;
            id_err = norm(A(:, piv_id(k_id+1:end)) - A_id_approx, 'fro') / norm(A, 'fro');
            fprintf('rrid_randomized: id_err=%e\n', id_err);
        end

    end


    methods (Static, Access = private)

        function X = _power_iteration(A, X, power)
            if nargin < 3, power = 0; end
            for ii = 1:power
                X = A' * (A * X);
                [X,~,~] = qr(X,0);
            end
        end

    end
end
