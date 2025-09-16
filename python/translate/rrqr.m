classdef rrqr
%RRQR  Rank-revealing QR factorization (MATLAB version)
%
%   This class implements the same public API that the original Python
%   module exposed.  All functions are static methods so that they can be
%   called exactly as the original functions, e.g.
%
%       [Q,R,p,k] = rrqr.rrqr(A,rtol);
%
%   The implementation follows the structure of the Python source:
%     * rrqr_lapack - thin wrapper around MATLAB's built-in QR (which
%                     already uses LAPACK).  It returns the orthogonal
%                     matrix Q, the upper-triangular factor R, a
%                     zero-based permutation vector p and the estimated
%                     numerical rank k.
%     * rrqr_native - pure MATLAB implementation that mimics the
%                     original NumPy version (useful for testing /
%                     education).
%   Helper routines (copysign_matlab, reflector, ...) are provided as
%   private static methods so that the code stays close to the original.
%
%   All functions keep the original Python name (e.g., rrqr_piv,
%   reflectorApply2, ...) to make a drop-in replacement possible.

    methods (Static)

        %==================================================================
        % PUBLIC API
        %==================================================================

        function [Q,R,p,k] = rrqr_lapack(A,rtol)
        %RRQR_LAPACK  QR factorization with column pivoting (LAPACK based).
        %
        %   [Q,R,p,k] = rrqr.rrqr_lapack(A,rtol) returns the thin Q factor,
        %   the leading k rows of R, a zero-based permutation vector p
        %   (such that A(:,p) = Q*R) and the estimated numerical rank k.
        %
        %   The computation uses MATLAB's built-in QR with pivoting,
        %   which already calls the optimal LAPACK driver.

            %--- 1. QR with column pivoting ---------------------------------
            %   'vector' returns the permutation as a vector of column indices
            %   (1-based, as MATLAB uses).
            [Qfull,Rfull,p] = qr(A,0);

            %--- 2. Determine numerical rank from the diagonal of R ----------
            diagR = abs(diag(Rfull));
            if isempty(diagR)
                atol = 0;
            else
                atol = rtol * norm(diagR);
            end
            k = sum(diagR > atol);

            %--- 3. Keep only the first k columns of Q and first k rows of R --
            Q = Qfull(:,1:k);
            R = triu(Rfull(1:k,:));
        end


        function [Q,R,p,k] = rrqr_native(A,rtol)
        %RRQR_NATIVE  Pure-MATLAB implementation (no LAPACK call).
        %
        %   This routine reproduces the original NumPy algorithm that
        %   builds the Householder reflectors explicitly.  It is slower
        %   but useful for unit-tests or teaching.

            [tau,p,k,H] = rrqr.rrqr_piv(A,rtol);   % pivoted reduction
            I = eye(size(A,1),k);                  % identity -> will become Q
            Q = rrqr.rrqr_q(H,tau,I,k);            % form Q from reflectors
            R = triu(H(1:k,:));                    % upper-triangular part
        end


        %==================================================================
        % INTERNAL ROUTINES (kept for compatibility with the original code)
        %==================================================================

        function [tau,p,k,a] = rrqr_piv(a,rtol)
        %RRQR_PIV  Perform the pivoted Householder reduction.
        %
        %   [tau,p,k,a] = rrqr.rrqr_piv(A,rtol) returns the scalar factors
        %   tau, the permutation vector p (zero-based), the estimated rank
        %   k and the matrix a that contains the reflectors in its strict
        %   lower-triangular part.

            [m,n] = size(a);
            maxRef = min(m,n);
            tau = zeros(maxRef,1,'like',a);
            p   = 1:n;

            % column 2-norms
            s = sqrt(sum(abs(a).^2,1));
            d = norm(s);
            atol = rtol * d;

            if d == 0
                % zero matrix - nothing to do
                k = 0;
                return;
            end

            k = 0;
            blasLevel = 2;                 % use level-2 update (outer product)

            for j = 1:maxRef
                %----- choose pivot column ---------------------------------
                [~,jpiv_rel] = max(s(j:end));
                jpiv = j-1 + jpiv_rel;     % absolute index (MATLAB 1-based)
                if jpiv ~= j
                    a(:,[j jpiv]) = a(:,[jpiv j]);   % swap columns
                    p([j jpiv])   = p([jpiv j]);     % swap permutation entries
                    s([j jpiv])   = s([jpiv j]);     % swap norms
                end

                %----- form current Householder reflector ------------------
                v = a(j:end,j);
                [tau_j, v] = rrqr.reflector(v);
                tau(j) = tau_j;
                a(j:end,j) = v;               % store reflector (with leading element)

                %----- apply reflector to trailing submatrix ---------------
                if blasLevel == 1
                    for i = j+1:n
                        a(j:end,i) = rrqr.reflectorApply_vector(v,tau_j,a(j:end,i));
                        s(i) = norm(a(j+1:end,i));
                    end
                else
                    a(j:end,j+1:end) = rrqr.reflectorApply2(v,tau_j,a(j:end,j+1:end));
                    for i = j+1:n
                        s(i) = norm(a(j+1:end,i));
                    end
                end

                %----- update rank estimate --------------------------------
                k = j;
                if norm(s(j+1:end),2) < atol
                    tau = tau(1:k);
                    return;
                end
            end
            tau = tau(1:k);
        end


        function Q = rrqr_q(a,tau,q,k)
        %RRQR_Q  Build the orthogonal matrix Q from stored Householder vectors.
        %
        %   Q = rrqr.rrqr_q(H,tau,I,k) returns the Q factor (size m-by-k)
        %   where H contains the reflectors, tau the scalar factors and I
        %   is the identity matrix that will be overwritten.

            m = size(q,1);
            for j = k:-1:1
                v = a(j:end,j);
                % apply the j-th reflector to the columns j…k of Q
                for i = k:-1:j
                    col = q(j:end,i);
                    col = rrqr.reflectorApply_vector(v,conj(tau(j)),col);
                    q(j:end,i) = col;
                end
            end
            Q = q;
        end


        function [U,S] = rrqr_breflector(H,tau,k)
        %RRQR_BREFLECTOR  Build a block reflector Q = I - U*S\U'.
        %
        %   [U,S] = rrqr.rrqr_breflector(H,tau,k) returns the matrix U
        %   (strictly lower part of H with ones on the diagonal) and the
        %   upper-triangular matrix S whose diagonal is 1./tau.

            U = tril(H(:,1:k),-1);
            idx = 1:k;
            U(sub2ind(size(U),idx,idx)) = 1;
            S = triu(U'*U);
            S(sub2ind(size(S),idx,idx)) = 1./tau(:);
        end


        %------------------------------------------------------------------
        % Householder utilities (exact copies of the Python versions)
        %------------------------------------------------------------------

        function y = copysign_matlab(x,y)
        %COPYSIGN_MATLAB  |x| with the sign of y (real part only).
            y = sign(real(y)) .* abs(x);
        end


        function [tau,x] = reflector(x)
        %REFLECTOR  Construct a Householder reflector for a vector x.
        %
        %   [tau,x] = REFLECTOR(x) overwrites x with the reflector vector
        %   (the leading element is set to -nu) and returns the scalar
        %   factor tau.

            n = numel(x);
            if n==0
                tau = 0;  x = zeros(0,1,'like',x);  return;
            end

            xi = x(1);
            norm_u = norm(x);
            if norm_u == 0
                tau = 0;  return;
            end

            nu = rrqr.copysign_matlab(norm_u,real(xi));
            x(1) = -nu;
            if n > 1
                x(2:end) = x(2:end) ./ (xi + nu);
            end
            tau = (xi + nu) / nu;
        end


        function A = reflectorApply(x,tau,A)
        %REFLECTORAPPLY  Apply a Householder reflector to a matrix A (level-1).
        %
        %   A = REFLECTORAPPLY(x,tau,A) updates A := (I - conj(tau)*v*v')*A
        %   where v = [1; x(2:end)].

            [m,n] = size(A);
            if m==0, return; end
            v = [1; x(2:end)];
            for j = 1:n
                vAj = conj(tau) * (A(1,j) + v(2:end)'*A(2:end,j));
                A(1,j) = A(1,j) - vAj;
                A(2:end,j) = A(2:end,j) - vAj * v(2:end);
            end
        end


        function a = reflectorApply_vector(x,tau,a)
        %REFLECTORAPPLY_VECTOR  Apply a reflector to a single vector.
            v = [1; x(2:end)];
            vAj = conj(tau) * (a(1) + v(2:end)'*a(2:end));
            a(1) = a(1) - vAj;
            a(2:end) = a(2:end) - vAj * v(2:end);
        end


        function A = reflectorApply1(x,tau,A)
        %REFLECTORAPPLY1  Level-1 update constructing the explicit vector.
            [m,n] = size(A);
            if m==0, return; end
            y = zeros(m,1,'like',A);
            y(1) = 1;
            y(2:end) = x(2:end);
            for j = 1:n
                vAj = conj(tau) * (y'*A(:,j));
                A(:,j) = A(:,j) - vAj * y;
            end
        end


        function A = reflectorApply2(x,tau,A)
        %REFLECTORAPPLY2  Level-2 update using an outer product.
            [m,~] = size(A);
            if m==0, return; end
            y = zeros(m,1,'like',A);
            y(1) = 1;
            y(2:end) = x(2:end);
            A = A - conj(tau) * (y * (y'*A));
        end

    end
end
