function [Q, R, p, k] = rrqr_rand(A,rtol)
    [m,n] = size(A);    
    block_size = 42;
    for i = 1:20
	if( isreal(A) )
%%%	    x = 2*rand(n,block_size)-1;
	    x = randn(n,block_size);
	else
%%%	    x = 2*(rand(n,block_size)+1i*rand(n,block_size))-1;
	    x = (randn(n,block_size)+1i*randn(n,block_size));
	end
	y = A*x;
	[tau, k, H] = rrqr_nopiv(y,rtol);
	k
	if( k >= block_size )
	    block_size = min(block_size*2,n)
	else	    
	    break;
	end
    end
    I = eye(m,k);
    q = rrqr_q(H,tau,I,k);
    r = q'*A;
    [Q,R,p,k] = rrqr(r,rtol);
    Q = q*Q;
end



function [tau, k, a] = rrqr_nopiv(a,rtol)
%
%  The traditional algorithm for the QR Factorization without column pivoting.
%  [1] P. A. Businger and G. H. Golub, Linear least squares solution by
%  Householder transformation, Numerische Mathematik, 7 (1965), pp. 269-276.
%
%  Note: This version evaluates column norms directly
%
%  Inputs:
%      a(m,n) - matrix A, real or complex
%      rtol - relative tolerance of QR decomposition    
%
%  Outputs:
%      tau - the scalar factors of the elementary reflectors
%      k - rank of QR decomposition
%      a - truncated Householder QR factorization
%
    [m,n] = size(a);
    tau = zeros(min(m,n),1);
    s = vecnorm(a);
    d = norm(s);
    atol = rtol*max(s);

    k = 0;
    blas_level = 2;
    
    if( d == 0 )
	tau = tau(1:k);
	return
    end
    
    for j = 1:min(m,n)

        v = a(j:m,j);
        [tau(j), v] = reflector(v);
        a(j:m,j) = v;

	k = j;
	
	if( j == n || j == m )
	    tau = tau(1:k);
	    return
	end

        for i = 1:j
            qi = a(i:m,j+1);
            qi = reflectorApply(a(i:m,i), tau(i), qi);
	    a(i:m,j+1) = qi;
        end

	s_next = norm(a(j:m,j+1));

	if( s_next < atol )
	    tau = tau(1:k);
	    return
	end
	
    end

    tau = tau(1:k);
    return
end


function q = rrqr_q(a, tau, q, k)
%  Inputs:
%      a(m,n) - truncated Householder QR factorization
%      p - column pivot (permutation) vector
%      k - rank of QR decomposition (or number of Q vector to be computed)
%      q - eye(m,k) identity matrix, if true Q vector are to be computed
%
%  Outputs:
%      q(m,k) - orthogonal matrix Q
%
    [m,n] = size(q);
    for j = k:-1:1
        v = a(j:m,j);
        for i = k:-1:j
            qi = q(j:m,i);
            qi = reflectorApply(v, conj(tau(j)), qi);
	    q(j:m,i) = qi;
        end
    end
    return
end


function [u, s] = rrqr_breflector(H,tau,k)
% build block reflector Q = I - u*(s\u')
    % extract Householder vectors
    [m,n] = size(H);
    u = tril(H(:,1:k));
    % diag(u) = 1
    u(1:(m+1):end) = 1;
    s = triu(u'*u);
    % diag(s) = 1./tau
    s(1:(k+1):end) = 1./tau;
    return
end


function xs = copysign_matlab(x,y)
% a simple workaround for missing copysign
    if( y >= 0 )
	xs = +abs(x);
    else
	xs = -abs(x);
    end
    return
end


function [tau,x] = reflector(x)
%
% julia-1.11/share/julia/stdlib/v1.11/LinearAlgebra/src
% Elementary reflection similar to LAPACK. The reflector is not Hermitian but
% ensures that tridiagonalization of Hermitian matrices become real. See lawn72
%
    n = length(x);
    if( n == 0 ), tau = zeros(1,1,class(x)); return; end
    xi = x(1);
    normu = norm(x);
    if( normu == 0 ), tau = zeros(1,1,class(x)); return; end
    nu = copysign_matlab(normu, real(xi));
    xi = xi + nu;
    x(1) = -nu;
    x(2:n) = x(2:n)/xi;
    tau = xi/nu;
    return
end


function A = reflectorApply(x,tau,A)
%
% julia-1.11/share/julia/stdlib/v1.11/LinearAlgebra/src
% Multiplies `A` in-place by a Householder reflection on the left.
% It is equivalent to `A .= (I - conj(tau)*[1; x[2:end]]*[1; x[2:end]]')*A`.
%
    % BLAS level 1 update
    [m,n] = size(A);
    if( m == 0 ), return; end    
    for j = 1:n
	vAj = conj(tau)*(A(1,j) + dot(x(2:m),A(2:m,j)));
	A(1,j) = A(1,j) - vAj;
	A(2:m,j) = A(2:m,j) - vAj*x(2:m);
    end
    return
end


function A = reflectorApply1(x,tau,A)
    % BLAS level 1 update
    [m,n] = size(A);
    if( m == 0 ), return; end
    y = [1.0; x(2:m)];
    for j = 1:n
	vAj = conj(tau)*dot(y,A(:,j));
	A(:,j) = A(:,j) - vAj*y;
    end
    return
end


function A = reflectorApply2(x,tau,A)
    % BLAS level 2 update
    [m,n] = size(A);
    if( m == 0 ), return; end
    y = [1.0; x(2:m)];
%%%    A = A - conj(tau)*(sum(conj(y).*A,1).*repmat(y,1,n));
%%%    A = A - conj(tau)*(sum(conj(y).*A,1).*y);
    A = A - conj(tau)*(y*(y'*A));
    return
end
