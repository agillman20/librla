function [proj, p, k, q] = id_house_rand1(A,rtol)
%
%  Inputs:
%      A(m,n) - matrix A, real or complex
%      rtol - relative tolerance of QR decomposition    
%
%  Outputs:
%      proj - the projector matrix 
%      p - column pivot (permutation) vector
%      k - rank of QR decomposition
%      q - rank of QR decomposition, adjusted via SVD post-processing
%
%      such that  a(:,p((k+1):n)) \approx a(:,p(1:k))*proj
%

    [R,p,k] = rrqr_piv_rand1(A,rtol);

%%    [m,n] = size(a);
%%    proj = triu(a(1:k,1:k))\a(1:k,k+1:n);
%%    return

%%    [m,n] = size(a);
%%    [L,U] = lu(triu(a(1:k,1:k)));
%%    proj = U\(L\a(1:k,k+1:n));
%%    return

    [U,D,V] = svd(triu(R(1:k,1:k)));
    atol = rtol*norm(diag(D),'fro');
    q = sum(diag(D) > atol);
    proj = V(:,1:q)*(D(1:q,1:q)\(U(:,1:q)'))*R(1:k, (k+1):size(R,2));
    return
end


function [R, p, k] = rrqr_piv_rand1(A,rtol)
    [m,n] = size(A);    
    block_size = 42;
%%   t1 = tic;
    for i = 1:20
%%	x = 2*rand(n,block_size)-1;
	x = rand(n,block_size)-0.5;
%%	x = randn(n,block_size);
%%	x = x ./ vecnorm(x);
	y = A*x;
	[Q, R, ~] = qr(y,0);

%%      randomized rank projection, rows of R approximates projector error
%%	vecnorm(R,2,2)
%%      randomized rank projection, diagonal of R approximates projector error
%%	diag(R(1:end,1:end))
%%	diag(R(1:end,1:end))./max(vecnorm(y))

%%	d = norm(R(end,:),'fro')/norm(A,'fro') / 10;
%%	d = norm(R(end,:),'fro')/norm(R(1,:));
%%	d = norm(R(end,:),'fro')/max(vecnorm(y));
	d = norm(R(end,end),'fro')/max(vecnorm(y));

	if( d > rtol )
	    block_size = min(block_size*4,min(m,n))
%%	    block_size = min(block_size+8,min(m,n))
	else	    
	    break;
	end
	if( block_size == min(m,n) )
            [Q,R,p] = qr(A,0);
            k = block_size;
            return
	end
    end
%%    toc(t1)
%%    t1 = tic;
    k = sum(vecnorm(R,2,2) >= rtol*norm(y,'fro'));
    q = Q(:,1:k);
%%    q = Q;
    r = q'*A;
    [Q,R,p] = qr(r,0);

    k = size(R,1);
    
    % estimate rank
    %%log10(vecnorm(R,2,2))
    %%vecnorm(R,2,2) >= rtol*norm(r,'fro')
    k = sum(vecnorm(R,2,2) >= rtol*norm(r,'fro'));

%%    Q = q*Q(:,1:k);
    R = R(1:k,:);

%%    Q = q*Q;
%%    toc(t1)
end



function [tau, p, k, r] = rrqr_piv_rand(A,rtol)
    [m,n] = size(A);    
    block_size = 42;
    for i = 1:20
	x = 2*rand(n,block_size)-1;
	y = A*x;    
	[tau, k, H] = rrqr_nopiv(y,rtol);
	if( k >= block_size )
	    block_size = min(block_size*2,n)
	else	    
	    break;
	end
    end
    I = eye(m,k);
    q = rrqr_q(H,tau,I,k);
    r = q'*A;
    [tau, p, k, r] = rrqr_piv(r,rtol);
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
    p = [1:n];
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


function [Q, R, p, k] = rrqr(A,rtol)
%
%  The traditional algorithm for the QR Factorization with column pivoting.
%  [1] P. A. Businger and G. H. Golub, Linear least squares solution by
%  Householder transformation, Numerische Mathematik, 7 (1965), pp. 269-276.
%
%  Inputs:
%      A(m,n) - matrix A, real or complex
%      rtol - relative tolerance of QR decomposition    
%
%  Outputs:
%      Q(m,k) - orthogonal matrix Q
%      R(k,n) - upper triangular matrix R
%      p - column pivot (permutation) vector
%      k - rank of QR decomposition
%
    [m,n] = size(A);
    [tau,p,k,H] = rrqr_piv(A,rtol);
    I = eye(m,k);
    Q = rrqr_q(H,tau,I,k);
    R = triu(H(1:k,:));
    return
end


function [tau, p, k, a] = rrqr_piv(a,rtol)
%
%  The traditional algorithm for the QR Factorization with column pivoting.
%  [1] P. A. Businger and G. H. Golub, Linear least squares solution by
%  Householder transformation, Numerische Mathematik, 7 (1965), pp. 269-276.
%
%  Note: This version evaluates pivot norms directly (no downdating)
%
%  Inputs:
%      a(m,n) - matrix A, real or complex
%      rtol - relative tolerance of QR decomposition    
%
%  Outputs:
%      tau - the scalar factors of the elementary reflectors
%      p - column pivot (permutation) vector
%      k - rank of QR decomposition
%      a - truncated Householder QR factorization
%
    [m,n] = size(a);
    tau = zeros(min(m,n),1);
    p = [1:n];
    s = vecnorm(a);
    d = norm(s);
    atol = rtol*d;

    k = 0;
    blas_level = 2;
    
    if( d == 0 )
	tau = tau(1:k);
	return
    end
    
    for j = 1:min(m,n)

        jpiv = j;
        spiv = s(j);
        for i = j:n
            if( spiv < s(i) )
                jpiv = i;
                spiv = s(i);
            end
        end
        if( jpiv ~= j )
            for l = 1:m
                tmp = a(l,j);
                a(l,j) = a(l,jpiv);
                a(l,jpiv) = tmp;
            end
            itmp = p(j);
            p(j) = p(jpiv);
            p(jpiv) = itmp;           
        end

        v = a(j:m,j);
        [tau(j), v] = reflector(v);
        a(j:m,j) = v;

	if( blas_level == 1 )
	
            for i = j+1:n
		a(j:m,i) = reflectorApply(v, tau(j), a(j:m,i));
		s(i) = norm(a(j+1:m,i));
            end

	end

	if( blas_level == 2 )

	    a(j:m,j+1:n) = reflectorApply2(v, tau(j), a(j:m,j+1:n));
            for i = j+1:n
		s(i) = norm(a(j+1:m,i));
            end

	end
	
        k = j;
        if( norm(s(j+1:n),2) < atol )
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


function [u, s] = breflector(H,tau,k)
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
