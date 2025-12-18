using LinearAlgebra, Random, FFTW

# P. Martinsson, V. Rokhlin, Y. Shkolnisky, M. Tygert,
# ID: a software package for low-rank approximation of matrices
# via interpolative decompositions, http://tygert.com/software.html

# Halko, Martinsson, Tropp, "Finding structure with randomness:
# probabilistic algorithms for constructing approximate
# matrix decompositions," SIAM Review, 53 (2): 217-288, 2011.

# Cheng, Gimbutas, Martinsson, Rokhlin, "On the compression of
# low-rank matrices," SIAM Journal on Scientific Computing,
# 26 (4): 1389-1404, 2005.



### Random orthogonal/unitary transformation, Rokhlin's random transform

function rtrans_init(::Type{T},n) where {T<:Number}
    ixs = randperm(n)
    rot = rand(T,2,n)*2 .- 1
    for i = 1:n
        d = hypot(rot[1,i],rot[2,i])
        if( d == 0 )
            rot[1,i] = 1
            rot[2,i] = 0
        else
            rot[1,i] = rot[1,i]/d
            rot[2,i] = rot[2,i]/d
        end
    end
    return rot, ixs
end
rtrans_init(n) = rtrans_init(Float64,n)


function rtrans(::Type{T},x,rot,ixs) where {T<:Number}
    n = length(x)
    y = x[ixs]
    for i = 1:n-1
        a = y[i]
        b = y[i+1]
        calpha = rot[1,i]
        salpha = rot[2,i]
        y[i  ] = conj(calpha)*a + salpha*b
        y[i+1] = conj(salpha)*a - calpha*b 
    end
    return y
end
rtrans(x,rot,ixs)=rtrans(Float64,x,rot,ixs)


function rtrans!(::Type{T},x,y,rot,ixs) where {T<:Number}
    n = length(x)
    for i = 1:n
        y[i] = x[ixs[i]]
    end
    for i = 1:n-1
        a = y[i]
        b = y[i+1]
        calpha = rot[1,i]
        salpha = rot[2,i]
        y[i  ] = conj(calpha)*a + salpha*b
        y[i+1] = conj(salpha)*a - calpha*b 
    end
end
rtrans!(x,y,rot,ixs)=rtrans!(Float64,x,y,rot,ixs)


### Random orthogonal/unitary transformation + subselect + fft + permute

### Truncated FFT

### Householder decomposition, QR with pivoting: specified precision or rank


function rrqr(a; tol=eps(real(eltype(a))), pivot::Bool=true)
    m, n = size(a)
    b = copy(a)
    tau, p, k = rrqr_piv(b, tol=tol)
    q = Matrix{eltype(a)}(I, m, k)
    rrqr_piv_q(b, tau, q)
    r = triu(b[1:k,:]);
    return q, r, p, k
end


function rrqr_rand(A; tol=eps(real(eltype(A))), pivot::Bool=true)
    m, n = size(A)
    local block_size = 42
    local tau, k, y
    T = eltype(A)
    for i = 1:20
###        x = 2*rand(T,n,block_size) .- 1;
	x = randn(T,n,block_size)
	y = A*x
	tau, k = rrqr_nopiv(y,tol=tol);
	if( k >= block_size && block_size < min(m,n) )
	    block_size = min(block_size*4,min(m,n))
	else	    
	    break;
	end
        println([k, block_size])
    end
    println("k = ", k)
    q = eye(T,m,k);
    rrqr_piv_q(y,tau,q);
    r = q'*A;
    Q, R, p, k = rrqr(r,tol=tol);
    Q = q*Q;
    return Q, R, p, k
end


function rrqr_rand1(A; tol=eps(real(eltype(A))), pivot::Bool=true)
    m, n = size(A)
    local block_size = 42
    local F
    T = eltype(A)
    for i = 1:20
###        x = 2*rand(T,n,block_size) .- 1;
###        x = rand(T,n,block_size) .- 0.5;
	x = randn(T,n,block_size)
	y = A*x
	F = qr(y,Val(true));
        R = F.R
#        show(size(R))
#        if( block_size > 2 )
#            d = norm(R[end-2,:])
#            println("error2 = ", d)
#        end
#        if( block_size > 1 ) 
#            d = norm(R[end-1,:])
#            println("error1 = ", d)
#        end
        ##d = norm(R[end,end])
        ##println("error0 = ", d)
        ##dy = norm(y)
        ##println("error = ", dy)
##        println("diag(R) = ", diag(R))
##        println("max(vecnorm(y)) = ", maximum(mapslices(norm,y,dims=1)))
        ##d = d/norm(A) / 10
        d = norm(R[end,end])/maximum(mapslices(norm,y,dims=1))
##        println("error_est = ", d)

##        d_approx = norm(F.Q*F.R - y[:,F.p])/norm(y)
##        println("error_est(y) = ", d_approx)

#        x = randn(T,n,12)
#        y = A*x
#        yp = y - Matrix(F.Q)*(Matrix(F.Q)'*y);
#        d = norm(yp)/norm(y) / 12;
#        println("error_12 = ", d)

###        println("block_size = ", [block_size])
	if( d > tol && block_size < min(m,n) )
	    block_size = min(block_size*4,min(m,n))
###	    block_size = block_size+4
	else	    
	    break;
	end
        if( block_size == min(m,n) )
            F = qr(A,Val(true));
            return Matrix(F.Q), F.R, F.p, block_size
            break;
        end
    end
    q = Matrix(F.Q)
    r = q'*A;

##    d_approx = norm(q*r - A)/norm(A)
##    println("error_est(A) = ", d_approx)

    F = qr(r,Val(true));
##    println(size(r),size(Matrix(F.Q)),size(F.R),size(F.p))
    k = size(F.R,1)

    ### estimate rank
    k = sum(mapslices(norm,F.R,dims=2) .> tol*norm(F.R))

    Q = q*Matrix(F.Q);
    return Q, F.R, F.p, k
end


function svd_rand1(A; tol=eps(real(eltype(A))), pivot::Bool=true)
    m, n = size(A)
    local block_size = 42
    local F
    T = eltype(A)
    for i = 1:20
###        x = 2*rand(T,n,block_size) .- 1;
###        x = rand(T,n,block_size) .- 0.5;
	x = randn(T,n,block_size)
	y = A*x
	F = qr(y,Val(true));
        R = F.R
#        show(size(R))
#        if( block_size > 2 )
#            d = norm(R[end-2,:])
#            println("error2 = ", d)
#        end
#        if( block_size > 1 ) 
#            d = norm(R[end-1,:])
#            println("error1 = ", d)
#        end
        ##d = norm(R[end,end])
        ##println("error0 = ", d)
        ##dy = norm(y)
        ##println("error = ", dy)
##        println("diag(R) = ", diag(R))
##        println("max(vecnorm(y)) = ", maximum(mapslices(norm,y,dims=1)))
        ##d = d/norm(A) / 10
        d = norm(R[end,end])/maximum(mapslices(norm,y,dims=1))
##        println("error_est = ", d)

##        d_approx = norm(F.Q*F.R - y[:,F.p])/norm(y)
##        println("error_est(y) = ", d_approx)

#        x = randn(T,n,12)
#        y = A*x
#        yp = y - Matrix(F.Q)*(Matrix(F.Q)'*y);
#        d = norm(yp)/norm(y) / 12;
#        println("error_12 = ", d)

###        println("block_size = ", [block_size])
	if( d > tol && block_size < min(m,n) )
	    block_size = min(block_size*4,min(m,n))
###	    block_size = block_size+4
	else	    
	    break;
	end
        if( block_size == min(m,n) )
            U, S, V = svd(r);
            return U, S, V
            break;
        end
    end
    q = Matrix(F.Q)
    r = q'*A;

##    d_approx = norm(q*r - A)/norm(A)
##    println("error_est(A) = ", d_approx)

    U, S, V = svd(r);
##    println(size(r),size(U),size(S),size(V))
    k = size(S,1)

    ### estimate rank
    k = sum( S .> tol*norm(r) )

    U = q*U[:,1:k]
    S = S[1:k]
    V = V[:,1:k]
##    println(size(U),size(S),size(V))
    
    return U, S, V
end


function rrqr_rand2(A; tol=eps(real(eltype(A))), pivot::Bool=true)
    m, n = size(A)
    T = eltype(A)
    tau, k, y = rrqr_nopiv2(A,tol=tol);
    println("k = ", k)
    q = eye(T,m,k);
    rrqr_piv_q(y,tau,q);
    r = q'*A;
    Q, R, p, k = rrqr(r,tol=tol);
    Q = q*Q;
    return Q, R, p, k
end


function rrqr_nopiv(a; tol=eps(real(eltype(a))) )
#  Golub and Van Loan, "Matrix Computations," 3rd edition,
#  Johns Hopkins University Press, 1996, Chapter 5.
    m, n = size(a)
    T = eltype(a)
    tau = Array{T}(undef,min(n,m))
    s = Array{real(T)}(undef,n)
    for j = 1:n
        s[j] = norm(view(a,1:m,j))
    end
    d = maximum(s)
    r = 0

    if( d == zero(T) )
        return tau[1:r], r
    end
    
    @inbounds    for j = 1:min(n,m)

        v = view(a,j:m,j)
        tau[j] = reflector!(v)

        r = j
        if( j == n || j == m )
            return tau[1:r], r
        end
        
        for i = 1:j
            ai = view(a,i:m,j+1)
            reflectorApply!(view(a, i:m, i), tau[i], ai)
        end

        s[j+1] = norm(view(a,j:n,j+1))
        
        if( s[j+1] < tol*d )
            return tau[1:r], r
        end
    end
    
    return tau[1:r], r
end


function rrqr_nopiv2(a; tol=eps(real(eltype(a))) )
#  Golub and Van Loan, "Matrix Computations," 3rd edition,
#  Johns Hopkins University Press, 1996, Chapter 5.
    m, n = size(a)
    T = eltype(a)
    tau = Array{T}(undef,min(n,m))
    s = Array{real(T)}(undef,n)
    for j = 1:n
        s[j] = norm(view(a,1:m,j))
    end
    d = maximum(s)
    r = 0

    H = zeros(T,m,n);
    
    if( d == zero(T) )
        return tau[1:r], r, H
    end

    x = randn(T,n,42)
    y = a*x
    for j = 1:42
        s[j] = norm(view(y,1:m,j))
    end
    d = maximum(s[1:42])
    
    x = randn(T,n)
    y = a*x
##    d = norm(y)
    
    @inbounds    for j = 1:min(n,m)

        H[:,j] = y;
        v = view(H,j:m,j)
        tau[j] = reflector!(v)
        
        r = j
        if( j == n || j == m )
            return tau[1:r], r, H
        end
        
        x = randn(T,n)
        y = a*x
        
        for i = 1:j
            ai = view(y,i:m)
            reflectorApply!(view(H, i:m, i), tau[i], ai)
        end

        s[j+1] = norm(view(y,j:n))
        
        if( s[j+1] < tol*d )
            return tau[1:r], r, H
        end
    end
    
    return tau[1:r], r, H
end


function rrqr_piv(a; tol=eps(real(eltype(a))) )
#  Golub and Van Loan, "Matrix Computations," 3rd edition,
#  Johns Hopkins University Press, 1996, Chapter 5.
    m, n = size(a)
    T = eltype(a)
    tau = Array{T}(undef,min(m,n))
    p = Array{Integer}(undef,n)
    for j = 1:n
        p[j] = j
    end
    s = Array{real(T)}(undef,n)
    for j = 1:n
        s[j] = norm(view(a,1:m,j))
    end
    d = norm(s)
    r = 0

    if( d == zero(T) )
        return tau[1:r], p, r
    end

    blas_level = 1
    
    @inbounds    for j = 1:min(m,n)

        jpiv = j
        spiv = s[j]
        for i = j:n
            if( spiv < s[i] )
                jpiv = i
                spiv = s[i]
            end
        end
        if( jpiv != j )
            for k = 1:m
                tmp = a[k,j]
                a[k,j] = a[k,jpiv]
                a[k,jpiv] = tmp
            end
            itmp = p[j]
            p[j] = p[jpiv]
            p[jpiv] = itmp            
        end

        v = view(a,j:m,j)
        tau[j] = reflector!(v)
        ###println(norm(view(a,j:m,j+1:n)))

        if( blas_level == 1 )
        for i = j+1:n
            ai = view(a,j:m,i)
            reflectorApply!(v, tau[j], ai)
            s[i] = norm(view(a,j+1:m,i))
        end
        end
        if( blas_level == 2 )
            ai = view(a,j:m,j+1:n)
            reflectorApply!(v, tau[j], ai)
            for i = j+1:n
                s[i] = norm(view(a,j+1:m,i))
            end
        end
        r = j
        if( norm(view(s,j+1:n)) < tol*d )
            return tau[1:r], p, r
        end
    end
    
    return tau[1:r], p, r
end



function rrqr_piv(a, b; tol=eps(real(eltype(a))) )
#  Golub and Van Loan, "Matrix Computations," 3rd edition,
#  Johns Hopkins University Press, 1996, Chapter 5.
    m, n = size(a)
    T = eltype(a)
    tau = Array{T}(undef,min(m,n))
    p = Array{Integer}(undef,n)
    for j = 1:n
        p[j] = j
    end
    s = Array{real(T)}(undef,n)
    for j = 1:n
        s[j] = norm(view(a,1:m,j))
    end
    d = norm(s)
    r = 0

    if( d == zero(T) )
        return tau[1:r], p, r
    end

    blas_level = 1
    
    @inbounds    for j = 1:min(m,n)

        jpiv = j
        spiv = s[j]
        for i = j:n
            if( spiv < s[i] )
                jpiv = i
                spiv = s[i]
            end
        end
        if( jpiv != j )
            for k = 1:m
                tmp = a[k,j]
                a[k,j] = a[k,jpiv]
                a[k,jpiv] = tmp
            end
            itmp = p[j]
            p[j] = p[jpiv]
            p[jpiv] = itmp            
        end

        v = view(a,j:m,j)
        tau[j] = reflector!(v)
        ###println(norm(view(a,j:m,j+1:n)))

        if( blas_level == 1 )
        for i = j+1:n
            ai = view(a,j:m,i)
            reflectorApply!(v, tau[j], ai)
            s[i] = norm(view(a,j+1:m,i))
        end
        for i = 1:size(b,2)
            bi = view(b,j:m,i)
            reflectorApply!(v, tau[j], bi)
        end
        end
        if( blas_level == 2 )
            ai = view(a,j:m,j+1:n)
            reflectorApply!(v, tau[j], ai)
            bi = view(b,j:m,:)
            reflectorApply!(v, tau[j], bi)
            for i = j+1:n
                s[i] = norm(view(a,j+1:m,i))
            end
        end
        r = j
        if( norm(view(s,j+1:n)) < tol*d )
            return tau[1:r], p, r
        end
    end
    
    return tau[1:r], p, r
end



function rrqr_piv_qt(a, tau, b; tol=eps(real(eltype(a))) )
#  Golub and Van Loan, "Matrix Computations," 3rd edition,
#  Johns Hopkins University Press, 1996, Chapter 5.
    m, n = size(b)
    r = length(tau)

    @inbounds    for j = 1:r
        v = view(a,j:m,j)
        for i = 1:m
            bi = view(b,j:m,i)
            reflectorApply!(v, tau[j], bi)
        end
    end
    
end

function rrqr_piv_q_full(a, tau, b; tol=eps(real(eltype(a))) )
#  Golub and Van Loan, "Matrix Computations," 3rd edition,
#  Johns Hopkins University Press, 1996, Chapter 5.
    m, n = size(b)
    r = length(tau)

    @inbounds    for j = r:-1:1
        v = view(a,j:m,j)
        for i = n:-1:j
            bi = view(b,j:m,i)
            reflectorApply!(v, conj(tau[j]), bi)
        end
    end
    
end

function rrqr_piv_q(a, tau, b; tol=eps(real(eltype(a))) )
#  Golub and Van Loan, "Matrix Computations," 3rd edition,
#  Johns Hopkins University Press, 1996, Chapter 5.
    m, n = size(b)
    r = length(tau)

    @inbounds    for j = r:-1:1
        v = view(a,j:m,j)
        for i = r:-1:j
            bi = view(b,j:m,i)
            reflectorApply!(v, conj(tau[j]), bi)
        end
    end
    
end


function breflector(H,tau,k)
# build block reflector Q = I - u*(s\u')
    # extract Householder vectors
    m, n = size(H)
    T = eltype(H)
    u = tril(H[:,1:k])
    # diag(u) = 1
    u[1:(m+1):end] .= one(T)
    s = triu(u'*u)
    # diag(s) = 1./tau
    s[1:(k+1):end] .= one(T) ./ tau
    return u, s
end


function id_lssolve!(a, r)
# BLAS Level 3 solver
    m, n = size(c)
    T = eltype(c)
    Au = UpperTriangular(view(a,1:r,1:r))
    rhs = view(a,1:r,r+1:n)
    a[1:r,r+1:n] = Au \ rhs;
end


function id_lssolve_libid!(a, r)
# libid library algorithm
    n, m = size(c)
    T = eltype(c)
    scale = 100/eps(real(T))^2
@inbounds    for k = 1:m-r
        for j = r:-1:1
            sum = zero(T)
            for l = j+1:r
                sum = sum + a[j,l]*a[l,r+k]
            end
            a[j,r+k] = a[j,r+k] - sum
            rnum = real(a[j,r+k]*conj(a[j,r+k]))
            rden = real(a[j,j]*conj(a[j,j]))
            if( rnum < scale*rden )
                a[j,r+k] = a[j,r+k] / a[j,j]
            else
                a[j,r+k] = zero(T)
            end
        end
    end
end



using LinearAlgebra.BLAS: @blasfunc, chkuplo

using LinearAlgebra: libblastrampoline, BlasFloat, BlasInt, LAPACKException,
    DimensionMismatch, SingularException, PosDefException,
    chkstride1, checksquare, triu, tril, dot

using Base: iszero, require_one_based_indexing

using LinearAlgebra.LAPACK: chklapackerror


# julia-1.11/share/julia/stdlib/v1.11/LinearAlgebra/src
# Elementary reflection similar to LAPACK. The reflector is not Hermitian but
# ensures that tridiagonalization of Hermitian matrices become real. See lawn72
@inline function reflector!(x::AbstractVector{T}) where {T}
    require_one_based_indexing(x)
    n = length(x)
    n == 0 && return zero(eltype(x))
    @inbounds begin
        ξ1 = x[1]
        normu = norm(x)
        if iszero(normu)
            return zero(ξ1/normu)
        end
        ν = T(copysign(normu, real(ξ1)))
        ξ1 += ν
        x[1] = -ν
        for i = 2:n
            x[i] /= ξ1
        end
    end
    ξ1/ν
end

"""
    reflectorApply!(x, τ, A)

Multiplies `A` in-place by a Householder reflection on the left. It is equivalent to `A .= (I - conj(τ)*[1; x[2:end]]*[1; x[2:end]]')*A`.
"""
@inline function reflectorApply!(x::AbstractVector, τ::Number, A::AbstractVecOrMat)
    require_one_based_indexing(x)
    m, n = size(A, 1), size(A, 2)
    if length(x) != m
        throw(DimensionMismatch(lazy"reflector has length $(length(x)), which must match the first dimension of matrix A, $m"))
    end
    m == 0 && return A
    @inbounds for j = 1:n
        Aj, xj = view(A, 2:m, j), view(x, 2:m)
        vAj = conj(τ)*(A[1, j] + dot(xj, Aj))
        A[1, j] -= vAj
        axpy!(-vAj, xj, Aj)
    end
    return A
end



### Interpolatory decomposition (ID): specified precision or rank

### Interpolatory decomposition (ID): projection and interpolation matrices

### Randomized QR, ID, SVD
### Matrix specified by routine

### Estimate spectral norm of a matrix
### Fast multi-threaded LAPACK routines
### ID to SVD wrapper





