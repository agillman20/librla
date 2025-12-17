#
#ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#
#        This is the end of the debugging code, and the beginning
#        of the special matrix code proper
#
#cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#
#	hilb - Hilbert matrix
#	invhilb - inverse of the Hilbert matrix
#	cauchy - Cauchy matrix
#	compan - companion matrix
#	vander - Vandermonde matrix
#	toeplitz - Toeplitz matrix
#	hankel - Hankel matrix
#	hadamard - Hadamard matrix
#	wilkinson - Wilkinson matrix
#


function hilb(::Type{T}, m::Integer, n::Integer) where {T<:Number}
    a=Array{T}(undef,m,n)
    for i=1:n
        for j=1:m
            a[j,i]=one(T)/(i+j-one(T))
        end
    end
    return a
end
hilb(::Type{T}, n::Integer) where {T<:Number} = hilb(T,n,n)

hilb(m::Integer, n::Integer) = hilb(typeof(1.0),m,n)
hilb(n::Integer) = hilb(typeof(1.0),n)


function invhilb(::Type{T}, m::Integer, n::Integer) where {T<:Number}
    a=Array{T}(undef,m,n)
    for i=1:n
        for j=1:m
            a[j,i]=(-one(T))^(i+j)*(i+j-1)*
               binomial(n+i-1,n-j)*binomial(n+j-1,n-i)*binomial(i+j-2,i-1)^2
        end
    end
    return a
end
invhilb(::Type{T}, n::Integer) where {T<:Number} = invhilb(T,n,n)

invhilb(m::Integer, n::Integer) = invhilb(typeof(1.0),m,n)
invhilb(n::Integer) = invhilb(typeof(1.0),n)


function cauchy(::Type{T},x,y) where {T<:Number}
    m = length(x)
    n = length(y)
    a = Array{T}(undef,m,n)
    for j=1:n
        for i=1:m
            a[i,j] = one(T)/(x[i]-y[j])
        end
    end
    return a
end
cauchy(x,y) = cauchy(eltype(x),x,y)


function cauchy(::Type{T},x,y,r,s)  where {T<:Number}
    m = length(x)
    n = length(y)
    a = Array{T}(undef,m,n)
    for j=1:n
        for i=1:m
            a[i,j] = r[i]*s[j]/(x[i]-y[j])
        end
    end
    return a
end
cauchy(x,y,r,s) = cauchy(eltype(x),x,y,r,s)


function compan(c)
    n = length(c)
    a = zeros(eltype(c),n-1,n-1)
    a[1,1:n-1] = -c[2:n] / c[1]
    for i = 1:n-2
        a[i+1,i] = one(eltype(c))
    end
    return a
end


function vander(c)
    n = length(c)
    x = Array{eltype(c)}(undef,n,n)  
    x[:,n] = 1
    a = Array{eltype(c)}(undef,n)
    a[1:n] = c[1:n]
    for i = n-1:-1:1
        x[:,i] = a .* x[:,i+1]
    end
    return x
end


function toeplitz(a,b)
    m = length(b)
    n = length(a)
    x = Array{eltype(a)}(undef,n,m)
    c = Array{eltype(a)}(undef,n+m-1)
    c[1:m] = b[end:-1:1]
    c[m:m+n-1] = a[1:end]
    for i = 1:m
        x[:,i] = c[m+1-i:m+n-i]
    end
    return x
end
toeplitz(b) = toeplitz(conj(b),b)


function hankel(a,b=[])
    if( isempty(b) ) b = zeros(eltype(a),length(a)) end
    m = length(b)
    n = length(a)
    x = Array{eltype(a)}(undef,n,m)
    c = Array{eltype(a)}(undef,n+m-1)
    c[n:m+n-1] = b[1:end]
    c[1:n] = a[1:end]
    for i = 1:m
        x[:,i] = c[i:i+n-1]
    end
    return x
end


function hadamard(n::Integer)
    if( n == 1 ) return [1];
    elseif( n == 12 ) s = 
   [1   1   1   1   1   1   1   1   1   1   1   1
    1  -1  -1   1  -1  -1  -1   1   1   1  -1   1
    1   1  -1  -1   1  -1  -1  -1   1   1   1  -1
    1  -1   1  -1  -1   1  -1  -1  -1   1   1   1
    1   1  -1   1  -1  -1   1  -1  -1  -1   1   1
    1   1   1  -1   1  -1  -1   1  -1  -1  -1   1
    1   1   1   1  -1   1  -1  -1   1  -1  -1  -1
    1  -1   1   1   1  -1   1  -1  -1   1  -1  -1
    1  -1  -1   1   1   1  -1   1  -1  -1   1  -1
    1  -1  -1  -1   1   1   1  -1   1  -1  -1   1
    1   1  -1  -1  -1   1   1   1  -1   1  -1  -1
    1  -1   1  -1  -1  -1   1   1   1  -1   1  -1];
    return s;
    elseif( n == 20 ) s = 
[1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
 1  -1  -1   1   1   1   1  -1   1  -1   1  -1  -1  -1  -1   1   1  -1  -1   1
 1  -1   1   1   1   1  -1   1  -1   1  -1  -1  -1  -1   1   1  -1  -1   1  -1
 1   1   1   1   1  -1   1  -1   1  -1  -1  -1  -1   1   1  -1  -1   1  -1  -1
 1   1   1   1  -1   1  -1   1  -1  -1  -1  -1   1   1  -1  -1   1  -1  -1   1
 1   1   1  -1   1  -1   1  -1  -1  -1  -1   1   1  -1  -1   1  -1  -1   1   1
 1   1  -1   1  -1   1  -1  -1  -1  -1   1   1  -1  -1   1  -1  -1   1   1   1
 1  -1   1  -1   1  -1  -1  -1  -1   1   1  -1  -1   1  -1  -1   1   1   1   1
 1   1  -1   1  -1  -1  -1  -1   1   1  -1  -1   1  -1  -1   1   1   1   1  -1
 1  -1   1  -1  -1  -1  -1   1   1  -1  -1   1  -1  -1   1   1   1   1  -1   1
 1   1  -1  -1  -1  -1   1   1  -1  -1   1  -1  -1   1   1   1   1  -1   1  -1
 1  -1  -1  -1  -1   1   1  -1  -1   1  -1  -1   1   1   1   1  -1   1  -1   1
 1  -1  -1  -1   1   1  -1  -1   1  -1  -1   1   1   1   1  -1   1  -1   1  -1
 1  -1  -1   1   1  -1  -1   1  -1  -1   1   1   1   1  -1   1  -1   1  -1  -1
 1  -1   1   1  -1  -1   1  -1  -1   1   1   1   1  -1   1  -1   1  -1  -1  -1
 1   1   1  -1  -1   1  -1  -1   1   1   1   1  -1   1  -1   1  -1  -1  -1  -1
 1   1  -1  -1   1  -1  -1   1   1   1   1  -1   1  -1   1  -1  -1  -1  -1   1
 1  -1  -1   1  -1  -1   1   1   1   1  -1   1  -1   1  -1  -1  -1  -1   1   1
 1  -1   1  -1  -1   1   1   1   1  -1   1  -1   1  -1  -1  -1  -1   1   1  -1
 1   1  -1  -1   1   1   1   1  -1   1  -1   1  -1  -1  -1  -1   1   1  -1  -1];
    return s;
    else        
       s = hadamard(div(n,2))
       return [ s s; s -s]
    end
end


function wilkinson(::Type{T}, n::Integer) where {T<:Number}
    a=zeros(T,n,n)
    for j=1:n
        if( j>1 ) a[j-1,j]=one(T) end
        a[j,j]=abs((n-2*j+one(T))/2) 
        if( j<n ) a[j+1,j]=one(T) end
    end
    return a
end
wilkinson(n::Integer) = wilkinson(typeof(1.0),n)

