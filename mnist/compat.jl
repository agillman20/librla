# compatibility routines

rscale!(A,b) = rmul!(A,Diagonal(b))
lscale!(b,A) = lmul!(Diagonal(b),A)

find(A::AbstractArray) = (LinearIndices(A))[findall(A)]

function eig(A)
    F = eigen(A)
    return F.values, F.vectors
end

eye(T,n,m) = Matrix{T}(I,n,m)
eye(T,n) = Matrix{T}(I,n,n)
eye(n) = eye(Float64,n)

repmat(a,n,m) = repeat(a,n,m)
