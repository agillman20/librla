# compatibility routines

rscale!(A,b) = rmul!(A,Diagonal(b))
lscale!(b,A) = lmul!(Diagonal(b),A)

find(A::AbstractArray) = (LinearIndices(A))[findall(A)]

function eig(A)
    F = eigen(A)
    return F.values, F.vectors
end

eye(T,m,n) = Matrix{T}(I,m,n)
eye(T,n) = Matrix{T}(I,n,n)
eye(m,n) = eye(Float64,m,n)
eye(n) = eye(Float64,n)

repmat(a,m,n) = repeat(a,m,n)
