using LinearAlgebra

include("libid_lapack.jl")
include("hilb.jl")

function main(T)

m = 4000
n = 2000

##T = Float64
##T = Complex{T}

a = hilb(T,m,n)
#a = rand(T,m,n)
#a = randn(T,m,n)

println(eltype(a))
println(size(a))

tol = 1e-15
@time b, tau, jpvt, k, abs_err, rel_err = geqp3rk!(copy(a),tol);
@time p = a[:,jpvt[1:k]] \ a;
println("k=",k)
err = norm(a - a[:,jpvt[1:k]]*p)
println("after geqp3rk, abserr = ",err)

@time proj = triu(b[1:k,1:k]) \ b[1:k,k+1:n];
abserr = norm(a[:,jpvt[k+1:n]] - a[:,jpvt[1:k]]*proj)
println("after idsolve, abserr = ",abserr)

end

main(Float64)
main(Float64)

##main(Commplex{Float64})
##main(Commplex{Float64})
