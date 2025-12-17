include("hilb.jl")
include("compat.jl")

include("libid.jl")


function ub_random(a,k;tol=eps(1.0))

    m, n = size(a)
    
    ###g = randn(eltype(a),n,k)
    g = rand(eltype(a),n,k).-1/2

    y = a*g

    F = qr(y,Val(true))
    U = Matrix(F.Q)

    B = Matrix(F.Q)'*a

    return U, B
    
end



if_randomized = 0

m = 2000
n = 4000

T = Float64
###T = Complex{T}

###a = randn(Complex{Float64},m,n)
a = hilb(T,m,n)

println(T)
println(size(a))

tol = 1e-15
println("tol = ", tol)

println("=======================")
@time q, r, p, k = rrqr(a, tol=tol);
@time q, r, p, k = rrqr(a, tol=tol);
println("k=",k)

println("=== rrqr ===")
@time q, r, p, k = rrqr(a, tol=tol);
println("k=",k)

println("relerr = ", norm(q*r - a[:,p])/norm(a))

