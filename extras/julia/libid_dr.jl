include("hilb.jl")
include("compat.jl")

include("libid.jl")


function ub_random(a,k;tol=eps(1.0))

    n,m = size(a)
    
    ###g = randn(eltype(a),m,k)
    g = rand(eltype(a),m,k).-1/2

    y = a*g

    F = qr(y,Val(true))
    U = Matrix(F.Q)

    B = Matrix(F.Q)'*a

    return U, B
    
end



if_randomized = 0

m = 4000
n = 2000

T = Float64
###T = Complex{T}


### testing pivoted rrqr, with optional randomization


d = eye(T,m,m)
###a = randn(Complex{Float64},m,n)
a = hilb(T,m,n)

println(T)
println(size(a))

println("if_randomized=",if_randomized)

if( if_randomized == 1 )
    println("=== randomization =====")
    @time Q, B = ub_random(a,42);
    @time Q, B = ub_random(a,42);

    @time abserr = norm(Q*B-a)
    println("error, randomized=",abserr)
    
    a_save = copy(a)
    a = copy(B)
end


x = randn(T,n)
y = copy(x)

b = copy(a)
c = copy(a)

tol = 1e-15

println("=======================")
@time tau, ipvt, k = rrqr_piv(copy(a'), tol=tol);
@time tau, ipvt, k = rrqr_piv(copy(a'), tol=tol);
println("k=",k)

println("=== rrqr_piv ===")
@time tau, jpvt, k = rrqr_piv(c, tol=tol);
println("k=",k)


if( if_randomized == 1 )
    a = copy(a_save)
end


proj = copy(c);
Au = UpperTriangular(view(proj,1:k,1:k));
rhs = view(proj,1:k,k+1:n);
@time proj[1:k,k+1:n] = Au \ rhs;
abserr = norm(a[:,jpvt] - a[:,jpvt[1:k]]*[Matrix(I,k,k) proj[1:k,k+1:n]])
println("abserr=",abserr)
abserr = norm(a[:,jpvt[k+1:n]] - a[:,jpvt[1:k]]*proj[1:k,k+1:n])
println("abserr=",abserr)


proj = copy(c)
id_lssolve!(proj, k)
proj = copy(c)
@time id_lssolve!(proj, k)
abserr = norm(a[:,jpvt] - a[:,jpvt[1:k]]*[Matrix(I,k,k) proj[1:k,k+1:n]])
println("abserr=",abserr)
abserr = norm(a[:,jpvt[k+1:n]] - a[:,jpvt[1:k]]*proj[1:k,k+1:n])
println("abserr=",abserr)

