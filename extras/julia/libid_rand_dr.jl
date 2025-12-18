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



function main()

if_randomized = 1

n = 2000
m = 4000
T = Float64
###a = randn(T,n,m)
a = hilb(T,n,m)

###n = 1000
###m = 2000
###T = Complex{T}
###a = randn(T,n,m)
###a = hilb(T,n,m)


### testing pivoted rrqr, with optional randomization

tol = 1e-15

println()
println(T)
println(size(a))

println("if_randomized=",if_randomized)

@time if( if_randomized == 1 )
    println("=== randomization =====")
    r = 42
    @time Q, B = ub_random(a,r);
##    @time Q, B = ub_random(a,r);

    for niter = 1:10
        println("randomization rank r = ", r)
        @time Q, B = ub_random(a,r);

        println("after randomization, size(Q) = ",size(Q))
        println("after randomization, size(B) = ",size(B))
        @time relerr = norm(Q*B-a)/norm(a)
        println("error, randomized=",relerr)
        if( relerr < tol*10 )
            break;
        else
            r = r*2
        end
    end
        
    A = copy(a)
    a = copy(B)
    n,m = size(B)
end


println("=======================")
@time tau, ipvt, k = rrqr_piv(copy(a'), tol=tol);
@time tau, ipvt, k = rrqr_piv(copy(a'), tol=tol);
println("k=",k)


if( if_randomized == 1 )
println("=== rrqr_piv, randomized ===")
c = copy(a)
@time tau, jpvt, k = rrqr_piv(c, tol=tol);
println("k=",k)
end


if( if_randomized == 1 )
###    a = copy(a_save)
end


if( 2 == 2 )
    println("=== rrqr_piv ===")
    c = copy(a)
    @time tau, jpvt, k = rrqr_piv(c, tol=tol);
    println("k=",k)
end


if( 2 == 2 )
    println("=== rrqr_piv ===")
    c = copy(a)
    q = eye(T,n,n)
    @time tau, jpvt, k = rrqr_piv(c, q, tol=tol);
    println("k=",k)

if( n <= m )
    println("error1=",norm(q' * triu(c) - a[:,jpvt]))
    println("error1=",norm((q[1:k,1:n])' * triu(c[1:k,1:m]) - a[:,jpvt]))
else
    println("error1=",norm((q[1:m,1:n])' * triu(c[1:m,1:m]) - a[:,jpvt]))
    println("error1=",norm((q[1:k,1:n])' * triu(c[1:k,1:m]) - a[:,jpvt]))
end

    if( if_randomized == 1 )
if( n <= m )
    println("error2=",norm(Q*q' * triu(c) - A[:,jpvt]))
    println("error2=",norm(Q*(q[1:k,1:n])' * triu(c[1:k,1:m]) - A[:,jpvt]))
else
    println("error2=",norm(Q*(q[1:m,1:n])' * triu(c[1:m,1:m]) - A[:,jpvt]))
    println("error2=",norm(Q*(q[1:k,1:n])' * triu(c[1:k,1:m]) - A[:,jpvt]))
end
    end        
    
end


if( 2 == 2 )
    println("=== rrqr_piv_qt ===")
    c = copy(a)
    q = eye(T,n,n)
    @time tau, jpvt, k = rrqr_piv(c, tol=tol);
    println("  --rrqr_piv_qt, form Qt ===")
    @time rrqr_piv_qt(c, tau, q, tol=tol);
    println("k=",k)

if( n <= m )
    println("error1=",norm(q' * triu(c) - a[:,jpvt]))
    println("error1=",norm((q[1:k,1:n])' * triu(c[1:k,1:m]) - a[:,jpvt]))
else
    println("error1=",norm((q[1:m,1:n])' * triu(c[1:m,1:m]) - a[:,jpvt]))
    println("error1=",norm((q[1:k,1:n])' * triu(c[1:k,1:m]) - a[:,jpvt]))
end

    if( if_randomized == 1 )
if( n <= m )
    println("error2=",norm(Q*q' * triu(c) - A[:,jpvt]))
    println("error2=",norm(Q*(q[1:k,1:n])' * triu(c[1:k,1:m]) - A[:,jpvt]))
else
    println("error2=",norm(Q*(q[1:m,1:n])' * triu(c[1:m,1:m]) - A[:,jpvt]))
    println("error2=",norm(Q*(q[1:k,1:n])' * triu(c[1:k,1:m]) - A[:,jpvt]))
end
    end        
    

end


if( 2 == 2 )
    println("=== rrqr_piv_q_full ===")
    c = copy(a)
    q = eye(T,n,n)
    @time tau, jpvt, k = rrqr_piv(c, tol=tol);
    println("  --rrqr_piv_q_full, form Q ===")
    @time rrqr_piv_q_full(c, tau, q, tol=tol);
    println("k=",k)

if( n <= m )
    println("error1=",norm(q * triu(c) - a[:,jpvt]))
    println("error1=",norm((q[1:n,1:k]) * triu(c[1:k,1:m]) - a[:,jpvt]))
else
    println("error1=",norm((q[1:n,1:m]) * triu(c[1:m,1:m]) - a[:,jpvt]))
    println("error1=",norm((q[1:n,1:k]) * triu(c[1:k,1:m]) - a[:,jpvt]))
end


    if( if_randomized == 1 )
if( n <= m )
    println("error2=",norm(Q*q * triu(c) - A[:,jpvt]))
    println("error2=",norm(Q*(q[1:n,1:k]) * triu(c[1:k,1:m]) - A[:,jpvt]))
else
    println("error2=",norm(Q*(q[1:n,1:m]) * triu(c[1:m,1:m]) - A[:,jpvt]))
    println("error2=",norm(Q*(q[1:n,1:k]) * triu(c[1:k,1:m]) - A[:,jpvt]))
end
    end        
    
end


if( 2 == 2 )
    println("=== rrqr_piv_q ===")
    c = copy(a)
    q = eye(T,n,n)
    @time tau, jpvt, k = rrqr_piv(c, tol=tol);
    println("  --rrqr_piv_q, form Q ===")
    @time rrqr_piv_q(c, tau, q, tol=tol);
    println("k=",k)

if( n <= m )
    println("error1=",norm(q * triu(c) - a[:,jpvt]))
    println("error1=",norm((q[1:n,1:k]) * triu(c[1:k,1:m]) - a[:,jpvt]))
else
    println("error1=",norm((q[1:n,1:m]) * triu(c[1:m,1:m]) - a[:,jpvt]))
    println("error1=",norm((q[1:n,1:k]) * triu(c[1:k,1:m]) - a[:,jpvt]))
end
end

    if( if_randomized == 1 )
if( n <= m )
    println("error2=",norm(Q*q * triu(c) - A[:,jpvt]))
    println("error2=",norm(Q*(q[1:n,1:k]) * triu(c[1:k,1:m]) - A[:,jpvt]))
else
    println("error2=",norm(Q*(q[1:n,1:m]) * triu(c[1:m,1:m]) - A[:,jpvt]))
    println("error2=",norm(Q*(q[1:n,1:k]) * triu(c[1:k,1:m]) - A[:,jpvt]))
end
    end        
    
end


main()
main()
