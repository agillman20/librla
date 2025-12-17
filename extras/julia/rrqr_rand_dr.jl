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



function main(T,m,n;post_flag=0)

    ###m = 4000
    ###n = 2000
    ###    T = Float64
    ###    T = Complex{T}

    a = rand(T,m,n)
    ###a = randn(T,m,n)
    a = hilb(T,m,n)

    ### testing pivoted rrqr, with optional randomization

    tol = 1e-15

    println()
    println(T)
    println(size(a))

    println("=== qr ===")

    @time    Q, R, p = qr(a,Val(true))
    println("k = ", length(p))
    println("error = ", norm(Q*R - a[:,p])/norm(a))
    
    @time    Q, R, p = qr(a,Val(true))
    println("k = ", length(p))
    println("error = ", norm(Q*R - a[:,p])/norm(a))

    println()
    if_randomized = 0
    
    println("=== rrqr ===")
    println("if_randomized=",if_randomized)

    @time    Q, R, p, k = rrqr(a,tol=tol)
    println("k = ", k)
    println("error = ", norm(Q*R - a[:,p])/norm(a))
    
    @time    Q, R, p, k = rrqr(a,tol=tol)
    println("k = ", k)
    println("error = ", norm(Q*R - a[:,p])/norm(a))
    
    println()
    if_randomized = 1
    println("=== randomized blocked rrqr_nopiv + rrqr ===")
    println("if_randomized=",if_randomized)

    @time    Q, R, p, k = rrqr_rand(a,tol=tol)
    println("k = ", k)
    println("error = ", norm(Q*R - a[:,p])/norm(a))
    
    @time    Q, R, p, k = rrqr_rand(a,tol=tol)
    println("k = ", k)
    println("error = ", norm(Q*R - a[:,p])/norm(a))
    
    println()
    if_randomized = 1
    println("=== randomized blocked qr + qr ===")
    println("if_randomized=",if_randomized)


    @time    Q, R, p, k = rrqr_rand1(a,tol=tol)
    println("k = ", k)
    println("error = ", norm(Q*R - a[:,p])/norm(a))

    if( post_flag == 1 )
    @time    Q1, R1, p1, k1 = rrqr_rand(R,tol=tol)
    println("k1 = ", k1)
    println("error_post = ", norm(Q*Q1*R1[:,p1] - a[:,p])/norm(a))
    end
    
    @time    Q, R, p, k = rrqr_rand1(a,tol=tol)
    println("k = ", k)
    println("error = ", norm(Q*R - a[:,p])/norm(a))
    
    if( post_flag == 1 )
    @time    Q1, R1, p1, k1 = rrqr_rand(R,tol=tol)
    println("k1 = ", k1)
    println("error_post = ", norm(Q*Q1*R1[:,p1] - a[:,p])/norm(a))
    end

#    println()
#    if_randomized = 1
#    println("if_randomized=",if_randomized)

#    @time    Q, R, p, k = rrqr_rand2(a,tol=tol)
#    println("k = ", k)
#    println("error = ", norm(Q*R - a[:,p])/norm(a))
    
#    @time    Q, R, p, k = rrqr_rand2(a,tol=tol)
#    println("k = ", k)
#    println("error = ", norm(Q*R - a[:,p])/norm(a))
    
    
end

main(T) = main(T,4000,2000)


main(Float64)
main(Float64)

###main(Complex{Float64})
###main(Complex{Float64})
