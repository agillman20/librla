# LAPACK: sgeqp3rk(), cgeqp3rk(), dgeqp3rk(), zgeqp3rk()
# LAPACK: slaqp2rk(), claqp2rk(), dlaqp2rk(), zlaqp2rk()
# LAPACK: slaqp3rk(), claqp3rk(), dlaqp3rk(), zlaqp3rk()

using LinearAlgebra.BLAS: @blasfunc, chkuplo

using LinearAlgebra: libblastrampoline, BlasFloat, BlasInt, LAPACKException,
    DimensionMismatch, SingularException, PosDefException,
    chkstride1, checksquare, triu, tril, dot

using Base: iszero, require_one_based_indexing

using LinearAlgebra.LAPACK: chklapackerror

# (GE) general matrices, direct decompositions
#
# These mutating functions take as arguments all the values they
# return, even if the value of the function does not depend on them
# (e.g. the tau argument).  This is so that a factorization can be
# updated in place.  The condensed mutating functions, usually a
# function of A only, are defined after this block.
for (geqp3rk, elty, relty) in
    ((:dgeqp3rk_,:Float64,:Float64),
     (:sgeqp3rk_,:Float32,:Float32),
     (:zgeqp3rk_,:ComplexF64,:Float64),
     (:cgeqp3rk_,:ComplexF32,:Float32))
    @eval begin

#       SUBROUTINE DGEQP3RK( M, N, NRHS, KMAX, ABSTOL, RELTOL, A, LDA,
#      $                     K, MAXC2NRMK, RELMAXC2NRMK, JPIV, TAU,
#      $                     WORK, LWORK, IWORK, INFO )
#       INTEGER            INFO, K, KMAX, LDA, LWORK, M, N, NRHS
#       DOUBLE PRECISION   ABSTOL, MAXC2NRMK, RELMAXC2NRMK, RELTOL
#       INTEGER            IWORK( * ), JPIV( * )
#       DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )

#       SUBROUTINE ZGEQP3RK( M, N, NRHS, KMAX, ABSTOL, RELTOL, A, LDA,
#      $                     K, MAXC2NRMK, RELMAXC2NRMK, JPIV, TAU,
#      $                     WORK, LWORK, RWORK, IWORK, INFO )
#       INTEGER            INFO, K, KMAX, LDA, LWORK, M, N, NRHS
#       DOUBLE PRECISION   ABSTOL, MAXC2NRMK, RELMAXC2NRMK, RELTOL
#       INTEGER            IWORK( * ), JPIV( * )
#       DOUBLE PRECISION   RWORK( * )
#       COMPLEX*16         A( LDA, * ), TAU( * ), WORK( * )

        function geqp3rk!(A::AbstractMatrix{$elty}, jpvt::AbstractVector{BlasInt}, tau::AbstractVector{$elty}, reltol0::BlasFloat)
            require_one_based_indexing(A, jpvt, tau)
            chkstride1(A, jpvt, tau)
            m, n  = size(A)
            if length(tau) != min(m,n)
                throw(DimensionMismatch(lazy"tau has length $(length(tau)), but needs length $(min(m,n))"))
            end
            if length(jpvt) != n
                throw(DimensionMismatch(lazy"jpvt has length $(length(jpvt)), but needs length $n"))
            end
            lda = stride(A,2)
            if lda == 0
                return A, tau, jpvt
            end # Early exit
            nrhs = Ref{BlasInt}(0)
            kmax = Ref{BlasInt}(min(m,n))
            abstol = $relty(-1.0)
            reltol = $relty(reltol0)
            k = Ref{BlasInt}(0)
            maxc2nmrk = Ref{$relty}(0)
            relmaxc2nmrk = Ref{$relty}(0)
            tau  = similar(A, $elty, min(m,n))
            work  = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iwork  = Vector{BlasInt}(undef, n)
            info  = Ref{BlasInt}()
            cmplx = eltype(A)<:Complex
            if cmplx
                rwork = Vector{$relty}(undef, 2*n)
            end
            for i = 1:2  # first call returns lwork as work[1]
                if cmplx
                    ccall((@blasfunc($geqp3rk), libblastrampoline), Cvoid,
                    (Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                     Ref{$relty},Ref{$relty},
                     Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, 
                     Ptr{$relty}, Ptr{$relty},
                     Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt}, Ptr{BlasInt}),
                          m, n, nrhs, kmax,
                          abstol, reltol,
                          A, lda, k, maxc2nmrk, relmaxc2nmrk, 
                          jpvt, tau, work,
                          lwork, rwork, iwork, info)
                else
                    ccall((@blasfunc($geqp3rk), libblastrampoline), Cvoid,
                    (Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                     Ref{$relty},Ref{$relty},
                     Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, 
                     Ptr{$relty}, Ptr{$relty},
                     Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                          m, n, nrhs, kmax,
                          abstol, reltol,
                          A, lda, k, maxc2nmrk, relmaxc2nmrk, 
                          jpvt, tau, work,
                          lwork, iwork, info)
                end
                chklapackerror(info[])
                println("info=",info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    println("cmplx=",cmplx)
                    println("lwork=",lwork)
                    resize!(work, lwork)
                end
            end
            return A, tau, jpvt, k[], maxc2nmrk[], relmaxc2nmrk[]
        end
    end
end

function geqp3rk!(A::AbstractMatrix{<:BlasFloat}, jpvt::AbstractVector{BlasInt}, reltol::BlasFloat)
    require_one_based_indexing(A, jpvt)
    m, n = size(A)
    geqp3rk!(A, jpvt, similar(A, min(m, n)), reltol)
end

function geqp3rk!(A::AbstractMatrix{<:BlasFloat}, reltol::BlasFloat)
    require_one_based_indexing(A)
    m, n = size(A)
    geqp3rk!(A, zeros(BlasInt, n), similar(A, min(m, n)), reltol)
end
