c
c Simple Fortan90 program that tests SGEQP3RK
c  A = Q R
c
        implicit real *8 (a-h,o-z)

c Define the floating point kind to be  single_precision
        integer, parameter :: fp_kind = kind(0.0e0) 

c Define 
        real (fp_kind), dimension(:,:), allocatable ::      A
        real (fp_kind), dimension(:), allocatable ::      TAU, WORK
        real (fp_kind), dimension(:), allocatable ::      RWORK
        integer, dimension(:), allocatable ::      IWORK, JPIV
        real (8) ::      time_start,time_end
        real (8) :: omp_get_wtime
        real (fp_kind) :: abstol, reltol, ABSMAXC2NRMK, RELMAXC2NRMK
        integer :: m,n,lda,nrhs,k,kmax,lwork,info
        
        do m=1024,1024*4,1024

        n=m*2
        allocate(A(m,n))
        allocate(TAU(min(m,n)+1000))
        allocate(RWORK(2*n+1000))
        allocate(JPIV(max(m,n)+1000))
        allocate(IWORK(2*n+1000))
        
c Initialize the matrices A,T and WORK
        call hilb(A,m,n)


c Compute the matrix product  computation
c        call cpu_time(time_start)
        time_start = omp_get_wtime()

        KMAX = min(m,n)
ccc        write(*,*) kmax
        LDA = M
        NRHS = 0
        abstol = 0
        reltol = 1d-7
        write(*,*) m,n
        allocate(work(100))
        LWORK = -1
        call SGEQP3RK( M, N, NRHS, KMAX, ABSTOL, RELTOL, A, LDA,
     $     K, MAXC2NRMK, RELMAXC2NRMK, JPIV, TAU,
     $     WORK, LWORK, IWORK, INFO )
        write(*,*) m,n
        LWORK = WORK(1)
        write(*,*) 'lwork=',lwork 
        deallocate(work)
        allocate(WORK(LWORK*100))
        call SGEQP3RK( M, N, NRHS, KMAX, ABSTOL, RELTOL, A, LDA,
     $     K, MAXC2NRMK, RELMAXC2NRMK, JPIV, TAU,
     $     WORK, LWORK, IWORK, INFO )
c        call cpu_time(time_end)
        time_end = omp_get_wtime()
        write(*,*) m,n
! Print timing information
        print "(i6,i6,1x,a,1x,f8.4,2x,a,f12.4)",
     $     m, n, " time =",time_end-time_start
        print "(a,i10)", " info =", info
        
        write(*,*) 'K=',K
        write(*,*) 'REL_ERROR=',RELMAXC2NRMK
        deallocate(A,TAU,WORK,RWORK,IWORK,JPIV)
        end do

        STOP
        end


        
        subroutine hilb(A,m,n)
        implicit real *8 (a-h,o-z)
        real(4) A(m,n)
        do i = 1,n
        do j = 1,m
        a(j,i) = 1.0/(i+j-1.0)
        enddo
        enddo
        return
        end
