! librla_lapack: explicit LAPACK/BLAS interfaces and thin typed wrappers
!
! The raw interfaces declare exactly the LAPACK/BLAS routines librla calls
! (LP64 integers, double precision, no complex-valued functions). The la_*
! wrappers take assumed-shape arrays, run workspace queries, handle empty
! dimensions and are generic over real(dp) and complex(dp). They are public
! so tests and demos can build reference factorizations with the same LAPACK.
!
! Author: Adrianna Gillman, Zydrunas Gimbutas
! SPDX-License-Identifier: MIT
! Assisted by: Claude Code (Anthropic)

! The raw interfaces bind either the plain Fortran symbols (dgemm_ ...) or,
! when compiled with -DLIBRLA_NEWLAPACK, Accelerate's LAPACK 3.11 interface
! (dgemm$NEWLAPACK ..., macOS 13.3+). The latter is a C interface with the
! same by-reference argument lists and no hidden string lengths; the
! interfaces use C-interoperable kinds so both bindings share one text.
#ifdef LIBRLA_NEWLAPACK
#define LA_BIND(label) bind(C, name=label)
#else
#define LA_BIND(label)
#endif

module librla_lapack
  use, intrinsic :: iso_fortran_env, only: error_unit
  use, intrinsic :: iso_c_binding, only: c_int, c_double, c_double_complex, &
       c_char
  implicit none
  private

  integer, parameter, public :: dp = kind(1.0d0)

  public :: dgeqp3, zgeqp3, dorgqr, zungqr, dgesdd, zgesdd, dgesvd, zgesvd
  public :: dgelsd, zgelsd, dtrsm, ztrsm, dgemm, zgemm, dlarnv, zlarnv
  public :: la_qrpiv, la_qrpiv_move, la_svd, la_lstsq, la_trsm_upper, la_gemm, la_larnv

  ! ------------------------------------------------------------------
  ! Raw LAPACK / BLAS interfaces
  ! ------------------------------------------------------------------
  interface
     subroutine dgeqp3(m, n, a, lda, jpvt, tau, work, lwork, info) LA_BIND('dgeqp3$NEWLAPACK')
       import :: c_int, c_double
       integer(c_int), intent(in) :: m, n, lda, lwork
       real(c_double), intent(inout) :: a(lda, *)
       integer(c_int), intent(inout) :: jpvt(*)
       real(c_double), intent(out) :: tau(*), work(*)
       integer(c_int), intent(out) :: info
     end subroutine dgeqp3

     subroutine zgeqp3(m, n, a, lda, jpvt, tau, work, lwork, rwork, &
          info) LA_BIND('zgeqp3$NEWLAPACK')
       import :: c_int, c_double, c_double_complex
       integer(c_int), intent(in) :: m, n, lda, lwork
       complex(c_double_complex), intent(inout) :: a(lda, *)
       integer(c_int), intent(inout) :: jpvt(*)
       complex(c_double_complex), intent(out) :: tau(*), work(*)
       real(c_double), intent(out) :: rwork(*)
       integer(c_int), intent(out) :: info
     end subroutine zgeqp3

     subroutine dorgqr(m, n, k, a, lda, tau, work, lwork, info) LA_BIND('dorgqr$NEWLAPACK')
       import :: c_int, c_double
       integer(c_int), intent(in) :: m, n, k, lda, lwork
       real(c_double), intent(inout) :: a(lda, *)
       real(c_double), intent(in) :: tau(*)
       real(c_double), intent(out) :: work(*)
       integer(c_int), intent(out) :: info
     end subroutine dorgqr

     subroutine zungqr(m, n, k, a, lda, tau, work, lwork, info) LA_BIND('zungqr$NEWLAPACK')
       import :: c_int, c_double_complex
       integer(c_int), intent(in) :: m, n, k, lda, lwork
       complex(c_double_complex), intent(inout) :: a(lda, *)
       complex(c_double_complex), intent(in) :: tau(*)
       complex(c_double_complex), intent(out) :: work(*)
       integer(c_int), intent(out) :: info
     end subroutine zungqr

     subroutine dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, &
          iwork, info) LA_BIND('dgesdd$NEWLAPACK')
       import :: c_int, c_double, c_char
       character(kind=c_char), intent(in) :: jobz
       integer(c_int), intent(in) :: m, n, lda, ldu, ldvt, lwork
       real(c_double), intent(inout) :: a(lda, *)
       real(c_double), intent(out) :: s(*), u(ldu, *), vt(ldvt, *), work(*)
       integer(c_int), intent(out) :: iwork(*), info
     end subroutine dgesdd

     subroutine zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, &
          rwork, iwork, info) LA_BIND('zgesdd$NEWLAPACK')
       import :: c_int, c_double, c_double_complex, c_char
       character(kind=c_char), intent(in) :: jobz
       integer(c_int), intent(in) :: m, n, lda, ldu, ldvt, lwork
       complex(c_double_complex), intent(inout) :: a(lda, *)
       real(c_double), intent(out) :: s(*), rwork(*)
       complex(c_double_complex), intent(out) :: u(ldu, *), vt(ldvt, *), work(*)
       integer(c_int), intent(out) :: iwork(*), info
     end subroutine zgesdd

     subroutine dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, &
          lwork, info) LA_BIND('dgesvd$NEWLAPACK')
       import :: c_int, c_double, c_char
       character(kind=c_char), intent(in) :: jobu, jobvt
       integer(c_int), intent(in) :: m, n, lda, ldu, ldvt, lwork
       real(c_double), intent(inout) :: a(lda, *)
       real(c_double), intent(out) :: s(*), u(ldu, *), vt(ldvt, *), work(*)
       integer(c_int), intent(out) :: info
     end subroutine dgesvd

     subroutine zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, &
          lwork, rwork, info) LA_BIND('zgesvd$NEWLAPACK')
       import :: c_int, c_double, c_double_complex, c_char
       character(kind=c_char), intent(in) :: jobu, jobvt
       integer(c_int), intent(in) :: m, n, lda, ldu, ldvt, lwork
       complex(c_double_complex), intent(inout) :: a(lda, *)
       real(c_double), intent(out) :: s(*), rwork(*)
       complex(c_double_complex), intent(out) :: u(ldu, *), vt(ldvt, *), work(*)
       integer(c_int), intent(out) :: info
     end subroutine zgesvd

     subroutine dgelsd(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, &
          lwork, iwork, info) LA_BIND('dgelsd$NEWLAPACK')
       import :: c_int, c_double
       integer(c_int), intent(in) :: m, n, nrhs, lda, ldb, lwork
       real(c_double), intent(inout) :: a(lda, *), b(ldb, *)
       real(c_double), intent(out) :: s(*), work(*)
       real(c_double), intent(in) :: rcond
       integer(c_int), intent(out) :: rank, iwork(*), info
     end subroutine dgelsd

     subroutine zgelsd(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, &
          lwork, rwork, iwork, info) LA_BIND('zgelsd$NEWLAPACK')
       import :: c_int, c_double, c_double_complex
       integer(c_int), intent(in) :: m, n, nrhs, lda, ldb, lwork
       complex(c_double_complex), intent(inout) :: a(lda, *), b(ldb, *)
       real(c_double), intent(out) :: s(*), rwork(*)
       complex(c_double_complex), intent(out) :: work(*)
       real(c_double), intent(in) :: rcond
       integer(c_int), intent(out) :: rank, iwork(*), info
     end subroutine zgelsd

     subroutine dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, &
          b, ldb) LA_BIND('dtrsm$NEWLAPACK')
       import :: c_int, c_double, c_char
       character(kind=c_char), intent(in) :: side, uplo, transa, diag
       integer(c_int), intent(in) :: m, n, lda, ldb
       real(c_double), intent(in) :: alpha, a(lda, *)
       real(c_double), intent(inout) :: b(ldb, *)
     end subroutine dtrsm

     subroutine ztrsm(side, uplo, transa, diag, m, n, alpha, a, lda, &
          b, ldb) LA_BIND('ztrsm$NEWLAPACK')
       import :: c_int, c_double_complex, c_char
       character(kind=c_char), intent(in) :: side, uplo, transa, diag
       integer(c_int), intent(in) :: m, n, lda, ldb
       complex(c_double_complex), intent(in) :: alpha, a(lda, *)
       complex(c_double_complex), intent(inout) :: b(ldb, *)
     end subroutine ztrsm

     subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, &
          c, ldc) LA_BIND('dgemm$NEWLAPACK')
       import :: c_int, c_double, c_char
       character(kind=c_char), intent(in) :: transa, transb
       integer(c_int), intent(in) :: m, n, k, lda, ldb, ldc
       real(c_double), intent(in) :: alpha, beta, a(lda, *), b(ldb, *)
       real(c_double), intent(inout) :: c(ldc, *)
     end subroutine dgemm

     subroutine zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, &
          c, ldc) LA_BIND('zgemm$NEWLAPACK')
       import :: c_int, c_double_complex, c_char
       character(kind=c_char), intent(in) :: transa, transb
       integer(c_int), intent(in) :: m, n, k, lda, ldb, ldc
       complex(c_double_complex), intent(in) :: alpha, beta, a(lda, *), b(ldb, *)
       complex(c_double_complex), intent(inout) :: c(ldc, *)
     end subroutine zgemm

     subroutine dlarnv(idist, iseed, n, x) LA_BIND('dlarnv$NEWLAPACK')
       import :: c_int, c_double
       integer(c_int), intent(in) :: idist, n
       integer(c_int), intent(inout) :: iseed(4)
       real(c_double), intent(out) :: x(*)
     end subroutine dlarnv

     subroutine zlarnv(idist, iseed, n, x) LA_BIND('zlarnv$NEWLAPACK')
       import :: c_int, c_double_complex
       integer(c_int), intent(in) :: idist, n
       integer(c_int), intent(inout) :: iseed(4)
       complex(c_double_complex), intent(out) :: x(*)
     end subroutine zlarnv
  end interface

  ! ------------------------------------------------------------------
  ! Generic wrappers
  ! ------------------------------------------------------------------

  !> Economic QR with column pivoting: a(:, jpvt) = q r, r upper (kk x n),
  !> q (m x kk) optional, kk = min(m, n). jpvt is 1-based.
  interface la_qrpiv
     module procedure qrpiv_r, qrpiv_c
  end interface la_qrpiv

  !> Same as la_qrpiv, but factors a in place: a is deallocated on return.
  !> Saves a full copy of a (and its fresh pages) when the caller is done
  !> with it.
  interface la_qrpiv_move
     module procedure qrpiv_move_r, qrpiv_move_c
  end interface la_qrpiv_move

  !> Economic SVD a = u diag(s) vh (gesdd, falling back to gesvd).
  interface la_svd
     module procedure svd_r, svd_c
  end interface la_svd

  !> Minimum-norm least squares x = argmin ||a x - b|| (gelsd, rcond = eps).
  interface la_lstsq
     module procedure lstsq_r, lstsq_c
  end interface la_lstsq

  !> b := r^{-1} b for upper triangular r (trsm, left side).
  interface la_trsm_upper
     module procedure trsm_r, trsm_c
  end interface la_trsm_upper

  !> c = op(a) op(b), op in {'N', 'T', 'C'} (gemm).
  interface la_gemm
     module procedure gemm_r, gemm_c
  end interface la_gemm

  !> Fill x with uniform(-1, 1) entries (larnv, idist = 2) in one call.
  interface la_larnv
     module procedure larnv_r, larnv_c
  end interface la_larnv

contains

  ! ------------------------------------------------------------------
  ! Pivoted QR
  ! ------------------------------------------------------------------
  subroutine qrpiv_r(a, r, jpvt, q)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable, intent(out) :: r(:,:)
    integer, allocatable, intent(out) :: jpvt(:)
    real(dp), allocatable, intent(out), optional :: q(:,:)
    real(dp), allocatable :: w(:,:)
    w = a
    call qrpiv_move_r(w, r, jpvt, q)
  end subroutine qrpiv_r

  subroutine qrpiv_move_r(a, r, jpvt, q)
    real(dp), allocatable, intent(inout) :: a(:,:)
    real(dp), allocatable, intent(out) :: r(:,:)
    integer, allocatable, intent(out) :: jpvt(:)
    real(dp), allocatable, intent(out), optional :: q(:,:)
    real(dp), allocatable :: w(:,:), tau(:), work(:)
    real(dp) :: wq(1)
    integer :: m, n, kk, lda, info, lwork, i

    m = size(a, 1); n = size(a, 2); kk = min(m, n); lda = max(1, m)
    allocate(jpvt(n), r(kk, n), tau(max(1, kk)))
    jpvt = 0
    call move_alloc(a, w)
    if (kk > 0) then
       call dgeqp3(m, n, w, lda, jpvt, tau, wq, -1, info)
       lwork = max(1, int(wq(1)))
       allocate(work(lwork))
       call dgeqp3(m, n, w, lda, jpvt, tau, work, lwork, info)
       call check_info('dgeqp3', info)
    else
       jpvt = [(i, i = 1, n)]
    end if
    r = w(1:kk, :)
    do i = 1, kk
       r(i+1:kk, i) = 0
    end do
    if (present(q)) then
       q = w(:, 1:kk)
       if (kk > 0) then
          call dorgqr(m, kk, kk, q, lda, tau, wq, -1, info)
          lwork = max(1, int(wq(1)))
          if (allocated(work)) deallocate(work)
          allocate(work(lwork))
          call dorgqr(m, kk, kk, q, lda, tau, work, lwork, info)
          call check_info('dorgqr', info)
       end if
    end if
  end subroutine qrpiv_move_r

  subroutine qrpiv_c(a, r, jpvt, q)
    complex(dp), intent(in) :: a(:,:)
    complex(dp), allocatable, intent(out) :: r(:,:)
    integer, allocatable, intent(out) :: jpvt(:)
    complex(dp), allocatable, intent(out), optional :: q(:,:)
    complex(dp), allocatable :: w(:,:)
    w = a
    call qrpiv_move_c(w, r, jpvt, q)
  end subroutine qrpiv_c

  subroutine qrpiv_move_c(a, r, jpvt, q)
    complex(dp), allocatable, intent(inout) :: a(:,:)
    complex(dp), allocatable, intent(out) :: r(:,:)
    integer, allocatable, intent(out) :: jpvt(:)
    complex(dp), allocatable, intent(out), optional :: q(:,:)
    complex(dp), allocatable :: w(:,:), tau(:), work(:)
    real(dp), allocatable :: rwork(:)
    complex(dp) :: wq(1)
    integer :: m, n, kk, lda, info, lwork, i

    m = size(a, 1); n = size(a, 2); kk = min(m, n); lda = max(1, m)
    allocate(jpvt(n), r(kk, n), tau(max(1, kk)))
    jpvt = 0
    call move_alloc(a, w)
    if (kk > 0) then
       allocate(rwork(2*n))
       call zgeqp3(m, n, w, lda, jpvt, tau, wq, -1, rwork, info)
       lwork = max(1, int(real(wq(1))))
       allocate(work(lwork))
       call zgeqp3(m, n, w, lda, jpvt, tau, work, lwork, rwork, info)
       call check_info('zgeqp3', info)
    else
       jpvt = [(i, i = 1, n)]
    end if
    r = w(1:kk, :)
    do i = 1, kk
       r(i+1:kk, i) = 0
    end do
    if (present(q)) then
       q = w(:, 1:kk)
       if (kk > 0) then
          call zungqr(m, kk, kk, q, lda, tau, wq, -1, info)
          lwork = max(1, int(real(wq(1))))
          if (allocated(work)) deallocate(work)
          allocate(work(lwork))
          call zungqr(m, kk, kk, q, lda, tau, work, lwork, info)
          call check_info('zungqr', info)
       end if
    end if
  end subroutine qrpiv_move_c

  ! ------------------------------------------------------------------
  ! Economic SVD; info > 0 only if both gesdd and gesvd fail to converge
  ! ------------------------------------------------------------------
  subroutine svd_r(a, u, s, vh, info)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable, intent(out) :: u(:,:), s(:), vh(:,:)
    integer, intent(out) :: info
    real(dp), allocatable :: w(:,:), work(:)
    integer, allocatable :: iwork(:)
    real(dp) :: wq(1)
    integer :: m, n, kk, lwork

    m = size(a, 1); n = size(a, 2); kk = min(m, n)
    allocate(u(m, kk), s(kk), vh(kk, n))
    info = 0
    if (kk == 0) return
    w = a
    allocate(iwork(8*kk))
    call dgesdd('S', m, n, w, m, s, u, m, vh, kk, wq, -1, iwork, info)
    lwork = max(1, int(wq(1)))
    allocate(work(lwork))
    call dgesdd('S', m, n, w, m, s, u, m, vh, kk, work, lwork, iwork, info)
    call check_info('dgesdd', min(info, 0))
    if (info == 0) return
    w = a
    deallocate(work)
    call dgesvd('S', 'S', m, n, w, m, s, u, m, vh, kk, wq, -1, info)
    lwork = max(1, int(wq(1)))
    allocate(work(lwork))
    call dgesvd('S', 'S', m, n, w, m, s, u, m, vh, kk, work, lwork, info)
    call check_info('dgesvd', min(info, 0))
  end subroutine svd_r

  subroutine svd_c(a, u, s, vh, info)
    complex(dp), intent(in) :: a(:,:)
    complex(dp), allocatable, intent(out) :: u(:,:), vh(:,:)
    real(dp), allocatable, intent(out) :: s(:)
    integer, intent(out) :: info
    complex(dp), allocatable :: w(:,:), work(:)
    real(dp), allocatable :: rwork(:)
    integer, allocatable :: iwork(:)
    complex(dp) :: wq(1)
    integer :: m, n, kk, mx, lwork

    m = size(a, 1); n = size(a, 2); kk = min(m, n); mx = max(m, n)
    allocate(u(m, kk), s(kk), vh(kk, n))
    info = 0
    if (kk == 0) return
    w = a
    allocate(iwork(8*kk))
    allocate(rwork(max(1, kk*max(5*kk + 7, 2*mx + 2*kk + 1))))
    call zgesdd('S', m, n, w, m, s, u, m, vh, kk, wq, -1, rwork, iwork, info)
    lwork = max(1, int(real(wq(1))))
    allocate(work(lwork))
    call zgesdd('S', m, n, w, m, s, u, m, vh, kk, work, lwork, rwork, iwork, &
         info)
    call check_info('zgesdd', min(info, 0))
    if (info == 0) return
    w = a
    deallocate(work)
    call zgesvd('S', 'S', m, n, w, m, s, u, m, vh, kk, wq, -1, rwork, info)
    lwork = max(1, int(real(wq(1))))
    allocate(work(lwork))
    call zgesvd('S', 'S', m, n, w, m, s, u, m, vh, kk, work, lwork, rwork, info)
    call check_info('zgesvd', min(info, 0))
  end subroutine svd_c

  ! ------------------------------------------------------------------
  ! Least squares (scipy.linalg.lstsq default: gelsd, rcond = eps)
  ! ------------------------------------------------------------------
  subroutine lstsq_r(a, b, x, info)
    real(dp), intent(in) :: a(:,:), b(:,:)
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, intent(out) :: info
    real(dp), allocatable :: w(:,:), bb(:,:), s(:), work(:)
    integer, allocatable :: iwork(:)
    real(dp) :: wq(1)
    integer :: m, n, nrhs, ldb, lwork, liwork, rank, iq(1)

    m = size(a, 1); n = size(a, 2); nrhs = size(b, 2); ldb = max(1, m, n)
    allocate(x(n, nrhs))
    x = 0
    info = 0
    if (min(m, n, nrhs) == 0) return
    w = a
    allocate(bb(ldb, nrhs), s(min(m, n)))
    bb = 0
    bb(1:m, :) = b
    iq = 0
    call dgelsd(m, n, nrhs, w, m, bb, ldb, s, epsilon(1.0_dp), rank, wq, -1, &
         iq, info)
    lwork = max(1, int(wq(1)))
    liwork = max(iq(1), gelsd_liwork(m, n))
    allocate(work(lwork), iwork(liwork))
    call dgelsd(m, n, nrhs, w, m, bb, ldb, s, epsilon(1.0_dp), rank, work, &
         lwork, iwork, info)
    call check_info('dgelsd', min(info, 0))
    x = bb(1:n, :)
  end subroutine lstsq_r

  subroutine lstsq_c(a, b, x, info)
    complex(dp), intent(in) :: a(:,:), b(:,:)
    complex(dp), allocatable, intent(out) :: x(:,:)
    integer, intent(out) :: info
    complex(dp), allocatable :: w(:,:), bb(:,:), work(:)
    real(dp), allocatable :: s(:), rwork(:)
    integer, allocatable :: iwork(:)
    complex(dp) :: wq(1)
    real(dp) :: rq(1)
    integer :: m, n, nrhs, ldb, lwork, lrwork, liwork, rank, iq(1)

    m = size(a, 1); n = size(a, 2); nrhs = size(b, 2); ldb = max(1, m, n)
    allocate(x(n, nrhs))
    x = 0
    info = 0
    if (min(m, n, nrhs) == 0) return
    w = a
    allocate(bb(ldb, nrhs), s(min(m, n)))
    bb = 0
    bb(1:m, :) = b
    iq = 0
    rq = 0
    call zgelsd(m, n, nrhs, w, m, bb, ldb, s, epsilon(1.0_dp), rank, wq, -1, &
         rq, iq, info)
    lwork = max(1, int(real(wq(1))))
    lrwork = max(int(rq(1)), zgelsd_lrwork(m, n, nrhs))
    liwork = max(iq(1), gelsd_liwork(m, n))
    allocate(work(lwork), rwork(lrwork), iwork(liwork))
    call zgelsd(m, n, nrhs, w, m, bb, ldb, s, epsilon(1.0_dp), rank, work, &
         lwork, rwork, iwork, info)
    call check_info('zgelsd', min(info, 0))
    x = bb(1:n, :)
  end subroutine lstsq_c

  ! Workspace bounds from the gelsd documentation. Older LAPACKs do not
  ! report IWORK/RWORK sizes in the workspace query, so take the documented
  ! minimum with SMLSIZ = 25 and NLVL computed for SMLSIZ = 1 (an upper bound).
  pure integer function gelsd_nlvl(mn) result(nlvl)
    integer, intent(in) :: mn
    nlvl = 1
    do while (2**nlvl < mn)
       nlvl = nlvl + 1
    end do
  end function gelsd_nlvl

  pure integer function gelsd_liwork(m, n) result(liwork)
    integer, intent(in) :: m, n
    integer :: mn
    mn = min(m, n)
    liwork = max(1, 3*mn*gelsd_nlvl(mn) + 11*mn)
  end function gelsd_liwork

  pure integer function zgelsd_lrwork(m, n, nrhs) result(lrwork)
    integer, intent(in) :: m, n, nrhs
    integer, parameter :: sml = 25
    integer :: mn
    mn = min(m, n)
    lrwork = 10*mn + 2*mn*sml + 8*mn*gelsd_nlvl(mn) + 3*sml*nrhs &
         + max((sml + 1)**2, n*(1 + nrhs) + 2*nrhs)
  end function zgelsd_lrwork

  ! ------------------------------------------------------------------
  ! Triangular solve, products, random fill
  ! ------------------------------------------------------------------
  subroutine trsm_r(r, b)
    real(dp), intent(in) :: r(:,:)
    real(dp), intent(inout) :: b(:,:)
    integer :: k, nrhs
    k = size(r, 1); nrhs = size(b, 2)
    if (k == 0 .or. nrhs == 0) return
    call dtrsm('L', 'U', 'N', 'N', k, nrhs, 1.0_dp, r, k, b, k)
  end subroutine trsm_r

  subroutine trsm_c(r, b)
    complex(dp), intent(in) :: r(:,:)
    complex(dp), intent(inout) :: b(:,:)
    integer :: k, nrhs
    k = size(r, 1); nrhs = size(b, 2)
    if (k == 0 .or. nrhs == 0) return
    call ztrsm('L', 'U', 'N', 'N', k, nrhs, (1.0_dp, 0.0_dp), r, k, b, k)
  end subroutine trsm_c

  subroutine gemm_r(ta, tb, a, b, c)
    character, intent(in) :: ta, tb
    real(dp), intent(in) :: a(:,:), b(:,:)
    real(dp), intent(out) :: c(:,:)
    integer :: m, n, k
    m = size(c, 1); n = size(c, 2)
    if (ta == 'N') then
       k = size(a, 2)
    else
       k = size(a, 1)
    end if
    if (m == 0 .or. n == 0) return
    call dgemm(ta, tb, m, n, k, 1.0_dp, a, max(1, size(a, 1)), b, &
         max(1, size(b, 1)), 0.0_dp, c, m)
  end subroutine gemm_r

  subroutine gemm_c(ta, tb, a, b, c)
    character, intent(in) :: ta, tb
    complex(dp), intent(in) :: a(:,:), b(:,:)
    complex(dp), intent(out) :: c(:,:)
    integer :: m, n, k
    m = size(c, 1); n = size(c, 2)
    if (ta == 'N') then
       k = size(a, 2)
    else
       k = size(a, 1)
    end if
    if (m == 0 .or. n == 0) return
    call zgemm(ta, tb, m, n, k, (1.0_dp, 0.0_dp), a, max(1, size(a, 1)), b, &
         max(1, size(b, 1)), (0.0_dp, 0.0_dp), c, m)
  end subroutine gemm_c

  subroutine larnv_r(iseed, x)
    integer, intent(inout) :: iseed(4)
    real(dp), intent(out) :: x(:,:)
    if (size(x) > 0) call dlarnv(2, iseed, size(x), x)
  end subroutine larnv_r

  subroutine larnv_c(iseed, x)
    integer, intent(inout) :: iseed(4)
    complex(dp), intent(out) :: x(:,:)
    if (size(x) > 0) call zlarnv(2, iseed, size(x), x)
  end subroutine larnv_c

  ! Negative info means an illegal argument: a librla bug, not a user error.
  subroutine check_info(name, info)
    character(len=*), intent(in) :: name
    integer, intent(in) :: info
    if (info < 0) then
       write(error_unit, '(a, a, a, i0)') 'librla: ', name, ' returned info = ', info
       stop 1
    end if
  end subroutine check_info

end module librla_lapack
