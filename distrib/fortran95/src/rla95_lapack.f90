! rla95_lapack: LAPACK/BLAS wrappers for librla95 (Fortran 95)
!
! Internal module: the public interface is the rla_* external subroutines in
! librla95.f90. LAPACK and BLAS are called through implicit interfaces under
! their standard names, so any LAPACK works (reference, OpenBLAS, MKL, or
! Accelerate's legacy symbols).
!
! Variable-size results are returned as POINTER arrays allocated here; the
! caller deallocates them. Fortran 95 has no allocatable dummy arguments.
!
! Author: Adrianna Gillman, Zydrunas Gimbutas
! SPDX-License-Identifier: MIT
! Assisted by: Claude Code (Anthropic)

module rla95_lapack
  implicit none
  private

  integer, parameter, public :: dp = kind(1.0d0)
  real(dp), parameter, public :: eps = epsilon(1.0_dp)

  public :: qrpiv_d, qrpiv_z, svd_d, svd_z, lstsq_d, lstsq_z
  public :: gemm_d, gemm_z, trsm_d, trsm_z, larnv_d, larnv_z

  external :: dgeqp3, dorgqr, dgesdd, dgesvd, dgelsd, dgemm, dtrsm, dlarnv
  external :: zgeqp3, zungqr, zgesdd, zgesvd, zgelsd, zgemm, ztrsm, zlarnv

contains

  ! ------------------------------------------------------------------
  ! Pivoted QR: a(:, jpvt) = q r, r upper triangular (kk x n), q (m x kk)
  ! formed only when wantq, kk = min(m, n). a (m x n) is consumed: it is
  ! factored in place and deallocated. jpvt is 1-based.
  ! ------------------------------------------------------------------
  subroutine qrpiv_d(m, n, a, r, jpvt, q, wantq, info)
    integer, intent(in) :: m, n
    real(dp), pointer :: a(:,:), r(:,:), q(:,:)
    integer, pointer :: jpvt(:)
    logical, intent(in) :: wantq
    integer, intent(out) :: info
    real(dp), allocatable :: tau(:), work(:)
    real(dp) :: wq(1)
    integer :: kk, lda, lwork, i

    kk = min(m, n); lda = max(1, m)
    allocate(r(kk, n), jpvt(n), tau(max(1, kk)))
    nullify(q)
    jpvt = 0
    info = 0
    if (kk > 0) then
       call dgeqp3(m, n, a, lda, jpvt, tau, wq, -1, info)
       lwork = max(1, int(wq(1)))
       allocate(work(lwork))
       call dgeqp3(m, n, a, lda, jpvt, tau, work, lwork, info)
       deallocate(work)
    else
       do i = 1, n
          jpvt(i) = i
       end do
    end if
    r = a(1:kk, :)
    do i = 1, kk
       r(i+1:kk, i) = 0
    end do
    if (wantq .and. info == 0) then
       allocate(q(m, kk))
       q = a(:, 1:kk)
       if (kk > 0) then
          call dorgqr(m, kk, kk, q, lda, tau, wq, -1, info)
          lwork = max(1, int(wq(1)))
          allocate(work(lwork))
          call dorgqr(m, kk, kk, q, lda, tau, work, lwork, info)
          deallocate(work)
       end if
    end if
    deallocate(a, tau)
  end subroutine qrpiv_d

  subroutine qrpiv_z(m, n, a, r, jpvt, q, wantq, info)
    integer, intent(in) :: m, n
    complex(dp), pointer :: a(:,:), r(:,:), q(:,:)
    integer, pointer :: jpvt(:)
    logical, intent(in) :: wantq
    integer, intent(out) :: info
    complex(dp), allocatable :: tau(:), work(:)
    real(dp), allocatable :: rwork(:)
    complex(dp) :: wq(1)
    integer :: kk, lda, lwork, i

    kk = min(m, n); lda = max(1, m)
    allocate(r(kk, n), jpvt(n), tau(max(1, kk)))
    nullify(q)
    jpvt = 0
    info = 0
    if (kk > 0) then
       allocate(rwork(2*n))
       call zgeqp3(m, n, a, lda, jpvt, tau, wq, -1, rwork, info)
       lwork = max(1, int(real(wq(1))))
       allocate(work(lwork))
       call zgeqp3(m, n, a, lda, jpvt, tau, work, lwork, rwork, info)
       deallocate(work, rwork)
    else
       do i = 1, n
          jpvt(i) = i
       end do
    end if
    r = a(1:kk, :)
    do i = 1, kk
       r(i+1:kk, i) = 0
    end do
    if (wantq .and. info == 0) then
       allocate(q(m, kk))
       q = a(:, 1:kk)
       if (kk > 0) then
          call zungqr(m, kk, kk, q, lda, tau, wq, -1, info)
          lwork = max(1, int(real(wq(1))))
          allocate(work(lwork))
          call zungqr(m, kk, kk, q, lda, tau, work, lwork, info)
          deallocate(work)
       end if
    end if
    deallocate(a, tau)
  end subroutine qrpiv_z

  ! ------------------------------------------------------------------
  ! Economic SVD a = u diag(s) vt (gesdd, falling back to gesvd when it
  ! does not converge). a (m x n) is not modified. info > 0: no convergence.
  ! ------------------------------------------------------------------
  subroutine svd_d(a, u, s, vt, info)
    real(dp), intent(in) :: a(:,:)
    real(dp), pointer :: u(:,:), s(:), vt(:,:)
    integer, intent(out) :: info
    real(dp), allocatable :: w(:,:), work(:)
    integer, allocatable :: iwork(:)
    real(dp) :: wq(1)
    integer :: m, n, kk, lwork

    m = size(a, 1); n = size(a, 2); kk = min(m, n)
    allocate(u(m, kk), s(kk), vt(kk, n))
    info = 0
    if (kk == 0) return
    allocate(w(m, n), iwork(8*kk))
    w = a
    call dgesdd('S', m, n, w, m, s, u, m, vt, kk, wq, -1, iwork, info)
    lwork = max(1, int(wq(1)))
    allocate(work(lwork))
    call dgesdd('S', m, n, w, m, s, u, m, vt, kk, work, lwork, iwork, info)
    deallocate(work)
    if (info > 0) then
       w = a
       call dgesvd('S', 'S', m, n, w, m, s, u, m, vt, kk, wq, -1, info)
       lwork = max(1, int(wq(1)))
       allocate(work(lwork))
       call dgesvd('S', 'S', m, n, w, m, s, u, m, vt, kk, work, lwork, info)
       deallocate(work)
    end if
    deallocate(w, iwork)
  end subroutine svd_d

  subroutine svd_z(a, u, s, vt, info)
    complex(dp), intent(in) :: a(:,:)
    complex(dp), pointer :: u(:,:), vt(:,:)
    real(dp), pointer :: s(:)
    integer, intent(out) :: info
    complex(dp), allocatable :: w(:,:), work(:)
    real(dp), allocatable :: rwork(:)
    integer, allocatable :: iwork(:)
    complex(dp) :: wq(1)
    integer :: m, n, kk, mx, lwork

    m = size(a, 1); n = size(a, 2); kk = min(m, n); mx = max(m, n)
    allocate(u(m, kk), s(kk), vt(kk, n))
    info = 0
    if (kk == 0) return
    allocate(w(m, n), iwork(8*kk))
    allocate(rwork(max(1, kk*max(5*kk + 7, 2*mx + 2*kk + 1))))
    w = a
    call zgesdd('S', m, n, w, m, s, u, m, vt, kk, wq, -1, rwork, iwork, info)
    lwork = max(1, int(real(wq(1))))
    allocate(work(lwork))
    call zgesdd('S', m, n, w, m, s, u, m, vt, kk, work, lwork, rwork, &
         iwork, info)
    deallocate(work)
    if (info > 0) then
       w = a
       call zgesvd('S', 'S', m, n, w, m, s, u, m, vt, kk, wq, -1, rwork, info)
       lwork = max(1, int(real(wq(1))))
       allocate(work(lwork))
       call zgesvd('S', 'S', m, n, w, m, s, u, m, vt, kk, work, lwork, &
            rwork, info)
       deallocate(work)
    end if
    deallocate(w, iwork, rwork)
  end subroutine svd_z

  ! ------------------------------------------------------------------
  ! Minimum-norm least squares x = argmin ||a x - b|| (gelsd, rcond = eps,
  ! the SciPy default). a is m x n, b is m x nrhs, x (n x nrhs) is allocated.
  ! ------------------------------------------------------------------
  subroutine lstsq_d(a, b, x, info)
    real(dp), intent(in) :: a(:,:), b(:,:)
    real(dp), pointer :: x(:,:)
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
    allocate(w(m, n), bb(ldb, nrhs), s(min(m, n)))
    w = a
    bb = 0
    bb(1:m, :) = b
    iq = 0
    call dgelsd(m, n, nrhs, w, m, bb, ldb, s, eps, rank, wq, -1, iq, info)
    lwork = max(1, int(wq(1)))
    liwork = max(iq(1), gelsd_liwork(m, n))
    allocate(work(lwork), iwork(liwork))
    call dgelsd(m, n, nrhs, w, m, bb, ldb, s, eps, rank, work, lwork, &
         iwork, info)
    x = bb(1:n, :)
    deallocate(w, bb, s, work, iwork)
  end subroutine lstsq_d

  subroutine lstsq_z(a, b, x, info)
    complex(dp), intent(in) :: a(:,:), b(:,:)
    complex(dp), pointer :: x(:,:)
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
    allocate(w(m, n), bb(ldb, nrhs), s(min(m, n)))
    w = a
    bb = 0
    bb(1:m, :) = b
    iq = 0
    rq = 0
    call zgelsd(m, n, nrhs, w, m, bb, ldb, s, eps, rank, wq, -1, rq, iq, &
         info)
    lwork = max(1, int(real(wq(1))))
    ! Some LAPACKs under-report rwork/iwork in the query: use the
    ! documented minimum when it is larger.
    lrwork = max(int(rq(1)), zgelsd_lrwork(m, n, nrhs))
    liwork = max(iq(1), gelsd_liwork(m, n))
    allocate(work(lwork), rwork(lrwork), iwork(liwork))
    call zgelsd(m, n, nrhs, w, m, bb, ldb, s, eps, rank, work, lwork, &
         rwork, iwork, info)
    x = bb(1:n, :)
    deallocate(w, bb, s, work, rwork, iwork)
  end subroutine lstsq_z

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
  ! c(m x n) = op(a) op(b), op in {'N', 'T'} (real) or {'N', 'C'} (complex),
  ! k the inner dimension
  ! ------------------------------------------------------------------
  subroutine gemm_d(ta, tb, m, n, k, a, lda, b, ldb, c, ldc)
    character, intent(in) :: ta, tb
    integer, intent(in) :: m, n, k, lda, ldb, ldc
    real(dp), intent(in) :: a(lda, *), b(ldb, *)
    real(dp), intent(inout) :: c(ldc, *)
    if (m == 0 .or. n == 0) return
    call dgemm(ta, tb, m, n, k, 1.0_dp, a, lda, b, ldb, 0.0_dp, c, ldc)
  end subroutine gemm_d

  subroutine gemm_z(ta, tb, m, n, k, a, lda, b, ldb, c, ldc)
    character, intent(in) :: ta, tb
    integer, intent(in) :: m, n, k, lda, ldb, ldc
    complex(dp), intent(in) :: a(lda, *), b(ldb, *)
    complex(dp), intent(inout) :: c(ldc, *)
    if (m == 0 .or. n == 0) return
    call zgemm(ta, tb, m, n, k, (1.0_dp, 0.0_dp), a, lda, b, ldb, &
         (0.0_dp, 0.0_dp), c, ldc)
  end subroutine gemm_z

  ! b(k x nrhs) := r^{-1} b, r upper triangular k x k
  subroutine trsm_d(k, nrhs, r, ldr, b, ldb)
    integer, intent(in) :: k, nrhs, ldr, ldb
    real(dp), intent(in) :: r(ldr, *)
    real(dp), intent(inout) :: b(ldb, *)
    if (k == 0 .or. nrhs == 0) return
    call dtrsm('L', 'U', 'N', 'N', k, nrhs, 1.0_dp, r, ldr, b, ldb)
  end subroutine trsm_d

  subroutine trsm_z(k, nrhs, r, ldr, b, ldb)
    integer, intent(in) :: k, nrhs, ldr, ldb
    complex(dp), intent(in) :: r(ldr, *)
    complex(dp), intent(inout) :: b(ldb, *)
    if (k == 0 .or. nrhs == 0) return
    call ztrsm('L', 'U', 'N', 'N', k, nrhs, (1.0_dp, 0.0_dp), r, ldr, b, ldb)
  end subroutine trsm_z

  ! x(1:nx) uniform(-1, 1) (larnv, idist = 2) in one call; complex: real
  ! and imaginary parts each uniform(-1, 1)
  subroutine larnv_d(iseed, nx, x)
    integer, intent(inout) :: iseed(4)
    integer, intent(in) :: nx
    real(dp), intent(out) :: x(*)
    if (nx > 0) call dlarnv(2, iseed, nx, x)
  end subroutine larnv_d

  subroutine larnv_z(iseed, nx, x)
    integer, intent(inout) :: iseed(4)
    integer, intent(in) :: nx
    complex(dp), intent(out) :: x(*)
    if (nx > 0) call zlarnv(2, iseed, nx, x)
  end subroutine larnv_z

end module rla95_lapack
