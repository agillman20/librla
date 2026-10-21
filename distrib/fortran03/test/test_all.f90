! test_all: librla Fortran test suite
!
! Prints one PASS/FAIL line per check and the CROSSPORT lines shared with
! the C++, Rust and Swift ports; stops with a nonzero exit code on failure.
!
! Author: Adrianna Gillman, Zydrunas Gimbutas
! SPDX-License-Identifier: MIT
! Assisted by: Claude Code (Anthropic)

module test_support
  use librla
  use librla_lapack, only: la_qrpiv, la_svd, la_gemm
  implicit none
  public

  integer :: npass = 0, nfail = 0
  integer :: ncols_applied = 0    ! columns pushed through operator applies

  ! Dense matrix behind an operator that does NOT expose dense()
  type, extends(rla_operator_r) :: wrap_r
     real(dp), allocatable :: a(:,:)
   contains
     procedure :: apply => wrap_apply_r
     procedure :: apply_adjoint => wrap_adjoint_r
  end type wrap_r

  type, extends(rla_operator_c) :: wrap_c
     complex(dp), allocatable :: a(:,:)
   contains
     procedure :: apply => wrap_apply_c
     procedure :: apply_adjoint => wrap_adjoint_c
  end type wrap_c

  ! Operator that DOES expose its matrix through dense()
  type, extends(rla_operator_r) :: exposed_r
     real(dp), pointer :: a(:,:) => null()
   contains
     procedure :: apply => exposed_apply_r
     procedure :: apply_adjoint => exposed_adjoint_r
     procedure :: dense => exposed_dense_r
  end type exposed_r

  ! Genuinely matrix-free low-rank product A = B C, applied as B (C x)
  type, extends(rla_operator_r) :: lowrank_r
     real(dp), allocatable :: b(:,:), c(:,:)
   contains
     procedure :: apply => lowrank_apply_r
     procedure :: apply_adjoint => lowrank_adjoint_r
  end type lowrank_r

  type, extends(rla_operator_c) :: lowrank_c
     complex(dp), allocatable :: b(:,:), c(:,:)
   contains
     procedure :: apply => lowrank_apply_c
     procedure :: apply_adjoint => lowrank_adjoint_c
  end type lowrank_c

  interface fro
     module procedure fro_r, fro_c
  end interface fro
  interface orth_err
     module procedure orth_err_r, orth_err_c
  end interface orth_err
  interface span_err
     module procedure span_err_r, span_err_c
  end interface span_err
  interface qr_err
     module procedure qr_err_r, qr_err_c
  end interface qr_err
  interface svd_err
     module procedure svd_err_r, svd_err_c
  end interface svd_err
  interface id_err
     module procedure id_err_r, id_err_c
  end interface id_err
  interface lower_max
     module procedure lower_max_r, lower_max_c
  end interface lower_max
  interface random_orth
     module procedure random_orth_r, random_orth_c
  end interface random_orth

contains

  ! ---------------- reporting ----------------
  subroutine check(name, ok)
    character(len=*), intent(in) :: name
    logical, intent(in) :: ok
    if (ok) then
       npass = npass + 1
       write(*, '(a, a)') 'PASS  ', name
    else
       nfail = nfail + 1
       write(*, '(a, a)') 'FAIL  ', name
    end if
  end subroutine check

  ! PASS iff val <= tol; prints the value
  subroutine check_le(name, val, tol)
    character(len=*), intent(in) :: name
    real(dp), intent(in) :: val, tol
    character(len=64) :: buf
    write(buf, '(a, es9.2, a, es8.1, a)') ' (', val, ' <= ', tol, ')'
    call check(name // trim(buf), val <= tol)
  end subroutine check_le

  ! C printf("%.15e") formatting
  function fmt_e(x) result(out)
    real(dp), intent(in) :: x
    character(len=:), allocatable :: out
    character(len=40) :: buf
    character(len=8) :: ebuf
    integer :: ie, ex
    write(buf, '(es24.15e3)') x
    buf = adjustl(buf)
    ie = index(buf, 'E')
    read(buf(ie+1:), *) ex
    write(ebuf, '(sp, i0.2)') ex
    out = buf(1:ie-1) // 'e' // trim(ebuf)
  end function fmt_e

  function itoa(i) result(out)
    integer, intent(in) :: i
    character(len=:), allocatable :: out
    character(len=16) :: buf
    write(buf, '(i0)') i
    out = trim(buf)
  end function itoa

  logical function is_perm(p, n)
    integer, intent(in) :: p(:), n
    logical :: seen(n)
    integer :: i
    is_perm = .false.
    if (size(p) /= n) return
    seen = .false.
    do i = 1, n
       if (p(i) < 1 .or. p(i) > n) return
       if (seen(p(i))) return
       seen(p(i)) = .true.
    end do
    is_perm = .true.
  end function is_perm

  logical function all_finite_r(x)
    real(dp), intent(in) :: x(:,:)
    all_finite_r = all(abs(x) <= huge(1.0_dp))
  end function all_finite_r

  logical function all_finite_c(x)
    complex(dp), intent(in) :: x(:,:)
    all_finite_c = all(abs(real(x)) <= huge(1.0_dp)) .and. &
         all(abs(aimag(x)) <= huge(1.0_dp))
  end function all_finite_c

  ! ---------------- norms and errors ----------------
  real(dp) function fro_r(a)
    real(dp), intent(in) :: a(:,:)
    fro_r = sqrt(sum(a**2))
  end function fro_r

  real(dp) function fro_c(a)
    complex(dp), intent(in) :: a(:,:)
    fro_c = sqrt(sum(abs(a)**2))
  end function fro_c

  real(dp) function orth_err_r(q)
    real(dp), intent(in) :: q(:,:)
    real(dp), allocatable :: g(:,:)
    integer :: i
    g = matmul(transpose(q), q)
    do i = 1, size(g, 1)
       g(i, i) = g(i, i) - 1
    end do
    orth_err_r = 0
    if (size(g) > 0) orth_err_r = maxval(abs(g))
  end function orth_err_r

  real(dp) function orth_err_c(q)
    complex(dp), intent(in) :: q(:,:)
    complex(dp), allocatable :: g(:,:)
    integer :: i
    g = matmul(conjg(transpose(q)), q)
    do i = 1, size(g, 1)
       g(i, i) = g(i, i) - 1
    end do
    orth_err_c = 0
    if (size(g) > 0) orth_err_c = maxval(abs(g))
  end function orth_err_c

  real(dp) function span_err_r(a, q)
    real(dp), intent(in) :: a(:,:), q(:,:)
    span_err_r = fro(a - matmul(q, matmul(transpose(q), a))) / fro(a)
  end function span_err_r

  real(dp) function span_err_c(a, q)
    complex(dp), intent(in) :: a(:,:), q(:,:)
    span_err_c = fro(a - matmul(q, matmul(conjg(transpose(q)), a))) / fro(a)
  end function span_err_c

  real(dp) function qr_err_r(a, q, r, p)
    real(dp), intent(in) :: a(:,:), q(:,:), r(:,:)
    integer, intent(in) :: p(:)
    qr_err_r = fro(a(:, p) - matmul(q, r)) / fro(a)
  end function qr_err_r

  real(dp) function qr_err_c(a, q, r, p)
    complex(dp), intent(in) :: a(:,:), q(:,:), r(:,:)
    integer, intent(in) :: p(:)
    qr_err_c = fro(a(:, p) - matmul(q, r)) / fro(a)
  end function qr_err_c

  real(dp) function svd_err_r(a, u, s, vh)
    real(dp), intent(in) :: a(:,:), u(:,:), s(:), vh(:,:)
    real(dp), allocatable :: us(:,:)
    integer :: j
    us = u
    do j = 1, size(s)
       us(:, j) = us(:, j) * s(j)
    end do
    svd_err_r = fro(a - matmul(us, vh)) / fro(a)
  end function svd_err_r

  real(dp) function svd_err_c(a, u, s, vh)
    complex(dp), intent(in) :: a(:,:), u(:,:), vh(:,:)
    real(dp), intent(in) :: s(:)
    complex(dp), allocatable :: us(:,:)
    integer :: j
    us = u
    do j = 1, size(s)
       us(:, j) = us(:, j) * s(j)
    end do
    svd_err_c = fro(a - matmul(us, vh)) / fro(a)
  end function svd_err_c

  real(dp) function id_err_r(a, k, piv, t)
    real(dp), intent(in) :: a(:,:), t(:,:)
    integer, intent(in) :: k, piv(:)
    id_err_r = fro(a(:, piv(k+1:)) - matmul(a(:, piv(1:k)), t)) / fro(a)
  end function id_err_r

  real(dp) function id_err_c(a, k, piv, t)
    complex(dp), intent(in) :: a(:,:), t(:,:)
    integer, intent(in) :: k, piv(:)
    id_err_c = fro(a(:, piv(k+1:)) - matmul(a(:, piv(1:k)), t)) / fro(a)
  end function id_err_c

  real(dp) function lower_max_r(r)
    real(dp), intent(in) :: r(:,:)
    integer :: j
    lower_max_r = 0
    do j = 1, min(size(r, 1), size(r, 2))
       if (j < size(r, 1)) &
            lower_max_r = max(lower_max_r, maxval(abs(r(j+1:, j))))
    end do
  end function lower_max_r

  real(dp) function lower_max_c(r)
    complex(dp), intent(in) :: r(:,:)
    integer :: j
    lower_max_c = 0
    do j = 1, min(size(r, 1), size(r, 2))
       if (j < size(r, 1)) &
            lower_max_c = max(lower_max_c, maxval(abs(r(j+1:, j))))
    end do
  end function lower_max_c

  ! ---------------- test matrices ----------------
  subroutine random_orth_r(g, m, n, q)
    type(rla_rng), intent(inout) :: g
    integer, intent(in) :: m, n
    real(dp), allocatable, intent(out) :: q(:,:)
    real(dp), allocatable :: x(:,:), r(:,:)
    integer, allocatable :: p(:)
    allocate(x(m, n))
    call rla_uniform(g, x)
    call la_qrpiv(x, r, p, q)
  end subroutine random_orth_r

  subroutine random_orth_c(g, m, n, q)
    type(rla_rng), intent(inout) :: g
    integer, intent(in) :: m, n
    complex(dp), allocatable, intent(out) :: q(:,:)
    complex(dp), allocatable :: x(:,:), r(:,:)
    integer, allocatable :: p(:)
    allocate(x(m, n))
    call rla_uniform(g, x)
    call la_qrpiv(x, r, p, q)
  end subroutine random_orth_c

  ! A = U diag(s) V^T with random orthonormal U (m x n), V (n x n), m >= n
  function with_svals_r(g, m, n, s) result(a)
    type(rla_rng), intent(inout) :: g
    integer, intent(in) :: m, n
    real(dp), intent(in) :: s(:)
    real(dp), allocatable :: a(:,:), u(:,:), v(:,:)
    integer :: j
    call random_orth(g, m, n, u)
    call random_orth(g, n, n, v)
    do j = 1, n
       u(:, j) = u(:, j) * s(j)
    end do
    a = matmul(u, transpose(v))
  end function with_svals_r

  function with_svals_c(g, m, n, s) result(a)
    type(rla_rng), intent(inout) :: g
    integer, intent(in) :: m, n
    real(dp), intent(in) :: s(:)
    complex(dp), allocatable :: a(:,:), u(:,:), v(:,:)
    integer :: j
    call random_orth(g, m, n, u)
    call random_orth(g, n, n, v)
    do j = 1, n
       u(:, j) = u(:, j) * s(j)
    end do
    a = matmul(u, conjg(transpose(v)))
  end function with_svals_c

  ! s_j = 10^(-(j-1)/4)
  function decay_svals(n) result(s)
    integer, intent(in) :: n
    real(dp) :: s(n)
    integer :: j
    s = [(10.0_dp**(-real(j - 1, dp)/4), j = 1, n)]
  end function decay_svals

  ! ---------------- operators ----------------
  subroutine wrap_apply_r(self, x, y)
    class(wrap_r), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(out) :: y(:,:)
    ncols_applied = ncols_applied + size(x, 2)
    y = matmul(self%a, x)
  end subroutine wrap_apply_r

  subroutine wrap_adjoint_r(self, x, y)
    class(wrap_r), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(out) :: y(:,:)
    ncols_applied = ncols_applied + size(x, 2)
    y = matmul(transpose(self%a), x)
  end subroutine wrap_adjoint_r

  subroutine wrap_apply_c(self, x, y)
    class(wrap_c), intent(in) :: self
    complex(dp), intent(in) :: x(:,:)
    complex(dp), intent(out) :: y(:,:)
    ncols_applied = ncols_applied + size(x, 2)
    y = matmul(self%a, x)
  end subroutine wrap_apply_c

  subroutine wrap_adjoint_c(self, x, y)
    class(wrap_c), intent(in) :: self
    complex(dp), intent(in) :: x(:,:)
    complex(dp), intent(out) :: y(:,:)
    ncols_applied = ncols_applied + size(x, 2)
    y = matmul(conjg(transpose(self%a)), x)
  end subroutine wrap_adjoint_c

  subroutine exposed_apply_r(self, x, y)
    class(exposed_r), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(out) :: y(:,:)
    call la_gemm('N', 'N', self%a, x, y)
  end subroutine exposed_apply_r

  subroutine exposed_adjoint_r(self, x, y)
    class(exposed_r), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(out) :: y(:,:)
    call la_gemm('T', 'N', self%a, x, y)
  end subroutine exposed_adjoint_r

  function exposed_dense_r(self) result(p)
    class(exposed_r), intent(in) :: self
    real(dp), pointer :: p(:,:)
    p => self%a
  end function exposed_dense_r

  subroutine lowrank_apply_r(self, x, y)
    class(lowrank_r), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(out) :: y(:,:)
    ncols_applied = ncols_applied + size(x, 2)
    y = matmul(self%b, matmul(self%c, x))
  end subroutine lowrank_apply_r

  subroutine lowrank_adjoint_r(self, x, y)
    class(lowrank_r), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(out) :: y(:,:)
    ncols_applied = ncols_applied + size(x, 2)
    y = matmul(transpose(self%c), matmul(transpose(self%b), x))
  end subroutine lowrank_adjoint_r

  subroutine lowrank_apply_c(self, x, y)
    class(lowrank_c), intent(in) :: self
    complex(dp), intent(in) :: x(:,:)
    complex(dp), intent(out) :: y(:,:)
    ncols_applied = ncols_applied + size(x, 2)
    y = matmul(self%b, matmul(self%c, x))
  end subroutine lowrank_apply_c

  subroutine lowrank_adjoint_c(self, x, y)
    class(lowrank_c), intent(in) :: self
    complex(dp), intent(in) :: x(:,:)
    complex(dp), intent(out) :: y(:,:)
    ncols_applied = ncols_applied + size(x, 2)
    y = matmul(conjg(transpose(self%c)), matmul(conjg(transpose(self%b)), x))
  end subroutine lowrank_adjoint_c

end module test_support


program test_all
  use librla
  use librla_lapack, only: la_qrpiv, la_svd
  use test_support
  implicit none

  character(len=5), parameter :: methods(3) = [character(len=5) :: &
       'fast', 'svd', 'lstsq']

  write(*, '(a, a)') 'librla Fortran port ', rla_version
  call test_rng()
  call test_real()
  call test_complex()
  call crossport()

  write(*, '(a, i0, a, i0, a)') 'SUMMARY ', npass, ' passed, ', nfail, ' failed'
  if (nfail > 0) stop 1

contains

  ! ==================================================================
  subroutine test_rng()
    real(dp), parameter :: ref(6) = [ &
         3.7327920546847082e-01_dp, 8.2093410748050388e-01_dp, &
         5.5866811353917711e-01_dp, 6.4291221902741569e-01_dp, &
         6.8760847451716955e-01_dp, 1.6449965895444763e-01_dp]
    integer, parameter :: ref_seed(4) = [2384, 3667, 635, 1229]
    type(rla_rng) :: g, h
    real(dp) :: x(6, 1)
    complex(dp) :: z(3, 1)
    integer :: info
    type(rla_rng) :: dflt   ! never assigned: default-initialized

    g = rla_rng(1, 2, 3, 5)
    call rla_uniform(g, x)
    call check_le('rng dlarnv reference values', maxval(abs(x(:, 1) - ref)), &
         1e-15_dp)
    call check('rng dlarnv iseed after', all(g%state() == ref_seed))

    g = rla_rng(1, 2, 3, 5)
    call rla_uniform(g, z)
    call check_le('rng zlarnv reference values', &
         max(maxval(abs(real(z(:, 1)) - ref(1:5:2))), &
             maxval(abs(aimag(z(:, 1)) - ref(2:6:2)))), 1e-15_dp)
    call check('rng zlarnv iseed after', all(g%state() == ref_seed))

    h = rla_rng(int(z'1002003005', selected_int_kind(18)))
    call check('rng 64-bit seed maps to iseed {1,2,3,5}', &
         all(h%state() == [1, 2, 3, 5]))
    h = rla_rng(4096)
    call check('rng seed 4096 -> {0,0,1,1}', all(h%state() == [0, 0, 1, 1]))
    call check('rng default-initialized iseed {0,0,0,1}', &
         all(dflt%state() == [0, 0, 0, 1]))
    h = rla_rng(1, 2, 3, 4, info)
    call check('rng even iseed(4) rejected', info == RLA_ERR_ARG)
    h = rla_rng(4096, 2, 3, 5, info)
    call check('rng iseed entry > 4095 rejected', info == RLA_ERR_ARG)
    h = rla_rng(1, 2, 3, 5, info)
    call check('rng valid iseed accepted', info == RLA_OK)
  end subroutine test_rng

  ! ==================================================================
  subroutine test_real()
    real(dp), allocatable, target :: a(:,:)
    real(dp), allocatable :: aw(:,:), gm(:,:), q(:,:), r(:,:), &
         u(:,:), vh(:,:), t(:,:), s(:), d(:), sf(:), uf(:,:), vf(:,:), &
         q2(:,:), r2(:,:), u2(:,:), vh2(:,:), s2(:), t2(:,:), lr(:,:), &
         z(:,:), sl(:)
    real(dp) :: e(3)
    integer, allocatable :: p(:), p2(:)
    type(rla_rng) :: g, g1, g2
    type(wrap_r) :: op
    type(exposed_r) :: xop
    type(lowrank_r) :: lop
    real(dp) :: nrm, tail, err
    integer :: flag, k, k2, info, im, i, j, ierr
    character(len=:), allocatable :: tag
    real(dp) :: bb(60, 5), cc(5, 40)
    type(wrap_r) :: opw, opg
    type(lowrank_r) :: lopw   ! wide: 200 x 300

    write(*, '(a)') '--- real ---'
    g = rla_rng(11, 22, 33, 45)
    a = with_svals_r(g, 400, 300, decay_svals(300))
    aw = transpose(a)
    nrm = fro(a)
    allocate(gm(100, 80))
    call rla_uniform(g, gm)
    call la_svd(a, uf, sf, vf, ierr)

    ! ---------------- orth_sketch ----------------
    call orth_sketch(a, 1e-6_dp, q, flag, d)
    call check('real orth tol: flag 0, accepted at block 42', &
         flag == 0 .and. size(q, 2) == 42 .and. size(d) == 42)
    call check_le('real orth tol: span error', span_err(a, q), 1e-6_dp)
    call check_le('real orth tol: orthonormality', orth_err(q), 1e-13_dp)
    call check('real orth tol: diagR decreasing', &
         all(d(1:size(d)-1) >= d(2:)))
    call orth_sketch(a, 1e-10_dp, q, flag, d)
    call check('real orth tol: grows 42 -> 168', flag == 0 .and. size(q, 2) == 168)
    call check_le('real orth tol 1e-10: span error', span_err(a, q), 1e-10_dp)
    call orth_sketch(a, 20.0_dp, q, flag, d)
    call check('real orth rank: 20 + 12 columns', &
         flag == 0 .and. size(q, 2) == 32 .and. size(d) == 32)
    call check_le('real orth rank: orthonormality', orth_err(q), 1e-13_dp)
    call orth_sketch(a(1:20, :), 15.7_dp, q, flag, d)
    call check('real orth rank: min(m, k+extra) columns', size(q, 2) == 20)
    call orth_sketch(a, 1e-6_dp, q, flag, d, extra_samples=0)
    call check('real orth extra_samples=0: legacy check accepts 42', &
         flag == 0 .and. size(q, 2) == 42)
    call check_le('real orth extra_samples=0: span error', span_err(a, q), &
         1e-6_dp)
    call orth_sketch(a, 1e-17_dp, q, flag, d)
    call check('real orth rtol < eps: flag 1, empty', &
         flag == 1 .and. size(q, 1) == 400 .and. size(q, 2) == 0 .and. size(d) == 0)
    call orth_sketch(a, 1e-6_dp, q, flag, d, block_size=300)
    call check('real orth block_size >= min(m,n): flag 1', &
         flag == 1 .and. size(q, 2) == 0)
    call orth_sketch(gm, 1e-6_dp, q, flag, d)
    call check('real orth growth reaches min(m,n): flag 1', &
         flag == 1 .and. size(q, 2) == 0)

    ! ---------------- qr_sketch ----------------
    call qr_sketch(a, 1e-6_dp, q, r, p)
    k = size(q, 2)
    call check('real qr tol: shapes and permutation', size(r, 1) == k .and. &
         size(r, 2) == 300 .and. is_perm(p, 300) .and. k > 20 .and. k < 42)
    call check_le('real qr tol: ||A(:,p) - QR||/||A||', qr_err(a, q, r, p), 1e-5_dp)
    call check('real qr tol: R upper triangular', lower_max(r) <= 0)
    call check_le('real qr tol: orthonormality', orth_err(q), 1e-13_dp)
    call qr_sketch(a, 25.0_dp, q, r, p)
    call la_qrpiv(a, r2, p2, q2)
    err = qr_err(a, q2(:, 1:25), r2(1:25, :), p2)
    call check('real qr rank: k = 25', size(q, 2) == 25 .and. size(r, 1) == 25)
    call check_le('real qr rank: error / pivoted-QR error', &
         qr_err(a, q, r, p) / err, 4.0_dp)
    call qr_sketch(aw, 1e-6_dp, q, r, p)
    call check_le('real qr wide tol: error', qr_err(aw, q, r, p), 1e-5_dp)
    call check('real qr wide tol: R upper, perm', lower_max(r) <= 0 .and. &
         is_perm(p, 400))
    call qr_sketch(aw, 25.0_dp, q, r, p)
    call check_le('real qr wide rank: error', qr_err(aw, q, r, p), 1e-5_dp)
    call qr_sketch(gm, 1e-6_dp, q, r, p)
    call la_qrpiv(gm, r2, p2, q2)
    call check('real qr fallback = deterministic pivoted QR', &
         all(p == p2) .and. size(q, 2) == 80 .and. &
         maxval(abs(q - q2)) <= 0 .and. maxval(abs(r - r2)) <= 0)

    ! ---------------- svd_sketch ----------------
    call svd_sketch(a, 1e-6_dp, u, s, vh)
    k = size(s)
    call check('real svd tol: rank within 1 of count(s >= rtol*s1)', &
         abs(k - count(sf >= 1e-6_dp*sf(1))) <= 1)
    call check_le('real svd tol: singular values', &
         maxval(abs(s - sf(1:k))) / sf(1), 1e-10_dp)
    call check_le('real svd tol: reconstruction', svd_err(a, u, s, vh), 1e-5_dp)
    call check_le('real svd tol: U, Vh orthonormal', &
         max(orth_err(u), orth_err(transpose(vh))), 1e-13_dp)
    call svd_sketch(a, 10.0_dp, u, s, vh)
    tail = sqrt(sum(sf(11:)**2)) / nrm
    call check('real svd rank: k = 10', size(s) == 10 .and. size(u, 2) == 10 &
         .and. size(vh, 1) == 10)
    call check_le('real svd rank: singular values', &
         maxval(abs(s - sf(1:10))) / sf(1), 1e-8_dp)
    call check_le('real svd rank: error / optimal', svd_err(a, u, s, vh) / tail, &
         4.0_dp)
    call svd_sketch(aw, 1e-6_dp, u, s, vh)
    k = size(s)
    call check('real svd wide tol: shapes', size(u, 1) == 300 .and. &
         size(vh, 2) == 400 .and. abs(k - count(sf >= 1e-6_dp*sf(1))) <= 1)
    call check_le('real svd wide tol: singular values', &
         maxval(abs(s - sf(1:k))) / sf(1), 1e-10_dp)
    call check_le('real svd wide tol: reconstruction', svd_err(aw, u, s, vh), &
         1e-5_dp)
    call check_le('real svd wide tol: U, Vh orthonormal', &
         max(orth_err(u), orth_err(transpose(vh))), 1e-13_dp)
    call svd_sketch(aw, 10.0_dp, u, s, vh)
    call check_le('real svd wide rank: error / optimal', &
         svd_err(aw, u, s, vh) / tail, 4.0_dp)
    call svd_sketch(gm, 1e-6_dp, u, s, vh)
    call la_svd(gm, u2, s2, vh2, ierr)
    call check('real svd fallback = full LAPACK SVD', size(s) == 80 .and. &
         maxval(abs(s - s2)) <= 0 .and. maxval(abs(u - u2)) <= 0)

    ! ---------------- id_sketch / id_qrpiv ----------------
    do im = 1, 3
       tag = trim(methods(im))
       call id_sketch(a, 1e-6_dp, k, p, t, method=tag)
       call check('real id_sketch tol ' // tag // ': shapes', is_perm(p, 300) &
            .and. size(t, 1) == k .and. size(t, 2) == 300 - k .and. k > 20)
       call check_le('real id_sketch tol ' // tag // ': error', &
            id_err(a, k, p, t), 1e-5_dp)
       call id_sketch(a, 20.0_dp, k, p, t, method=tag)
       call check_le('real id_sketch rank ' // tag // ': error', &
            id_err(a, k, p, t), 1e-3_dp)
       call check('real id_sketch rank ' // tag // ': k = 20', k == 20)
       call id_qrpiv(a, 1e-6_dp, k, p, t, method=tag)
       call check_le('real id_qrpiv tol ' // tag // ': error', &
            id_err(a, k, p, t), 1e-5_dp)
       call id_qrpiv(a, 20.0_dp, k, p, t, method=tag)
       call check_le('real id_qrpiv rank ' // tag // ': error', &
            id_err(a, k, p, t), 1e-3_dp)
       call id_sketch(aw, 1e-6_dp, k, p, t, method=tag)
       call check_le('real id_sketch wide tol ' // tag // ': error', &
            id_err(aw, k, p, t), 1e-5_dp)
    end do

    ! rank-deficient: requested rank 10 > true rank 5
    allocate(lr(60, 40))
    call rla_uniform(g, bb)
    call rla_uniform(g, cc)
    lr = matmul(bb, cc)
    do im = 1, 3
       tag = trim(methods(im))
       call id_sketch(lr, 10.0_dp, k, p, t, method=tag)
       call check('real id_sketch rank-deficient ' // tag // ': k = 10, T finite', &
            k == 10 .and. all_finite_r(t))
       call check_le('real id_sketch rank-deficient ' // tag // ': error', &
            id_err(lr, k, p, t), 1e-12_dp)
       call id_qrpiv(lr, 10.0_dp, k, p, t, method=tag)
       call check('real id_qrpiv rank-deficient ' // tag // ': k = 10, T finite', &
            k == 10 .and. all_finite_r(t))
       call check_le('real id_qrpiv rank-deficient ' // tag // ': error', &
            id_err(lr, k, p, t), 1e-12_dp)
    end do

    ! zero matrix: k = 0
    allocate(z(30, 20))
    z = 0
    do im = 1, 3
       tag = trim(methods(im))
       call id_qrpiv(z, 1e-6_dp, k, p, t, method=tag)
       call check('real id_qrpiv zero matrix ' // tag // ': k = 0, T 0 x 20', &
            k == 0 .and. size(t, 1) == 0 .and. size(t, 2) == 20 .and. is_perm(p, 20))
       call id_sketch(z, 1e-6_dp, k, p, t, method=tag)
       call check('real id_sketch zero matrix ' // tag // ': k = 0, T 0 x 20', &
            k == 0 .and. size(t, 1) == 0 .and. size(t, 2) == 20)
    end do

    ! k == n
    do im = 1, 3
       tag = trim(methods(im))
       call id_qrpiv(gm(:, 1:20), 20.0_dp, k, p, t, method=tag)
       call check('real id_qrpiv k == n ' // tag // ': T 20 x 0', &
            k == 20 .and. size(t, 1) == 20 .and. size(t, 2) == 0)
       call id_sketch(gm(:, 1:20), 20.0_dp, k, p, t, method=tag)
       call check('real id_sketch k == n ' // tag // ': T 20 x 0', &
            k == 20 .and. size(t, 1) == 20 .and. size(t, 2) == 0)
    end do

    call id_sketch(a, 1e-6_dp, k, p, t, method='qr', info=info)
    call check('real id_sketch invalid method -> RLA_ERR_METHOD', &
         info == RLA_ERR_METHOD)
    call id_qrpiv(a, 1e-6_dp, k, p, t, method='bogus', info=info)
    call check('real id_qrpiv invalid method -> RLA_ERR_METHOD', &
         info == RLA_ERR_METHOD)
    call svd_sketch(a, 1e-6_dp, u, s, vh, power_iter=-1, info=info)
    call check('real svd negative power_iter -> RLA_ERR_ARG', info == RLA_ERR_ARG)

    ! ---------------- matrix-free ----------------
    op%m = 400; op%n = 300; op%a = a
    g1 = rla_rng(1, 2, 3, 5); g2 = g1
    call orth_sketch(a, 20.0_dp, q, flag, d, rng=g1)
    call orth_sketch(op, 20.0_dp, q2, flag, s2, rng=g2)
    ! trailing columns of Q amplify rounding by 1/diagR(j): compare diagR
    ! and the leading columns only
    call check_le('real operator orth rank = dense: diagR', &
         maxval(abs(d - s2)) / d(1), 1e-13_dp)
    call check_le('real operator orth rank = dense: Q(:, 1:10)', &
         maxval(abs(q(:, 1:10) - q2(:, 1:10))), 1e-11_dp)
    call check('real operator advances rng like dense', &
         all(g1%state() == g2%state()))
    ! rtol 1.3e-6 lies between singular values 24 and 25; at 1e-6 = s(25)
    ! rounding decides the rank
    g1 = rla_rng(1, 2, 3, 5); g2 = g1
    call svd_sketch(a, 1.3e-6_dp, u, s, vh, rng=g1, power_iter=1)
    call svd_sketch(op, 1.3e-6_dp, u2, s2, vh2, rng=g2, power_iter=1)
    call check_le('real operator svd tol (power_iter=1) = dense', &
         maxval(abs(s - s2)) / s(1), 1e-12_dp)
    g1 = rla_rng(1, 2, 3, 5); g2 = g1
    call id_sketch(a, 1.3e-6_dp, k, p, t, method='lstsq', rng=g1)
    call id_sketch(op, 1.3e-6_dp, k2, p2, t2, method='lstsq', rng=g2)
    call check('real operator id_sketch lstsq = dense: k, piv', &
         k == k2 .and. all(p == p2))
    call check_le('real operator id_sketch lstsq = dense: T', &
         maxval(abs(t - t2)), 1e-9_dp)
    ! wide operator: svd_sketch goes through the adjoint
    opw%m = 300; opw%n = 400; opw%a = aw
    g1 = rla_rng(1, 2, 3, 5); g2 = g1
    call svd_sketch(aw, 15.0_dp, u, s, vh, rng=g1)
    call svd_sketch(opw, 15.0_dp, u2, s2, vh2, rng=g2)
    call check_le('real operator svd wide rank = dense', &
         maxval(abs(s - s2)) / s(1), 1e-12_dp)
    call check_le('real operator svd wide rank: reconstruction', &
         svd_err(aw, u2, s2, vh2) / tail, 4.0_dp)
    ! tolerance-mode fallback materializes the operator
    opg%m = 100; opg%n = 80; opg%a = gm
    g1 = rla_rng(1, 2, 3, 5); g2 = g1
    call qr_sketch(gm, 1e-6_dp, q, r, p, rng=g1)
    ncols_applied = 0
    call qr_sketch(opg, 1e-6_dp, q2, r2, p2, rng=g2)
    call check('real operator qr fallback = dense (materialized)', &
         all(p == p2) .and. maxval(abs(q - q2)) <= 1e-12_dp .and. &
         maxval(abs(r - r2)) <= 1e-12_dp)
    call check('real operator qr fallback applied >= n columns', &
         ncols_applied >= 80)
    call svd_sketch(opg, 1e-6_dp, u2, s2, vh2)
    call svd_sketch(gm, 1e-6_dp, u, s, vh)
    call check_le('real operator svd fallback = dense', &
         maxval(abs(s - s2)), 1e-12_dp)
    call id_qrpiv(opg, 30.0_dp, k, p, t)
    call id_qrpiv(gm, 30.0_dp, k2, p2, t2)
    call check('real operator id_qrpiv = dense', k == k2 .and. all(p == p2) &
         .and. maxval(abs(t - t2)) <= 1e-12_dp)
    ! operator exposing dense(): bitwise identical to the dense path
    xop%m = 400; xop%n = 300; xop%a => a
    g1 = rla_rng(1, 2, 3, 5); g2 = g1
    call qr_sketch(a, 1e-6_dp, q, r, p, rng=g1)
    call qr_sketch(xop, 1e-6_dp, q2, r2, p2, rng=g2)
    call check('real operator with dense() = dense path', all(p == p2) .and. &
         maxval(abs(q - q2)) <= 0 .and. maxval(abs(r - r2)) <= 0)
    ! genuinely matrix-free low-rank product, rank mode
    allocate(lop%b(300, 8), lop%c(8, 200))
    call rla_uniform(g, lop%b)
    call rla_uniform(g, lop%c)
    lop%m = 300; lop%n = 200
    lr = matmul(lop%b, lop%c)
    ncols_applied = 0
    call svd_sketch(lop, 8.0_dp, u, s, vh)
    call check_le('real low-rank operator svd rank 8: reconstruction', &
         svd_err(lr, u, s, vh), 1e-12_dp)
    call check('real low-rank operator svd: no materialization', &
         ncols_applied < 200)
    call id_sketch(lop, 8.0_dp, k, p, t)
    call check_le('real low-rank operator id_sketch rank 8: error', &
         id_err(lr, k, p, t), 1e-12_dp)
    call qr_sketch(lop, 8.0_dp, q, r, p)
    call check_le('real low-rank operator qr_sketch rank 8: error', &
         qr_err(lr, q, r, p), 1e-12_dp)
    lopw%m = 200; lopw%n = 300
    lopw%b = transpose(lop%c); lopw%c = transpose(lop%b)
    call svd_sketch(lopw, 8.0_dp, u, s, vh)
    call check_le('real low-rank wide operator svd rank 8: reconstruction', &
         svd_err(transpose(lr), u, s, vh), 1e-12_dp)

    ! ---------------- power iteration ----------------
    allocate(sl(150))
    sl = [(1.0_dp / real(j, dp), j = 1, 150)]
    lr = with_svals_r(g, 200, 150, sl)
    tail = sqrt(sum(sl(11:)**2)) / fro(lr)
    do i = 0, 2
       g1 = rla_rng(1, 2, 3, 5)
       call svd_sketch(lr, 10.0_dp, u, s, vh, power_iter=i, extra_samples=2, &
            rng=g1)
       e(i + 1) = svd_err(lr, u, s, vh) / tail
    end do
    call check_le('real power_iter=1 vs 0 (error ratio)', e(2) / e(1), 1.0_dp)
    call check_le('real power_iter=2 vs 1 (error ratio)', e(3) / e(2), 1.01_dp)

    ! ---------------- seeds ----------------
    g1 = rla_rng(1, 2, 3, 5); g2 = g1
    call svd_sketch(a, 10.0_dp, u, s, vh, rng=g1)
    call svd_sketch(a, 10.0_dp, u2, s2, vh2, rng=g2)
    call check('real same seed -> identical output', &
         maxval(abs(u - u2)) <= 0 .and. maxval(abs(s - s2)) <= 0)
    call check('real rng advances', any(g1%state() /= [1, 2, 3, 5]))
    call svd_sketch(a, 10.0_dp, u2, s2, vh2, rng=g1)
    call check('real advanced rng -> different sketch', maxval(abs(u - u2)) > 0)
    call svd_sketch(a, 10.0_dp, u, s, vh)
    call svd_sketch(a, 10.0_dp, u2, s2, vh2)
    call check('real default rng is deterministic', maxval(abs(u - u2)) <= 0)
  end subroutine test_real

  ! ==================================================================
  subroutine test_complex()
    complex(dp), allocatable :: a(:,:), aw(:,:), gm(:,:), q(:,:), r(:,:), &
         u(:,:), vh(:,:), t(:,:), uf(:,:), vf(:,:), q2(:,:), r2(:,:), &
         u2(:,:), vh2(:,:), t2(:,:), lr(:,:), z(:,:)
    real(dp), allocatable :: s(:), d(:), sf(:), s2(:), sl(:), e(:)
    integer, allocatable :: p(:), p2(:)
    type(rla_rng) :: g, g1, g2
    type(wrap_c) :: op
    type(lowrank_c) :: lop
    real(dp) :: nrm, tail, err
    integer :: flag, k, k2, info, im, i, j, ierr
    character(len=:), allocatable :: tag
    complex(dp) :: bb(60, 5), cc(5, 40)
    type(wrap_c) :: opw, opg

    write(*, '(a)') '--- complex ---'
    allocate(e(3))
    g = rla_rng(7, 8, 9, 11)
    a = with_svals_c(g, 400, 300, decay_svals(300))
    aw = conjg(transpose(a))
    nrm = fro(a)
    allocate(gm(100, 80))
    call rla_uniform(g, gm)
    call la_svd(a, uf, sf, vf, ierr)

    ! ---------------- orth_sketch ----------------
    call orth_sketch(a, 1e-6_dp, q, flag, d)
    call check('cplx orth tol: flag 0, accepted at block 42', &
         flag == 0 .and. size(q, 2) == 42 .and. size(d) == 42)
    call check_le('cplx orth tol: span error', span_err(a, q), 1e-6_dp)
    call check_le('cplx orth tol: orthonormality', orth_err(q), 1e-13_dp)
    call orth_sketch(a, 1e-10_dp, q, flag, d)
    call check('cplx orth tol: grows 42 -> 168', flag == 0 .and. size(q, 2) == 168)
    call check_le('cplx orth tol 1e-10: span error', span_err(a, q), 1e-10_dp)
    call orth_sketch(a, 20.0_dp, q, flag, d)
    call check('cplx orth rank: 20 + 12 columns', &
         flag == 0 .and. size(q, 2) == 32 .and. size(d) == 32)
    call orth_sketch(a(1:20, :), 15.7_dp, q, flag, d)
    call check('cplx orth rank: min(m, k+extra) columns', size(q, 2) == 20)
    call orth_sketch(a, 1e-6_dp, q, flag, d, extra_samples=0)
    call check('cplx orth extra_samples=0: legacy check accepts 42', &
         flag == 0 .and. size(q, 2) == 42)
    call orth_sketch(a, 1e-17_dp, q, flag, d)
    call check('cplx orth rtol < eps: flag 1, empty', &
         flag == 1 .and. size(q, 2) == 0 .and. size(d) == 0)
    call orth_sketch(a, 1e-6_dp, q, flag, d, block_size=300)
    call check('cplx orth block_size >= min(m,n): flag 1', flag == 1)
    call orth_sketch(gm, 1e-6_dp, q, flag, d)
    call check('cplx orth growth reaches min(m,n): flag 1', &
         flag == 1 .and. size(q, 2) == 0)

    ! ---------------- qr_sketch ----------------
    call qr_sketch(a, 1e-6_dp, q, r, p)
    k = size(q, 2)
    call check('cplx qr tol: shapes and permutation', size(r, 1) == k .and. &
         is_perm(p, 300) .and. k > 20 .and. k < 42)
    call check_le('cplx qr tol: ||A(:,p) - QR||/||A||', qr_err(a, q, r, p), 1e-5_dp)
    call check('cplx qr tol: R upper triangular', lower_max(r) <= 0)
    call check_le('cplx qr tol: orthonormality', orth_err(q), 1e-13_dp)
    call qr_sketch(a, 25.0_dp, q, r, p)
    call la_qrpiv(a, r2, p2, q2)
    err = qr_err(a, q2(:, 1:25), r2(1:25, :), p2)
    call check('cplx qr rank: k = 25', size(q, 2) == 25 .and. size(r, 1) == 25)
    call check_le('cplx qr rank: error / pivoted-QR error', &
         qr_err(a, q, r, p) / err, 4.0_dp)
    call qr_sketch(aw, 1e-6_dp, q, r, p)
    call check_le('cplx qr wide tol: error', qr_err(aw, q, r, p), 1e-5_dp)
    call check('cplx qr wide tol: R upper, perm', lower_max(r) <= 0 .and. &
         is_perm(p, 400))
    call qr_sketch(aw, 25.0_dp, q, r, p)
    call check_le('cplx qr wide rank: error', qr_err(aw, q, r, p), 1e-5_dp)
    call qr_sketch(gm, 1e-6_dp, q, r, p)
    call la_qrpiv(gm, r2, p2, q2)
    call check('cplx qr fallback = deterministic pivoted QR', &
         all(p == p2) .and. size(q, 2) == 80 .and. &
         maxval(abs(q - q2)) <= 0 .and. maxval(abs(r - r2)) <= 0)

    ! ---------------- svd_sketch ----------------
    call svd_sketch(a, 1e-6_dp, u, s, vh)
    k = size(s)
    call check('cplx svd tol: rank within 1 of count(s >= rtol*s1)', &
         abs(k - count(sf >= 1e-6_dp*sf(1))) <= 1)
    call check_le('cplx svd tol: singular values', &
         maxval(abs(s - sf(1:k))) / sf(1), 1e-10_dp)
    call check_le('cplx svd tol: reconstruction', svd_err(a, u, s, vh), 1e-5_dp)
    call check_le('cplx svd tol: U, Vh orthonormal', &
         max(orth_err(u), orth_err(conjg(transpose(vh)))), 1e-13_dp)
    call svd_sketch(a, 10.0_dp, u, s, vh)
    tail = sqrt(sum(sf(11:)**2)) / nrm
    call check('cplx svd rank: k = 10', size(s) == 10)
    call check_le('cplx svd rank: singular values', &
         maxval(abs(s - sf(1:10))) / sf(1), 1e-8_dp)
    call check_le('cplx svd rank: error / optimal', svd_err(a, u, s, vh) / tail, &
         4.0_dp)
    call svd_sketch(aw, 1e-6_dp, u, s, vh)
    k = size(s)
    call check('cplx svd wide tol: shapes', size(u, 1) == 300 .and. &
         size(vh, 2) == 400 .and. abs(k - count(sf >= 1e-6_dp*sf(1))) <= 1)
    call check_le('cplx svd wide tol: reconstruction', svd_err(aw, u, s, vh), &
         1e-5_dp)
    call check_le('cplx svd wide tol: U, Vh orthonormal', &
         max(orth_err(u), orth_err(conjg(transpose(vh)))), 1e-13_dp)
    call svd_sketch(aw, 10.0_dp, u, s, vh)
    call check_le('cplx svd wide rank: error / optimal', &
         svd_err(aw, u, s, vh) / tail, 4.0_dp)
    call svd_sketch(gm, 1e-6_dp, u, s, vh)
    call la_svd(gm, u2, s2, vh2, ierr)
    call check('cplx svd fallback = full LAPACK SVD', size(s) == 80 .and. &
         maxval(abs(s - s2)) <= 0 .and. maxval(abs(u - u2)) <= 0)

    ! ---------------- id_sketch / id_qrpiv ----------------
    do im = 1, 3
       tag = trim(methods(im))
       call id_sketch(a, 1e-6_dp, k, p, t, method=tag)
       call check('cplx id_sketch tol ' // tag // ': shapes', is_perm(p, 300) &
            .and. size(t, 1) == k .and. size(t, 2) == 300 - k .and. k > 20)
       call check_le('cplx id_sketch tol ' // tag // ': error', &
            id_err(a, k, p, t), 1e-5_dp)
       call id_sketch(a, 20.0_dp, k, p, t, method=tag)
       call check_le('cplx id_sketch rank ' // tag // ': error', &
            id_err(a, k, p, t), 1e-3_dp)
       call check('cplx id_sketch rank ' // tag // ': k = 20', k == 20)
       call id_qrpiv(a, 1e-6_dp, k, p, t, method=tag)
       call check_le('cplx id_qrpiv tol ' // tag // ': error', &
            id_err(a, k, p, t), 1e-5_dp)
       call id_qrpiv(a, 20.0_dp, k, p, t, method=tag)
       call check_le('cplx id_qrpiv rank ' // tag // ': error', &
            id_err(a, k, p, t), 1e-3_dp)
       call id_sketch(aw, 1e-6_dp, k, p, t, method=tag)
       call check_le('cplx id_sketch wide tol ' // tag // ': error', &
            id_err(aw, k, p, t), 1e-5_dp)
    end do

    allocate(lr(60, 40))
    call rla_uniform(g, bb)
    call rla_uniform(g, cc)
    lr = matmul(bb, cc)
    do im = 1, 3
       tag = trim(methods(im))
       call id_sketch(lr, 10.0_dp, k, p, t, method=tag)
       call check('cplx id_sketch rank-deficient ' // tag // ': k = 10, T finite', &
            k == 10 .and. all_finite_c(t))
       call check_le('cplx id_sketch rank-deficient ' // tag // ': error', &
            id_err(lr, k, p, t), 1e-12_dp)
       call id_qrpiv(lr, 10.0_dp, k, p, t, method=tag)
       call check('cplx id_qrpiv rank-deficient ' // tag // ': k = 10, T finite', &
            k == 10 .and. all_finite_c(t))
       call check_le('cplx id_qrpiv rank-deficient ' // tag // ': error', &
            id_err(lr, k, p, t), 1e-12_dp)
    end do

    allocate(z(30, 20))
    z = 0
    do im = 1, 3
       tag = trim(methods(im))
       call id_qrpiv(z, 1e-6_dp, k, p, t, method=tag)
       call check('cplx id_qrpiv zero matrix ' // tag // ': k = 0, T 0 x 20', &
            k == 0 .and. size(t, 1) == 0 .and. size(t, 2) == 20)
       call id_sketch(z, 1e-6_dp, k, p, t, method=tag)
       call check('cplx id_sketch zero matrix ' // tag // ': k = 0, T 0 x 20', &
            k == 0 .and. size(t, 1) == 0 .and. size(t, 2) == 20)
    end do

    do im = 1, 3
       tag = trim(methods(im))
       call id_qrpiv(gm(:, 1:20), 20.0_dp, k, p, t, method=tag)
       call check('cplx id_qrpiv k == n ' // tag // ': T 20 x 0', &
            k == 20 .and. size(t, 1) == 20 .and. size(t, 2) == 0)
       call id_sketch(gm(:, 1:20), 20.0_dp, k, p, t, method=tag)
       call check('cplx id_sketch k == n ' // tag // ': T 20 x 0', &
            k == 20 .and. size(t, 1) == 20 .and. size(t, 2) == 0)
    end do

    call id_sketch(a, 1e-6_dp, k, p, t, method='qr', info=info)
    call check('cplx id_sketch invalid method -> RLA_ERR_METHOD', &
         info == RLA_ERR_METHOD)
    call id_qrpiv(a, 1e-6_dp, k, p, t, method='bogus', info=info)
    call check('cplx id_qrpiv invalid method -> RLA_ERR_METHOD', &
         info == RLA_ERR_METHOD)

    ! ---------------- matrix-free ----------------
    op%m = 400; op%n = 300; op%a = a
    g1 = rla_rng(1, 2, 3, 5); g2 = g1
    call orth_sketch(a, 20.0_dp, q, flag, d, rng=g1)
    call orth_sketch(op, 20.0_dp, q2, flag, s2, rng=g2)
    ! trailing columns of Q amplify rounding by 1/diagR(j): compare diagR
    ! and the leading columns only
    call check_le('cplx operator orth rank = dense: diagR', &
         maxval(abs(d - s2)) / d(1), 1e-13_dp)
    call check_le('cplx operator orth rank = dense: Q(:, 1:10)', &
         maxval(abs(q(:, 1:10) - q2(:, 1:10))), 1e-11_dp)
    ! rtol 1.3e-6 lies between singular values 24 and 25; at 1e-6 = s(25)
    ! rounding decides the rank
    g1 = rla_rng(1, 2, 3, 5); g2 = g1
    call svd_sketch(a, 1.3e-6_dp, u, s, vh, rng=g1, power_iter=1)
    call svd_sketch(op, 1.3e-6_dp, u2, s2, vh2, rng=g2, power_iter=1)
    call check_le('cplx operator svd tol (power_iter=1) = dense', &
         maxval(abs(s - s2)) / s(1), 1e-12_dp)
    g1 = rla_rng(1, 2, 3, 5); g2 = g1
    call id_sketch(a, 1.3e-6_dp, k, p, t, method='lstsq', rng=g1)
    call id_sketch(op, 1.3e-6_dp, k2, p2, t2, method='lstsq', rng=g2)
    call check('cplx operator id_sketch lstsq = dense: k, piv', &
         k == k2 .and. all(p == p2))
    call check_le('cplx operator id_sketch lstsq = dense: T', &
         maxval(abs(t - t2)), 1e-9_dp)
    opw%m = 300; opw%n = 400; opw%a = aw
    g1 = rla_rng(1, 2, 3, 5); g2 = g1
    call svd_sketch(aw, 15.0_dp, u, s, vh, rng=g1)
    call svd_sketch(opw, 15.0_dp, u2, s2, vh2, rng=g2)
    call check_le('cplx operator svd wide rank = dense', &
         maxval(abs(s - s2)) / s(1), 1e-12_dp)
    call check_le('cplx operator svd wide rank: reconstruction', &
         svd_err(aw, u2, s2, vh2) / tail, 4.0_dp)
    opg%m = 100; opg%n = 80; opg%a = gm
    g1 = rla_rng(1, 2, 3, 5); g2 = g1
    call qr_sketch(gm, 1e-6_dp, q, r, p, rng=g1)
    ncols_applied = 0
    call qr_sketch(opg, 1e-6_dp, q2, r2, p2, rng=g2)
    call check('cplx operator qr fallback = dense (materialized)', &
         all(p == p2) .and. maxval(abs(q - q2)) <= 1e-12_dp .and. &
         maxval(abs(r - r2)) <= 1e-12_dp)
    call check('cplx operator qr fallback applied >= n columns', &
         ncols_applied >= 80)
    call id_qrpiv(opg, 30.0_dp, k, p, t, method='svd')
    call id_qrpiv(gm, 30.0_dp, k2, p2, t2, method='svd')
    call check('cplx operator id_qrpiv = dense', k == k2 .and. all(p == p2) &
         .and. maxval(abs(t - t2)) <= 1e-12_dp)
    allocate(lop%b(300, 8), lop%c(8, 200))
    call rla_uniform(g, lop%b)
    call rla_uniform(g, lop%c)
    lop%m = 300; lop%n = 200
    lr = matmul(lop%b, lop%c)
    ncols_applied = 0
    call svd_sketch(lop, 8.0_dp, u, s, vh)
    call check_le('cplx low-rank operator svd rank 8: reconstruction', &
         svd_err(lr, u, s, vh), 1e-12_dp)
    call check('cplx low-rank operator svd: no materialization', &
         ncols_applied < 200)
    call id_sketch(lop, 8.0_dp, k, p, t, method='svd')
    call check_le('cplx low-rank operator id_sketch rank 8: error', &
         id_err(lr, k, p, t), 1e-12_dp)
    call qr_sketch(lop, 8.0_dp, q, r, p)
    call check_le('cplx low-rank operator qr_sketch rank 8: error', &
         qr_err(lr, q, r, p), 1e-12_dp)

    ! ---------------- power iteration ----------------
    allocate(sl(150))
    sl = [(1.0_dp / real(j, dp), j = 1, 150)]
    lr = with_svals_c(g, 200, 150, sl)
    tail = sqrt(sum(sl(11:)**2)) / fro(lr)
    do i = 0, 2
       g1 = rla_rng(1, 2, 3, 5)
       call svd_sketch(lr, 10.0_dp, u, s, vh, power_iter=i, extra_samples=2, &
            rng=g1)
       e(i + 1) = svd_err(lr, u, s, vh) / tail
    end do
    call check_le('cplx power_iter=1 vs 0 (error ratio)', e(2) / e(1), 1.0_dp)
    call check_le('cplx power_iter=2 vs 1 (error ratio)', e(3) / e(2), 1.01_dp)

    ! ---------------- seeds ----------------
    g1 = rla_rng(1, 2, 3, 5); g2 = g1
    call svd_sketch(a, 10.0_dp, u, s, vh, rng=g1)
    call svd_sketch(a, 10.0_dp, u2, s2, vh2, rng=g2)
    call check('cplx same seed -> identical output', &
         maxval(abs(u - u2)) <= 0 .and. maxval(abs(s - s2)) <= 0)
    call check('cplx rng advances', any(g1%state() /= [1, 2, 3, 5]))
    call svd_sketch(a, 10.0_dp, u2, s2, vh2, rng=g1)
    call check('cplx advanced rng -> different sketch', maxval(abs(u - u2)) > 0)
    call svd_sketch(a, 10.0_dp, u, s, vh)
    call svd_sketch(a, 10.0_dp, u2, s2, vh2)
    call check('cplx default rng is deterministic', maxval(abs(u - u2)) <= 0)
  end subroutine test_complex

  ! ==================================================================
  subroutine crossport()
    real(dp), allocatable :: ar(:,:), q(:,:), d(:), t(:,:)
    complex(dp), allocatable :: ac(:,:), qc(:,:), u(:,:), vh(:,:)
    real(dp), allocatable :: s(:)
    integer, allocatable :: piv(:)
    type(rla_rng) :: g
    integer :: flag, k, i, seed(4)
    character(len=:), allocatable :: line

    write(*, '(a)') '--- crossport ---'
    call hilbert_r(50, 30, ar)
    g = rla_rng(1, 2, 3, 5)
    call orth_sketch(ar, 5.0_dp, q, flag, d, extra_samples=3, power_iter=0, rng=g)
    seed = g%state()
    line = 'CROSSPORT orth_real'
    do i = 1, size(d)
       line = line // ' ' // fmt_e(d(i))
    end do
    write(*, '(a)') line

    call hilbert_c(50, 30, ac)
    g = rla_rng(1, 2, 3, 5)
    call orth_sketch(ac, 5.0_dp, qc, flag, d, extra_samples=3, power_iter=0, rng=g)
    line = 'CROSSPORT orth_cplx'
    do i = 1, size(d)
       line = line // ' ' // fmt_e(d(i))
    end do
    write(*, '(a)') line

    line = 'CROSSPORT seed_after'
    do i = 1, 4
       line = line // ' ' // itoa(seed(i))
    end do
    write(*, '(a)') line

    call hilbert_r(300, 200, ar)
    g = rla_rng(1, 2, 3, 5)
    call id_sketch(ar, 1e-10_dp, k, piv, t, rng=g)
    line = 'CROSSPORT id_real ' // itoa(k)
    do i = 1, 10
       line = line // ' ' // itoa(piv(i) - 1)
    end do
    write(*, '(a)') line

    call hilbert_c(120, 80, ac)
    g = rla_rng(1, 2, 3, 5)
    call svd_sketch(ac, 10.0_dp, u, s, vh, power_iter=1, rng=g)
    line = 'CROSSPORT svd_cplx ' // itoa(size(s))
    do i = 1, 5
       line = line // ' ' // fmt_e(s(i))
    end do
    write(*, '(a)') line
  end subroutine crossport

  ! Ar(i, j) = 1 / (i + j + 1), 0-based
  subroutine hilbert_r(m, n, a)
    integer, intent(in) :: m, n
    real(dp), allocatable, intent(out) :: a(:,:)
    integer :: i, j
    allocate(a(m, n))
    do j = 1, n
       do i = 1, m
          a(i, j) = 1.0_dp / real(i + j - 1, dp)
       end do
    end do
  end subroutine hilbert_r

  ! Ac(i, j) = (1 + 1i*(i - j)/80) / (i + j + 1), 0-based
  subroutine hilbert_c(m, n, a)
    integer, intent(in) :: m, n
    complex(dp), allocatable, intent(out) :: a(:,:)
    integer :: i, j
    allocate(a(m, n))
    do j = 1, n
       do i = 1, m
          a(i, j) = cmplx(1.0_dp, real(i - j, dp) / 80, dp) / real(i + j - 1, dp)
       end do
    end do
  end subroutine hilbert_c

end program test_all
