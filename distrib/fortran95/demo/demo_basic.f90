! demo_basic: librla95 on a dense Hilbert-like matrix and a matrix-free
! operator. Calls the library without `use librla95`, as Fortran 77 code
! would; adding `use librla95` would check every argument at compile time.
!
! Author: Adrianna Gillman, Zydrunas Gimbutas
! SPDX-License-Identifier: MIT
! Assisted by: Claude Code (Anthropic)

program demo_basic
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: m = 1000, n = 500, kc = 500
  external :: rla95_version, rla_dsvd_sketch, rla_dqr_sketch, rla_did_sketch
  external :: rla_dsvd_sketchf, dlarnv, product
  real(dp), allocatable :: a(:,:), u(:,:), s(:), vt(:,:), q(:,:), r(:,:), t(:,:)
  real(dp), allocatable :: bc(:), b(:,:), c(:,:)
  integer, allocatable :: piv(:)
  integer :: iseed(4), ipar(1), i, j, k, ier
  complex(dp) :: zdum(1)
  character(len=16) :: version

  ! Dense: A(i, j) = 1 / (i + j - 1), 1000 x 500
  allocate(a(m, n), u(m, kc), s(kc), vt(kc, n), q(m, kc), r(kc, n), &
       t(kc, n), piv(n))
  do j = 1, n
     do i = 1, m
        a(i, j) = 1.0_dp / real(i + j - 1, dp)
     end do
  end do
  call rla95_version(version)
  print '(a, a)', 'librla Fortran 95 port ', trim(version)

  ! Defaults: block size 42, no power iterations, 12 extra samples
  iseed = (/ 0, 0, 0, 1 /)
  call rla_dsvd_sketch(m, n, a, m, 1e-10_dp, 42, 0, 12, iseed, kc, k, u, m, &
       s, vt, kc, ier)
  print '(a, i0, a, es9.2)', 'svd_sketch  rtol=1e-10: k = ', k, &
       ', ||A - U S Vt||_F / ||A||_F = ', &
       rel(a, matmul(u(:, 1:k), spread(s(1:k), 2, n) * vt(1:k, :)))

  call rla_dqr_sketch(m, n, a, m, 20.0_dp, 42, 1, 12, iseed, kc, k, q, m, &
       r, kc, piv, ier)
  print '(a, i0, a, es9.2)', 'qr_sketch   rank 20:    k = ', k, &
       ', ||A(:,p) - Q R||_F / ||A||_F = ', &
       rel(a(:, piv), matmul(q(:, 1:k), r(1:k, :)))

  ! method 3 = lstsq
  call rla_did_sketch(m, n, a, m, 1e-10_dp, 42, 0, 12, 3, iseed, kc, k, piv, &
       t, kc, ier)
  print '(a, i0, a, es9.2)', 'id_sketch   rtol=1e-10: k = ', k, &
       ', ID error = ', &
       rel(a(:, piv(k+1:n)), matmul(a(:, piv(1:k)), t(1:k, 1:n-k)))

  ! Matrix-free: rank-12 product of 2000 x 12 and 12 x 1500 factors, stored
  ! back to back in rpar and applied by the external subroutine product
  deallocate(u, s, vt)
  allocate(bc(2000*12 + 12*1500), b(2000, 12), c(12, 1500))
  allocate(u(2000, 12), s(12), vt(12, 1500))
  iseed = (/ 0, 0, 0, 2027 /)
  call dlarnv(2, iseed, size(bc), bc)
  b = reshape(bc(1:2000*12), (/ 2000, 12 /))
  c = reshape(bc(2000*12+1:), (/ 12, 1500 /))
  ipar(1) = 12
  zdum = 0
  call rla_dsvd_sketchf(2000, 1500, product, ipar, bc, zdum, 12.0_dp, 42, 0, &
       12, iseed, 12, k, u, 2000, s, vt, 12, ier)
  print '(a, i0, a, es9.2)', 'matrix-free svd_sketch rank 12: k = ', k, &
       ', error = ', rel(matmul(b, c), matmul(u, spread(s, 2, 1500) * vt))
  print '(a, 3es12.4)', '  leading singular values:', s(1:3)

contains

  real(dp) function rel(x, y)
    real(dp), intent(in) :: x(:,:), y(:,:)
    rel = sqrt(sum((x - y)**2) / sum(x**2))
  end function rel

end program demo_basic

! A = B C with B (m x r) then C (r x n) stored in rpar, r = ipar(1):
! itrans = 0: y (m x k) = B (C x); itrans = 1: y (n x k) = C^T (B^T x)
subroutine product(itrans, m, n, k, x, y, ipar, rpar, zpar)
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, intent(in) :: itrans, m, n, k
  real(dp), intent(in) :: x(*)
  real(dp), intent(inout) :: y(*)
  integer :: ipar(*)
  real(dp) :: rpar(*)
  complex(dp) :: zpar(*)
  integer :: r
  r = ipar(1)
  if (itrans == 0) then
     call apply(m, n, r, rpar, rpar(m*r + 1), x, y)
  else
     call apply_t(m, n, r, rpar, rpar(m*r + 1), x, y)
  end if
  if (abs(zpar(1)) < 0) y(1) = 0   ! zpar is unused here
contains
  subroutine apply(m, n, r, b, c, x, y)
    integer, intent(in) :: m, n, r
    real(dp), intent(in) :: b(m, r), c(r, n), x(n, k)
    real(dp), intent(out) :: y(m, k)
    y = matmul(b, matmul(c, x))
  end subroutine apply
  subroutine apply_t(m, n, r, b, c, x, y)
    integer, intent(in) :: m, n, r
    real(dp), intent(in) :: b(m, r), c(r, n), x(m, k)
    real(dp), intent(out) :: y(n, k)
    y = matmul(transpose(c), matmul(transpose(b), x))
  end subroutine apply_t
end subroutine product
