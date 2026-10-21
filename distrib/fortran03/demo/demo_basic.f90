! demo_basic: librla on a dense Hilbert-like matrix and a matrix-free operator
!
! Author: Adrianna Gillman, Zydrunas Gimbutas
! SPDX-License-Identifier: MIT
! Assisted by: Claude Code (Anthropic)

module demo_ops
  use librla
  implicit none

  ! Matrix-free low-rank product A = B C, applied as B (C x)
  type, extends(rla_operator_r) :: product_op
     real(dp), allocatable :: b(:,:), c(:,:)
   contains
     procedure :: apply => product_apply
     procedure :: apply_adjoint => product_adjoint
  end type product_op

contains

  subroutine product_apply(self, x, y)
    class(product_op), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(out) :: y(:,:)
    y = matmul(self%b, matmul(self%c, x))
  end subroutine product_apply

  subroutine product_adjoint(self, x, y)
    class(product_op), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(out) :: y(:,:)
    y = matmul(transpose(self%c), matmul(transpose(self%b), x))
  end subroutine product_adjoint

end module demo_ops

program demo_basic
  use librla
  use demo_ops
  implicit none

  real(dp), allocatable :: a(:,:), u(:,:), s(:), vh(:,:), t(:,:), q(:,:), r(:,:)
  integer, allocatable :: piv(:)
  type(product_op) :: op
  type(rla_rng) :: rng
  integer :: i, j, k

  ! Dense: A(i, j) = 1 / (i + j - 1), 1000 x 500
  allocate(a(1000, 500))
  do j = 1, 500
     do i = 1, 1000
        a(i, j) = 1.0_dp / real(i + j - 1, dp)
     end do
  end do
  print '(a, a)', 'librla Fortran port ', rla_version

  call svd_sketch(a, 1e-10_dp, u, s, vh)
  print '(a, i0, a, es9.2)', 'svd_sketch  rtol=1e-10: k = ', size(s), &
       ', ||A - U S Vh||_F / ||A||_F = ', rel(a, matmul(u, spread(s, 2, &
       size(vh, 2)) * vh))

  call qr_sketch(a, 20.0_dp, q, r, piv, power_iter=1)
  print '(a, i0, a, es9.2)', 'qr_sketch   rank 20:    k = ', size(q, 2), &
       ', ||A(:,p) - Q R||_F / ||A||_F = ', rel(a(:, piv), matmul(q, r))

  call id_sketch(a, 1e-10_dp, k, piv, t, method='lstsq')
  print '(a, i0, a, es9.2)', 'id_sketch   rtol=1e-10: k = ', k, &
       ', ID error = ', rel(a(:, piv(k+1:)), matmul(a(:, piv(1:k)), t))

  ! Matrix-free: rank-12 product of 2000 x 12 and 12 x 1500 factors
  rng = rla_rng(2026)
  allocate(op%b(2000, 12), op%c(12, 1500))
  call rla_uniform(rng, op%b)
  call rla_uniform(rng, op%c)
  op%m = 2000; op%n = 1500
  call svd_sketch(op, 12.0_dp, u, s, vh, rng=rng)
  print '(a, i0, a, es9.2)', 'matrix-free svd_sketch rank 12: k = ', size(s), &
       ', error = ', rel(matmul(op%b, op%c), matmul(u, spread(s, 2, &
       size(vh, 2)) * vh))
  print '(a, 3es12.4)', '  leading singular values:', s(1:3)

contains

  real(dp) function rel(x, y)
    real(dp), intent(in) :: x(:,:), y(:,:)
    rel = sqrt(sum((x - y)**2) / sum(x**2))
  end function rel

end program demo_basic
