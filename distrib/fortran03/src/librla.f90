! librla: randomized low-rank factorizations (Fortran 2003 port)
!
! Randomized algorithms for low-rank matrix approximations:
!   orth_sketch  - approximate orthonormal basis for the column space
!   qr_sketch    - truncated QR factorization with column pivoting
!   svd_sketch   - truncated singular value decomposition
!   id_sketch    - interpolative decomposition (ID)
! Deterministic:
!   id_qrpiv     - interpolative decomposition via QR with column pivoting
!
! Usage (tolerance mode: rtol < 1, rank mode: rtol >= 1, k = floor(rtol)):
!   call orth_sketch(a, rtol, q, flag, diagr)
!   call qr_sketch(a, rtol, q, r, piv)          ! a(:, piv) ~ q r
!   call svd_sketch(a, rtol, u, s, vh)          ! a ~ u diag(s) vh
!   call id_sketch(a, rtol, k, piv, t)          ! a(:, piv(k+1:)) ~ a(:, piv(:k)) t
!   call id_qrpiv(a, rtol, k, piv, t)
! a is a dense real(dp)/complex(dp) array or a class(rla_operator_r/_c)
! matrix-free operator. Optional keywords: block_size, power_iter,
! extra_samples, rng, method ('fast', 'svd', 'lstsq'), info.
!
! Author: Adrianna Gillman, Zydrunas Gimbutas
! SPDX-License-Identifier: MIT
! Assisted by: Claude Code (Anthropic)

module librla
  use, intrinsic :: iso_fortran_env, only: error_unit
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  use librla_lapack, only: dp, la_qrpiv_move, la_svd, la_lstsq, la_trsm_upper, &
       la_gemm, la_larnv
  implicit none
  private

  public :: dp, rla_version
  public :: rla_rng, rla_uniform
  public :: rla_operator_r, rla_operator_c, rla_get_matrix
  public :: orth_sketch, qr_sketch, svd_sketch, id_sketch, id_qrpiv
  public :: RLA_OK, RLA_ERR_METHOD, RLA_ERR_ARG, RLA_ERR_LAPACK, rla_errmsg

  !> Library version (the single place it is defined for this port).
  character(len=*), parameter :: rla_version = '0.1.0'

  !> Values returned in the optional info argument.
  integer, parameter :: RLA_OK = 0          ! success
  integer, parameter :: RLA_ERR_METHOD = 1  ! method not 'fast', 'svd' or 'lstsq'
  integer, parameter :: RLA_ERR_ARG = 2     ! invalid argument
  integer, parameter :: RLA_ERR_LAPACK = 3  ! SVD / least squares did not converge

  ! 64-bit integer kind for seeds (int64 itself is Fortran 2008)
  integer, parameter :: i8 = selected_int_kind(18)

  integer, parameter :: METHOD_FAST = 1, METHOD_SVD = 2, METHOD_LSTSQ = 3
  real(dp), parameter :: eps = epsilon(1.0_dp)
  integer, parameter :: materialize_block = 64

  ! ------------------------------------------------------------------
  ! Random number generator: the LAPACK larnv seed iseed(4)
  ! ------------------------------------------------------------------

  !> RNG state for the test matrices: LAPACK iseed(4), entries in
  !> [0, 4095], iseed(4) odd. A default-initialized rla_rng holds
  !> [0, 0, 0, 1], the state used when no rng argument is passed.
  type :: rla_rng
     private
     integer :: iseed(4) = [0, 0, 0, 1]
   contains
     procedure :: state => rng_state
  end type rla_rng

  !> rla_rng(i1, i2, i3, i4 [, info]) or rla_rng(seed) (default or 64-bit integer).
  interface rla_rng
     module procedure rng_from_ints, rng_from_seed, rng_from_seed64
  end interface rla_rng

  !> call rla_uniform(rng, x): fill a 2-D real/complex array with
  !> uniform(-1, 1) entries (complex: real and imaginary parts), the same
  !> draw librla uses for its test matrices.
  interface rla_uniform
     module procedure uniform_r, uniform_c
  end interface rla_uniform

  ! ------------------------------------------------------------------
  ! Operators
  ! ------------------------------------------------------------------

  !> Real m x n linear operator. Extend it and implement
  !>   apply(x, y):          y = A x     (x: n x k, y: m x k)
  !>   apply_adjoint(w, z):  z = A^T w   (w: m x k, z: n x k)
  !> Override dense() to return a pointer to an explicit matrix if there
  !> is one; the default (null) marks the operator matrix-free.
  type, abstract :: rla_operator_r
     integer :: m = 0, n = 0
   contains
     procedure(apply_r_iface), deferred :: apply
     procedure(apply_r_iface), deferred :: apply_adjoint
     procedure :: dense => no_dense_r
  end type rla_operator_r

  !> Complex m x n linear operator; apply_adjoint is z = A^H w.
  type, abstract :: rla_operator_c
     integer :: m = 0, n = 0
   contains
     procedure(apply_c_iface), deferred :: apply
     procedure(apply_c_iface), deferred :: apply_adjoint
     procedure :: dense => no_dense_c
  end type rla_operator_c

  abstract interface
     subroutine apply_r_iface(self, x, y)
       import :: rla_operator_r, dp
       class(rla_operator_r), intent(in) :: self
       real(dp), intent(in) :: x(:,:)
       real(dp), intent(out) :: y(:,:)
     end subroutine apply_r_iface

     subroutine apply_c_iface(self, x, y)
       import :: rla_operator_c, dp
       class(rla_operator_c), intent(in) :: self
       complex(dp), intent(in) :: x(:,:)
       complex(dp), intent(out) :: y(:,:)
     end subroutine apply_c_iface
  end interface

  ! Dense array viewed as an operator (how dense input enters the core)
  type, extends(rla_operator_r) :: dense_r
     real(dp), pointer :: a(:,:) => null()
   contains
     procedure :: apply => dense_apply_r
     procedure :: apply_adjoint => dense_adjoint_r
     procedure :: dense => dense_dense_r
  end type dense_r

  type, extends(rla_operator_c) :: dense_c
     complex(dp), pointer :: a(:,:) => null()
   contains
     procedure :: apply => dense_apply_c
     procedure :: apply_adjoint => dense_adjoint_c
     procedure :: dense => dense_dense_c
  end type dense_c

  ! Adjoint of another operator (svd_sketch of a wide matrix)
  type, extends(rla_operator_r) :: adjoint_r
     class(rla_operator_r), pointer :: base => null()
   contains
     procedure :: apply => adj_apply_r
     procedure :: apply_adjoint => adj_adjoint_r
  end type adjoint_r

  type, extends(rla_operator_c) :: adjoint_c
     class(rla_operator_c), pointer :: base => null()
   contains
     procedure :: apply => adj_apply_c
     procedure :: apply_adjoint => adj_adjoint_c
  end type adjoint_c

  type :: opts_t
     integer :: bs = 42, pit = 0, es = 12
  end type opts_t

  ! ------------------------------------------------------------------
  ! Public generic interfaces
  ! ------------------------------------------------------------------

  !> call orth_sketch(a, rtol, q, flag, diagr [, block_size, power_iter,
  !>                  extra_samples, rng, info])
  interface orth_sketch
     module procedure orth_dense_r, orth_dense_c, orth_op_r, orth_op_c
  end interface orth_sketch

  !> call qr_sketch(a, rtol, q, r, piv [, block_size, power_iter,
  !>                extra_samples, rng, info])
  interface qr_sketch
     module procedure qr_dense_r, qr_dense_c, qr_op_r, qr_op_c
  end interface qr_sketch

  !> call svd_sketch(a, rtol, u, s, vh [, block_size, power_iter,
  !>                 extra_samples, rng, info])
  interface svd_sketch
     module procedure svd_dense_r, svd_dense_c, svd_op_r, svd_op_c
  end interface svd_sketch

  !> call id_sketch(a, rtol, k, piv, t [, block_size, power_iter,
  !>                extra_samples, method, rng, info])
  interface id_sketch
     module procedure ids_dense_r, ids_dense_c, ids_op_r, ids_op_c
  end interface id_sketch

  !> call id_qrpiv(a, rtol, k, piv, t [, method, info])
  interface id_qrpiv
     module procedure idq_dense_r, idq_dense_c, idq_op_r, idq_op_c
  end interface id_qrpiv

  !> call rla_get_matrix(op, a): materialize an operator as a dense array
  !> (a copy of dense() if available, else one apply per block of columns).
  interface rla_get_matrix
     module procedure get_matrix_pub_r, get_matrix_pub_c
  end interface rla_get_matrix

contains

  ! ==================================================================
  ! RNG
  ! ==================================================================

  function rng_from_ints(i1, i2, i3, i4, info) result(rng)
    integer, intent(in) :: i1, i2, i3, i4
    integer, intent(out), optional :: info
    type(rla_rng) :: rng
    integer :: s(4)
    s = [i1, i2, i3, i4]
    if (any(s < 0) .or. any(s > 4095) .or. mod(i4, 2) == 0) then
       call raise(info, RLA_ERR_ARG, &
            'rla_rng: entries must be in [0, 4095] and the last one odd')
       return
    end if
    rng%iseed = s
    if (present(info)) info = RLA_OK
  end function rng_from_ints

  function rng_from_seed64(seed) result(rng)
    integer(i8), intent(in) :: seed
    type(rla_rng) :: rng
    integer(i8), parameter :: mask = 4095_i8
    rng%iseed(1) = int(iand(ishft(seed, -36), mask))
    rng%iseed(2) = int(iand(ishft(seed, -24), mask))
    rng%iseed(3) = int(iand(ishft(seed, -12), mask))
    rng%iseed(4) = int(ior(iand(seed, mask), 1_i8))
  end function rng_from_seed64

  function rng_from_seed(seed) result(rng)
    integer, intent(in) :: seed
    type(rla_rng) :: rng
    rng = rng_from_seed64(int(seed, i8))
  end function rng_from_seed

  !> Current iseed(4).
  function rng_state(self) result(iseed)
    class(rla_rng), intent(in) :: self
    integer :: iseed(4)
    iseed = self%iseed
  end function rng_state

  subroutine uniform_r(rng, x)
    type(rla_rng), intent(inout) :: rng
    real(dp), intent(out) :: x(:,:)
    call la_larnv(rng%iseed, x)
  end subroutine uniform_r

  subroutine uniform_c(rng, x)
    type(rla_rng), intent(inout) :: rng
    complex(dp), intent(out) :: x(:,:)
    call la_larnv(rng%iseed, x)
  end subroutine uniform_c

  ! ==================================================================
  ! Errors and option handling
  ! ==================================================================

  !> Text for an info code.
  function rla_errmsg(code) result(msg)
    integer, intent(in) :: code
    character(len=:), allocatable :: msg
    select case (code)
    case (RLA_OK)
       msg = 'success'
    case (RLA_ERR_METHOD)
       msg = "method must be one of 'fast', 'svd', 'lstsq'"
    case (RLA_ERR_ARG)
       msg = 'invalid argument'
    case (RLA_ERR_LAPACK)
       msg = 'LAPACK SVD or least squares did not converge'
    case default
       msg = 'unknown error'
    end select
  end function rla_errmsg

  ! Report an error through info, or stop when the caller passed no info.
  subroutine raise(info, code, msg)
    integer, intent(out), optional :: info
    integer, intent(in) :: code
    character(len=*), intent(in), optional :: msg
    if (present(info)) then
       info = code
       return
    end if
    if (present(msg)) then
       write(error_unit, '(a, a)') 'librla: ', msg
    else
       write(error_unit, '(a, a)') 'librla: ', rla_errmsg(code)
    end if
    stop 1
  end subroutine raise

  subroutine setup(m, n, rtol, block_size, power_iter, extra_samples, rng, &
       o, r, ierr)
    integer, intent(in) :: m, n
    real(dp), intent(in) :: rtol
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    type(rla_rng), intent(in), optional :: rng
    type(opts_t), intent(out) :: o
    type(rla_rng), intent(out) :: r
    integer, intent(out) :: ierr
    if (present(block_size)) o%bs = block_size
    if (present(power_iter)) o%pit = power_iter
    if (present(extra_samples)) o%es = extra_samples
    if (present(rng)) r = rng
    ierr = RLA_OK
    if (m < 0 .or. n < 0 .or. o%bs < 0 .or. o%pit < 0 .or. o%es < 0) &
         ierr = RLA_ERR_ARG
    if (ieee_is_nan(rtol)) then
       ierr = RLA_ERR_ARG
    else if (rtol >= real(huge(1) - max(o%es, 0), dp)) then
       ierr = RLA_ERR_ARG
    end if
  end subroutine setup

  subroutine parse_method(method, meth, ierr)
    character(len=*), intent(in), optional :: method
    integer, intent(out) :: meth, ierr
    ierr = RLA_OK
    meth = METHOD_FAST
    if (.not. present(method)) return
    select case (method)
    case ('fast')
       meth = METHOD_FAST
    case ('svd')
       meth = METHOD_SVD
    case ('lstsq')
       meth = METHOD_LSTSQ
    case default
       ierr = RLA_ERR_METHOD
    end select
  end subroutine parse_method

  pure integer function rank_from_diag(d, rtol) result(rank)
    real(dp), intent(in) :: d(:), rtol
    rank = 0
    if (size(d) == 0) return
    if (d(1) <= 0) return
    rank = count(d >= rtol*d(1))
  end function rank_from_diag

  pure integer function kmax_of(rtol) result(k)
    real(dp), intent(in) :: rtol
    k = int(rtol)   ! rtol >= 1: truncation is floor
  end function kmax_of

  ! ==================================================================
  ! Public wrappers: dense real
  ! ==================================================================

  subroutine orth_dense_r(a, rtol, q, flag, diagr, block_size, power_iter, &
       extra_samples, rng, info)
    real(dp), intent(in), target :: a(:,:)
    real(dp), intent(in) :: rtol
    real(dp), allocatable, intent(out) :: q(:,:), diagr(:)
    integer, intent(out) :: flag
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(dense_r) :: op
    op%m = size(a, 1); op%n = size(a, 2); op%a => a
    call orth_op_r(op, rtol, q, flag, diagr, block_size, power_iter, &
         extra_samples, rng, info)
  end subroutine orth_dense_r

  subroutine qr_dense_r(a, rtol, q, r, piv, block_size, power_iter, &
       extra_samples, rng, info)
    real(dp), intent(in), target :: a(:,:)
    real(dp), intent(in) :: rtol
    real(dp), allocatable, intent(out) :: q(:,:), r(:,:)
    integer, allocatable, intent(out) :: piv(:)
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(dense_r) :: op
    op%m = size(a, 1); op%n = size(a, 2); op%a => a
    call qr_op_r(op, rtol, q, r, piv, block_size, power_iter, &
         extra_samples, rng, info)
  end subroutine qr_dense_r

  subroutine svd_dense_r(a, rtol, u, s, vh, block_size, power_iter, &
       extra_samples, rng, info)
    real(dp), intent(in), target :: a(:,:)
    real(dp), intent(in) :: rtol
    real(dp), allocatable, intent(out) :: u(:,:), s(:), vh(:,:)
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(dense_r) :: op
    op%m = size(a, 1); op%n = size(a, 2); op%a => a
    call svd_op_r(op, rtol, u, s, vh, block_size, power_iter, &
         extra_samples, rng, info)
  end subroutine svd_dense_r

  subroutine ids_dense_r(a, rtol, k, piv, t, block_size, power_iter, &
       extra_samples, method, rng, info)
    real(dp), intent(in), target :: a(:,:)
    real(dp), intent(in) :: rtol
    integer, intent(out) :: k
    integer, allocatable, intent(out) :: piv(:)
    real(dp), allocatable, intent(out) :: t(:,:)
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    character(len=*), intent(in), optional :: method
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(dense_r) :: op
    op%m = size(a, 1); op%n = size(a, 2); op%a => a
    call ids_op_r(op, rtol, k, piv, t, block_size, power_iter, &
         extra_samples, method, rng, info)
  end subroutine ids_dense_r

  subroutine idq_dense_r(a, rtol, k, piv, t, method, info)
    real(dp), intent(in), target :: a(:,:)
    real(dp), intent(in) :: rtol
    integer, intent(out) :: k
    integer, allocatable, intent(out) :: piv(:)
    real(dp), allocatable, intent(out) :: t(:,:)
    character(len=*), intent(in), optional :: method
    integer, intent(out), optional :: info
    type(dense_r) :: op
    op%m = size(a, 1); op%n = size(a, 2); op%a => a
    call idq_op_r(op, rtol, k, piv, t, method, info)
  end subroutine idq_dense_r

  ! ==================================================================
  ! Public wrappers: operator real
  ! ==================================================================

  subroutine orth_op_r(a, rtol, q, flag, diagr, block_size, power_iter, &
       extra_samples, rng, info)
    class(rla_operator_r), intent(in), target :: a
    real(dp), intent(in) :: rtol
    real(dp), allocatable, intent(out) :: q(:,:), diagr(:)
    integer, intent(out) :: flag
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(opts_t) :: o
    type(rla_rng) :: r
    integer :: ierr
    call setup(a%m, a%n, rtol, block_size, power_iter, extra_samples, rng, &
         o, r, ierr)
    if (ierr /= RLA_OK) then
       flag = 1
       allocate(q(max(a%m, 0), 0), diagr(0))
       call raise(info, ierr)
       return
    end if
    call orth_core_r(a, rtol, o, r, q, flag, diagr)
    if (present(rng)) rng = r
    if (present(info)) info = RLA_OK
  end subroutine orth_op_r

  subroutine qr_op_r(a, rtol, q, r, piv, block_size, power_iter, &
       extra_samples, rng, info)
    class(rla_operator_r), intent(in), target :: a
    real(dp), intent(in) :: rtol
    real(dp), allocatable, intent(out) :: q(:,:), r(:,:)
    integer, allocatable, intent(out) :: piv(:)
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(opts_t) :: o
    type(rla_rng) :: g
    integer :: ierr
    call setup(a%m, a%n, rtol, block_size, power_iter, extra_samples, rng, &
         o, g, ierr)
    if (ierr /= RLA_OK) then
       allocate(q(max(a%m, 0), 0), r(0, max(a%n, 0)), piv(0))
       call raise(info, ierr)
       return
    end if
    call qr_core_r(a, rtol, o, g, q, r, piv)
    if (present(rng)) rng = g
    if (present(info)) info = RLA_OK
  end subroutine qr_op_r

  subroutine svd_op_r(a, rtol, u, s, vh, block_size, power_iter, &
       extra_samples, rng, info)
    class(rla_operator_r), intent(in), target :: a
    real(dp), intent(in) :: rtol
    real(dp), allocatable, intent(out) :: u(:,:), s(:), vh(:,:)
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(opts_t) :: o
    type(rla_rng) :: g
    integer :: ierr
    call setup(a%m, a%n, rtol, block_size, power_iter, extra_samples, rng, &
         o, g, ierr)
    if (ierr == RLA_OK) call svd_core_r(a, rtol, o, g, u, s, vh, ierr)
    if (ierr /= RLA_OK) then
       if (allocated(u)) deallocate(u, s, vh)
       allocate(u(max(a%m, 0), 0), s(0), vh(0, max(a%n, 0)))
       call raise(info, ierr)
       return
    end if
    if (present(rng)) rng = g
    if (present(info)) info = RLA_OK
  end subroutine svd_op_r

  subroutine ids_op_r(a, rtol, k, piv, t, block_size, power_iter, &
       extra_samples, method, rng, info)
    class(rla_operator_r), intent(in), target :: a
    real(dp), intent(in) :: rtol
    integer, intent(out) :: k
    integer, allocatable, intent(out) :: piv(:)
    real(dp), allocatable, intent(out) :: t(:,:)
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    character(len=*), intent(in), optional :: method
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(opts_t) :: o
    type(rla_rng) :: g
    integer :: ierr, meth
    k = 0
    call parse_method(method, meth, ierr)
    if (ierr == RLA_OK) call setup(a%m, a%n, rtol, block_size, power_iter, &
         extra_samples, rng, o, g, ierr)
    if (ierr == RLA_OK) call ids_core_r(a, rtol, o, meth, g, k, piv, t, ierr)
    if (ierr /= RLA_OK) then
       k = 0
       if (allocated(piv)) deallocate(piv)
       if (allocated(t)) deallocate(t)
       allocate(piv(0), t(0, 0))
       call raise(info, ierr)
       return
    end if
    if (present(rng)) rng = g
    if (present(info)) info = RLA_OK
  end subroutine ids_op_r

  subroutine idq_op_r(a, rtol, k, piv, t, method, info)
    class(rla_operator_r), intent(in), target :: a
    real(dp), intent(in) :: rtol
    integer, intent(out) :: k
    integer, allocatable, intent(out) :: piv(:)
    real(dp), allocatable, intent(out) :: t(:,:)
    character(len=*), intent(in), optional :: method
    integer, intent(out), optional :: info
    type(opts_t) :: o
    type(rla_rng) :: g
    integer :: ierr, meth
    k = 0
    call parse_method(method, meth, ierr)
    if (ierr == RLA_OK) call setup(a%m, a%n, rtol, o=o, r=g, ierr=ierr)
    if (ierr == RLA_OK) call idq_core_r(a, rtol, meth, k, piv, t, ierr)
    if (ierr /= RLA_OK) then
       k = 0
       if (allocated(piv)) deallocate(piv)
       if (allocated(t)) deallocate(t)
       allocate(piv(0), t(0, 0))
       call raise(info, ierr)
       return
    end if
    if (present(info)) info = RLA_OK
  end subroutine idq_op_r

  subroutine get_matrix_pub_r(op, a)
    class(rla_operator_r), intent(in) :: op
    real(dp), allocatable, intent(out) :: a(:,:)
    call get_matrix_r(op, a)
  end subroutine get_matrix_pub_r

  ! ==================================================================
  ! Operator plumbing: real
  ! ==================================================================

  function no_dense_r(self) result(p)
    class(rla_operator_r), intent(in) :: self
    real(dp), pointer :: p(:,:)
    p => null()
    if (self%m < 0) return   ! self is unused: matrix-free by default
  end function no_dense_r

  function dense_dense_r(self) result(p)
    class(dense_r), intent(in) :: self
    real(dp), pointer :: p(:,:)
    p => self%a
  end function dense_dense_r

  subroutine dense_apply_r(self, x, y)
    class(dense_r), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(out) :: y(:,:)
    call la_gemm('N', 'N', self%a, x, y)
  end subroutine dense_apply_r

  subroutine dense_adjoint_r(self, x, y)
    class(dense_r), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(out) :: y(:,:)
    call la_gemm('T', 'N', self%a, x, y)
  end subroutine dense_adjoint_r

  subroutine adj_apply_r(self, x, y)
    class(adjoint_r), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(out) :: y(:,:)
    call self%base%apply_adjoint(x, y)
  end subroutine adj_apply_r

  subroutine adj_adjoint_r(self, x, y)
    class(adjoint_r), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(out) :: y(:,:)
    call self%base%apply(x, y)
  end subroutine adj_adjoint_r

  ! y = A x (allocates y)
  subroutine matvec_r(a, x, y)
    class(rla_operator_r), intent(in) :: a
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable, intent(out) :: y(:,:)
    allocate(y(a%m, size(x, 2)))
    if (size(y) == 0) return
    if (a%n == 0) then
       y = 0
    else
       call a%apply(x, y)
    end if
  end subroutine matvec_r

  ! z = A^T w (allocates z)
  subroutine rmatvec_r(a, w, z)
    class(rla_operator_r), intent(in) :: a
    real(dp), intent(in) :: w(:,:)
    real(dp), allocatable, intent(out) :: z(:,:)
    allocate(z(a%n, size(w, 2)))
    if (size(z) == 0) return
    if (a%m == 0) then
       z = 0
    else
       call a%apply_adjoint(w, z)
    end if
  end subroutine rmatvec_r

  ! b = Q^T A; for an operator without dense(), (A^T Q)^T
  subroutine qh_times_a_r(q, a, b)
    real(dp), intent(in) :: q(:,:)
    class(rla_operator_r), intent(in) :: a
    real(dp), allocatable, intent(out) :: b(:,:)
    real(dp), pointer :: p(:,:)
    real(dp), allocatable :: z(:,:)
    p => a%dense()
    if (associated(p)) then
       allocate(b(size(q, 2), a%n))
       call la_gemm('T', 'N', q, p, b)
    else
       call rmatvec_r(a, q, z)
       b = transpose(z)
    end if
  end subroutine qh_times_a_r

  ! Dense copy of A: dense() if available, the transpose of the base of an
  ! adjoint, otherwise A applied to blocks of identity columns.
  recursive subroutine get_matrix_r(a, am)
    class(rla_operator_r), intent(in) :: a
    real(dp), allocatable, intent(out) :: am(:,:)
    real(dp), pointer :: p(:,:)
    real(dp), allocatable :: e(:,:), bm(:,:)
    integer :: j0, j1, j
    p => a%dense()
    if (associated(p)) then
       am = p
       return
    end if
    select type (a)
    type is (adjoint_r)
       call get_matrix_r(a%base, bm)
       am = transpose(bm)
       return
    end select
    allocate(am(a%m, a%n))
    if (size(am) == 0) return
    do j0 = 1, a%n, materialize_block
       j1 = min(j0 + materialize_block - 1, a%n)
       allocate(e(a%n, j1 - j0 + 1))
       e = 0
       do j = j0, j1
          e(j, j - j0 + 1) = 1
       end do
       call a%apply(e, am(:, j0:j1))
       deallocate(e)
    end do
  end subroutine get_matrix_r

  ! ==================================================================
  ! Algorithms: real
  ! ==================================================================

  ! y = A x with x = Omega (n x l) after power_iter power iterations
  subroutine sample_r(a, l, pit, rng, y)
    class(rla_operator_r), intent(in) :: a
    integer, intent(in) :: l, pit
    type(rla_rng), intent(inout) :: rng
    real(dp), allocatable, intent(out) :: y(:,:)
    real(dp), allocatable :: x(:,:), z(:,:), r(:,:)
    integer, allocatable :: jp(:)
    integer :: it
    allocate(x(a%n, l))
    call la_larnv(rng%iseed, x)
    do it = 1, pit
       call matvec_r(a, x, y)
       call rmatvec_r(a, y, z)
       call la_qrpiv_move(z, r, jp, x)
    end do
    call matvec_r(a, x, y)
  end subroutine sample_r

  subroutine orth_core_r(a, rtol, o, rng, q, flag, diagr)
    class(rla_operator_r), intent(in) :: a
    real(dp), intent(in) :: rtol
    type(opts_t), intent(in) :: o
    type(rla_rng), intent(inout) :: rng
    real(dp), allocatable, intent(out) :: q(:,:), diagr(:)
    integer, intent(out) :: flag
    real(dp), allocatable :: y(:,:), r(:,:)
    integer, allocatable :: jp(:)
    integer :: mn, l, nd, i

    mn = min(a%m, a%n)
    flag = 0

    ! Rank mode: one sketch of floor(rtol) + extra_samples columns
    if (rtol >= 1) then
       call sample_r(a, kmax_of(rtol) + o%es, o%pit, rng, y)
       call la_qrpiv_move(y, r, jp, q)
       diagr = [(abs(r(i, i)), i = 1, size(r, 1))]
       return
    end if

    ! Tolerance mode: fresh sketch per pass, block size grows 4x
    if (rtol < eps .or. o%bs >= mn) then
       flag = 1
       allocate(q(a%m, 0), diagr(0))
       return
    end if
    l = o%bs
    do
       call sample_r(a, l, o%pit, rng, y)
       call la_qrpiv_move(y, r, jp, q)
       diagr = [(abs(r(i, i)), i = 1, size(r, 1))]
       nd = size(diagr)
       if (nd == 0) return
       if (nd > o%es) then
          if (diagr(nd - o%es) <= rtol*diagr(1)) return
       end if
       l = min(4*l, mn)
       if (l >= mn) then
          flag = 1
          deallocate(q, diagr)
          allocate(q(a%m, 0), diagr(0))
          return
       end if
    end do
  end subroutine orth_core_r

  subroutine qr_core_r(a, rtol, o, rng, q, r, piv)
    class(rla_operator_r), intent(in) :: a
    real(dp), intent(in) :: rtol
    type(opts_t), intent(in) :: o
    type(rla_rng), intent(inout) :: rng
    real(dp), allocatable, intent(out) :: q(:,:), r(:,:)
    integer, allocatable, intent(out) :: piv(:)
    real(dp), allocatable :: qs(:,:), d(:), b(:,:), qp(:,:), rp(:,:), am(:,:)
    logical :: rank_mode, fallback
    integer :: mn, flag, k, rank, i

    mn = min(a%m, a%n)
    rank_mode = rtol >= 1
    call orth_core_r(a, rtol, o, rng, qs, flag, d)
    k = mn
    if (flag == 0) k = size(qs, 2)
    fallback = (flag /= 0 .or. k >= mn) .and. .not. rank_mode

    if (fallback) then
       call get_matrix_r(a, am)
       call la_qrpiv_move(am, rp, piv, qp)
       if (rank_mode) then
          rank = min(kmax_of(rtol), size(qp, 2))
       else
          rank = rank_from_diag([(abs(rp(i, i)), i = 1, size(rp, 1))], rtol)
       end if
       q = qp(:, 1:rank)
       r = rp(1:rank, :)
       return
    end if

    call qh_times_a_r(qs, a, b)
    call la_qrpiv_move(b, rp, piv, qp)
    if (rank_mode) then
       rank = min(kmax_of(rtol), size(qp, 2))
    else
       rank = rank_from_diag([(abs(rp(i, i)), i = 1, size(rp, 1))], rtol)
    end if
    allocate(q(a%m, rank))
    call la_gemm('N', 'N', qs, qp(:, 1:rank), q)
    r = rp(1:rank, :)
  end subroutine qr_core_r

  recursive subroutine svd_core_r(a, rtol, o, rng, u, s, vh, ierr)
    class(rla_operator_r), intent(in), target :: a
    real(dp), intent(in) :: rtol
    type(opts_t), intent(in) :: o
    type(rla_rng), intent(inout) :: rng
    real(dp), allocatable, intent(out) :: u(:,:), s(:), vh(:,:)
    integer, intent(out) :: ierr
    type(adjoint_r) :: adj
    real(dp), allocatable :: qs(:,:), d(:), b(:,:), up(:,:), sp(:), vp(:,:)
    logical :: rank_mode, fallback
    integer :: mn, flag, k, rank

    ! Wide matrix: factor the adjoint and swap the factors
    if (a%m < a%n) then
       adj%m = a%n; adj%n = a%m; adj%base => a
       call svd_core_r(adj, rtol, o, rng, up, s, vp, ierr)
       if (ierr /= RLA_OK) return
       u = transpose(vp)
       vh = transpose(up)
       return
    end if

    mn = min(a%m, a%n)
    rank_mode = rtol >= 1
    call orth_core_r(a, rtol, o, rng, qs, flag, d)
    k = mn
    if (flag == 0) k = size(qs, 2)
    fallback = (flag /= 0 .or. k >= mn) .and. .not. rank_mode

    if (fallback) then
       call get_matrix_r(a, b)
       call la_svd(b, up, sp, vp, ierr)
       if (ierr /= 0) then
          ierr = RLA_ERR_LAPACK
          return
       end if
       if (rank_mode) then
          rank = min(kmax_of(rtol), size(sp))
       else
          rank = rank_from_diag(sp, rtol)
       end if
       u = up(:, 1:rank)
       s = sp(1:rank)
       vh = vp(1:rank, :)
       return
    end if

    call qh_times_a_r(qs, a, b)
    call la_svd(b, up, sp, vp, ierr)
    if (ierr /= 0) then
       ierr = RLA_ERR_LAPACK
       return
    end if
    if (rank_mode) then
       rank = min(kmax_of(rtol), size(sp))
    else
       rank = rank_from_diag(sp, rtol)
    end if
    allocate(u(a%m, rank))
    call la_gemm('N', 'N', qs, up(:, 1:rank), u)
    s = sp(1:rank)
    vh = vp(1:rank, :)
  end subroutine svd_core_r

  subroutine ids_core_r(a, rtol, o, meth, rng, k, piv, t, ierr)
    class(rla_operator_r), intent(in) :: a
    real(dp), intent(in) :: rtol
    type(opts_t), intent(in) :: o
    integer, intent(in) :: meth
    type(rla_rng), intent(inout) :: rng
    integer, intent(out) :: k, ierr
    integer, allocatable, intent(out) :: piv(:)
    real(dp), allocatable, intent(out) :: t(:,:)
    real(dp), allocatable :: q(:,:), r(:,:)
    real(dp) :: rts

    call qr_core_r(a, rtol, o, rng, q, r, piv)
    k = size(r, 1)
    rts = rtol
    if (rtol >= 1) rts = 0
    call compute_t_r(a, r, piv, k, meth, rts, t, ierr)
  end subroutine ids_core_r

  subroutine idq_core_r(a, rtol, meth, k, piv, t, ierr)
    class(rla_operator_r), intent(in) :: a
    real(dp), intent(in) :: rtol
    integer, intent(in) :: meth
    integer, intent(out) :: k, ierr
    integer, allocatable, intent(out) :: piv(:)
    real(dp), allocatable, intent(out) :: t(:,:)
    real(dp), allocatable :: am(:,:), w(:,:), r(:,:)
    real(dp) :: rts
    integer :: n, i

    n = a%n
    ierr = RLA_OK
    call get_matrix_r(a, am)
    ! The QR overwrites its input: keep A only when lstsq needs it
    if (meth == METHOD_LSTSQ) then
       w = am
       call la_qrpiv_move(w, r, piv)
    else
       call la_qrpiv_move(am, r, piv)
    end if
    if (rtol >= 1) then
       k = min(kmax_of(rtol), min(a%m, n))
       rts = 0
    else
       k = rank_from_diag([(abs(r(i, i)), i = 1, size(r, 1))], rtol)
       rts = rtol
    end if
    if (k == 0) then
       allocate(t(0, n))
       return
    end if
    if (k == n) then
       allocate(t(k, 0))
       return
    end if
    if (meth == METHOD_LSTSQ) then
       call t_lstsq_r(am, piv, k, t, ierr)
    else
       call compute_t_r(a, r, piv, k, meth, rts, t, ierr)
    end if
  end subroutine idq_core_r

  subroutine compute_t_r(a, r, piv, k, meth, rts, t, ierr)
    class(rla_operator_r), intent(in) :: a
    real(dp), intent(in) :: r(:,:), rts
    integer, intent(in) :: piv(:), k, meth
    real(dp), allocatable, intent(out) :: t(:,:)
    integer, intent(out) :: ierr
    real(dp), allocatable :: am(:,:)
    ierr = RLA_OK
    select case (meth)
    case (METHOD_LSTSQ)
       if (k == 0 .or. k >= a%n) then
          allocate(t(k, a%n - k))
          t = 0
       else
          call get_matrix_r(a, am)
          call t_lstsq_r(am, piv, k, t, ierr)
       end if
    case (METHOD_SVD)
       call t_svd_r(r, k, rts, t, ierr)
    case default
       call t_fast_r(r, k, rts, t, ierr)
    end select
  end subroutine compute_t_r

  subroutine t_lstsq_r(am, piv, k, t, ierr)
    real(dp), intent(in) :: am(:,:)
    integer, intent(in) :: piv(:), k
    real(dp), allocatable, intent(out) :: t(:,:)
    integer, intent(out) :: ierr
    call la_lstsq(am(:, piv(1:k)), am(:, piv(k+1:)), t, ierr)
    if (ierr /= 0) ierr = RLA_ERR_LAPACK
  end subroutine t_lstsq_r

  ! T = pinv(R11) R12, dropping singular values below max(rts, eps) * smax
  subroutine t_svd_r(r, k, rts, t, ierr)
    real(dp), intent(in) :: r(:,:), rts
    integer, intent(in) :: k
    real(dp), allocatable, intent(out) :: t(:,:)
    integer, intent(out) :: ierr
    real(dp), allocatable :: u(:,:), s(:), vh(:,:), w(:,:)
    integer, allocatable :: keep(:)
    real(dp) :: smax
    integer :: n, i

    n = size(r, 2)
    ierr = RLA_OK
    allocate(t(k, n - k))
    t = 0
    if (k == 0) return
    call la_svd(r(1:k, 1:k), u, s, vh, ierr)
    if (ierr /= 0) then
       ierr = RLA_ERR_LAPACK
       return
    end if
    smax = maxval(s)
    if (smax <= 0) return
    keep = pack([(i, i = 1, k)], s >= max(rts, eps)*smax)
    if (size(keep) == 0) return
    allocate(w(size(keep), n - k))
    call la_gemm('T', 'N', u(:, keep), r(1:k, k+1:n), w)
    do i = 1, size(keep)
       w(i, :) = w(i, :) / s(keep(i))
    end do
    call la_gemm('T', 'N', vh(keep, :), w, t)
  end subroutine t_svd_r

  ! T = R11 \ R12 by triangular solve, or t_svd_r if R11 is near-singular
  subroutine t_fast_r(r, k, rts, t, ierr)
    real(dp), intent(in) :: r(:,:), rts
    integer, intent(in) :: k
    real(dp), allocatable, intent(out) :: t(:,:)
    integer, intent(out) :: ierr
    real(dp), allocatable :: d(:)
    integer :: n, i

    n = size(r, 2)
    ierr = RLA_OK
    if (k == 0) then
       allocate(t(0, n))
       return
    end if
    d = [(abs(r(i, i)), i = 1, k)]
    if (minval(d) <= eps*maxval(d)) then
       call t_svd_r(r, k, rts, t, ierr)
       return
    end if
    t = r(1:k, k+1:n)
    call la_trsm_upper(r(1:k, 1:k), t)
  end subroutine t_fast_r

  ! ==================================================================
  ! Public wrappers: dense complex
  ! ==================================================================

  subroutine orth_dense_c(a, rtol, q, flag, diagr, block_size, power_iter, &
       extra_samples, rng, info)
    complex(dp), intent(in), target :: a(:,:)
    real(dp), intent(in) :: rtol
    complex(dp), allocatable, intent(out) :: q(:,:)
    real(dp), allocatable, intent(out) :: diagr(:)
    integer, intent(out) :: flag
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(dense_c) :: op
    op%m = size(a, 1); op%n = size(a, 2); op%a => a
    call orth_op_c(op, rtol, q, flag, diagr, block_size, power_iter, &
         extra_samples, rng, info)
  end subroutine orth_dense_c

  subroutine qr_dense_c(a, rtol, q, r, piv, block_size, power_iter, &
       extra_samples, rng, info)
    complex(dp), intent(in), target :: a(:,:)
    real(dp), intent(in) :: rtol
    complex(dp), allocatable, intent(out) :: q(:,:), r(:,:)
    integer, allocatable, intent(out) :: piv(:)
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(dense_c) :: op
    op%m = size(a, 1); op%n = size(a, 2); op%a => a
    call qr_op_c(op, rtol, q, r, piv, block_size, power_iter, &
         extra_samples, rng, info)
  end subroutine qr_dense_c

  subroutine svd_dense_c(a, rtol, u, s, vh, block_size, power_iter, &
       extra_samples, rng, info)
    complex(dp), intent(in), target :: a(:,:)
    real(dp), intent(in) :: rtol
    complex(dp), allocatable, intent(out) :: u(:,:), vh(:,:)
    real(dp), allocatable, intent(out) :: s(:)
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(dense_c) :: op
    op%m = size(a, 1); op%n = size(a, 2); op%a => a
    call svd_op_c(op, rtol, u, s, vh, block_size, power_iter, &
         extra_samples, rng, info)
  end subroutine svd_dense_c

  subroutine ids_dense_c(a, rtol, k, piv, t, block_size, power_iter, &
       extra_samples, method, rng, info)
    complex(dp), intent(in), target :: a(:,:)
    real(dp), intent(in) :: rtol
    integer, intent(out) :: k
    integer, allocatable, intent(out) :: piv(:)
    complex(dp), allocatable, intent(out) :: t(:,:)
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    character(len=*), intent(in), optional :: method
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(dense_c) :: op
    op%m = size(a, 1); op%n = size(a, 2); op%a => a
    call ids_op_c(op, rtol, k, piv, t, block_size, power_iter, &
         extra_samples, method, rng, info)
  end subroutine ids_dense_c

  subroutine idq_dense_c(a, rtol, k, piv, t, method, info)
    complex(dp), intent(in), target :: a(:,:)
    real(dp), intent(in) :: rtol
    integer, intent(out) :: k
    integer, allocatable, intent(out) :: piv(:)
    complex(dp), allocatable, intent(out) :: t(:,:)
    character(len=*), intent(in), optional :: method
    integer, intent(out), optional :: info
    type(dense_c) :: op
    op%m = size(a, 1); op%n = size(a, 2); op%a => a
    call idq_op_c(op, rtol, k, piv, t, method, info)
  end subroutine idq_dense_c

  ! ==================================================================
  ! Public wrappers: operator complex
  ! ==================================================================

  subroutine orth_op_c(a, rtol, q, flag, diagr, block_size, power_iter, &
       extra_samples, rng, info)
    class(rla_operator_c), intent(in), target :: a
    real(dp), intent(in) :: rtol
    complex(dp), allocatable, intent(out) :: q(:,:)
    real(dp), allocatable, intent(out) :: diagr(:)
    integer, intent(out) :: flag
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(opts_t) :: o
    type(rla_rng) :: r
    integer :: ierr
    call setup(a%m, a%n, rtol, block_size, power_iter, extra_samples, rng, &
         o, r, ierr)
    if (ierr /= RLA_OK) then
       flag = 1
       allocate(q(max(a%m, 0), 0), diagr(0))
       call raise(info, ierr)
       return
    end if
    call orth_core_c(a, rtol, o, r, q, flag, diagr)
    if (present(rng)) rng = r
    if (present(info)) info = RLA_OK
  end subroutine orth_op_c

  subroutine qr_op_c(a, rtol, q, r, piv, block_size, power_iter, &
       extra_samples, rng, info)
    class(rla_operator_c), intent(in), target :: a
    real(dp), intent(in) :: rtol
    complex(dp), allocatable, intent(out) :: q(:,:), r(:,:)
    integer, allocatable, intent(out) :: piv(:)
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(opts_t) :: o
    type(rla_rng) :: g
    integer :: ierr
    call setup(a%m, a%n, rtol, block_size, power_iter, extra_samples, rng, &
         o, g, ierr)
    if (ierr /= RLA_OK) then
       allocate(q(max(a%m, 0), 0), r(0, max(a%n, 0)), piv(0))
       call raise(info, ierr)
       return
    end if
    call qr_core_c(a, rtol, o, g, q, r, piv)
    if (present(rng)) rng = g
    if (present(info)) info = RLA_OK
  end subroutine qr_op_c

  subroutine svd_op_c(a, rtol, u, s, vh, block_size, power_iter, &
       extra_samples, rng, info)
    class(rla_operator_c), intent(in), target :: a
    real(dp), intent(in) :: rtol
    complex(dp), allocatable, intent(out) :: u(:,:), vh(:,:)
    real(dp), allocatable, intent(out) :: s(:)
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(opts_t) :: o
    type(rla_rng) :: g
    integer :: ierr
    call setup(a%m, a%n, rtol, block_size, power_iter, extra_samples, rng, &
         o, g, ierr)
    if (ierr == RLA_OK) call svd_core_c(a, rtol, o, g, u, s, vh, ierr)
    if (ierr /= RLA_OK) then
       if (allocated(u)) deallocate(u, s, vh)
       allocate(u(max(a%m, 0), 0), s(0), vh(0, max(a%n, 0)))
       call raise(info, ierr)
       return
    end if
    if (present(rng)) rng = g
    if (present(info)) info = RLA_OK
  end subroutine svd_op_c

  subroutine ids_op_c(a, rtol, k, piv, t, block_size, power_iter, &
       extra_samples, method, rng, info)
    class(rla_operator_c), intent(in), target :: a
    real(dp), intent(in) :: rtol
    integer, intent(out) :: k
    integer, allocatable, intent(out) :: piv(:)
    complex(dp), allocatable, intent(out) :: t(:,:)
    integer, intent(in), optional :: block_size, power_iter, extra_samples
    character(len=*), intent(in), optional :: method
    type(rla_rng), intent(inout), optional :: rng
    integer, intent(out), optional :: info
    type(opts_t) :: o
    type(rla_rng) :: g
    integer :: ierr, meth
    k = 0
    call parse_method(method, meth, ierr)
    if (ierr == RLA_OK) call setup(a%m, a%n, rtol, block_size, power_iter, &
         extra_samples, rng, o, g, ierr)
    if (ierr == RLA_OK) call ids_core_c(a, rtol, o, meth, g, k, piv, t, ierr)
    if (ierr /= RLA_OK) then
       k = 0
       if (allocated(piv)) deallocate(piv)
       if (allocated(t)) deallocate(t)
       allocate(piv(0), t(0, 0))
       call raise(info, ierr)
       return
    end if
    if (present(rng)) rng = g
    if (present(info)) info = RLA_OK
  end subroutine ids_op_c

  subroutine idq_op_c(a, rtol, k, piv, t, method, info)
    class(rla_operator_c), intent(in), target :: a
    real(dp), intent(in) :: rtol
    integer, intent(out) :: k
    integer, allocatable, intent(out) :: piv(:)
    complex(dp), allocatable, intent(out) :: t(:,:)
    character(len=*), intent(in), optional :: method
    integer, intent(out), optional :: info
    type(opts_t) :: o
    type(rla_rng) :: g
    integer :: ierr, meth
    k = 0
    call parse_method(method, meth, ierr)
    if (ierr == RLA_OK) call setup(a%m, a%n, rtol, o=o, r=g, ierr=ierr)
    if (ierr == RLA_OK) call idq_core_c(a, rtol, meth, k, piv, t, ierr)
    if (ierr /= RLA_OK) then
       k = 0
       if (allocated(piv)) deallocate(piv)
       if (allocated(t)) deallocate(t)
       allocate(piv(0), t(0, 0))
       call raise(info, ierr)
       return
    end if
    if (present(info)) info = RLA_OK
  end subroutine idq_op_c

  subroutine get_matrix_pub_c(op, a)
    class(rla_operator_c), intent(in) :: op
    complex(dp), allocatable, intent(out) :: a(:,:)
    call get_matrix_c(op, a)
  end subroutine get_matrix_pub_c

  ! ==================================================================
  ! Operator plumbing: complex
  ! ==================================================================

  function no_dense_c(self) result(p)
    class(rla_operator_c), intent(in) :: self
    complex(dp), pointer :: p(:,:)
    p => null()
    if (self%m < 0) return   ! self is unused: matrix-free by default
  end function no_dense_c

  function dense_dense_c(self) result(p)
    class(dense_c), intent(in) :: self
    complex(dp), pointer :: p(:,:)
    p => self%a
  end function dense_dense_c

  subroutine dense_apply_c(self, x, y)
    class(dense_c), intent(in) :: self
    complex(dp), intent(in) :: x(:,:)
    complex(dp), intent(out) :: y(:,:)
    call la_gemm('N', 'N', self%a, x, y)
  end subroutine dense_apply_c

  subroutine dense_adjoint_c(self, x, y)
    class(dense_c), intent(in) :: self
    complex(dp), intent(in) :: x(:,:)
    complex(dp), intent(out) :: y(:,:)
    call la_gemm('C', 'N', self%a, x, y)
  end subroutine dense_adjoint_c

  subroutine adj_apply_c(self, x, y)
    class(adjoint_c), intent(in) :: self
    complex(dp), intent(in) :: x(:,:)
    complex(dp), intent(out) :: y(:,:)
    call self%base%apply_adjoint(x, y)
  end subroutine adj_apply_c

  subroutine adj_adjoint_c(self, x, y)
    class(adjoint_c), intent(in) :: self
    complex(dp), intent(in) :: x(:,:)
    complex(dp), intent(out) :: y(:,:)
    call self%base%apply(x, y)
  end subroutine adj_adjoint_c

  subroutine matvec_c(a, x, y)
    class(rla_operator_c), intent(in) :: a
    complex(dp), intent(in) :: x(:,:)
    complex(dp), allocatable, intent(out) :: y(:,:)
    allocate(y(a%m, size(x, 2)))
    if (size(y) == 0) return
    if (a%n == 0) then
       y = 0
    else
       call a%apply(x, y)
    end if
  end subroutine matvec_c

  subroutine rmatvec_c(a, w, z)
    class(rla_operator_c), intent(in) :: a
    complex(dp), intent(in) :: w(:,:)
    complex(dp), allocatable, intent(out) :: z(:,:)
    allocate(z(a%n, size(w, 2)))
    if (size(z) == 0) return
    if (a%m == 0) then
       z = 0
    else
       call a%apply_adjoint(w, z)
    end if
  end subroutine rmatvec_c

  subroutine qh_times_a_c(q, a, b)
    complex(dp), intent(in) :: q(:,:)
    class(rla_operator_c), intent(in) :: a
    complex(dp), allocatable, intent(out) :: b(:,:)
    complex(dp), pointer :: p(:,:)
    complex(dp), allocatable :: z(:,:)
    p => a%dense()
    if (associated(p)) then
       allocate(b(size(q, 2), a%n))
       call la_gemm('C', 'N', q, p, b)
    else
       call rmatvec_c(a, q, z)
       b = conjg(transpose(z))
    end if
  end subroutine qh_times_a_c

  recursive subroutine get_matrix_c(a, am)
    class(rla_operator_c), intent(in) :: a
    complex(dp), allocatable, intent(out) :: am(:,:)
    complex(dp), pointer :: p(:,:)
    complex(dp), allocatable :: e(:,:), bm(:,:)
    integer :: j0, j1, j
    p => a%dense()
    if (associated(p)) then
       am = p
       return
    end if
    select type (a)
    type is (adjoint_c)
       call get_matrix_c(a%base, bm)
       am = conjg(transpose(bm))
       return
    end select
    allocate(am(a%m, a%n))
    if (size(am) == 0) return
    do j0 = 1, a%n, materialize_block
       j1 = min(j0 + materialize_block - 1, a%n)
       allocate(e(a%n, j1 - j0 + 1))
       e = 0
       do j = j0, j1
          e(j, j - j0 + 1) = 1
       end do
       call a%apply(e, am(:, j0:j1))
       deallocate(e)
    end do
  end subroutine get_matrix_c

  ! ==================================================================
  ! Algorithms: complex
  ! ==================================================================

  subroutine sample_c(a, l, pit, rng, y)
    class(rla_operator_c), intent(in) :: a
    integer, intent(in) :: l, pit
    type(rla_rng), intent(inout) :: rng
    complex(dp), allocatable, intent(out) :: y(:,:)
    complex(dp), allocatable :: x(:,:), z(:,:), r(:,:)
    integer, allocatable :: jp(:)
    integer :: it
    allocate(x(a%n, l))
    call la_larnv(rng%iseed, x)
    do it = 1, pit
       call matvec_c(a, x, y)
       call rmatvec_c(a, y, z)
       call la_qrpiv_move(z, r, jp, x)
    end do
    call matvec_c(a, x, y)
  end subroutine sample_c

  subroutine orth_core_c(a, rtol, o, rng, q, flag, diagr)
    class(rla_operator_c), intent(in) :: a
    real(dp), intent(in) :: rtol
    type(opts_t), intent(in) :: o
    type(rla_rng), intent(inout) :: rng
    complex(dp), allocatable, intent(out) :: q(:,:)
    real(dp), allocatable, intent(out) :: diagr(:)
    integer, intent(out) :: flag
    complex(dp), allocatable :: y(:,:), r(:,:)
    integer, allocatable :: jp(:)
    integer :: mn, l, nd, i

    mn = min(a%m, a%n)
    flag = 0

    if (rtol >= 1) then
       call sample_c(a, kmax_of(rtol) + o%es, o%pit, rng, y)
       call la_qrpiv_move(y, r, jp, q)
       diagr = [(abs(r(i, i)), i = 1, size(r, 1))]
       return
    end if

    if (rtol < eps .or. o%bs >= mn) then
       flag = 1
       allocate(q(a%m, 0), diagr(0))
       return
    end if
    l = o%bs
    do
       call sample_c(a, l, o%pit, rng, y)
       call la_qrpiv_move(y, r, jp, q)
       diagr = [(abs(r(i, i)), i = 1, size(r, 1))]
       nd = size(diagr)
       if (nd == 0) return
       if (nd > o%es) then
          if (diagr(nd - o%es) <= rtol*diagr(1)) return
       end if
       l = min(4*l, mn)
       if (l >= mn) then
          flag = 1
          deallocate(q, diagr)
          allocate(q(a%m, 0), diagr(0))
          return
       end if
    end do
  end subroutine orth_core_c

  subroutine qr_core_c(a, rtol, o, rng, q, r, piv)
    class(rla_operator_c), intent(in) :: a
    real(dp), intent(in) :: rtol
    type(opts_t), intent(in) :: o
    type(rla_rng), intent(inout) :: rng
    complex(dp), allocatable, intent(out) :: q(:,:), r(:,:)
    integer, allocatable, intent(out) :: piv(:)
    complex(dp), allocatable :: qs(:,:), b(:,:), qp(:,:), rp(:,:), am(:,:)
    real(dp), allocatable :: d(:)
    logical :: rank_mode, fallback
    integer :: mn, flag, k, rank, i

    mn = min(a%m, a%n)
    rank_mode = rtol >= 1
    call orth_core_c(a, rtol, o, rng, qs, flag, d)
    k = mn
    if (flag == 0) k = size(qs, 2)
    fallback = (flag /= 0 .or. k >= mn) .and. .not. rank_mode

    if (fallback) then
       call get_matrix_c(a, am)
       call la_qrpiv_move(am, rp, piv, qp)
       if (rank_mode) then
          rank = min(kmax_of(rtol), size(qp, 2))
       else
          rank = rank_from_diag([(abs(rp(i, i)), i = 1, size(rp, 1))], rtol)
       end if
       q = qp(:, 1:rank)
       r = rp(1:rank, :)
       return
    end if

    call qh_times_a_c(qs, a, b)
    call la_qrpiv_move(b, rp, piv, qp)
    if (rank_mode) then
       rank = min(kmax_of(rtol), size(qp, 2))
    else
       rank = rank_from_diag([(abs(rp(i, i)), i = 1, size(rp, 1))], rtol)
    end if
    allocate(q(a%m, rank))
    call la_gemm('N', 'N', qs, qp(:, 1:rank), q)
    r = rp(1:rank, :)
  end subroutine qr_core_c

  recursive subroutine svd_core_c(a, rtol, o, rng, u, s, vh, ierr)
    class(rla_operator_c), intent(in), target :: a
    real(dp), intent(in) :: rtol
    type(opts_t), intent(in) :: o
    type(rla_rng), intent(inout) :: rng
    complex(dp), allocatable, intent(out) :: u(:,:), vh(:,:)
    real(dp), allocatable, intent(out) :: s(:)
    integer, intent(out) :: ierr
    type(adjoint_c) :: adj
    complex(dp), allocatable :: qs(:,:), b(:,:), up(:,:), vp(:,:)
    real(dp), allocatable :: d(:), sp(:)
    logical :: rank_mode, fallback
    integer :: mn, flag, k, rank

    if (a%m < a%n) then
       adj%m = a%n; adj%n = a%m; adj%base => a
       call svd_core_c(adj, rtol, o, rng, up, s, vp, ierr)
       if (ierr /= RLA_OK) return
       u = conjg(transpose(vp))
       vh = conjg(transpose(up))
       return
    end if

    mn = min(a%m, a%n)
    rank_mode = rtol >= 1
    call orth_core_c(a, rtol, o, rng, qs, flag, d)
    k = mn
    if (flag == 0) k = size(qs, 2)
    fallback = (flag /= 0 .or. k >= mn) .and. .not. rank_mode

    if (fallback) then
       call get_matrix_c(a, b)
       call la_svd(b, up, sp, vp, ierr)
       if (ierr /= 0) then
          ierr = RLA_ERR_LAPACK
          return
       end if
       if (rank_mode) then
          rank = min(kmax_of(rtol), size(sp))
       else
          rank = rank_from_diag(sp, rtol)
       end if
       u = up(:, 1:rank)
       s = sp(1:rank)
       vh = vp(1:rank, :)
       return
    end if

    call qh_times_a_c(qs, a, b)
    call la_svd(b, up, sp, vp, ierr)
    if (ierr /= 0) then
       ierr = RLA_ERR_LAPACK
       return
    end if
    if (rank_mode) then
       rank = min(kmax_of(rtol), size(sp))
    else
       rank = rank_from_diag(sp, rtol)
    end if
    allocate(u(a%m, rank))
    call la_gemm('N', 'N', qs, up(:, 1:rank), u)
    s = sp(1:rank)
    vh = vp(1:rank, :)
  end subroutine svd_core_c

  subroutine ids_core_c(a, rtol, o, meth, rng, k, piv, t, ierr)
    class(rla_operator_c), intent(in) :: a
    real(dp), intent(in) :: rtol
    type(opts_t), intent(in) :: o
    integer, intent(in) :: meth
    type(rla_rng), intent(inout) :: rng
    integer, intent(out) :: k, ierr
    integer, allocatable, intent(out) :: piv(:)
    complex(dp), allocatable, intent(out) :: t(:,:)
    complex(dp), allocatable :: q(:,:), r(:,:)
    real(dp) :: rts

    call qr_core_c(a, rtol, o, rng, q, r, piv)
    k = size(r, 1)
    rts = rtol
    if (rtol >= 1) rts = 0
    call compute_t_c(a, r, piv, k, meth, rts, t, ierr)
  end subroutine ids_core_c

  subroutine idq_core_c(a, rtol, meth, k, piv, t, ierr)
    class(rla_operator_c), intent(in) :: a
    real(dp), intent(in) :: rtol
    integer, intent(in) :: meth
    integer, intent(out) :: k, ierr
    integer, allocatable, intent(out) :: piv(:)
    complex(dp), allocatable, intent(out) :: t(:,:)
    complex(dp), allocatable :: am(:,:), w(:,:), r(:,:)
    real(dp) :: rts
    integer :: n, i

    n = a%n
    ierr = RLA_OK
    call get_matrix_c(a, am)
    ! The QR overwrites its input: keep A only when lstsq needs it
    if (meth == METHOD_LSTSQ) then
       w = am
       call la_qrpiv_move(w, r, piv)
    else
       call la_qrpiv_move(am, r, piv)
    end if
    if (rtol >= 1) then
       k = min(kmax_of(rtol), min(a%m, n))
       rts = 0
    else
       k = rank_from_diag([(abs(r(i, i)), i = 1, size(r, 1))], rtol)
       rts = rtol
    end if
    if (k == 0) then
       allocate(t(0, n))
       return
    end if
    if (k == n) then
       allocate(t(k, 0))
       return
    end if
    if (meth == METHOD_LSTSQ) then
       call t_lstsq_c(am, piv, k, t, ierr)
    else
       call compute_t_c(a, r, piv, k, meth, rts, t, ierr)
    end if
  end subroutine idq_core_c

  subroutine compute_t_c(a, r, piv, k, meth, rts, t, ierr)
    class(rla_operator_c), intent(in) :: a
    complex(dp), intent(in) :: r(:,:)
    real(dp), intent(in) :: rts
    integer, intent(in) :: piv(:), k, meth
    complex(dp), allocatable, intent(out) :: t(:,:)
    integer, intent(out) :: ierr
    complex(dp), allocatable :: am(:,:)
    ierr = RLA_OK
    select case (meth)
    case (METHOD_LSTSQ)
       if (k == 0 .or. k >= a%n) then
          allocate(t(k, a%n - k))
          t = 0
       else
          call get_matrix_c(a, am)
          call t_lstsq_c(am, piv, k, t, ierr)
       end if
    case (METHOD_SVD)
       call t_svd_c(r, k, rts, t, ierr)
    case default
       call t_fast_c(r, k, rts, t, ierr)
    end select
  end subroutine compute_t_c

  subroutine t_lstsq_c(am, piv, k, t, ierr)
    complex(dp), intent(in) :: am(:,:)
    integer, intent(in) :: piv(:), k
    complex(dp), allocatable, intent(out) :: t(:,:)
    integer, intent(out) :: ierr
    call la_lstsq(am(:, piv(1:k)), am(:, piv(k+1:)), t, ierr)
    if (ierr /= 0) ierr = RLA_ERR_LAPACK
  end subroutine t_lstsq_c

  subroutine t_svd_c(r, k, rts, t, ierr)
    complex(dp), intent(in) :: r(:,:)
    real(dp), intent(in) :: rts
    integer, intent(in) :: k
    complex(dp), allocatable, intent(out) :: t(:,:)
    integer, intent(out) :: ierr
    complex(dp), allocatable :: u(:,:), vh(:,:), w(:,:)
    real(dp), allocatable :: s(:)
    integer, allocatable :: keep(:)
    real(dp) :: smax
    integer :: n, i

    n = size(r, 2)
    ierr = RLA_OK
    allocate(t(k, n - k))
    t = 0
    if (k == 0) return
    call la_svd(r(1:k, 1:k), u, s, vh, ierr)
    if (ierr /= 0) then
       ierr = RLA_ERR_LAPACK
       return
    end if
    smax = maxval(s)
    if (smax <= 0) return
    keep = pack([(i, i = 1, k)], s >= max(rts, eps)*smax)
    if (size(keep) == 0) return
    allocate(w(size(keep), n - k))
    call la_gemm('C', 'N', u(:, keep), r(1:k, k+1:n), w)
    do i = 1, size(keep)
       w(i, :) = w(i, :) / s(keep(i))
    end do
    call la_gemm('C', 'N', vh(keep, :), w, t)
  end subroutine t_svd_c

  subroutine t_fast_c(r, k, rts, t, ierr)
    complex(dp), intent(in) :: r(:,:)
    real(dp), intent(in) :: rts
    integer, intent(in) :: k
    complex(dp), allocatable, intent(out) :: t(:,:)
    integer, intent(out) :: ierr
    real(dp), allocatable :: d(:)
    integer :: n, i

    n = size(r, 2)
    ierr = RLA_OK
    if (k == 0) then
       allocate(t(0, n))
       return
    end if
    d = [(abs(r(i, i)), i = 1, k)]
    if (minval(d) <= eps*maxval(d)) then
       call t_svd_c(r, k, rts, t, ierr)
       return
    end if
    t = r(1:k, k+1:n)
    call la_trsm_upper(r(1:k, 1:k), t)
  end subroutine t_fast_c

end module librla
