// The five librla routines and their private helpers (port of librla.py).
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

use crate::error::Error;
use crate::lapack::{gemm, lstsq, qr_piv, solve_upper, svd_econ};
use crate::mat::Mat;
use crate::operator::{Adjoint, ByRef, LinearOperator};
use crate::rng::Rng;
use crate::scalar::Scalar;
use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;

type Op<'a, T> = &'a dyn LinearOperator<T>;

// --------------------------------------------------------------
// Options and results
// --------------------------------------------------------------

/// How `id_sketch` / `id_qrpiv` compute the interpolation matrix `T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Method {
    /// Triangular solve `R11 \ R12` (default); falls back to `Svd` when
    /// `R11` is numerically singular.
    #[default]
    Fast,
    /// SVD-based pseudoinverse of `R11`.
    Svd,
    /// Least squares on the columns of `A` (most accurate, slowest).
    Lstsq,
}

impl Method {
    /// `"fast"`, `"svd"` or `"lstsq"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Method::Fast => "fast",
            Method::Svd => "svd",
            Method::Lstsq => "lstsq",
        }
    }
}

impl FromStr for Method {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        match s {
            "fast" => Ok(Method::Fast),
            "svd" => Ok(Method::Svd),
            "lstsq" => Ok(Method::Lstsq),
            _ => Err(Error::InvalidMethod(s.to_string())),
        }
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Optional parameters of the sketching routines.
///
/// ```
/// use librla::{Method, Options, Rng};
/// let mut rng = Rng::from_seed(7);
/// let opts = Options { power_iter: 1, rng: Some(&mut rng), ..Default::default() };
/// let opts2 = Options::new().extra_samples(5).method(Method::Lstsq);
/// # let _ = (opts, opts2);
/// ```
#[derive(Debug)]
pub struct Options<'a> {
    /// Initial number of random test vectors in tolerance mode (default 42);
    /// ignored in rank mode.
    pub block_size: usize,
    /// Number of power iterations `x <- orth(A^H A x)` (default 0).
    pub power_iter: usize,
    /// Buffer columns beyond the target rank (default 12); see the crate docs.
    pub extra_samples: usize,
    /// Interpolation-matrix method for `id_sketch` (default `Fast`); the other
    /// routines ignore it.
    pub method: Method,
    /// Random number generator, advanced in place; `None` uses a fresh
    /// `Rng::default()` (seed `[0, 0, 0, 1]`), so results are deterministic.
    pub rng: Option<&'a mut Rng>,
}

impl Default for Options<'_> {
    fn default() -> Self {
        Options {
            block_size: 42,
            power_iter: 0,
            extra_samples: 12,
            method: Method::Fast,
            rng: None,
        }
    }
}

impl<'a> Options<'a> {
    /// Defaults (same as `Options::default()`).
    pub fn new() -> Self {
        Self::default()
    }
    /// Set `block_size`.
    pub fn block_size(mut self, v: usize) -> Self {
        self.block_size = v;
        self
    }
    /// Set `power_iter`.
    pub fn power_iter(mut self, v: usize) -> Self {
        self.power_iter = v;
        self
    }
    /// Set `extra_samples`.
    pub fn extra_samples(mut self, v: usize) -> Self {
        self.extra_samples = v;
        self
    }
    /// Set `method`.
    pub fn method(mut self, v: Method) -> Self {
        self.method = v;
        self
    }
    /// Use (and advance) `rng`.
    pub fn rng(mut self, rng: &'a mut Rng) -> Self {
        self.rng = Some(rng);
        self
    }
}

/// Result of [`orth_sketch`].
#[derive(Debug, Clone)]
pub struct OrthSketch<T> {
    /// Orthonormal basis (`m x k`) including the `extra_samples` buffer columns;
    /// `m x 0` when `flag == 1`.
    pub q: Mat<T>,
    /// 0 on success; 1 on early termination in tolerance mode
    /// (`rtol < eps`, or the sketch would reach `min(m, n)` columns).
    pub flag: i32,
    /// `|diag(R)|` of the pivoted QR of the sketch (non-increasing); empty when `flag == 1`.
    pub diag_r: Vec<f64>,
}

/// Result of [`qr_sketch`]: `A[:, p] ~= Q R`.
#[derive(Debug, Clone)]
pub struct QrSketch<T> {
    /// `m x k`, orthonormal columns.
    pub q: Mat<T>,
    /// `k x n`, upper triangular.
    pub r: Mat<T>,
    /// Column permutation of length `n` (0-based).
    pub p: Vec<usize>,
}

/// Result of [`svd_sketch`]: `A ~= U diag(s) Vh`.
#[derive(Debug, Clone)]
pub struct SvdSketch<T> {
    /// `m x k` left singular vectors.
    pub u: Mat<T>,
    /// `k` singular values, descending.
    pub s: Vec<f64>,
    /// `k x n` right singular vectors, conjugate-transposed.
    pub vh: Mat<T>,
}

/// Result of [`id_sketch`] / [`id_qrpiv`]:
/// `A[:, piv[k..]] ~= A[:, piv[..k]] T`.
#[derive(Debug, Clone)]
pub struct IdDecomp<T> {
    /// Rank (number of skeleton columns).
    pub k: usize,
    /// Column permutation of length `n` (0-based); `piv[..k]` is the skeleton.
    pub piv: Vec<usize>,
    /// Interpolation matrix, `k x (n - k)`.
    pub t: Mat<T>,
}

// --------------------------------------------------------------
// Public entry points
// --------------------------------------------------------------

/// Run `f` with the caller's RNG, or with a fresh `Rng::default()`.
fn with_rng<R>(rng: Option<&mut Rng>, f: impl FnOnce(&mut Rng) -> R) -> R {
    match rng {
        Some(r) => f(r),
        None => f(&mut Rng::default()),
    }
}

fn check_rtol(rtol: f64) -> Result<(), Error> {
    if rtol.is_finite() {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!("rtol must be finite, got {rtol}")))
    }
}

/// Approximate orthonormal basis for the column space of `A` by randomized
/// sketching (`Y = A Omega`, then pivoted QR of `Y`).
///
/// * Tolerance mode (`rtol < 1`): fresh sketches of `block_size`,
///   `4 block_size`, ... columns until `diag_r[len - 1 - extra_samples]
///   <= rtol * diag_r[0]`; `flag = 1` with an empty basis if `rtol < eps`
///   or the sketch would reach `min(m, n)` columns.
/// * Rank mode (`rtol >= 1`): one sketch of `floor(rtol) + extra_samples`
///   columns.
///
/// The basis includes the `extra_samples` buffer columns in both modes.
pub fn orth_sketch<T, A>(a: &A, rtol: f64, opts: Options<'_>) -> Result<OrthSketch<T>, Error>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
{
    check_rtol(rtol)?;
    let op = ByRef(a);
    let Options { block_size, power_iter, extra_samples, rng, .. } = opts;
    with_rng(rng, |rng| orth_impl(&op, rtol, block_size, power_iter, extra_samples, rng))
}

/// Truncated QR with column pivoting by randomized sketching:
/// `A[:, p] ~= Q R` with `Q` `m x k`, `R` `k x n` upper triangular.
///
/// Tolerance mode keeps the columns with `|R[i,i]| >= rtol |R[0,0]|`; rank
/// mode returns `k = floor(rtol)` columns (fewer if `min(m, n)` is smaller).
/// When the sketch terminates early in tolerance mode, a deterministic
/// pivoted QR of the full (possibly materialized) matrix is used.
pub fn qr_sketch<T, A>(a: &A, rtol: f64, opts: Options<'_>) -> Result<QrSketch<T>, Error>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
{
    check_rtol(rtol)?;
    let op = ByRef(a);
    let Options { block_size, power_iter, extra_samples, rng, .. } = opts;
    with_rng(rng, |rng| qr_impl(&op, rtol, block_size, power_iter, extra_samples, rng))
}

/// Truncated SVD by randomized sketching: `A ~= U diag(s) Vh`.
///
/// Tolerance mode keeps the singular values `s[i] >= rtol s[0]`; rank mode
/// returns `k = floor(rtol)` triplets. Wide matrices (`m < n`) are handled
/// through the adjoint. When the sketch terminates early in tolerance mode,
/// a deterministic SVD of the full (possibly materialized) matrix is used.
pub fn svd_sketch<T, A>(a: &A, rtol: f64, opts: Options<'_>) -> Result<SvdSketch<T>, Error>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
{
    check_rtol(rtol)?;
    let op = ByRef(a);
    let op: Op<T> = &op;
    let Options { block_size, power_iter, extra_samples, rng, .. } = opts;
    with_rng(rng, |rng| {
        let (m, n) = op.shape();
        if m < n {
            // Recurse on the adjoint and swap/adjoint the factors.
            let dense_h;
            let wrapped;
            let adj: Op<T> = match op.as_dense() {
                Some(d) => {
                    dense_h = d.adjoint();
                    &dense_h
                }
                None => {
                    wrapped = Adjoint(op);
                    &wrapped
                }
            };
            let r = svd_tall(adj, rtol, block_size, power_iter, extra_samples, rng)?;
            return Ok(SvdSketch { u: r.vh.adjoint(), s: r.s, vh: r.u.adjoint() });
        }
        svd_tall(op, rtol, block_size, power_iter, extra_samples, rng)
    })
}

/// Interpolative decomposition by randomized sketching:
/// `A[:, piv[k..]] ~= A[:, piv[..k]] T`.
///
/// The permutation and rank come from [`qr_sketch`]; `T` is computed by
/// `opts.method`.
pub fn id_sketch<T, A>(a: &A, rtol: f64, opts: Options<'_>) -> Result<IdDecomp<T>, Error>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
{
    check_rtol(rtol)?;
    let op = ByRef(a);
    let op: Op<T> = &op;
    let Options { block_size, power_iter, extra_samples, method, rng } = opts;
    with_rng(rng, |rng| {
        let qr = qr_impl(op, rtol, block_size, power_iter, extra_samples, rng)?;
        let k = qr.r.nrows();
        let piv = qr.p;
        // Rank mode: minimal filtering (only exact zeros)
        let rtol_for_svd = if rtol >= 1.0 { 0.0 } else { rtol };
        let t = match method {
            Method::Lstsq => compute_t_lstsq(op, &piv, k)?,
            Method::Svd => compute_t_svd(&qr.r, k, rtol_for_svd)?,
            Method::Fast => compute_t_fast(&qr.r, k, rtol_for_svd)?,
        };
        Ok(IdDecomp { k, piv, t })
    })
}

/// Interpolative decomposition by deterministic QR with column pivoting
/// (geqp3) of the full (possibly materialized) matrix:
/// `A[:, piv[k..]] ~= A[:, piv[..k]] T`.
pub fn id_qrpiv<T, A>(a: &A, rtol: f64, method: Method) -> Result<IdDecomp<T>, Error>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
{
    check_rtol(rtol)?;
    let op = ByRef(a);
    let op: Op<T> = &op;
    let (m, n) = op.shape();

    let a_mat = get_matrix(op)?;
    let (_, r, piv) = qr_piv(a_mat.into_owned(), false)?;

    let (k, rtol_for_svd) = if rtol >= 1.0 {
        ((rtol.floor() as usize).min(m.min(n)), 0.0)
    } else {
        (rank_from_diag(&r.diag(), rtol), rtol)
    };

    if k == 0 {
        return Ok(IdDecomp { k: 0, piv, t: Mat::zeros(0, n) });
    }
    if k == n {
        return Ok(IdDecomp { k, piv, t: Mat::zeros(k, 0) });
    }

    let t = match method {
        Method::Lstsq => compute_t_lstsq(op, &piv, k)?,
        Method::Svd => compute_t_svd(&r, k, rtol_for_svd)?,
        Method::Fast => compute_t_fast(&r, k, rtol_for_svd)?,
    };
    Ok(IdDecomp { k, piv, t })
}

// --------------------------------------------------------------
// Algorithms
// --------------------------------------------------------------

fn orth_impl<T: Scalar>(
    a: Op<T>, rtol: f64, block_size: usize, power_iter: usize, extra_samples: usize, rng: &mut Rng,
) -> Result<OrthSketch<T>, Error> {
    let (m, n) = a.shape();
    let sketch = |l: usize, rng: &mut Rng| -> Result<(Mat<T>, Vec<f64>), Error> {
        let x = rng.uniform::<T>(n, l);
        let x = power_iteration(a, x, power_iter)?;
        let y = apply(a, &x)?;
        let (q, r, _) = qr_piv(y, true)?;
        let diag_r = r.diag().iter().map(|d| d.abs()).collect();
        Ok((q.expect("Q requested"), diag_r))
    };
    let early = || OrthSketch { q: Mat::zeros(m, 0), flag: 1, diag_r: Vec::new() };

    // Rank mode (rtol >= 1): single oversampled sketch, buffer included
    if rtol >= 1.0 {
        let kmax = rtol.floor() as usize;
        let (q, diag_r) = sketch(kmax + extra_samples, rng)?;
        return Ok(OrthSketch { q, flag: 0, diag_r });
    }

    // Tolerance mode (rtol < 1): geometric growth with tolerance checking
    if rtol < f64::EPSILON {
        return Ok(early());
    }
    let mut block_size = block_size;
    if block_size >= m.min(n) {
        return Ok(early());
    }
    loop {
        let (q, diag_r) = sketch(block_size, rng)?;
        // Buffered tolerance test (diag_r is non-increasing; the multiplied
        // form stays valid when diag_r[0] == 0)
        let len = diag_r.len();
        if len == 0 || (len > extra_samples && diag_r[len - 1 - extra_samples] <= rtol * diag_r[0])
        {
            return Ok(OrthSketch { q, flag: 0, diag_r });
        }
        block_size = (block_size * 4).min(m.min(n));
        if block_size >= m.min(n) {
            return Ok(early());
        }
    }
}

fn qr_impl<T: Scalar>(
    a: Op<T>, rtol: f64, block_size: usize, power_iter: usize, extra_samples: usize, rng: &mut Rng,
) -> Result<QrSketch<T>, Error> {
    let (m, n) = a.shape();
    let rank_mode = rtol >= 1.0;
    let kmax = if rank_mode { rtol.floor() as usize } else { 0 };

    // The basis includes the extra_samples buffer columns (truncated after QR)
    let orth = orth_impl(a, rtol, block_size, power_iter, extra_samples, rng)?;
    let k = if orth.flag == 0 { orth.q.ncols() } else { m.min(n) };

    let needs_fallback = (orth.flag != 0 || k >= m.min(n)) && !rank_mode;

    let (q, r, p) = if needs_fallback {
        // Deterministic pivoted QR of the full matrix
        let a_mat = get_matrix(a)?;
        let (q, r, p) = qr_piv(a_mat.into_owned(), true)?;
        (q.expect("Q requested"), r, p)
    } else {
        // Project and compute QR
        let b = matmat_left(&orth.q, a)?;
        let (qp, r, p) = qr_piv(b, true)?;
        (orth.q.matmul(&qp.expect("Q requested")), r, p)
    };

    let rank = if rank_mode { kmax.min(q.ncols()) } else { rank_from_diag(&r.diag(), rtol) };
    Ok(QrSketch { q: q.columns(0..rank), r: r.rows(0..rank), p })
}

fn svd_tall<T: Scalar>(
    a: Op<T>, rtol: f64, block_size: usize, power_iter: usize, extra_samples: usize, rng: &mut Rng,
) -> Result<SvdSketch<T>, Error> {
    let (m, n) = a.shape();
    let rank_mode = rtol >= 1.0;
    let kmax = if rank_mode { rtol.floor() as usize } else { 0 };

    // The basis includes the extra_samples buffer columns (truncated after SVD)
    let orth = orth_impl(a, rtol, block_size, power_iter, extra_samples, rng)?;
    let k = if orth.flag == 0 { orth.q.ncols() } else { m.min(n) };

    let needs_fallback = (orth.flag != 0 || k >= m.min(n)) && !rank_mode;

    let (u, s, vh) = if needs_fallback {
        // Deterministic SVD of the full matrix
        let a_mat = get_matrix(a)?;
        svd_econ(&a_mat)?
    } else {
        // Project and compute SVD
        let b = matmat_left(&orth.q, a)?;
        let (up, s, vh) = svd_econ(&b)?;
        (orth.q.matmul(&up), s, vh)
    };

    let rank = if rank_mode { kmax.min(s.len()) } else { rank_from_svals(&s, rtol) };
    Ok(SvdSketch { u: u.columns(0..rank), s: s[..rank].to_vec(), vh: vh.rows(0..rank) })
}

// --------------------------------------------------------------
// Operator helpers
// --------------------------------------------------------------

fn shape_check<T: Scalar>(
    call: &'static str, y: Mat<T>, expected: (usize, usize),
) -> Result<Mat<T>, Error> {
    if y.shape() == expected {
        Ok(y)
    } else {
        Err(Error::ShapeMismatch { call, expected, got: y.shape() })
    }
}

/// `Y = A X`.
fn apply<T: Scalar>(a: Op<T>, x: &Mat<T>) -> Result<Mat<T>, Error> {
    let expected = (a.shape().0, x.ncols());
    shape_check("apply", a.apply(x), expected)
}

/// `Z = A^H W`.
fn apply_adjoint<T: Scalar>(a: Op<T>, w: &Mat<T>) -> Result<Mat<T>, Error> {
    let expected = (a.shape().1, w.ncols());
    shape_check("apply_adjoint", a.apply_adjoint(w), expected)
}

/// Columns of identity applied per operator call when materializing.
const MATERIALIZE_BLOCK: usize = 64;

/// `A` as a dense matrix: borrowed when the operator exposes one, otherwise
/// materialized by applying it to blocks of identity columns (O(n) matvecs).
fn get_matrix<T: Scalar>(a: Op<'_, T>) -> Result<Cow<'_, Mat<T>>, Error> {
    if let Some(d) = a.as_dense() {
        return Ok(Cow::Borrowed(d));
    }
    let (m, n) = a.shape();
    let mut out = Mat::zeros(m, n);
    let mut j0 = 0;
    while j0 < n {
        let b = MATERIALIZE_BLOCK.min(n - j0);
        let mut e = Mat::zeros(n, b);
        for c in 0..b {
            e[(j0 + c, c)] = T::one();
        }
        let y = apply(a, &e)?;
        out.as_mut_slice()[j0 * m..(j0 + b) * m].copy_from_slice(y.as_slice());
        j0 += b;
    }
    Ok(Cow::Owned(out))
}

/// `Q^H A`; for a matrix-free operator `(A^H Q)^H`.
fn matmat_left<T: Scalar>(q: &Mat<T>, a: Op<T>) -> Result<Mat<T>, Error> {
    match a.as_dense() {
        Some(d) => Ok(gemm(b'C', q, b'N', d)),
        None => Ok(apply_adjoint(a, q)?.adjoint()),
    }
}

/// `power_iter` times: `x = A^H (A x)`, then `x = Q` of the pivoted QR of `x`.
fn power_iteration<T: Scalar>(a: Op<T>, mut x: Mat<T>, power_iter: usize) -> Result<Mat<T>, Error> {
    for _ in 0..power_iter {
        let y = apply(a, &x)?;
        x = apply_adjoint(a, &y)?;
        x = qr_piv(x, true)?.0.expect("Q requested");
    }
    Ok(x)
}

// --------------------------------------------------------------
// Rank determination helpers
// --------------------------------------------------------------

/// Number of singular values `>= rtol * s[0]` (0 if `s` is empty or `s[0] <= 0`).
fn rank_from_svals(s: &[f64], rtol: f64) -> usize {
    match s.first() {
        Some(&s0) if s0 > 0.0 => s.iter().filter(|&&v| v >= rtol * s0).count(),
        _ => 0,
    }
}

/// Number of `|d[i]| >= rtol * |d[0]|` (0 if `d` is empty or `|d[0]| <= 0`).
fn rank_from_diag<T: Scalar>(d: &[T], rtol: f64) -> usize {
    let abs: Vec<f64> = d.iter().map(|x| x.abs()).collect();
    rank_from_svals(&abs, rtol)
}

// --------------------------------------------------------------
// Interpolation matrix T for the ID
// --------------------------------------------------------------

/// `T` by least squares on the columns of `A`: `A[:, piv[..k]] T = A[:, piv[k..]]`.
fn compute_t_lstsq<T: Scalar>(a: Op<T>, piv: &[usize], k: usize) -> Result<Mat<T>, Error> {
    let n = a.shape().1;
    if k == 0 || k >= n {
        return Ok(Mat::zeros(k, n.saturating_sub(k)));
    }
    let a_mat = get_matrix(a)?;
    lstsq(&a_mat.select_columns(&piv[..k]), &a_mat.select_columns(&piv[k..]))
}

/// `T = pinv(R11) R12` through the SVD of `R11`, dropping singular values
/// below `max(rtol_for_svd, eps) * s_max`; an all-zero `R11` gives `T = 0`.
fn compute_t_svd<T: Scalar>(r: &Mat<T>, k: usize, rtol_for_svd: f64) -> Result<Mat<T>, Error> {
    let n = r.ncols();
    if k == 0 {
        return Ok(Mat::zeros(0, n));
    }
    let r11 = r.rows(0..k).columns(0..k);
    let r12 = r.rows(0..k).columns(k..n);

    let (u, s, vh) = svd_econ(&r11)?;

    // Floor the relative threshold at machine precision so a rank-deficient
    // R11 drops near-zero singular values instead of inverting them.
    let smax = s.iter().copied().fold(0.0, f64::max);
    if smax == 0.0 {
        return Ok(Mat::zeros(k, n - k));
    }
    let tol = rtol_for_svd.max(f64::EPSILON);
    let keep: Vec<usize> = (0..s.len()).filter(|&i| s[i] >= tol * smax).collect();
    if keep.is_empty() {
        return Ok(Mat::zeros(k, n - k));
    }
    let inv_s: Vec<f64> = keep.iter().map(|&i| 1.0 / s[i]).collect();
    let u_keep = u.select_columns(&keep);
    let v_keep = vh.adjoint().select_columns(&keep);
    let w = gemm(b'C', &u_keep, b'N', &r12).scale_rows(&inv_s);
    Ok(v_keep.matmul(&w))
}

/// `T = R11 \ R12` by triangular solve, or `compute_t_svd` when
/// `min |R11[i,i]| <= eps * max |R11[i,i]|`.
fn compute_t_fast<T: Scalar>(r: &Mat<T>, k: usize, rtol_for_svd: f64) -> Result<Mat<T>, Error> {
    let n = r.ncols();
    if k == 0 {
        return Ok(Mat::zeros(0, n));
    }
    let r11 = r.rows(0..k).columns(0..k);
    let d: Vec<f64> = r11.diag().iter().map(|x| x.abs()).collect();
    let dmin = d.iter().copied().fold(f64::INFINITY, f64::min);
    let dmax = d.iter().copied().fold(0.0, f64::max);
    if dmin <= f64::EPSILON * dmax {
        return compute_t_svd(r, k, rtol_for_svd);
    }
    let r12 = r.rows(0..k).columns(k..n);
    Ok(solve_upper(&r11, r12))
}
