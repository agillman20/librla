// Safe wrappers over the LAPACK/BLAS routines (workspace queries, zero-size
// guards, info checks). `qr_pivoted` and `svd` are re-exported as `librla::dense`.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

use crate::error::Error;
use crate::ffi::Int;
use crate::mat::Mat;
use crate::scalar::Scalar;

/// Convert a dimension to a LAPACK integer (LP64); panics beyond `i32::MAX`.
pub(crate) fn int(x: usize) -> Int {
    Int::try_from(x).expect("librla: dimension exceeds the 32-bit LAPACK integer range")
}

/// Leading dimension: at least 1, as LAPACK requires even for empty arrays.
fn ld(m: usize) -> Int {
    int(m.max(1))
}

/// Workspace length from a query answer (rounded up, at least 1).
fn lwork<T: Scalar>(w: T) -> usize {
    (w.work_size().ceil() as usize).max(1)
}

/// `(Q, R, p)` from [`qr_piv`], `Q` only when requested.
pub(crate) type PivQr<T> = (Option<Mat<T>>, Mat<T>, Vec<usize>);

fn check(routine: &'static str, info: Int) -> Result<(), Error> {
    if info == 0 {
        Ok(())
    } else {
        Err(Error::Lapack { routine, info })
    }
}

/// `op(A) op(B)` with `op` in {`N`, `T`, `C`} (BLAS gemm).
pub(crate) fn gemm<T: Scalar>(ta: u8, a: &Mat<T>, tb: u8, b: &Mat<T>) -> Mat<T> {
    let (am, ak) = if ta == b'N' { a.shape() } else { (a.ncols(), a.nrows()) };
    let (bk, bn) = if tb == b'N' { b.shape() } else { (b.ncols(), b.nrows()) };
    assert_eq!(ak, bk, "matrix product: inner dimensions differ ({ak} vs {bk})");
    let mut c = Mat::zeros(am, bn);
    if am == 0 || bn == 0 || ak == 0 {
        return c;
    }
    // SAFETY: dimensions and leading dimensions match the buffers.
    unsafe {
        T::gemm(
            ta,
            tb,
            int(am),
            int(bn),
            int(ak),
            T::one(),
            a.as_slice().as_ptr(),
            ld(a.nrows()),
            b.as_slice().as_ptr(),
            ld(b.nrows()),
            T::zero(),
            c.as_mut_slice().as_mut_ptr(),
            ld(am),
        );
    }
    c
}

/// Economic QR with column pivoting of an `m x n` matrix (geqp3 + orgqr/ungqr):
/// returns `(Q, R, p)` with `Q` `m x min(m,n)` (only when `want_q`),
/// `R` `min(m,n) x n` upper triangular, and `A[:, p] = Q R` (0-based `p`).
pub(crate) fn qr_piv<T: Scalar>(mut a: Mat<T>, want_q: bool) -> Result<PivQr<T>, Error> {
    let (m, n) = a.shape();
    let kk = m.min(n);
    if kk == 0 {
        let q = want_q.then(|| Mat::zeros(m, 0));
        return Ok((q, Mat::zeros(0, n), (0..n).collect()));
    }
    let mut jpvt = vec![0 as Int; n];
    let mut tau = vec![T::zero(); kk];
    let mut rwork = vec![0.0f64; if T::IS_COMPLEX { 2 * n } else { 0 }];
    let mut info: Int = 0;
    let mut wq = [T::zero()];
    // SAFETY: a is m x n with lda = m; jpvt has n entries, tau min(m,n), rwork 2n (complex).
    unsafe {
        T::geqp3(
            int(m),
            int(n),
            a.as_mut_slice().as_mut_ptr(),
            ld(m),
            jpvt.as_mut_ptr(),
            tau.as_mut_ptr(),
            wq.as_mut_ptr(),
            -1,
            rwork.as_mut_ptr(),
            &mut info,
        );
    }
    check("geqp3", info)?;
    let lw = lwork(wq[0]);
    let mut work = vec![T::zero(); lw];
    unsafe {
        T::geqp3(
            int(m),
            int(n),
            a.as_mut_slice().as_mut_ptr(),
            ld(m),
            jpvt.as_mut_ptr(),
            tau.as_mut_ptr(),
            work.as_mut_ptr(),
            int(lw),
            rwork.as_mut_ptr(),
            &mut info,
        );
    }
    check("geqp3", info)?;

    // R: upper triangle of the leading min(m,n) rows; below the diagonal geqp3
    // stores Householder vectors, which must not leak into R.
    let r = Mat::from_fn(kk, n, |i, j| if i <= j { a[(i, j)] } else { T::zero() });
    let p = jpvt.iter().map(|&j| (j - 1) as usize).collect();

    let q = if want_q {
        let mut q = Mat::from_col_major(m, kk, a.into_vec()[..m * kk].to_vec());
        let mut info: Int = 0;
        let mut wq = [T::zero()];
        unsafe {
            T::orgqr(
                int(m),
                int(kk),
                int(kk),
                q.as_mut_slice().as_mut_ptr(),
                ld(m),
                tau.as_ptr(),
                wq.as_mut_ptr(),
                -1,
                &mut info,
            );
        }
        check("orgqr", info)?;
        let lw = lwork(wq[0]);
        let mut work = vec![T::zero(); lw];
        unsafe {
            T::orgqr(
                int(m),
                int(kk),
                int(kk),
                q.as_mut_slice().as_mut_ptr(),
                ld(m),
                tau.as_ptr(),
                work.as_mut_ptr(),
                int(lw),
                &mut info,
            );
        }
        check("orgqr", info)?;
        Some(q)
    } else {
        None
    };
    Ok((q, r, p))
}

/// Economic SVD `A = U diag(s) Vh` (gesdd with jobz = 'S'; gesvd if gesdd
/// does not converge). `U` is `m x min(m,n)`, `Vh` is `min(m,n) x n`.
pub(crate) fn svd_econ<T: Scalar>(a: &Mat<T>) -> Result<dense::SvdFactors<T>, Error> {
    let (m, n) = a.shape();
    let kk = m.min(n);
    if kk == 0 {
        return Ok((Mat::zeros(m, 0), Vec::new(), Mat::zeros(0, n)));
    }
    let mut s = vec![0.0f64; kk];
    let mut u = Mat::zeros(m, kk);
    let mut vt = Mat::zeros(kk, n);
    let mut info: Int = 0;

    let mut acopy = a.clone();
    let mx = m.max(n);
    let lrwork = if T::IS_COMPLEX { kk * (5 * kk + 7).max(2 * mx + 2 * kk + 1) } else { 0 };
    let mut rwork = vec![0.0f64; lrwork];
    let mut iwork = vec![0 as Int; 8 * kk];
    let mut wq = [T::zero()];
    // SAFETY: buffer sizes follow the LAPACK gesdd documentation for jobz = 'S'.
    unsafe {
        T::gesdd(
            b'S',
            int(m),
            int(n),
            acopy.as_mut_slice().as_mut_ptr(),
            ld(m),
            s.as_mut_ptr(),
            u.as_mut_slice().as_mut_ptr(),
            ld(m),
            vt.as_mut_slice().as_mut_ptr(),
            ld(kk),
            wq.as_mut_ptr(),
            -1,
            rwork.as_mut_ptr(),
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    check("gesdd", info)?;
    let lw = lwork(wq[0]);
    let mut work = vec![T::zero(); lw];
    unsafe {
        T::gesdd(
            b'S',
            int(m),
            int(n),
            acopy.as_mut_slice().as_mut_ptr(),
            ld(m),
            s.as_mut_ptr(),
            u.as_mut_slice().as_mut_ptr(),
            ld(m),
            vt.as_mut_slice().as_mut_ptr(),
            ld(kk),
            work.as_mut_ptr(),
            int(lw),
            rwork.as_mut_ptr(),
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    if info < 0 {
        return Err(Error::Lapack { routine: "gesdd", info });
    }
    if info == 0 {
        return Ok((u, s, vt));
    }

    // gesdd did not converge: fall back to the QR-iteration driver.
    let mut acopy = a.clone();
    let mut rwork = vec![0.0f64; if T::IS_COMPLEX { 5 * kk } else { 0 }];
    let mut wq = [T::zero()];
    unsafe {
        T::gesvd(
            b'S',
            b'S',
            int(m),
            int(n),
            acopy.as_mut_slice().as_mut_ptr(),
            ld(m),
            s.as_mut_ptr(),
            u.as_mut_slice().as_mut_ptr(),
            ld(m),
            vt.as_mut_slice().as_mut_ptr(),
            ld(kk),
            wq.as_mut_ptr(),
            -1,
            rwork.as_mut_ptr(),
            &mut info,
        );
    }
    check("gesvd", info)?;
    let lw = lwork(wq[0]);
    let mut work = vec![T::zero(); lw];
    unsafe {
        T::gesvd(
            b'S',
            b'S',
            int(m),
            int(n),
            acopy.as_mut_slice().as_mut_ptr(),
            ld(m),
            s.as_mut_ptr(),
            u.as_mut_slice().as_mut_ptr(),
            ld(m),
            vt.as_mut_slice().as_mut_ptr(),
            ld(kk),
            work.as_mut_ptr(),
            int(lw),
            rwork.as_mut_ptr(),
            &mut info,
        );
    }
    check("gesvd", info)?;
    Ok((u, s, vt))
}

/// Minimum-norm least-squares solution `X = argmin ||A X - B||` (gelsd,
/// rcond = machine epsilon as in scipy.linalg.lstsq). `A` is `m x k`,
/// `B` is `m x nrhs`, `X` is `k x nrhs`.
pub(crate) fn lstsq<T: Scalar>(a: &Mat<T>, b: &Mat<T>) -> Result<Mat<T>, Error> {
    let (m, k) = a.shape();
    let nrhs = b.ncols();
    assert_eq!(b.nrows(), m, "lstsq: A and B must have the same number of rows");
    if k == 0 || nrhs == 0 {
        return Ok(Mat::zeros(k, nrhs));
    }
    if m == 0 {
        return Ok(Mat::zeros(k, nrhs));
    }
    let ldb = m.max(k);
    let mut bb = Mat::zeros(ldb, nrhs);
    for j in 0..nrhs {
        bb.col_mut(j)[..m].copy_from_slice(b.col(j));
    }
    let mut acopy = a.clone();
    let minmn = m.min(k);
    let mut s = vec![0.0f64; minmn];
    let rcond = f64::EPSILON;
    let mut rank: Int = 0;
    let mut info: Int = 0;

    // Documented minimum sizes, used if a backend's query leaves them unset.
    const SMLSIZ: usize = 25;
    let nlvl =
        ((((minmn as f64) / (SMLSIZ as f64 + 1.0)).ln() / 2f64.ln()) as i64 + 1).max(0) as usize;
    let liwork_min = (3 * minmn * nlvl + 11 * minmn).max(1);
    let lrwork_min = if T::IS_COMPLEX {
        10 * minmn
            + 2 * minmn * SMLSIZ
            + 8 * minmn * nlvl
            + 3 * SMLSIZ * nrhs
            + ((SMLSIZ + 1) * (SMLSIZ + 1)).max(minmn * (1 + nrhs) + 2 * nrhs)
    } else {
        1
    };

    let mut wq = [T::zero()];
    let mut rq = [0.0f64];
    let mut iq = [0 as Int];
    // SAFETY: workspace query; a is m x k (lda = m), bb is ldb x nrhs.
    unsafe {
        T::gelsd(
            int(m),
            int(k),
            int(nrhs),
            acopy.as_mut_slice().as_mut_ptr(),
            ld(m),
            bb.as_mut_slice().as_mut_ptr(),
            ld(ldb),
            s.as_mut_ptr(),
            rcond,
            &mut rank,
            wq.as_mut_ptr(),
            -1,
            rq.as_mut_ptr(),
            iq.as_mut_ptr(),
            &mut info,
        );
    }
    check("gelsd", info)?;
    let lw = lwork(wq[0]);
    let lrwork = (rq[0].ceil() as usize).max(lrwork_min);
    let liwork = (iq[0].max(0) as usize).max(liwork_min);
    let mut work = vec![T::zero(); lw];
    let mut rwork = vec![0.0f64; if T::IS_COMPLEX { lrwork } else { 0 }];
    let mut iwork = vec![0 as Int; liwork];
    unsafe {
        T::gelsd(
            int(m),
            int(k),
            int(nrhs),
            acopy.as_mut_slice().as_mut_ptr(),
            ld(m),
            bb.as_mut_slice().as_mut_ptr(),
            ld(ldb),
            s.as_mut_ptr(),
            rcond,
            &mut rank,
            work.as_mut_ptr(),
            int(lw),
            rwork.as_mut_ptr(),
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    check("gelsd", info)?;
    Ok(bb.rows(0..k))
}

/// Solve `R11 X = B` for upper-triangular `R11` (`k x k`) in place of `B` (trsm).
pub(crate) fn solve_upper<T: Scalar>(r11: &Mat<T>, mut b: Mat<T>) -> Mat<T> {
    let (k, nrhs) = b.shape();
    assert_eq!(r11.shape(), (k, k), "solve_upper: R11 must be k x k");
    if k == 0 || nrhs == 0 {
        return b;
    }
    // SAFETY: r11 is k x k (lda = k), b is k x nrhs (ldb = k).
    unsafe {
        T::trsm(
            b'L',
            b'U',
            b'N',
            b'N',
            int(k),
            int(nrhs),
            T::one(),
            r11.as_slice().as_ptr(),
            ld(k),
            b.as_mut_slice().as_mut_ptr(),
            ld(k),
        );
    }
    b
}

/// Dense LAPACK factorizations used by librla, exposed for reference
/// computations and tests.
pub mod dense {
    use super::*;

    /// `(Q, R, p)`: `A[:, p] = Q R`.
    pub type QrFactors<T> = (Mat<T>, Mat<T>, Vec<usize>);
    /// `(U, s, Vh)`: `A = U diag(s) Vh`.
    pub type SvdFactors<T> = (Mat<T>, Vec<f64>, Mat<T>);

    /// Economic QR with column pivoting (geqp3 + orgqr/ungqr, all columns free):
    /// `(Q, R, p)` with `A[:, p] = Q R`, `Q` `m x min(m,n)`, `R` `min(m,n) x n`
    /// upper triangular, `p` 0-based.
    pub fn qr_pivoted<T: Scalar>(a: &Mat<T>) -> Result<QrFactors<T>, Error> {
        let (q, r, p) = qr_piv(a.clone(), true)?;
        Ok((q.expect("Q requested"), r, p))
    }

    /// Economic SVD `(U, s, Vh)` with `A = U diag(s) Vh`, `s` descending
    /// (gesdd, falling back to gesvd).
    pub fn svd<T: Scalar>(a: &Mat<T>) -> Result<SvdFactors<T>, Error> {
        svd_econ(a)
    }
}
