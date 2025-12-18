function q = idd_qinqr(out,krank)
%
%       constructs the matrix q from iddp_qrpiv or iddr_qrpiv
%       (see the routine iddp_qrpiv or iddr_qrpiv
%       for more information).
%
%       input:
%       m -- first dimension of a; also, right now, q is m x m
%       n -- second dimension of a
%       a -- matrix output by iddp_qrpiv or iddr_qrpiv
%            (and denoted the same there)
%       krank -- numerical rank output by iddp_qrpiv or iddr_qrpiv
%                (and denoted the same there)
%
%       output:
%       q -- orthogonal matrix implicitly specified by the data in a
%            from iddp_qrpiv or iddr_qrpiv
%
%       Note:
%       Right now, this routine simply multiplies
%       one after another the krank Householder matrices
%       in the full QR decomposition of a,
%       in order to obtain the complete m x m Q factor in the QR.
%       This routine should instead use the following 
%       (more elaborate but more efficient) scheme
%       to construct a q dimensioned q(krank,m); this scheme
%       was introduced by Robert Schreiber and Charles Van Loan
%       in "A Storage-Efficient _WY_ Representation
%       for Products of Householder Transformations,"
%       _SIAM Journal on Scientific and Statistical Computing_,
%       Vol. 10, No. 1, pp. 53-57, January, 1989:
%
%       Theorem 1. Suppose that Q = _1_ + YTY^T is
%       an m x m orthogonal real matrix,
%       where Y is an m x k real matrix
%       and T is a k x k upper triangular real matrix.
%       Suppose also that P = _1_ - 2 v v^T is
%       a real Householder matrix and Q_+ = QP,
%       where v is an m x 1 real vector,
%       normalized so that v^T v = 1.
%       Then, Q_+ = _1_ + Y_+ T_+ Y_+^T,
%       where Y_+ = (Y v) is the m x (k+1) matrix
%       formed by adjoining v to the right of Y,
%                 ( T   z )
%       and T_+ = (       ) is
%                 ( 0  -2 )
%       the (k+1) x (k+1) upper triangular matrix
%       formed by adjoining z to the right of T
%       and the vector (0 ... 0 -2) with k zeroes below (T z),
%       where z = -2 T Y^T v.
%
%       Now, suppose that A is a (rank-deficient) matrix
%       whose complete QR decomposition has
%       the blockwise partioned form
%           ( Q_11 Q_12 ) ( R_11 R_12 )   ( Q_11 )
%       A = (           ) (           ) = (      ) (R_11 R_12).
%           ( Q_21 Q_22 ) (  0    0   )   ( Q_21 )
%       Then, the only blocks of the orthogonal factor
%       in the above QR decomposition of A that matter are
%                                                        ( Q_11 )
%       Q_11 and Q_21, _i.e._, only the block of columns (      )
%                                                        ( Q_21 )
%       interests us.
%       Suppose in addition that Q_11 is a k x k matrix,
%       Q_21 is an (m-k) x k matrix, and that
%       ( Q_11 Q_12 )
%       (           ) = _1_ + YTY^T, as in Theorem 1 above.
%       ( Q_21 Q_22 )
%       Then, Q_11 = _1_ + Y_1 T Y_1^T
%       and Q_21 = Y_2 T Y_1^T,
%       where Y_1 is the k x k matrix and Y_2 is the (m-k) x k matrix
%                   ( Y_1 )
%       so that Y = (     ).
%                   ( Y_2 )
%
%       So, you can calculate T and Y via the above recursions,
%       and then use these to compute the desired Q_11 and Q_21.
%
%

[m, n] = size(out);
q = zeros(m,m);

mex_id_ = 'iddqinqr(c i int[x], c i int[x], c io double[], c i int[x], c io double[])';
[out, q] = libid(mex_id_, m, n, out, krank, q, 1, 1, 1);


