function [out,krank,list,rnorms] = idzr_qrpiv(a)
%
%       computes the pivoted QR decomposition
%       of the matrix input into a, using Householder transformations,
%       _i.e._, transforms the matrix a from its input value in
%       to the matrix out with entry
%
%                               m
%       out(j,indprod(k))  =  Sigma  q(l,j) * in(l,k),
%                              l=1
%
%       for all j = 1, ..., krank, and k = 1, ..., n,
%
%       where in = the a from before the routine runs,
%       out = the a from after the routine runs,
%       out(j,k) = 0 when j > k (so that out is triangular),
%       q(1:m,1), ..., q(1:m,krank) are orthonormal,
%       indprod is the product of the permutations given by ind,
%       (as computable via the routine permmult,
%       with the permutation swapping 1 and ind(1) taken leftmost
%       in the product, that swapping 2 and ind(2) taken next leftmost,
%       ..., that swapping krank and ind(krank) taken rightmost),
%       and with the matrix out satisfying
%
%                  min(krank,m,n)
%       in(j,k)  =     Sigma      q(j,l) * out(l,indprod(k))
%                       l=1
%
%                +  epsilon(j,k),
%
%       for all j = 1, ..., m, and k = 1, ..., n,
%
%       for some matrix epsilon whose norm is (hopefully) minimized
%       by the pivoting procedure.
%       Well, technically, this routine outputs the Householder vectors
%       (or, rather, their second through last entries)
%       in the part of a that is supposed to get zeroed, that is,
%       in a(j,k) with m >= j > k >= 1.
%
%       input:
%       m -- first dimension of a and q
%       n -- second dimension of a
%       a -- matrix whose QR decomposition gets computed
%       krank -- desired rank of the output matrix
%                (please note that if krank > m or krank > n,
%                then the rank of the output matrix will be
%                less than krank)
%
%       output:
%       a -- triangular (R) factor in the QR decompositon
%            of the matrix input into the same storage locations, 
%            with the Householder vectors stored in the part of a
%            that would otherwise consist entirely of zeroes, that is,
%            in a(j,k) with m >= j > k >= 1
%       ind(k) -- index of the k^th pivot vector;
%                 the following code segment will correctly rearrange
%                 the product b of q and the upper triangle of out
%                 so that b best matches the input matrix in:
%
%                 copy the non-rearranged product of q and out into b
%                 set k to krank
%                 [start of loop]
%                   swap b(1:m,k) and b(1:m,ind(k))
%                   decrement k by 1
%                 if k > 0, then go to [start of loop]
%
%       work:
%       ss -- must be at least n real*8 words long
%
%       _N.B._: This routine outputs the Householder vectors
%       (or, rather, their second through last entries)
%       in the part of a that is supposed to get zeroed, that is,
%       in a(j,k) with m >= j > k >= 1.
%
%       reference:
%       Golub and Van Loan, "Matrix Computations," 3rd edition,
%            Johns Hopkins University Press, 1996, Chapter 5.
%
%
%       computes the pivoted QR decomposition
%       of the matrix input into a, using Householder transformations,
%       _i.e._, transforms the matrix a from its input value in
%       to the matrix out with entry
%
%                               m
%       out(j,indprod(k))  =  Sigma  q(l,j) * in(l,k),
%                              l=1
%
%       for all j = 1, ..., krank, and k = 1, ..., n,
%
%       where in = the a from before the routine runs,
%       out = the a from after the routine runs,
%       out(j,k) = 0 when j > k (so that out is triangular),
%       q(1:m,1), ..., q(1:m,krank) are orthonormal,
%       indprod is the product of the permutations given by ind,
%       (as computable via the routine permmult,
%       with the permutation swapping 1 and ind(1) taken leftmost
%       in the product, that swapping 2 and ind(2) taken next leftmost,
%       ..., that swapping krank and ind(krank) taken rightmost),
%       and with the matrix out satisfying
%
%                  min(krank,m,n)
%       in(j,k)  =     Sigma      q(j,l) * out(l,indprod(k))
%                       l=1
%
%                +  epsilon(j,k),
%
%       for all j = 1, ..., m, and k = 1, ..., n,
%
%       for some matrix epsilon whose norm is (hopefully) minimized
%       by the pivoting procedure.
%       Well, technically, this routine outputs the Householder vectors
%       (or, rather, their second through last entries)
%       in the part of a that is supposed to get zeroed, that is,
%       in a(j,k) with m >= j > k >= 1.
%
%       input:
%       m -- first dimension of a and q
%       n -- second dimension of a
%       a -- matrix whose QR decomposition gets computed
%       krank -- desired rank of the output matrix
%                (please note that if krank > m or krank > n,
%                then the rank of the output matrix will be
%                less than krank)
%
%       output:
%       a -- triangular (R) factor in the QR decompositon
%            of the matrix input into the same storage locations, 
%            with the Householder vectors stored in the part of a
%            that would otherwise consist entirely of zeroes, that is,
%            in a(j,k) with m >= j > k >= 1
%       ind(k) -- index of the k^th pivot vector;
%                 the following code segment will correctly rearrange
%                 the product b of q and the upper triangle of out
%                 so that b best matches the input matrix in:
%
%                 copy the non-rearranged product of q and out into b
%                 set k to krank
%                 [start of loop]
%                   swap b(1:m,k) and b(1:m,ind(k))
%                   decrement k by 1
%                 if k > 0, then go to [start of loop]
%
%       work:
%       ss -- must be at least n real*8 words long
%
%       _N.B._: This routine outputs the Householder vectors
%       (or, rather, their second through last entries)
%       in the part of a that is supposed to get zeroed, that is,
%       in a(j,k) with m >= j > k >= 1.
%
%       reference:
%       Golub and Van Loan, "Matrix Computations," 3rd edition,
%            Johns Hopkins University Press, 1996, Chapter 5.
%

if( nargin < 2 ), tol = 10*eps(); end

[m, n] = size(a);

list = zeros(1,n);
krank = zeros(1,1);
rnorms = zeros(1,n);

mex_id_ = 'idzrqrpiv(c i int[x], c i int[x], c io dcomplex[], c io int[x], c io int[], c io double[])';
[a, krank, list, rnorms] = libid(mex_id_, m, n, a, krank, list, rnorms, 1, 1, 1);

out = a;
list = list(1:krank);


