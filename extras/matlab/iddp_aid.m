function [proj,krank,list] = iddp_aid(a, tol)
%
%       computes the ID of the matrix a, i.e., lists in list
%       the indices of krank columns of a such that
%
%       a(j,list(k))  =  a(j,list(k))
%
%       for all j = 1, ..., m; k = 1, ..., krank, and
%
%                        krank
%       a(j,list(k))  =  Sigma  a(j,list(l)) * proj(l,k-krank)       (*)
%                         l=1
%
%                     +  epsilon(j,k-krank)
%
%       for all j = 1, ..., m; k = krank+1, ..., n,
%
%       for some matrix epsilon dimensioned epsilon(m,n-krank)
%       such that the greatest singular value of epsilon
%       <= the greatest singular value of a * eps.
%
%       input:
%       eps -- precision to which the ID is to be computed
%       m -- first dimension of a
%       n -- second dimension of a
%       a -- matrix to be decomposed; the present routine does not
%            alter a
%       work -- initialization array that has been constructed
%               by routine idd_frmi
%
%       output:
%       krank -- numerical rank of a to precision eps
%       list -- indices of the columns in the ID
%       proj -- matrix of coefficients needed to interpolate
%               from the selected columns to the other columns
%               in the original matrix being ID'd;
%               proj doubles as a work array in the present routine, so
%               proj must be at least n*(2*n2+1)+n2+1 real*8 elements
%               long, where n2 is the greatest integer less than
%               or equal to m, such that n2 is a positive integer
%               power of two.
%
%       _N.B._: The algorithm used by this routine is randomized.
%               proj must be at least n*(2*n2+1)+n2+1 real*8 elements
%               long, where n2 is the greatest integer less than
%               or equal to m, such that n2 is a positive integer
%               power of two.
%
%       reference:
%       Halko, Martinsson, Tropp, "Finding structure with randomness:
%            probabilistic algorithms for constructing approximate
%            matrix decompositions," SIAM Review, 53 (2): 217-288,
%            2011.
%

if( nargin < 2 ), tol = 10*eps(); end

[m, n] = size(a);

n2 = zeros(1,1);
work = zeros(30*m+100,1);

mex_id_ = 'iddfrmi(c i int[x], c io int[x], c io double[])';
[n2, work] = libid(mex_id_, m, n2, work, 1, 1);

list = zeros(1,n);
krank = zeros(1,1);
proj = zeros(n*(2*n2+1)+n2+1,1);

mex_id_ = 'iddpaid(c i double[], c i int[x], c i int[x], c io double[], c io double[], c io int[x], c io int[], c io double[])';
[a, work, krank, list, proj] = libid(mex_id_, tol, m, n, a, work, krank, list, proj, 1, 1, 1);

proj = reshape(proj(1:krank*(n-krank)),krank,n-krank);




