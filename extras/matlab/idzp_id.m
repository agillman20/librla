function [proj,krank,list,rnorms] = idzp_id(a, tol)
%
%       computes the ID of a, i.e., lists in list the indices
%       of krank columns of a such that 
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
%       The present routine stores the krank x (n-krank) matrix proj
%       in the memory initially occupied by a.
%
%       input:
%       tol -- relative precision of the resulting ID
%       m -- first dimension of a
%       n -- second dimension of a, as well as the dimension required
%            of list
%       a -- matrix to be ID'd
%
%       output:
%       a -- the first krank*(n-krank) elements of a constitute
%            the krank x (n-krank) interpolation matrix proj
%       krank -- numerical rank
%       list -- list of the indices of the krank columns of a
%               through which the other columns of a are expressed;
%               also, list describes the permutation of proj
%               required to reconstruct a as indicated in (*) above
%       rnorms -- absolute values of the entries on the diagonal
%                 of the triangular matrix used to compute the ID
%                 (these may be used to check the stability of the ID)
%
%       _N.B._: This routine changes a.
%
%       reference:
%       Cheng, Gimbutas, Martinsson, Rokhlin, "On the compression of
%            low-rank matrices," SIAM Journal on Scientific Computing,
%            26 (4): 1389-1404, 2005.
%

if( nargin < 2 ), tol = 10*eps(); end

[m, n] = size(a);

list = zeros(1,n);
krank = zeros(1,1);
rnorms = zeros(1,n);

mex_id_ = 'idzpid(c i double[], c i int[x], c i int[x], c io dcomplex[], c io int[x], c io int[], c io double[])';
[a, krank, list, rnorms] = libid(mex_id_, tol, m, n, a, krank, list, rnorms, 1, 1, 1);

proj = reshape(a(1:krank*(n-krank)),krank,n-krank);


