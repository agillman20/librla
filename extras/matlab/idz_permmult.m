function indprod = idz_permmult(m,ind,n)
%
%       multiplies together the series of permutations in ind.
%
%       input:
%       m -- length of ind
%       ind(k) -- number of the slot with which to swap
%                 the k^th slot
%       n -- length of indprod and indprodinv
%
%       output:
%       indprod -- product of the permutations in ind,
%                  with the permutation swapping 1 and ind(1)
%                  taken leftmost in the product,
%                  that swapping 2 and ind(2) taken next leftmost,
%                  ..., that swapping krank and ind(krank)
%                  taken rightmost; indprod(k) is the number
%                  of the slot with which to swap the k^th slot
%                  in the product permutation

indprod = zeros(1,n);

mex_id_ = 'idzpermmult(c i int[x], c i int[], c i int[x], c io int[])';
[indprod] = libid(mex_id_, m, ind, n, indprod, 1, 1);


