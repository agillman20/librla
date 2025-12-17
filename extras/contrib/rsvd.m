%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the "basic" randomized sampling algorithm.
% It works well when the svds of A decay rapidly.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U,D,V] = rsvd(A,k,p)

n         = size(A,2);
ell       = k + p;
Omega     = randn(n,ell);
Y         = A*Omega;
[Q,~,~]   = qr(Y,'econ');
B         = Q'*A;
[UU,D,V]  = svd(B,'econ');
U         = Q*UU(:,1:k);
D         = D(1:k,1:k);
V         = V(:,1:k);

return

