%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the "power method" randomized sampling algorithm.
% The input parameter "flag_power" indicates nr of steps in power iteration.
%    Y = (A * A')^flag_power * A * Omega
% There is an internal tuning parameter "flag_reorth" that indicates
% how often re-orthogonalization should be done.
%    flag_reort = 0 -> never reorthogonalize
%    flag_reort = 1 -> reorthogonalize between EVERY step
%    flag_reort = 2*j -> reorthogonalize EVERY "2j" steps (must be even)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U,D,V] = psvd(A,k,p,flag_power)

flag_reorth = 2;
n           = size(A,2);
ell         = k + p;
Omega       = randn(n,ell);
Y           = A*Omega;
if (flag_reorth == 0) % Never orthonormalize.
  for icount = 1:flag_power
    Y = A'*Y;
    Y = A*Y;
  end
  [Q,~,~] = qr(Y,0);
elseif (flag_reorth == 1) % Orthonormalize after every step.
  [Q,~,~]   = qr(Y,0);
  for icount = 1:flag_power
    Y = A'*Q;
    [Q,~,~] = qr(Y,0);
    Y = A*Q;
    [Q,~,~] = qr(Y,0);
  end
elseif (mod(flag_reorth,2) == 1)
  fprintf(1,'The only admissible odd value for "flag_reorth" is 1.\n')
  keyboard
else % Orthonormalize occasionally.
  for icount = 1:flag_power
    Y = A'*Y;
    if (mod(icount,2) == round(flag_reorth/2))
      [Q,~,~]   = qr(Y,0);
      Y = A*Q;
    else
      Y = A*Y;
    end
  end
  [Q,~,~] = qr(Y,0);
end
B         = Q'*A;
[UU,D,V]  = svd(B,'econ');
U         = Q*UU(:,1:k);
D         = D(1:k,1:k);
V         = V(:,1:k);

return
