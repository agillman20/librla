% This program implements several versions of the "randomized SVD",
% as described in Halko, Martinsson, Tropp [SIREV 2011].

function main_rsvd

%%% Initialize the random number generator.
rng(0)

%%% Set parameters.
%%% The matrix is of size m x n.
%%% The target rank is k.
%%% We over sample by an amount p.
m   = 1000;
n   =  800;
k   =  100;
p   =   20;

%%% Choose text matrix. Possible choices include:
%%% flag_matrix = 1  -> svds decay FAST, full matrices generated
%%% flag_matrix = 2  -> svds decay slowly, full matrices generated
%%% flag_matrix = 3  -> A(m,n) = 1/(m+n) + noise
%%% flag_matrix = 4  -> svds decay fast, partial U and V generated
%%% flag_matrix = 5  -> svds decay slowly, partial U and V generated
%%% NOTE: If min(m,n) is large, then options 1 and 2 are expensive!
flag_matrix = 3;

%%% Set how many steps of "power-iteration" should be used.
%%%    Y = (A * A')^flag_power * A * Omega
%%% (In other words, flag_power is the parameter sometimes called "q".)
flag_power  = 2;

%%% Generate the matrix.
fprintf(1,'Generating A ... ')
tic
A = LOCAL_get_matrix(m,n,k,flag_matrix);
fprintf(1,'time required = %10.3f\n',toc)

%%% If "A" is small enough, then compute full SVD.
if (min(m,n) <= 2000)
   tic
   [U,D,V] = svd(A,0);
   t_svd = toc;
   RES   = A - U(:,1:k)*D(1:k,1:k)*V(:,1:k)';
   E_max = max(max(abs(RES)));
   E_L2  = D(k+1,k+1);
   E_fro = sqrt(sum(sum(RES.*RES)));
   n_fro = sqrt(sum(sum(A.*A)));
   fprintf(1,'t_svd  = %10.5f    E_max = %12.5e     E_L2  = %12.5e     E_fro = %12.5e     E_fro_rel = %12.5e\n',...
           t_svd,E_max,E_L2,E_fro,E_fro/n_fro)
end

%%% Perform the "basic" randomized SVD.
tic
[U,D,V] = LOCAL_rsvd(A,k,p);
t_rsvd = toc;
RES   = A - U(:,1:k)*D(1:k,1:k)*V(:,1:k)';
E_max = max(max(abs(RES)));
if (min([m,n]) < 2000)
  E_L2  = norm(RES);
else
  E_L2  = NaN;
end
E_fro = sqrt(sum(sum(RES.*RES)));
n_fro = sqrt(sum(sum(A.*A)));
fprintf(1,'t_rsvd = %10.5f    E_max = %12.5e     E_L2  = %12.5e     E_fro = %12.5e     E_fro_rel = %12.5e\n',...
        t_rsvd,E_max,E_L2,E_fro,E_fro/n_fro)

%%% Perform the "power method" randomized SVD.
tic
[U,D,V] = LOCAL_psvd(A,k,p,flag_power);
t_psvd = toc;
RES   = A - U(:,1:k)*D(1:k,1:k)*V(:,1:k)';
E_max = max(max(abs(RES)));
if (min([m,n]) < 2000)
  E_L2  = norm(RES);
else
  E_L2  = NaN;
end
E_fro = sqrt(sum(sum(RES.*RES)));
n_fro = sqrt(sum(sum(A.*A)));
fprintf(1,'t_psvd = %10.5f    E_max = %12.5e     E_L2  = %12.5e     E_fro = %12.5e     E_fro_rel = %12.5e\n',...
        t_psvd,E_max,E_L2,E_fro,E_fro/n_fro)

if( ~is_octave() )
%%% If the system has a CUDA-enabled GPU, then run the GPU-rsvd.
n = gpuDeviceCount;
if (n > 0)
  tic
  [U,D,V] = LOCAL_gpusvd(A,k,p,flag_power);
  t_gsvd = toc;
  RES   = A - U(:,1:k)*D(1:k,1:k)*V(:,1:k)';
  E_max = max(max(abs(RES)));
  if (min([m,n]) < 2000)
    E_L2  = norm(RES);
  else
    E_L2  = NaN;
  end
  E_fro = sqrt(sum(sum(RES.*RES)));
  n_fro = sqrt(sum(sum(A.*A)));
  fprintf(1,'t_gsvd = %10.5f    E_max = %12.5e     E_L2  = %12.5e     E_fro = %12.5e     E_fro_rel = %12.5e\n',...
          t_gsvd,E_max,E_L2,E_fro,E_fro/n_fro)
end
end

keyboard
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the "basic" randomized sampling algorithm.
% It works well when the svds of A decay rapidly.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U,D,V] = LOCAL_rsvd(A,k,p)

n         = size(A,2);
ell       = k + p;
Omega     = randn(n,ell);
Y         = A*Omega;
[Q,~,~]   = qr(Y,0);
B         = Q'*A;
[UU,D,V]  = svd(B,'econ');
U         = Q*UU(:,1:k);
D         = D(1:k,1:k);
V         = V(:,1:k);

return

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

function [U,D,V] = LOCAL_psvd(A,k,p,flag_power)

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function executes the rsvd on the GPU.
% It can do power iteration if requested (with "full" reorthogonalization).
% Note that the "SVD" command on the GPU does not come with the "econ" option.
% For this reason, we manually perform QR on the "B" matrix before SVD.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [U,D,V] = LOCAL_gpusvd(A,k,p,flag_power)

n        = size(A,2);
ell      = k + p;
gA       = gpuArray(A);
gOmega   = gpuArray(randn(n,ell));
gY       = gA*gOmega;
[gQ,~]   = qr(gY,0);
for icount = 1:flag_power
  gY     = gA'*gQ;
  [gQ,~] = qr(gY,0);
  gY     = gA*gQ;
  [gQ,~] = qr(gY,0);
end
gB           = gQ'*gA;
[gQQ,gRR]    = qr(gB',0);
[gUU,gD,gVV] = svd(gRR');
U            = gather(gQ*gUU(:,1:k));
D            = gather(gD(1:k,1:k));
V            = gather(gQQ*gVV(:,1:k));

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = LOCAL_get_matrix(m,n,k,flag_matrix)

if (flag_matrix == 1)

  acc   = 1e-10;
  kbig  = min([m,n]);
  [U,~] = qr(randn(m,kbig),0);
  [V,~] = qr(randn(n,kbig),0);
  beta  = acc^(1/(k-1));
  ss    = beta.^(0:(kbig-1));
  ss    = ss.*(1.1 - 0.2*rand(1,kbig));
  A     = U*diag(ss)*V';

elseif (flag_matrix == 2)
  
  kbig  = min([m,n]);
  [U,~] = qr(randn(m,kbig),0);
  [V,~] = qr(randn(n,kbig),0);
  ss    = 1./((1:kbig).^0.25);
  ss    = ss.*(1.1 - 0.2*rand(1,kbig));
  A     = U*diag(ss)*V';

elseif (flag_matrix == 3)
  
  M   = (1./(1:m))' * ones(1,n);
  N   = ones(m,1)*(1./(1:n));
  A   = 1./(M+N);
  acc = 1e-7;
  A   = A.*(1 + acc*(1-2*rand(m,n)));

elseif (flag_matrix == 4)
  
  acc  = 1e-10;
  kbig = min([m,n,3*k]);
  U    = orth(randn(m,kbig));
  V    = orth(randn(n,kbig));
  beta = acc^(1/(k-1));
  ss   = beta.^(0:(kbig-1));
  ss   = ss.*(1.1 - 0.2*rand(1,kbig));
  A    = U*diag(ss)*V';

elseif (flag_matrix == 5)
  
  kbig = min([m,n,3*k]);
  U    = (1/sqrt(m))*randn(m,kbig);
  V    = (1/sqrt(n))*randn(n,kbig);
%  acc  = 1e-5;
%  beta = acc^(1/(k-1));
%  ss   = beta.^(0:(kbig-1));
%  ss   = ss.*(1.1 - 0.2*rand(1,kbig));
  ss   = 1./((1:kbig).^0.55);
  ss   = ss.*(1.1 - 0.2*rand(1,kbig));
  A    = U*diag(ss)*V';

end
  
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

