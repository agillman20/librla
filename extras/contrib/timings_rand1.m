% This script tests the function "id_decomp.m"

function timings_rand1()

rng('default')
rng(0)
m   = 4000;
n   = 2000;
k   = 31;
acc = 1e-12;
flag_complex = 0;
A   = LOCAL_rankdefmatrix(m, n, k, acc, flag_complex);
%%%A   = hilb(m,n); if( flag_complex==1 ), A = complex(A)*(1+2i); end

size(A)

fprintf(1,'\n=== id_decomp with fixed accuracy:\n')
tic
[T,I]   = id_decomp(A,acc);
t       = toc;
k       = size(T,1);
fprintf(1,'Time consumed    = %6.3f (sec) and k = %d\n', t, k)
fprintf(1,'Error            = %12.5e\n', max(max(abs(A(:,I((k+1):n)) - A(:,I(1:k))*T))))
fprintf(1,'max(max(abs(T))) = %12.5e\n', max(max(abs(T))))

fprintf(1,'\n=== id_decomp with fixed rank:\n')
tic
[T,I]   = id_decomp(A,k);
t       = toc;
k       = size(T,1);
fprintf(1,'Time consumed    = %6.3f (sec) and k = %d\n', t, k)
fprintf(1,'Error            = %12.5e\n', max(max(abs(A(:,I((k+1):n)) - A(:,I(1:k))*T))))
fprintf(1,'max(max(abs(T))) = %12.5e\n', max(max(abs(T))))

fprintf(1,'\n=== id_decomp with fixed accuracy (forcing Gram-Schmidt):\n')
tic
[T,I]   = id_decomp(A,acc,'PGS');
t       = toc;
k       = size(T,1);
fprintf(1,'Time consumed    = %6.3f (sec) and k = %d\n', t, k)
fprintf(1,'Error            = %12.5e\n', max(max(abs(A(:,I((k+1):n)) - A(:,I(1:k))*T))))
fprintf(1,'max(max(abs(T))) = %12.5e\n', max(max(abs(T))))

fprintf(1,'\n=== id_decomp with fixed rank (forcing Gram-Schmidt):\n')
tic
[T,I]   = id_decomp(A,k,'PGS');
t       = toc;
k       = size(T,1);
fprintf(1,'Time consumed    = %6.3f (sec) and k = %d\n', t, k)
fprintf(1,'Error            = %12.5e\n', max(max(abs(A(:,I((k+1):n)) - A(:,I(1:k))*T))))
fprintf(1,'max(max(abs(T))) = %12.5e\n', max(max(abs(T))))

fprintf(1,'\n=== id_house with fixed accuracy (forcing random Householder):\n')
tic
[T,I]   = id_house_rand1(A,acc);
t       = toc;
k       = size(T,1);
fprintf(1,'Time consumed    = %6.3f (sec) and k = %d\n', t, k)
fprintf(1,'Error            = %12.5e\n', max(max(abs(A(:,I((k+1):n)) - A(:,I(1:k))*T))))
fprintf(1,'max(max(abs(T))) = %12.5e\n', max(max(abs(T))))


return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = LOCAL_rankdefmatrix(m, n, k, acc, flag_complex)

% This function generates a matrix of size m x n
% whose rank to accuracy acc is k

ii      = 1:min(m,n);
log_dd  = max((ii-1)*log10(acc)/(k-1), -15);
dd      = 10.^log_dd;
if (flag_complex == 0)
   [U,tmp] = qr(rand(m,min(m,n)),0);
   [V,tmp] = qr(rand(n,min(m,n)),0);
else
   [U,tmp] = qr(rand(m,min(m,n)) + i*rand(m, min(m,n)),0);
   [V,tmp] = qr(rand(n,min(m,n)) + i*rand(n, min(m,n)),0);
end
A       = U*diag(dd)*V';

return
