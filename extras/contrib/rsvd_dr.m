
m = 4000
n = 2000

a = hilb(m,n);
tol = 1e-15

disp('== real, double ==')

tic; [U,S,V] = rsvd(a,42,5); toc
k = size(S,1)

relerr = norm(U*S*V' - a,'fro')/norm(a,'fro')

disp('== complex, double ==')

a = hilb(m,n)*(1+2i);
tol = 1e-15;

tic; [U,S,V] = rsvd(a,42,5); toc
k = size(S,1)

relerr = norm(U*S*V' - a,'fro')/norm(a,'fro')

