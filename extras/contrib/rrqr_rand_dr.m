
m = 4000
n = 2000

a = hilb(m,n);
tol = 1e-15

disp('== real, double ==')

tic; [q,r,p] = rrqr_rand(a,tol); toc
k = size(r,1)

relerr = norm(q*r - a(:,p),'fro')/norm(a,'fro')

disp('== complex, double ==')

a = hilb(m,n)*(1+2i);
tol = 1e-15;

tic; [q,r,p] = rrqr_rand(a,tol); toc
k = size(r,1)

relerr = norm(q*r - a(:,p),'fro')/norm(a,'fro')


