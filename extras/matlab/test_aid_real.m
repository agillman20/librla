n = 2000
m = n*2
%%a = hilb(n);
a = hilb(m,n);
tol = 1e-12

tic; [p,k,cols] = iddp_aid(a, tol); toc
err = norm(a(:,cols(k+1:end)) - a(:,cols(1:k))*p,'fro')
k

a = hilb(n,m);
tol = 1e-12

tic; [p,k,cols] = iddp_aid(a, tol); toc
err = norm(a(:,cols(k+1:end)) - a(:,cols(1:k))*p,'fro')
k
