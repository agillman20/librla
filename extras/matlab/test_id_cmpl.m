n = 2000
m = 2*n
%%a = hilb(n)*(1+1i);
a = hilb(m,n)*(1+1i);
tol = 1e-15

tic; [p,k,cols,rnorms] = idzp_id(a, tol); toc
err = norm(a(:,cols(k+1:end)) - a(:,cols(1:k))*p,'fro')
k

a = hilb(n,m)*(1+1i);
tol = 1e-15

tic; [p,k,cols,rnorms] = idzp_id(a, tol); toc
err = norm(a(:,cols(k+1:end)) - a(:,cols(1:k))*p,'fro')
k
