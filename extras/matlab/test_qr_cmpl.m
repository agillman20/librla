n = 2000
m = n*2
%%a = hilb(n)*(1+1i);
a = hilb(m,n)*(1+1i);
tol = 1e-12

tic
[out,k,cols] = idzp_qrpiv(a,tol);
toc
cols
ncols = length(cols)

tic
q = idz_qinqr(out,k);
r = triu(out);
ip = idz_permmult(k,cols,n);
p = 1:length(ip);
p(ip) = p;
toc

b = q*r;
norm(b(:,ip) - a,'fro')
norm(b - a(:,p),'fro')



a = hilb(n,m)*(1+1i);
tol = 1e-12

tic
[out,k,cols] = idzp_qrpiv(a,tol);
toc
cols
ncols = length(cols)

tic
q = idz_qinqr(out,k);
r = triu(out);
ip = idz_permmult(k,cols,m);
p = 1:length(ip);
p(ip) = p;
toc

b = q*r;
norm(b(:,ip) - a,'fro')
norm(b - a(:,p),'fro')
