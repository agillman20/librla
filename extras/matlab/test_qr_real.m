n = 2000
m = n*2
%%a = hilb(n);
a = hilb(m,n);
tol = 1e-12

tic
[out,k,cols] = iddp_qrpiv(a,tol);
toc
cols
ncols = length(cols)

tic
q = idd_qinqr(out,k);
r = triu(out);
ip = idd_permmult(k,cols,n);
p = 1:length(ip);
p(ip) = p;
toc

b = q*r;
norm(b(:,ip) - a,'fro')
norm(b - a(:,p),'fro')


a = hilb(n,m);
tol = 1e-12

tic
[out,k,cols] = iddp_qrpiv(a,tol);
toc
cols
ncols = length(cols)

tic
q = idd_qinqr(out,k);
r = triu(out);
ip = idd_permmult(k,cols,m);
p = 1:length(ip);
p(ip) = p;
toc

b = q*r;
norm(b(:,ip) - a,'fro')
norm(b - a(:,p),'fro')
