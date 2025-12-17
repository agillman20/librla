
m = 4000
n = 2000

a = hilb(m,n);
tol = 1e-15

disp('== real, double ==')

tic; [T,I] = id_house_rand(a,tol); toc
k = size(T,1)

relerr = norm(a(:,I(k+1:n)) - a(:,I(1:k))*T,'fro')/norm(a,'fro')


disp('== complex, double ==')

a = hilb(m,n)*(1+2i);

tic; [T,I] = id_house_rand(a,tol); toc
k = size(T,1)

relerr = norm(a(:,I(k+1:n)) - a(:,I(1:k))*T,'fro')/norm(a,'fro')


