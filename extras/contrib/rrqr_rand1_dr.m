
if_post = 0
m = 4000
n = 2000

a = hilb(m,n);
tol = 1e-15

disp('== real, double ==')

tic; [q,r,p] = rrqr_rand1(a,tol); toc
k = size(r,1)

relerr = norm(q*r - a(:,p),'fro')/norm(a,'fro')

%% projector residuals, post
%%log10(vecnorm(r,2,2))

if( if_post == 1 )
'post-processing'
tic; [Q,R,pp] = rrqr(r,tol); toc
%%tic; [Q,R,pp] = qr(r,0); toc
k = size(R,1)

%% pp should not permute in this test, permute_flag = 0
if( any(diff(pp) ~= 1 ) )
    post_permute_flag = true
else
    post_permute_flag = false
end

%% pp should not permute in this test
relerr = norm(q*Q*R(:,pp) - a(:,p),'fro')/norm(a,'fro')

%% projector residuals, post
%%log10(vecnorm(R,2,2))
end

disp('== complex, double ==')

a = hilb(m,n)*(1+2i);
tol = 1e-15;

tic; [q,r,p] = rrqr_rand1(a,tol); toc
k = size(r,1)

relerr = norm(q*r - a(:,p),'fro')/norm(a,'fro')

%% projector residuals
%%log10(vecnorm(r,2,2))

if( if_post == 1 )
'post-processing'
tic; [Q,R,pp] = rrqr(r,tol); toc
%%tic; [Q,R,pp] = qr(r,0); toc
k = size(R,1)

%% pp should not permute in this test, permute_flag = 0
if( any(diff(pp) ~= 1 ) )
    post_permute_flag = true
else
    post_permute_flag = false
end

%% pp should not permute in this test
relerr = norm(q*Q*R(:,pp) - a(:,p),'fro')/norm(a,'fro')

%% projector residuals, post
%%log10(vecnorm(R,2,2))
end
