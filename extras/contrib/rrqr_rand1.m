function [Q, R, p, k] = rrqr_rand1(A,rtol)
    [m,n] = size(A);    
    block_size = 42;
%%   t1 = tic;
    for i = 1:20
%%	x = 2*rand(n,block_size)-1;
	x = rand(n,block_size)-0.5;
%%	x = randn(n,block_size);
%%	x = x ./ vecnorm(x);
	y = A*x;
	[Q, R, ~] = qr(y,0);

%%      randomized rank projection, rows of R approximates projector error
%%	vecnorm(R,2,2)
%%      randomized rank projection, diagonal of R approximates projector error
%%	diag(R(1:end,1:end))
%%	diag(R(1:end,1:end))./max(vecnorm(y))

%%	d = norm(R(end,:),'fro')/norm(A,'fro') / 10;
%%	d = norm(R(end,:),'fro')/norm(R(1,:));
%%	d = norm(R(end,:),'fro')/max(vecnorm(y));
	d = norm(R(end,end),'fro')/max(vecnorm(y));

	if( d > rtol )
	    block_size = min(block_size*4,min(m,n))
%%	    block_size = min(block_size+8,min(m,n))
	else	    
	    break;
	end
	if( block_size == min(m,n) )
            [Q,R,p] = qr(A,0);
            k = block_size;
            return
	end
    end
%%    toc(t1)
%%    t1 = tic;
    q = Q;
    r = q'*A;
    [Q,R,p] = qr(r,0);

    k = size(R,1);
    
    % estimate rank
    %%log10(vecnorm(R,2,2))
    %%vecnorm(R,2,2) >= rtol*norm(r,'fro')
    k = sum(vecnorm(R,2,2) >= rtol*norm(r,'fro'));

    Q = q*Q(:,1:k);
    R = R(1:k,:);

%%    Q = q*Q;
%%    toc(t1)
end
