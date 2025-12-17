%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = get_matrix(m,n,k,flag_matrix)
%%% Choose text matrix. Possible choices include:
%%% flag_matrix = 1  -> svds decay FAST, full matrices generated
%%% flag_matrix = 2  -> svds decay slowly, full matrices generated
%%% flag_matrix = 3  -> A(m,n) = 1/(m+n) + noise
%%% flag_matrix = 4  -> svds decay fast, partial U and V generated
%%% flag_matrix = 5  -> svds decay slowly, partial U and V generated
%%% NOTE: If min(m,n) is large, then options 1 and 2 are expensive!

if (flag_matrix == 1)

  acc   = 1e-10;
  kbig  = min([m,n]);
  [U,~] = qr(randn(m,kbig),0);
  [V,~] = qr(randn(n,kbig),0);
  beta  = acc^(1/(k-1));
  ss    = beta.^(0:(kbig-1));
  ss    = ss.*(1.1 - 0.2*rand(1,kbig));
  A     = U*diag(ss)*V';

elseif (flag_matrix == 2)
  
  kbig  = min([m,n]);
  [U,~] = qr(randn(m,kbig),0);
  [V,~] = qr(randn(n,kbig),0);
  ss    = 1./((1:kbig).^0.25);
  ss    = ss.*(1.1 - 0.2*rand(1,kbig));
  A     = U*diag(ss)*V';

elseif (flag_matrix == 3)
  
  M   = (1./(1:m))' * ones(1,n);
  N   = ones(m,1)*(1./(1:n));
  A   = 1./(M+N);
  acc = 1e-7;
  A   = A.*(1 + acc*(1-2*rand(m,n)));

elseif (flag_matrix == 4)
  
  acc  = 1e-10;
  kbig = min([m,n,3*k]);
  U    = orth(randn(m,kbig));
  V    = orth(randn(n,kbig));
  beta = acc^(1/(k-1));
  ss   = beta.^(0:(kbig-1));
  ss   = ss.*(1.1 - 0.2*rand(1,kbig));
  A    = U*diag(ss)*V';

elseif (flag_matrix == 5)
  
  kbig = min([m,n,3*k]);
  U    = (1/sqrt(m))*randn(m,kbig);
  V    = (1/sqrt(n))*randn(n,kbig);
%  acc  = 1e-5;
%  beta = acc^(1/(k-1));
%  ss   = beta.^(0:(kbig-1));
%  ss   = ss.*(1.1 - 0.2*rand(1,kbig));
  ss   = 1./((1:kbig).^0.55);
  ss   = ss.*(1.1 - 0.2*rand(1,kbig));
  A    = U*diag(ss)*V';

end
  
return
