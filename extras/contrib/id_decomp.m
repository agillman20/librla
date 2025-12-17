% Given an input matrix A, this function determines an index 
% vector I and a "coefficient matrix" T such that
%
%   A(:,I) \approx A(:,I(1:k))*[I,T]
%
% The function has two modes of operating:
%
% (1) If the second input is smaller than 1, then the function
%     determines the rank k adaptively to precision acc=INPUT2
%
% (2) If the second input is larger than of equal to 1, then it
%     sets k = INPUT2 
%
% In addition, there are two modes of computing:
%
% flag_mode='MAT'    This uses Matlab's built-in function qr.
%                    This is sometimes very fast due to the efficient implementation.
%                    It is sometimes very wasteful, though, since the full
%                    decomposition (to rank min(m,n)) is computed.
%
% flag_mode='PGS'    This forces column pivoted Gram-Schmidt to compute the skeleton.
%                    This usually gives better skeletons but can be slow.
%
% NOTE: If possible, a mex-file verion of "id_decomp" should be used!
%       The only purpose of the present routine is for convenience to avoid 
%       having to compile the mex-file (this is _very_ system dependent).

function [T, I] = id_decomp(A,INPUT2,flag_mode)

if (nargin == 2)
  flag_mode = 'MAT';
end

if (INPUT2 < 1) % Execute in "fixed precision mode"
  acc = INPUT2;
  if strcmp(flag_mode,'MAT')
    [T,I] = skel_col2(A,acc);
  else
    [T,I] = skel_col4(A,acc);
  end
else % Execute in "fixed rank mode"
  k = INPUT2;
  if strcmp(flag_mode,'MAT')
    [T,I] = skel_col3(A,k);
  else
    [T,I] = skel_col5(A,k);
  end
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [T, I] = skel_col2(A,acc)

% Compute the skeleton for a fixed accuracy using Matlab's built-in functions.

ss = svd(A);
k  = sum(ss > acc);
[tmp, R, I] = qr(A,0);

[U,D,V] = svd(R(1:k,1:k));
q = sum(diag(D) > acc);
T = V(:,1:q)*(D(1:q,1:q)\(U(:,1:q)'))*R(1:k, (k+1):size(R,2));

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [T, I] = skel_col3(A,k)

% Compute the skeleton for a fixed rank using Matlab's built-in functions.
% NOTE: The function HARDCODES the least-square solve precision to 1e-12.

[tmp, R, I] = qr(A,0);
[U,D,V] = svd(R(1:k,1:k));
q = sum(diag(D) > 1e-12);
T = V(:,1:q)*(D(1:q,1:q)\(U(:,1:q)'))*R(1:k, (k+1):size(R,2));

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [T, I] = skel_col4(A,acc)

% Compute the skeleton for a fixed accuracy using pivoted Gram-Schmidt.

[tmp, R, I] = pivoted_QR4(A,acc);
k = size(R,1);
[U,D,V] = svd(R(1:k,1:k));
q = sum(diag(D) > acc);
T = V(:,1:q)*(D(1:q,1:q)\(U(:,1:q)'))*R(1:k, (k+1):size(R,2));

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [T, I] = skel_col5(A,k)

% Compute the skeleton for a fixed rank using pivoted Gram-Schmidt.
% NOTE: The function HARDCODES the least-square solve precition to 1e-12.

[tmp, R, I] = pivoted_QR5(A,k);

[U,D,V] = svd(R(1:k,1:k));
q = sum(diag(D) > 1e-12);
T = V(:,1:q)*(D(1:q,1:q)\(U(:,1:q)'))*R(1:k, (k+1):size(R,2));

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Q,R,ind] = pivoted_QR4(A,acc)

% This function performs column pivoted Gram-Schmidt to construct a 
% QR-factorization of A.
%
%     A(:, ind) = Q*R    --> to accuracy 'acc'
%
% The factorization stops once ||R22||_{F} < acc.
%
% The "paranoid" reorthogonalization scheme of Bjorck is used.

m = size(A,1);
n = size(A,2);

R = zeros(min(m,n),n);
Q = A;
ind = 1:n;

k = -1;
for j = 1:min(m,n)
        
    % Find the pivots and break the iteration if required accuracy has been achieved
    piv_vec = conj(Q(:,j:n)).*Q(:,j:n);
    [tmp, j_max] = max(sum(piv_vec));
    j_max          = j_max + j - 1;
    if (sum(piv_vec) < acc*acc)
        k = j-1;
        break
    end
    
    % Swap the cols to place the pivot col in j'th position
    Q(:,[j,j_max]) = Q(:,[j_max,j]);
    R(:,[j,j_max]) = R(:,[j_max,j]);
    ind([j,j_max]) = ind([j_max,j]);
    
    % Compute the pivot
    r_jj   = norm(Q(:,j));
    R(j,j) = r_jj;
    Q(:,j) = Q(:,j)/r_jj;
    
    % Perform the "paranoid" reorthogonalization of Q(:,j) w.r.t. Q(:,1:(j-1))
    Q(:,j) = Q(:,j) - Q(:,1:(j-1))*(Q(:,1:(j-1))'*Q(:,j));
    Q(:,j) = Q(:,j)/norm(Q(:,j));

    % Orthonormalize Q(:,(j+1):n) w.r.t. Q(:,j)
    rr     = Q(:,j)'*Q(:,(j+1):n);
    R(j,(j+1):n) = rr;
    Q(:,(j+1):n) = Q(:,(j+1):n) - Q(:,j)*rr;
    
end

if (k == -1)
   Q = Q(:, 1:min(m,n));
else
   Q = Q(:, 1:k);
   R = R(1:k, :);
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Q,R,ind] = pivoted_QR5(A,k)

% This function performs column pivoted Gram-Schmidt to construct a 
% rank-k QR-factorization of A.
%
%     A(:, ind) = Q*R    
%
% where ind is of length k.
%
% The "paranoid" reorthogonalization scheme of Bjorck is used.

m = size(A,1);
n = size(A,2);

R = zeros(min(m,n),n);
Q = A;
ind = 1:n;

for j = 1:k
        
    % Find the pivot.
    piv_vec      = conj(Q(:,j:n)).*Q(:,j:n);
    [tmp, j_max] = max(sum(piv_vec));
    j_max        = j_max + j - 1;
    
    % Swap the cols to place the pivot col in j'th position
    Q(:,[j,j_max]) = Q(:,[j_max,j]);
    R(:,[j,j_max]) = R(:,[j_max,j]);
    ind([j,j_max]) = ind([j_max,j]);
    
    % Compute the pivot
    r_jj   = norm(Q(:,j));
    R(j,j) = r_jj;
    Q(:,j) = Q(:,j)/r_jj;
    
    % Perform the "paranoid" reorthogonalization of Q(:,j) w.r.t. Q(:,1:(j-1))
    Q(:,j) = Q(:,j) - Q(:,1:(j-1))*(Q(:,1:(j-1))'*Q(:,j));
    Q(:,j) = Q(:,j)/norm(Q(:,j));

    % Orthonormalize Q(:,(j+1):n) w.r.t. Q(:,j)
    rr     = Q(:,j)'*Q(:,(j+1):n);
    R(j,(j+1):n) = rr;
    Q(:,(j+1):n) = Q(:,(j+1):n) - Q(:,j)*rr;
    
end

Q = Q(:, 1:k);
R = R(1:k, :);

return
