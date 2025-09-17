function H = hilb(m,n,method)
%HILB  Construct an m‑by‑n Hilbert matrix.
%
%   H = HILB(m)      returns the square m‑by‑m Hilbert matrix.
%   H = HILB(m,n)    returns the rectangular m‑by‑n Hilbert matrix.
%   H = HILB(m,n,method)  selects the construction method:
%        'vectorized' – default, uses implicit expansion.
%        'hankel'     – builds the matrix via the built‑in hankel().
%        'loops'      – explicit double‑for‑loop (educational only).
%
%   The (i,j)‑entry of a Hilbert matrix is
%          H(i,j) = 1/(i + j - 1)
%
%   Example
%       H = hilb(5)          % 5‑by‑5 Hilbert matrix
%       H = hilb(3,6)        % 3‑by‑6 Hilbert matrix
%       H = hilb(4,4,'hankel')
%
%   See also hankel, toeplitz, gallery.
    
%-----------------------------------------------------------------------
% Argument handling
%-----------------------------------------------------------------------
if nargin < 2               % only one size supplied → square matrix
    n = m;
end

if nargin < 3               % default construction method
    method = 'vectorized';
end

% -------------------------------------------------------------------------
% 1) Vectorized construction (default)
% -------------------------------------------------------------------------
if strcmp(method,'vectorized')
    % Row indices as a column vector (m‑by‑1)
    i = (1:m).';
    % Column indices as a row vector (1‑by‑n)
    j = 1:n;
    % Implicit expansion gives the required m‑by‑n matrix
    H = 1./(i + j - 1);
    return
end

% -------------------------------------------------------------------------
% 2) Using the built‑in HANKEL function
% -------------------------------------------------------------------------
if strcmp(method,'hankel')
    c = 1./(1:m);               % first column
    r = 1./(m + (1:n) - 1);     % last row
    H = hankel(c,r);            % MATLAB built-in hankel
    return
end

% -------------------------------------------------------------------------
% 3) Explicit double‑loop construction (slowest)
% -------------------------------------------------------------------------
if strcmp(method,'loops')
    H = zeros(m,n);             % pre‑allocate for speed
    for i = 1:m                 % rows
        for j = 1:n             % columns
            H(i,j) = 1/(i + j - 1);
        end
    end
    return
end

end
