import numpy as np
from numpy.linalg import norm

def  rrqr(A,rtol):
     """
  The traditional algorithm for the QR Factorization with column pivoting.
  [1] P. A. Businger and G. H. Golub, Linear least squares solution by
  Householder transformation, Numerische Mathematik, 7 (1965), pp. 269-276.

  Inputs:
      A(m,n) - matrix A, real or complex
      rtol - relative tolerance of QR decomposition    

  Outputs:
      Q(m,k) - orthogonal matrix Q
      R(k,n) - upper triangular matrix R
      p - column pivot (permutation) vector
      k - rank of QR decomposition
     """
     [m,n] = A.shape
     [tau,p,k,H] = rrqr_piv(A,rtol)
     print('tau = ', tau)
     I = np.eye(m,k)
     Q = rrqr_q(H,tau,I,k)
# need to investigate this triu in python
     R = np.triu(H[0:k,:])
     return (Q,R,p,k)

def rrqr_piv(a,rtol):
     """
%  The traditional algorithm for the QR Factorization with column pivoting.
%  [1] P. A. Businger and G. H. Golub, Linear least squares solution by
%  Householder transformation, Numerische Mathematik, 7 (1965), pp. 269-276.
%
%  Note: This version evaluates pivot norms directly (no downdating)
%
%  Inputs:
%      a(m,n) - matrix A, real or complex
%      rtol - relative tolerance of QR decomposition    
%
%  Outputs:
%      tau - the scalar factors of the elementary reflectors
%      p - column pivot (permutation) vector
%      k - rank of QR decomposition
%      a - truncated Householder QR factorization
     """
     [m,n] = a.shape
     nn = min(m,n)
     tau = np.zeros(nn)
     p = np.arange(0,n,1)
     s = norm(a,axis=0)
     d = norm(s)
     atol = rtol*d

     k = 0
   
     if( d == 0 ):
        tau = tau[0:k]
        return (tau,p, k,a)
    
     for j in range(1): #range(nn):

        jpiv = j
        
        spiv = s[j]
        for i in range(j,n):
            if( spiv < s[i] ):
                jpiv = i
                spiv = s[i]
        
#        print('spiv = ', spiv)

        if( jpiv != j ):
            for l in range(m):
                tmp = a[l,j]
                a[l,j] = a[l,jpiv]
                a[l,jpiv] = tmp
          
            itmp = p[j]
            p[j] = p[jpiv]
            p[jpiv] = itmp           
        
        v = a[j:m,j]
        [tau[j], v] = reflector(v)
        a[j:m,j] = v
        

        v2= reflectorApply2(v, tau[j], a[j:m,j+1:n])
        print('v2 = ', v2)
        print('norm(v2) = ', norm(v2))
        a[j:m,j+1:n] = reflectorApply2(v, tau[j], a[j:m,j+1:n])
        
        print('norm matrix = ', norm(a[j:m,j+1:n] ))
        
        for i in range(j+1,n):
                s[i] = norm(a[j+1:m,i])

        k = j
        if( norm(s[j+1:n]) < atol ):
            tau = tau[0:k]
            return(tau,p,k,a)

     tau = tau[1:k]
     """
	
     """
     return (tau, p, k,a) 
     
def reflectorApply2(x,tau,a):

     [m,n] = a.shape
     if (m==0): return
     y = x
     y[0] = 1
#     tmp = np.conj(y.T)@a
     print('np.conj(tau) = ', np.conj(tau))
     print(' yp *a = ', np.conj(y.T)@a)
     a = a-np.conj(tau)*np.outer(y,np.conj(y.T)@a)
     return a
     
def reflector(x):

     """
%
% julia-1.11/share/julia/stdlib/v1.11/LinearAlgebra/src
% Elementary reflection similar to LAPACK. The reflector is not Hermitian but
% ensures that tridiagonalization of Hermitian matrices become real. See lawn72
%
     """
     
     n = len(x)
     if (n==0): 
        tau = 0
        return(tau, x)
     xi = x[0]
     normu = norm(x)
     if (normu ==0): 
        tau = 0
        return(tau,x)
     nu = copysign_py(normu, xi.real)
     xi = xi + nu
     x[0] = -nu
     x[1:n] = x[1:n]/xi
     tau = xi/nu
     return [tau,x]      
     
     
def copysign_py(x,y):
     if (y>=0):
        xs = abs(x)    
     else: 
        xs = -abs(x)
     return xs

def  rrqr_q(a, tau, q, k):

     """
%  Inputs:
%      a(m,n) - truncated Householder QR factorization
%      p - column pivot (permutation) vector
%      k - rank of QR decomposition (or number of Q vector to be computed)
%      q - eye(m,k) identity matrix, if true Q vector are to be computed
%
%  Outputs:
%      q(m,k) - orthogonal matrix Q
%
     """
     [m,n] = q.shape
     for j in range(k,-1,1):
        v = a[j:m,j]
        for i in range(k,-1,j):
            qi = q[j:m,i]
            qi = reflectorApply(v, np.conj(tau[j]), qi)
            q[j:m,i] = qi
     return q

def reflectorApply(x,tau,A):

     """
%
% julia-1.11/share/julia/stdlib/v1.11/LinearAlgebra/src
% Multiplies `A` in-place by a Householder reflection on the left.
% It is equivalent to `A .= (I - conj(tau)*[1; x[2:end]]*[1; x[2:end]]')*A`.
%
    % BLAS level 1 update
     """
     [m,n] = A.shape
     if( m == 0 ): return    
     for j in range(0,n):
        vAj = np.conj(tau)*(A[1,j] + np.dot(x[1:m],A[1:m,j]))
        A[0,j] = A[0,j] - vAj
        A[1:m,j] = A[0:m,j] - vAj*x[2:m]
     return A

