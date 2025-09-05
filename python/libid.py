import numpy as np
from scipy import linalg
from numpy.linalg import norm


def range_randomized(A,rtol,block_size=42,flag_power=0):

    def _range_power(A,x,flag_power=0):

        for j in range(flag_power):
            x = A.T @ (A @ x)
            x, _R, _p = linalg.qr(x, mode='economic', pivoting=True)
        return x

    m, n = A.shape

    if (block_size >= min(m,n)):
        return min(m,n), np.empty_like(A, shape=(0, 0))
    
    while 1:
        x = 2*np.random.uniform(size=(n, block_size))-1
        x = _range_power(A,x,flag_power)
        y = A @ x
        Q, R, p = linalg.qr(y, mode='economic', pivoting=True)
        r = R.diagonal()
        d = max(abs(r[-1:]))/max(norm(y,axis=0))

        if (d <= rtol): 
            return block_size, Q

        if (d > rtol): 
            block_size = min(block_size*4,min(m,n))

        if (block_size >= min(m,n)):
            return min(m,n), np.empty_like(A, shape=(0, 0))


def rrqr_randomized(A,rtol,block_size=42,flag_power=0):

    m, n = A.shape
    k, q = range_randomized(A,rtol,block_size,flag_power)

    if (k >= min(m,n)):
        Q, R, p = linalg.qr(A, mode='economic', pivoting=True)
        k = sum(norm(R,axis=1) >= rtol*norm(A))
        return Q[:,:k],R[:k,:],p
  
    Aproj = q.T @ A
    Qproj, R, p = linalg.qr(Aproj, mode='economic', pivoting=True)   
    Q = q @ Qproj
    k = sum(norm(R,axis=1) >= rtol*norm(Aproj))
    return Q[:,:k],R[:k,:],p


def rrsvd_randomized(A,rtol,block_size=42,flag_power=0):

    m, n = A.shape
    k, q = range_randomized(A,rtol,block_size,flag_power)

    if (k >= min(m,n)):
        U, s, V = linalg.svd(A,full_matrices=False)
        k = sum(abs(s) >= rtol*norm(A))
        return U[:,:k],s[:k],V[:k,:]
  
    Aproj = q.T @ A
    Uproj, s, V = linalg.svd(Aproj,full_matrices=False)
    U = q @ Uproj
    k = sum(abs(s) >= rtol*norm(Aproj))
    return U[:,:k],s[:k],V[:k,:]


def rrid_randomized(A,rtol,block_size=42,flag_power=0):

    Q, R, p = rrqr_randomized(A,rtol,block_size,flag_power)
    k = R.shape[0]
    proj = linalg.solve(np.triu(R[:k,:k]), R[:,k:])
    return k, p, proj


def image_randomized(A,rtol,block_size=42,flag_power=0):
    return range_randomized(A.T,rtol,block_size,flag_power)

