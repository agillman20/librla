import numpy as np
from scipy import linalg
from numpy.linalg import norm



"""
(local) $ ollama run gpt-oss:120b "Summarize file: $(cat libid.py)"

### TL;DR
The file provides a **fast, adaptive RRQR factorisation** that first 
builds a randomised low‑dimensional basis for the column space of a 
matrix, then refines it with a pivoted QR to obtain a numerically 
rank‑revealing decomposition `A ≈ Q·R·Pᵀ`. It returns the orthogonal 
factor `Q`, the upper‑triangular factor `R`, and the column‑pivot 
permutation `p`, with the numerical rank automatically inferred from the 
user‑specified tolerance `rtol`.
"""

"""
(local) $ ollama run gpt-oss:120b "Summarize file: $(cat libid.py)"

### TL;DR (as in the original docstring)
The module supplies **fast, adaptive randomized RRQR and RR‑SVD**.  
1. Build a cheap random basis for the column space of `A`.  
2. Project `A` onto that basis, perform a standard pivoted QR (or SVD) on 
the tiny projected matrix, and lift the result back.  
3. Return the orthogonal factor, the triangular (or diagonal) factor, and 
the column permutation, with the numerical rank inferred from a 
user‑supplied relative tolerance `rtol`.
"""

"""
(local) $ ollama run gpt-oss:120b "Summarize file: $(cat libid.py)"

`libid.py` implements cheap, adaptive randomised versions of rank<E2><80><91>revealing QR, SVD and Interpolative Decomposition that first sketch the column space of a matrix, then perform a standard factorisation on the tiny sketch and lift the result back, automatically returning the numerical rank based on a user<E2><80><91>specified relative tolerance.
"""


def range_randomized(A,rtol,block_size=42):

    [m,n] = np.shape(A)

    if (block_size >= min(m,n)):
        return min(m,n), []
    
    for i in range(20):
        x = 2*np.random.uniform(size=(n, block_size))-1
        y = A @ x
        [Q,R,p] = linalg.qr(y, mode='economic', pivoting=True)
        r = R.diagonal()
        d = max(abs(r[-1:]))/max(norm(y,axis=0))

        if (d <= rtol): 
            break

        if (d > rtol): 
            block_size = min(block_size*4,min(m,n))

        if (block_size >= min(m,n)):
            return min(m,n), []

    return block_size, Q


def rrqr_randomized(A,rtol,block_size=42):

    [m,n] = np.shape(A)

    k, q = range_randomized(A,rtol,block_size)

    if (k >= min(m,n)):
        [Q,R,p] = linalg.qr(A, mode='economic', pivoting=True)
        k = sum(norm(R,axis=1) >= rtol*norm(A))
        return Q[:,:k],R[:k,:],p
  
    Aproj = q.T @ A
    [Qproj,R,p] = linalg.qr(Aproj, mode='economic', pivoting=True)   
    Q = q @ Qproj
    k = sum(norm(R,axis=1) >= rtol*norm(Aproj))
    return Q[:,:k],R[:k,:],p


def rrsvd_randomized(A,rtol,block_size=42):

    [m,n] = np.shape(A)
#    if (m < n):
#        [Vt,s,Ut] = rrsvd_randomized(A.T,rtol,block_size)
#        return Ut.T, s, Vt.T

    k, q = range_randomized(A,rtol,block_size)

    if (k >= min(m,n)):
        [U,s,V] = linalg.svd(A,full_matrices=False)
        k = sum(abs(s) >= rtol*norm(A))
        return U[:,:k],s[:k],V[:k,:]
  
    Aproj = q.T @ A
    [Uproj,s,V] = linalg.svd(Aproj,full_matrices=False)
    U = q @ Uproj
    k = sum(abs(s) >= rtol*norm(Aproj))
    return U[:,:k],s[:k],V[:k,:]


def rrid_randomized(A,rtol,block_size=42):

    Q, R, p = rrqr_randomized(A,rtol,block_size)
    k = R.shape[0]
    proj = linalg.solve(np.triu(R[:k,:k]), R[:,k:])
    return k, p, proj


def image_randomized(A,rtol,block_size=42):
    return range_randomized(A.T, rtol, block_size)


