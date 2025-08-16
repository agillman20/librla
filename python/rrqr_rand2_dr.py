import numpy as np
import hilb as h
import rrqr_rand2 as rand2
import time
from scipy.linalg import norm
from scipy import linalg

def driver():

    n = 4000
    m = 2000
    
    a = h.hilb(m,n)
    
    a = a[0:m,:]
    
    print('a shape', a.shape)
    
    tol = 1e-15
    
    start_time = time.perf_counter()
    [q,r,p] = rand2.rrqr_rand1(a,m,n,tol)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
    print('r.shape = ', r.shape)
    k= r.shape[0]
    
    relerr = norm(np.matmul(q,r)-a[:,p],'fro')/norm(a,'fro')
    
    print('k =', k)
    print('relerr = ', relerr)
    
    
    return
    
driver()    
