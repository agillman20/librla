import numpy as np
import hilb as h
import rrqr_rand2 as rand2
import time
from scipy.linalg import norm
from scipy import linalg

def driver():

    m = 4000
    n = 2000
    
    a = h.hilb(m,n)
#    a = 2*np.random.uniform(size=(m,n))-1
#    a = np.random.normal(size=(m,n))

    print('a shape', a.shape)
    
    tol = 1e-15
    
    for i in range(1,10):
        start_time = time.perf_counter()
        [q,r,p] = rand2.rrqr_rand1(a,m,n,tol)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
    print('r.shape = ', r.shape)
    k = r.shape[0]
    
    relerr = norm(np.matmul(q,r)-a[:,p],'fro')/norm(a,'fro')
    
    print('k =', k)
    print('relerr = ', relerr)

    """
    # np arrays are mutable
    # r.shape is in economic format, as requested
    
    for i in range(1,10):
        start_time = time.perf_counter()
        [q,r,p] = linalg.qr(a, mode='economic', pivoting=True)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
        print('r.shape = ', r.shape)
        k = r.shape[0]
    
        start_time = time.perf_counter()
        relerr = norm(np.matmul(q,r)-a[:,p],'fro')/norm(a,'fro')
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
        print('k =', k)
        print('relerr = ', relerr)
    """
    
    return


driver()    
