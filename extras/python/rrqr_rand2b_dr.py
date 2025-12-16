import numpy as np
import hilb as h
import rrqr_rand2 as rand2
import rrqr as rrqr
import time
from scipy.linalg import norm
from scipy import linalg
import scipy.linalg.interpolative as sli

def driver():

    m = 4000
    n = 2000
    
    a = h.hilb(m,n)
#    a = 2*np.random.uniform(size=(m,n))-1
#    a = np.random.normal(size=(m,n))

    print('a shape', a.shape)
    
    if_post = 1
    
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
    
    if (if_post==1):
        [Q,R,p,k] = rrqr.rrqr(r,tol)

    relerr = norm(np.matmul(q@Q,R[:,p])-a[:,p],'fro')/norm(a,'fro')
    print('relerr from post process= ', relerr)

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
    
    for i in range(1,10):
        start_time = time.perf_counter()
        [k, J, proj] =sli.interp_decomp(a,tol)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    B = sli.reconstruct_skel_matrix(a, k, J)
    P = sli.reconstruct_interp_matrix(J, proj)
     
    print('Flam k=',k) 
    err = norm(a-B@P)
    print('Flam relerr = ', err/norm(a))

    return


driver()    
