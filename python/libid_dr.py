import math
import numpy as np
import hilb as h
import libid as libid
import time
from scipy import linalg
from scipy.linalg import norm
import scipy.linalg.interpolative as sli


def rankdef_matrix(m,n,k,rtol,flag_complex=0):
    d = (1.0-(np.arange(0,min(m,n))/min(m,n)))**(k-1)
    a = np.random.normal(size=(m,n))
    u, s, v = linalg.svd(a,full_matrices=False)
    return np.matmul(np.matmul(u,np.diag(d)),v)


def driver():

    m = 4000
    n = 2000
    
    a = h.hilb(m,n)
#    a = rankdef_matrix(m,n,k=5,rtol=1e-12)
#    a = 2*np.random.uniform(size=(m,n))-1
#    a = np.random.normal(size=(m,n))

    print('a shape', a.shape)
    
#    rtol = 1e-15
#    rtol = 1e-15*max(m,n)
    rtol = np.max(a)*np.finfo(a.dtype).eps*max(m,n)
    print('rtol =', rtol)

    print('### rrqr')
        
    for i in range(1,10):
        start_time = time.perf_counter()
        [Q,R,p] = libid.rrqr_randomized(a,rtol)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
    print('r.shape =', R.shape)
    k = R.shape[0]
    
    relerr = norm(np.matmul(Q,R)-a[:,p],'fro')/norm(a,'fro')
    
    print('k =', k)
    print('relerr =', relerr)

    print('### rrsvd')
        
    for i in range(1,10):
        start_time = time.perf_counter()
        [U,s,Vt] = libid.rrsvd_randomized(a,rtol)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
    print('s.shape =', s.shape)
    k = s.shape[0]
    
    print(np.shape(U),np.shape(s),np.shape(Vt))
    relerr = norm(np.matmul(np.matmul(U,np.diag(s)),Vt)-a,'fro')/norm(a,'fro')
    
    print('k =', k)
    print('relerr =', relerr)

    """
    # np arrays are mutable
    # r.shape is in economic format, as requested
    
    for i in range(1,10):
        start_time = time.perf_counter()
        [q,r,p] = linalg.qr(a, mode='economic', pivoting=True)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
        print('r.shape =', r.shape)
        k = r.shape[0]
    
        start_time = time.perf_counter()
        relerr = norm(np.matmul(q,r)-a[:,p],'fro')/norm(a,'fro')
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
        print('k =', k)
        print('relerr =', relerr)
    """

    print('### rrid')    

    for i in range(1,10):
        start_time = time.perf_counter()
        [k, J, proj] =libid.rrid_randomized(a,rtol)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    B = sli.reconstruct_skel_matrix(a, k, J)
    P = sli.reconstruct_interp_matrix(J, proj)
     
    print('rrid_randomized, k =',k) 
    err = norm(a-B@P)
    print('rrid_randomized, relerr =', err/norm(a))
    
    print('### rrid_flam')
        
    for i in range(1,10):
        start_time = time.perf_counter()
        [k, J, proj] =sli.interp_decomp(a,rtol)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    B = sli.reconstruct_skel_matrix(a, k, J)
    P = sli.reconstruct_interp_matrix(J, proj)
     
    print('rrid_flam, k =',k) 
    err = norm(a-B@P)
    print('rrid_flam, relerr =', err/norm(a))
       
    return


driver()    
