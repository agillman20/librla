import jax
import numpy as np
import jax.numpy as jnp
import rrqr_rand1 as rand1
import time
from scipy.linalg import norm
jax.config.update('jax_enable_x64', True)

def driver():

    m = 4000
    n = 2000
    
#    a = h.hilb(m,n)
    a = jax.scipy.linalg.hilbert(max(m,n))
#    a = 2*np.random.uniform(size=(m,n))-1
#    a = np.random.normal(size=(m,n))
    
    a = a[0:m,0:n]
    
    print('a shape', a.shape)
    
    tol = 1e-15

    for i in range(1,10):
        start_time = time.perf_counter()
        [q,r,p] = rand1.rrqr_rand1(a,m,n,tol)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
    print('r.shape = ', r.shape)
    k = r.shape[0]
    
    start_time = time.perf_counter()
    relerr = norm(jnp.matmul(q,r)-a[:,p],'fro')/norm(a,'fro')
    
    print('k =', k)
    print('relerr = ', relerr)

    """
    # jax arrays are immutable, lazy evaluation?
    # WARNING: r.shape is not returned in economic format

    for i in range(1,10):
        start_time = time.perf_counter()
        [q,r,p] = jax.scipy.linalg.qr(a, 'economic', pivoting=True)
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
