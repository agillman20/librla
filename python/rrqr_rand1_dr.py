import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
import rrqr_rand1 as rand1
import time
from jax.numpy.linalg import norm

def driver():

    n = 4000
    m = 2000
    
#    a = h.hilb(m,n)
    a = jax.scipy.linalg.hilbert(n)
    
    a = a[0:m,:]
    
    print('a shape', a.shape)
    
    tol = 1e-15
    
    start_time = time.perf_counter()
    [q,r,p] = rand1.rrqr_rand1(a,m,n,tol)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
    print('r.shape = ', r.shape)
    k= r.shape[1]
    
    relerr = norm(jnp.matmul(q,r)-a[:,p],'fro')/norm(a,'fro')
    
    print('k =', k)
    print('relerr = ', relerr)
    
    
    return


    
driver()    
