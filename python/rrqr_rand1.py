import numpy as np
from jax.numpy.linalg import norm
from scipy.linalg import norm
import jax.numpy as jnp
import jax

def rrqr_rand1(A,m,n,rtol):

    block_size = 42
   
    for i in range(20):
       x = 2*np.random.uniform(size=(n, block_size))-1
       x = jnp.array(x)
       y = jnp.matmul(A,x)
       [q,r] = jax.scipy.linalg.qr(y, mode = 'economic')
       nn = r.shape[0]
       #d = norm(r[nn-1,:])/norm(A,'fro')/10
       d = norm(r[nn-1,nn-1])/max(norm(y,axis=0))
       if (d > rtol): 
          block_size = min(block_size*4,min(m,n))
       else:
          break
       if (block_size >= min(m,n)):
          [q,r,p] = jax.scipy.linalg.qr(A, 'economic', pivoting=True)
          k = block_size
    rr = jnp.matmul(jnp.transpose(q),A)
    [Q2,R,p] = jax.scipy.linalg.qr(rr, 'economic', pivoting=True)   
    Q = jnp.matmul(q,Q2)

    k = sum(norm(R,axis=1) >= rtol*norm(rr))
    #print("k = ", k)

    Q = Q[:,0:k]
    R = R[0:k,:]
    
    return(Q, R,p)
           

