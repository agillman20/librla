import numpy as np
from scipy import linalg
from numpy.linalg import norm
#from numpy import linalg

   
def rrqr_rand1(A,m,n,rtol):

    block_size = 42
   
    for i in range(20):
       x = 2*np.random.uniform(size=(n, block_size))-1
       y = np.matmul(A,x)
       [q,r] = linalg.qr(y, mode='economic')
       nn = r.shape[0]
#       err = norm(np.matmul(q,r)-y)
       d = norm(r[nn-1,:])/norm(A,'fro')/10
       if (d > rtol): 
          block_size = min(block_size*4,min(m,n))
       else:
          break
       if (block_size == min(m,n)):
          [q,r,p] = linalg.qr(A, mode='economic', pivoting=True)
          k = block_size
    rr = q.T@A
    [Q2,R,p] = linalg.qr(rr, mode='economic', pivoting=True)   
    Q = q@Q2
    
    return(Q, R,p)
           

