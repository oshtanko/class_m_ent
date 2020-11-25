from __future__ import division  
import numpy as np
from evo import normalize

def renyi_entropy(psi,indx,n):
    Lm = psi[indx]
    #--- performing SVD to find eigenstates ---
    s = np.linalg.svd(Lm,compute_uv=False)
    if n==0:
        return np.log(np.count_nonzero(s>1e-12))
    if n==1:
        return -np.sum((s**2*np.log(s**2)))
    if n>1:
        return -np.log(np.sum(s**(2*n)))

def bp_renyi_entropy(n,psi,L,x):
    #--- computing Shmidt decomposition matrix ---
    A = np.meshgrid(np.arange(2**(L-x)),np.arange(2**x))
    indx = A[1]*2**(L-x)+A[0]
    # --- computing Renyi entropy ---
    return renyi_entropy(psi,indx,n)

def tp_mutual_info(psi,L,n):
    S,R,G = 2**int(L/4),2**int(L/2),2**int(3*L/4)
    
    #--- A --------------------------------------------------------------------
    indx = np.arange(2**L).reshape(S,G)
    S_A = renyi_entropy(psi,indx,n)
    
    #--- B --------------------------------------------------------------------
    indx = np.tile(np.tile(np.arange(R),S)+np.repeat(np.arange(S)*G,R),S).reshape(S,G)+\
           np.tile(np.arange(S)*R,G).reshape(G,S).T
    S_B = renyi_entropy(psi,indx,n)
    
    #--- C --------------------------------------------------------------------
    a = np.kron(np.ones(R,int),np.arange(S))+np.kron(np.arange(R)*R,np.ones(S,int))
    indx = np.tile(a,S).reshape(S,G)+np.tile(np.arange(S)*S,G).reshape(G,S).T
    S_C = renyi_entropy(psi,indx,n)
    
    #--- D --------------------------------------------------------------------
    indx = np.arange(2**L).reshape(G,S).T
    S_D = renyi_entropy(psi,indx,n)
    
    #--- AB -------------------------------------------------------------------
    indx = np.arange(2**L).reshape(2**int(L/2),2**int(L/2))
    S_AB = renyi_entropy(psi,indx,n)
    
    #--- BC -------------------------------------------------------------------
    indx = np.tile(np.arange(R)*S,R).reshape(R,R)+\
           np.tile(np.tile(np.arange(S),S)+np.repeat(np.arange(S)*G,S),R).reshape(R,R).T
    S_BC = renyi_entropy(psi,indx,n)
    
    #--- AC -------------------------------------------------------------------
    a = np.kron(np.ones(S,int),np.arange(S))+np.kron(np.arange(S)*R,np.ones(S,int))
    b = np.tile(a,R).reshape(R,R)
    indx = b+b.T*S
    S_AC = renyi_entropy(psi,indx,n)
    
    return S_A+S_B+S_C-S_AB-S_AC-S_BC+S_D