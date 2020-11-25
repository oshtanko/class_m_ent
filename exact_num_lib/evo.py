from __future__ import division  
import numpy as np

def unitary1(psi,u,x,sublocal):
    psi2 = 0j*psi
    zers,ones = sublocal[x]
    psi2[zers] = u[0,0]*psi[zers]+u[0,1]*psi[ones]
    psi2[ones] = u[1,0]*psi[zers]+u[1,1]*psi[ones]
    psi = psi2.copy()
    return psi

def unitary2_nn(psi,u,x,sublocal2_nn):
    psi2 = 0j*psi
    zers_zers,ones_zers,zers_ones,ones_ones = sublocal2_nn[x]
    psi2[zers_zers] = u[0,0]*psi[zers_zers]+u[0,1]*psi[zers_ones]+u[0,2]*psi[ones_zers]+u[0,3]*psi[ones_ones]
    psi2[zers_ones] = u[1,0]*psi[zers_zers]+u[1,1]*psi[zers_ones]+u[1,2]*psi[ones_zers]+u[1,3]*psi[ones_ones]
    psi2[ones_zers] = u[2,0]*psi[zers_zers]+u[2,1]*psi[zers_ones]+u[2,2]*psi[ones_zers]+u[2,3]*psi[ones_ones]
    psi2[ones_ones] = u[3,0]*psi[zers_zers]+u[3,1]*psi[zers_ones]+u[3,2]*psi[ones_zers]+u[3,3]*psi[ones_ones]
    return psi2

def unitary2(psi,u,x1,x2,sublocal2):
    psi2 = 0j*psi
    zers_zers,ones_zers,zers_ones,ones_ones = sublocal2[x1,x2]
    psi2[zers_zers] = u[0,0]*psi[zers_zers]+u[0,1]*psi[zers_ones]+u[0,2]*psi[ones_zers]+u[0,3]*psi[ones_ones]
    psi2[zers_ones] = u[1,0]*psi[zers_zers]+u[1,1]*psi[zers_ones]+u[1,2]*psi[ones_zers]+u[1,3]*psi[ones_ones]
    psi2[ones_zers] = u[2,0]*psi[zers_zers]+u[2,1]*psi[zers_ones]+u[2,2]*psi[ones_zers]+u[2,3]*psi[ones_ones]
    psi2[ones_ones] = u[3,0]*psi[zers_zers]+u[3,1]*psi[zers_ones]+u[3,2]*psi[ones_zers]+u[3,3]*psi[ones_ones]
    return psi2

def measurement(psi,L,x,sublocal):
    zers,ones = sublocal[x]
    p0 = np.round(np.sum(np.abs(psi[zers])**2),4)
    out_is_0 = np.random.choice([True,False],p=(p0,1-p0))
    if out_is_0:
        psi[ones] = 0
    else:
        psi[zers] = 0
    return psi

def normalize(psi):
    return psi/np.sqrt(np.sum(np.abs(psi)**2))