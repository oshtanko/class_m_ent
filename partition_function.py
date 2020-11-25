#------------------------------------------------------------------------------
# Partition function computation using approach described in
# http://dx.doi.org/10.1103/PhysRevLett.97.227205
#------------------------------------------------------------------------------

import numpy as np
from transforms import Y_transform, D_transform, X_transform, L_transform

#------------------------------------------------------------------------------
# mpmath: free (BSD licensed) Python library for real and complex floating-point 
# arithmetic with arbitrary precision. http://mpmath.org/

from mpmath import mp

#------------------------------------------------------------------------------

def log_partition_function(J_matrix,L,T,q=2):#(E,j_matrix,L,T):
    E = mp.mpc(0)
    j_matrix = (-J_matrix).apply(mp.exp)
    func_choice = [kill_right,kill_left]
    count_choice = [np.flip(np.arange(L-1)),np.arange(L-1)]
    for y1 in range(T-1):
        func,count = func_choice[y1%2],count_choice[y1%2]
        for y2 in np.flip(np.arange(y1+1)):
            for x in count:
                E,j_matrix = func(E,j_matrix,x,y2,L,T)
    for y in np.flip(np.arange(T-1)):
        for x in np.flip(np.arange(L-1)):
            E,j_matrix = kill_cell(E,j_matrix,x,y,L,T)
        E,j_matrix = kill_vertical(E,j_matrix,0,y,L,T)
    for x in np.flip(np.arange(L-1)):
        E,j_matrix = kill_horizontal(E,j_matrix,x,0,L,T)
    logPF = E.real+len(j_matrix)*mp.log(2)
    return logPF

def kill_right(E,j_matrix,x,y,L,T):
    if x == L-2 and y==0:
        j0 = j_matrix[L+x+1,x]
        if j0!=1:
            j1 = j_matrix[x,x+1]
            j2 = j_matrix[x+1,L+x+1]
            j1p,j2p,dF = D_transform(j0,j1,j2)
            j_matrix[x,L+x+1],j_matrix[L+x+1,x] = 1,1
            j_matrix[x,x+1],j_matrix[x+1,x] = j1p,j1p
            j_matrix[x+1,L+x+1],j_matrix[L+x+1,x+1] = j2p,j2p
            E += dF
    if x<L-2 and y==0:
        j0 = j_matrix[L+x+1,x]
        if j0!=1:
            j1 = j_matrix[x,x+1]
            j2 = j_matrix[x+1,L+x+1]
            j3 = j_matrix[x+1,x+2]
            j1p,j2p,j3p,dF = Y_transform(j0,j1,j2,j3)
            j_matrix[x,L+x+1],j_matrix[L+x+1,x] = 1,1
            j_matrix[x,x+1],j_matrix[x+1,x] = j1p,j1p
            j_matrix[x+1,L+x+1],j_matrix[L+x+1,x+1] = j2p,j2p
            j_matrix[x+1,x+2],j_matrix[x+2,x+1] = j3p,j3p
            E += dF
    if y>0:
        y0,y1,y2 = y-1,y,y+1
        if x == L-2:
            j0 = j_matrix[y1*L+x,y2*L+x+1]
            if j0!=1:
                j1 = j_matrix[y1*L+x,y1*L+x+1]
                j2 = j_matrix[y1*L+x+1,y2*L+x+1]
                j3 = j_matrix[y0*L+x+1,y1*L+x+1]
                j1p,j2p,j3p,dF = Y_transform(j0,j1,j2,j3)
                j_matrix[y1*L+x,y2*L+x+1],j_matrix[y2*L+x+1,y1*L+x] = 1,1
                j_matrix[y1*L+x,y1*L+x+1],j_matrix[y1*L+x+1,y1*L+x] = j1p,j1p
                j_matrix[y1*L+x+1,y2*L+x+1],j_matrix[y2*L+x+1,y1*L+x+1] = j2p,j2p
                j_matrix[y0*L+x+1,y1*L+x+1],j_matrix[y1*L+x+1,y0*L+x+1] = j3p,j3p
                E += dF
        if x < L-2:
            j0 = j_matrix[y1*L+x,y2*L+x+1]
            if j0!=1:
                j1 = j_matrix[y1*L+x,y1*L+x+1]
                j2 = j_matrix[y1*L+x+1,y2*L+x+1]
                j3 = j_matrix[y0*L+x+1,y1*L+x+1]
                j4 = j_matrix[y1*L+x+1,y1*L+x+2]
                j1p,j2p,j3p,j4p,j5p,dF = X_transform(j0,j1,j2,j3,j4)
                j_matrix[y1*L+x,y2*L+x+1],j_matrix[y2*L+x+1,y1*L+x] = 1,1
                j_matrix[y1*L+x,y1*L+x+1],j_matrix[y1*L+x+1,y1*L+x] = j1p,j1p
                j_matrix[y1*L+x+1,y2*L+x+1],j_matrix[y2*L+x+1,y1*L+x+1] = j2p,j2p
                j_matrix[y0*L+x+1,y1*L+x+1],j_matrix[y1*L+x+1,y0*L+x+1] = j3p,j3p
                j_matrix[y1*L+x+1,y1*L+x+2],j_matrix[y1*L+x+2,y1*L+x+1] = j4p,j4p
                j_matrix[y0*L+x+1,y1*L+x+2],j_matrix[y1*L+x+2,y0*L+x+1] = j5p,j5p 
                E += dF
    return E,j_matrix

def kill_left(E,j_matrix,x,y,L,T):
    if x == 0 and y==0:
        j0 = j_matrix[L+x,x+1]
        if j0!=1:
            j1 = j_matrix[x,L+x]
            j2 = j_matrix[x,x+1]
            j1p,j2p,dF = D_transform(j0,j1,j2)
            j_matrix[L+x,x+1],j_matrix[x+1,L+x] = 1,1
            j_matrix[x,L+x],j_matrix[L+x,x] = j1p,j1p
            j_matrix[x,x+1],j_matrix[x+1,x] = j2p,j2p
            E += dF
    if x>0 and y==0:
        j0 = j_matrix[L+x,x+1]
        if j0!=1:
            j1 = j_matrix[x,L+x]
            j2 = j_matrix[x,x+1]
            j3 = j_matrix[x-1,x]
            j1p,j2p,j3p,dF = Y_transform(j0,j1,j2,j3)
            j_matrix[L+x,x+1],j_matrix[x+1,L+x] = 1,1
            j_matrix[x,L+x],j_matrix[L+x,x] = j1p,j1p
            j_matrix[x,x+1],j_matrix[x+1,x] = j2p,j2p
            j_matrix[x-1,x],j_matrix[x,x-1] = j3p,j3p
            E += dF
    if y>0:
        y0,y1,y2 = y-1,y,y+1
        if x == 0:
            j0 = j_matrix[y2*L+x,y1*L+x+1]
            if j0!=1:
                j1 = j_matrix[y1*L+x,y2*L+x]
                j2 = j_matrix[y1*L+x,y1*L+x+1]
                j3 = j_matrix[y0*L+x,y1*L+x]
                j1p,j2p,j3p,dF = Y_transform(j0,j1,j2,j3)
                j_matrix[y2*L+x,y1*L+x+1],j_matrix[y1*L+x+1,y2*L+x] = 1,1
                j_matrix[y1*L+x,y2*L+x],j_matrix[y2*L+x,y1*L+x] = j1p,j1p
                j_matrix[y1*L+x,y1*L+x+1],j_matrix[y1*L+x+1,y1*L+x] = j2p,j2p
                j_matrix[y0*L+x,y1*L+x],j_matrix[y1*L+x,y0*L+x] = j3p,j3p
                E += dF
        if x>0:
            j0 = j_matrix[y2*L+x,y1*L+x+1]
            if j0!=1:
                j1 = j_matrix[y2*L+x,y1*L+x]
                j2 = j_matrix[y1*L+x,y1*L+x+1]
                j3 = j_matrix[y0*L+x,y1*L+x]
                j4 = j_matrix[y1*L+x,y1*L+x-1]
                j1p,j2p,j3p,j4p,j5p,dF = X_transform(j0,j1,j2,j3,j4)
                j_matrix[y2*L+x,y1*L+x+1],j_matrix[y1*L+x+1,y2*L+x] = 1,1
                j_matrix[y2*L+x,y1*L+x],j_matrix[y1*L+x,y2*L+x] = j1p,j1p
                j_matrix[y1*L+x,y1*L+x+1],j_matrix[y1*L+x+1,y1*L+x] = j2p,j2p
                j_matrix[y0*L+x,y1*L+x],j_matrix[y1*L+x,y0*L+x] = j3p,j3p
                j_matrix[y1*L+x,y1*L+x-1],j_matrix[y1*L+x-1,y1*L+x] = j4p,j4p
                j_matrix[y0*L+x,y1*L+x-1],j_matrix[y1*L+x-1,y0*L+x] = j5p,j5p
                E += dF
    return E,j_matrix

def kill_cell(E,j_matrix,x,y,L,T):
    y0,y1 = y,y+1
    j1 = j_matrix[y1*L+x,y1*L+x+1]
    j2 = j_matrix[y0*L+x+1,y1*L+x+1]
    jp,dF = L_transform(j1,j2)
    j_matrix[y1*L+x,y1*L+x+1],j_matrix[y1*L+x+1,y1*L+x] = 1,1
    j_matrix[y0*L+x+1,y1*L+x+1],j_matrix[y1*L+x+1,y0*L+x+1] = 1,1
    j_matrix[y1*L+x,y0*L+x+1],j_matrix[y0*L+x+1,y1*L+x] = jp,jp
    E += dF-mp.log(2)

    xm = 1*x
    for y in np.flip(np.arange(y+1)):
        E,j_matrix = kill_left(E,j_matrix,xm,y,L,T)
        xm = xm-1
        if xm<0:
            break
    return E,j_matrix

def kill_vertical(E,j_matrix,x,y,L,T):
    y0,y1 = y,y+1
    j0 = j_matrix[y0*L+x,y1*L+x]
    j_matrix[y0*L+x,y1*L+x],j_matrix[y1*L+x,y0*L+x] = 1,1
    dF = mp.log(j0+1/j0)-mp.log(2)
    E += dF
    return E,j_matrix

def kill_horizontal(E,j_matrix,x,y,L,T):
    if x>0:
        j1 = j_matrix[y*L+x,y*L+x+1]
        j2 = j_matrix[y*L+x-1,y*L+x]
        jp,dF = L_transform(j1,j2)
        j_matrix[y*L+x+1,y*L+x],j_matrix[y*L+x,y*L+x+1] = 1,1
        j_matrix[y*L+x-1,y*L+x],j_matrix[y*L+x,y*L+x-1] = jp,jp
        E += dF-mp.log(2)
    if x==0:
        j0 = j_matrix[0,1]
        j_matrix[0,1],j_matrix[1,0] = 1,1
        E += mp.log(j0+1/j0)-mp.log(2)
    return E,j_matrix
    
