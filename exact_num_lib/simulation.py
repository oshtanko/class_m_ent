from __future__ import division  
import numpy as np
from scipy.stats import unitary_group
from evo import unitary2, unitary2_nn, measurement, normalize
from basis import random_product_state
from local import bp_renyi_entropy, tp_mutual_info
from basis import generate_sublocal,generate_sublocal2,generate_sublocal2_nn
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def brickwork_ruc(L,tmax,meas,n,x_cut,ts=1,tp='entropy',bc='open'):
    sublocal = generate_sublocal(L)
    sublocal2_nn = generate_sublocal2_nn(L)
    psi = random_product_state(L)
    Ts = np.arange(0,tmax+ts,ts)
    S,c = np.zeros(len(Ts)),1
    for ti in range(tmax):
        for x in range(int(L/2)):
            u = unitary_group.rvs(4)
            if bc=='periodic' or 2*x+ti%2<L-1:
                psi = unitary2_nn(psi,u,2*x+ti%2,sublocal2_nn)
        measured = meas[ti]#np.random.choice([True,False],p=(p,1-p),size=L)
        for x in np.arange(L)[measured]:
            psi = measurement(psi,L,x,sublocal)
        psi = normalize(psi)
        if ti==Ts[c] and tp=='entropy':
            S[c] = bp_renyi_entropy(n,psi,L,x_cut)
            c+=1
        if ti==Ts[c] and tp=='tp_info':
            S[c] = tp_mutual_info(psi,L,n)
            c+=1
    return Ts[:-1],S[:-1]

def plot_brickwork_ruc_uavg(L,tmax,meas,n=2):
    x_cut = int(L/2)
    samples = 100
    entropy_av = np.zeros(tmax)
    for si in range(samples):
        time,entropy = brickwork_ruc(L=10,tmax=tmax,meas=meas,n=2,x_cut=x_cut)
        entropy_av = (entropy_av*si+entropy)/(si+1)
        plt.plot(time[::2],entropy[::2],c='0.9')
    plt.plot(time[::2],entropy_av[::2],c='k')
    return 0
    
#------------------------------------------------------------------------------
    
def brickwork_ruc_surface(L,tmax,p,n,ts=1):
    sublocal = generate_sublocal(L+1)
    sublocal2_nn = generate_sublocal2_nn(L+1)
    psi = random_product_state(L+1)
    for ti in range(tmax):
        for x in range(int(L/2)-ti%2):
            u = unitary_group.rvs(4)
            psi = unitary2_nn(psi,u,2*x+ti%2,sublocal2_nn)
        measured = np.random.choice([True,False],p=(p,1-p),size=L)
        for x in np.arange(L)[measured]:
            psi = measurement(psi,L,x,sublocal)
        psi = normalize(psi)
    u = unitary_group.rvs(4)
    psi = unitary2_nn(psi,u,L-1,sublocal2_nn)
    Ts = np.arange(0,tmax+ts,ts)
    S,c = np.zeros(len(Ts)),1
    S[0] = bp_renyi_entropy(n,psi,L+1,L)
    for ti in range(tmax):
        for x in range(int(L/2)-ti%2):
            u = unitary_group.rvs(4)
            psi = unitary2_nn(psi,u,2*x+ti%2,sublocal2_nn)
        measured = np.random.choice([True,False],p=(p,1-p),size=L)
        for x in np.arange(L)[measured]:
            psi = measurement(psi,L,x,sublocal)
        psi = normalize(psi)
        if ti==Ts[c]:
            S[c] = bp_renyi_entropy(n,psi,L+1,L)
            c+=1
    return Ts[:-1],S[:-1]

#------------------------------------------------------------------------------
    
def grph_ruc_surface(L,tmax,p,n,ts=1):
    sublocal = generate_sublocal(L+1)
    sublocal2 = generate_sublocal2(L+1)
    psi = random_product_state(L+1)
#    for ti in range(tmax):
#        a = np.random.choice(L,size=L,replace=False)
#        M1,M2 = a[:int(L/2)],a[int(L/2):]
#        for x in range(int(L/2)):
#            u = unitary_group.rvs(4)
#            psi = unitary2(psi,u,M1[x],M2[x],sublocal2)
#        measured = np.random.choice([True,False],p=(p,1-p),size=L)
#        for x in np.arange(L)[measured]:
#            psi = measurement(psi,L,x,sublocal)
#        psi = normalize(psi)
    u = unitary_group.rvs(4)
    psi = unitary2(psi,u,L-1,L,sublocal2)
    Ts = np.arange(0,tmax+ts,ts)
    S,c = np.zeros(len(Ts)),1
    S[0] = bp_renyi_entropy(n,psi,L+1,L)
    for ti in range(tmax):
        a = np.random.choice(L,size=L,replace=False)
        M1,M2 = a[:int(L/2)],a[int(L/2):]
        for x in range(int(L/2)):
            u = unitary_group.rvs(4)
            psi = unitary2(psi,u,M1[x],M2[x],sublocal2)
        measured = np.random.choice([True,False],p=(p,1-p),size=L)
        for x in np.arange(L)[measured]:
            psi = measurement(psi,L,x,sublocal)
        psi = normalize(psi)
        if ti==Ts[c]:
            S[c] = bp_renyi_entropy(n,psi,L+1,L)
            c+=1
    return Ts[:-1],S[:-1]