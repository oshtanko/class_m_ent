from __future__ import division  
import numpy as np
import itertools
from files import savedata, loaddata, exist

X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])

def spin_basis(num_sites):
    full_basis = np.array(list(itertools.product([0, 1], repeat=num_sites))) 
    return full_basis

def basis_1qubit(L,x):
    d = 2**(L-x-1)
    indx_opp  = np.arange(2**L)+np.kron((-1)**(np.arange(2**(x+1))),np.ones(d,int)*d)
    indx_zers = np.arange(2**L)[np.kron(np.arange(2**(x+1))%2==0,np.ones(d,bool))]
    indx_ones = indx_opp[np.kron(np.arange(2**(x+1))%2==0,np.ones(d,bool))]
    return indx_zers,indx_ones

def basis_2qubit(L,x1,x2):
    zers1,ones1 = basis_1qubit(L,x1)
    zers2,ones2 = basis_1qubit(L,x2)
    zers_zers = np.intersect1d(zers1,zers2)
    ones_zers = np.intersect1d(ones1,zers2)
    zers_ones = np.intersect1d(zers1,ones2)
    ones_ones = np.intersect1d(ones1,ones2)
    return zers_zers,ones_zers,zers_ones,ones_ones

def generate_sublocal(L):
    filename = "sublocal_L"+str(L)
    if not exist(filename):
        separated1 = np.empty(L,np.ndarray)
        for x in range(L):
            separated1[x] = np.empty(2,np.ndarray)
            separated1[x] = basis_1qubit(L,x)
        savedata(separated1,filename)
    separated1 = loaddata(filename)   
    return separated1

def generate_sublocal2_nn(L):
    filename = "sublocal2_nn_L"+str(L)
    if not exist(filename):
        separated2 = np.empty(L,np.ndarray)
        for x in range(L):
            separated2[x] = np.empty(4,np.ndarray)
            separated2[x] = basis_2qubit(L,x,(x+1)%L)
        savedata(separated2,filename)
    separated2 = loaddata(filename)   
    return separated2

def generate_sublocal2(L):
    filename = "sublocal2_L"+str(L)
    if not exist(filename):
        separated2 = np.empty([L,L],np.ndarray)
        for x1 in range(L):
            for x2 in range(x1+1,L):
                separated2[x1,x2] = np.empty(4,np.ndarray)
                separated2[x1,x2] = basis_2qubit(L,x1,x2)
                separated2[x2,x1] = separated2[x1,x2]
        savedata(separated2,filename)
    separated2 = loaddata(filename)   
    return separated2

def random_product_state(L):
    psi = 1
    for x in range(L):
        theta,phi =2*np.pi*np.random.rand(),2*np.pi*np.random.rand()
        v = [np.cos(theta),np.exp(1j*phi)*np.sin(theta)]
        psi = np.kron(psi,v)
    return psi