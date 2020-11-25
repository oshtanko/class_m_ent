#sfrom __future__ import division  
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpmath import mp
import sys

#mp.dps = 50

def qtrt(x):
    return mp.sqrt(mp.sqrt(x+0j)+0j)

def Y_transform(j0,j1,j2,j3):
    
    d = mp.mpf(10**(-mp.dps+1))
    
    z0,z1,z2,z3 = 1/(j0*j1*j2),j1*j0/j2,j2*j0/j1,j1*j2/j0
    c0,c1,c2,c3 = z0+z1+z2+z3,z0+z1-z2-z3,z0+z2-z1-z3,z0+z3-z1-z2 
    
    t1 = mp.sqrt(c2)*mp.sqrt(c3)/(mp.sqrt(c0)*mp.sqrt(c1))
    t2 = mp.sqrt(c1)*mp.sqrt(c3)/(mp.sqrt(c0)*mp.sqrt(c2))
    t3 = mp.sqrt(c1)*mp.sqrt(c2)/(mp.sqrt(c0)*mp.sqrt(c3))
    
    if abs(1+t1)<d:
        t1 = -1+d
    if abs(1+t2)<d:
        t2 = -1+d
    if abs(1+t3)<d:
        t3 = -1+d
        
    j1p = d + mp.sqrt((1-t1)/(1+t1))
    j2p = d + mp.sqrt((1-t2)/(1+t2))
    jD  = d + mp.sqrt((1-t3)/(1+t3))
    
    dF1 = mp.log(z0/(j1p*j2p*jD+1/(j1p*j2p*jD)))
    j3p,dF2 = L_transform(j3,jD)
    
    return j1p,j2p,j3p,dF1+dF2

def D_transform(j0,j1,j2):
    
    d = mp.mpf(10**(-mp.dps+1))
    
    z0,z1,z2,z3 = 1/(j0*j1*j2),j1*j0/j2,j2*j0/j1,j1*j2/j0
    c0,c1,c2,c3 = z0+z1+z2+z3,z0+z1-z2-z3,z0+z2-z1-z3,z0+z3-z1-z2 
    
    t1 = mp.sqrt(c2+0j)*mp.sqrt(c3+0j)/(mp.sqrt(c0+0j)*mp.sqrt(c1+0j))
    t2 = mp.sqrt(c1+0j)*mp.sqrt(c3+0j)/(mp.sqrt(c0+0j)*mp.sqrt(c2+0j))
    t3 = mp.sqrt(c1+0j)*mp.sqrt(c2+0j)/(mp.sqrt(c0+0j)*mp.sqrt(c3+0j))
    
    if abs(1+t1)<d:
        t1 = -1+d
    if abs(1+t2)<d:
        t2 = -1+d
    if abs(1+t3)<d:
        t3 = -1+d
        
    j1p = d + mp.sqrt((1-t1)/(1+t1))
    j2p = d + mp.sqrt((1-t2)/(1+t2))
    jD  = d + mp.sqrt((1-t3)/(1+t3))
    
    dF = mp.log(z0/(j1p*j2p*jD+1/(j1p*j2p*jD))+0j)+mp.log(jD+1/jD)
    
    return j1p,j2p,dF

def L_transform(j1,j2):
    
    d = mp.mpf(10**(-mp.dps+1))
    
    z0,z1 = j1*j2+1/(j1*j2), j1/j2+j2/j1
    jp = d + mp.sqrt(z1)/mp.sqrt(z0)
    dF = mp.log(mp.sqrt(z1)*mp.sqrt(z0))
    
    return jp,dF
    
def X_transform(j0,j1,j2,j3,j4):
        
    d = mp.mpf(10**(-mp.dps+1))
    #print d
        
    z0,z1,z2,z3 = 1/(j0*j1*j2),j1*j0/j2,j2*j0/j1,j1*j2/j0
    c0,c1,c2,c3 = z0+z1+z2+z3,z0+z1-z2-z3,z0+z2-z1-z3,z0+z3-z1-z2 
    
    t1 = mp.sqrt(c2+0j)*mp.sqrt(c3+0j)/(mp.sqrt(c0+0j)*mp.sqrt(c1+0j))
    t2 = mp.sqrt(c1+0j)*mp.sqrt(c3+0j)/(mp.sqrt(c0+0j)*mp.sqrt(c2+0j))
    t3 = mp.sqrt(c1+0j)*mp.sqrt(c2+0j)/(mp.sqrt(c0+0j)*mp.sqrt(c3+0j))
    
    if abs(1+t1)<d:
        t1 = -1+d
    if abs(1+t2)<d:
        t2 = -1+d
    if abs(1+t3)<d:
        t3 = -1+d
        
    j1p,j2p,jD = d+mp.sqrt((1-t1)/(1+t1)),d+mp.sqrt((1-t2)/(1+t2)),d+mp.sqrt((1-t3)/(1+t3))
    
    #------------------------------------------------------------------------------
    
    m0 = 1/(j3*j4*jD)+j3*j4*jD
    m3 = j3/(j4*jD)+(j4*jD)/j3
    m4 = j4/(j3*jD)+(j3*jD)/j4
    mD = jD/(j3*j4)+(j3*j4)/jD
    
    j3p = d + qtrt(mD)*qtrt(m3)/(qtrt(m0)*qtrt(m4))
    j4p = d + qtrt(mD)*qtrt(m4)/(qtrt(m0)*qtrt(m3))
    j5p = d + qtrt(m3)*qtrt(m4)/(qtrt(m0)*qtrt(mD))
    
    #------------------------------------------------------------------------------
    
    dF1 = mp.log(z0/(j1p*j2p*jD+1/(j1p*j2p*jD))+0j)
    dF2 = mp.log(qtrt(m3+0j)*qtrt(m4+0j)*qtrt(m0+0j)*qtrt(mD+0j))

    return j1p,j2p,j3p,j4p,j5p,dF1+dF2

#------------------------------------------------------------------------------

def log_partition_function(E,J_matrix,L,T):
    q=2
    beta = np.log((q**2+1)/q)
    basis = spin_basis(len(J_matrix))
    PF = mp.mpc(0)
    Eb = []
    for i in range(len(basis)):
        v = basis[i]
        Eb += [-0.5*np.dot(v,np.dot(J_matrix,v))/beta]
        PF += mp.exp(0.5*np.dot(v,np.dot(J_matrix,v))+0j)
    
    #np.set_printoptions(threshold=sys.maxsize)
    indx = np.argsort(Eb)
    basis = basis[indx]
    k=0
    plot_configuration(basis[k],L,T,basis,Eb,E)
    print('energy shift:',-E/beta)
    print('energy of trajectory:',Eb[k])
    return E+mp.log(PF)

def plot_configuration(state,L,T,basis,Eb,E):
    basis_x = np.tile(np.arange(L),T)
    basis_y = np.repeat(np.arange(T),L)
    for i in range(len(state)):
        plt.scatter([basis_x[i]-0.5*(basis_y[i]%2)],[basis_y[i]],c=str(0.5*(1+state[i])),s=100,edgecolor='k')
    
    return 0

def spin_basis(num_sites):
    full_basis = np.array(list(itertools.product([-1, 1], repeat=num_sites))) 
    return full_basis

def show_configuration():
    fig = plt.figure(figsize=(8, 6))    
    
    ax0 = fig.add_subplot(2,2,1)
    ax0.text(0.5,0.0125,r'$J_1$')
    ax0.text(1.5,0.0125,r'$J_2$')
    ax0.plot([0,1],[0.01,0.01],marker='o',ms=10,c='k')
    ax0.plot([1,2],[0.01,0.01],marker='o',ms=10,c='k')
    ax0.text(1,-0.0075,r'$J^\prime$')
    ax0.plot([0,2],[-0.01,-0.01],marker='o',ms=10,c='k')
    ax0.set_ylim(-0.04,0.04)
    ax0.set_xlim(-0.5,2.5)
    ax0.set_axis_off()
    ax0.set_title('L-transform')
    
    
    ax1 = fig.add_subplot(2,2,2)
    ax1.text(1,0.0125,r'$J_0$')
    ax1.text(-0.1,0.004,r'$J_1$')
    ax1.text(1.8,0.004,r'$J_2$')
    ax1.plot([0,2],[0.01,0.01],marker='o',ms=10,c='k')
    ax1.plot([0,1],[0.01,-0.0],marker='o',ms=10,c='k')
    ax1.plot([1,2],[-0.0,0.01],marker='o',ms=10,c='k')

    x0=4
    ax1.text(x0-0.1,0.004,r'$J_1^\prime$')
    ax1.text(x0+1.8,0.004,r'$J_2^\prime$')
    ax1.plot([x0,x0+1],[0.01,-0.0],marker='o',ms=10,c='k')
    ax1.plot([x0+1,x0+2],[-0.0,0.01],marker='o',ms=10,c='k')

    ax1.set_ylim(-0.01,0.02)
    ax1.set_axis_off()
    ax1.set_title('D-transform')
    
    ax7 = fig.add_subplot(2,2,3)
    ax7.text(1,0.0125,r'$J_0$')
    ax7.text(-0.1,0.004,r'$J_1$')
    ax7.text(1.8,0.004,r'$J_2$')
    ax7.text(1.5,-0.006,r'$J_3$')
    ax7.plot([0,2],[0.01,0.01],marker='o',ms=10,c='k')
    ax7.plot([0,1],[0.01,-0.0],marker='o',ms=10,c='k')
    ax7.plot([1,2],[-0.0,0.01],marker='o',ms=10,c='k')
    ax7.plot([1,1],[-0.0,-0.01],marker='o',ms=10,c='k')

    x0=4
    ax7.text(x0-0.1,0.004,r'$J_1^\prime$')
    ax7.text(x0+1.8,0.004,r'$J_2^\prime$')
    ax7.text(x0+1.5,-0.006,r'$J_3^\prime$')
    ax7.plot([x0,x0+1],[0.01,-0.0],marker='o',ms=10,c='k')
    ax7.plot([x0+1,x0+2],[-0.0,0.01],marker='o',ms=10,c='k')
    ax7.plot([x0+1,x0+1],[-0.0,-0.01],marker='o',ms=10,c='k')

    ax7.set_ylim(-0.02,0.02)
    ax7.set_axis_off()
    ax7.set_title('Y-transform')
    
    
    ax2 = fig.add_subplot(2,2,4)
    ax2.text(1,0.0125,r'$J_0$')
    ax2.text(-0.1,0.004,r'$J_1$')
    ax2.text(1.8,0.004,r'$J_2$')
    ax2.text(-0.1,-0.004,r'$J_4$')
    ax2.text(1.8,-0.004,r'$J_3$')
    ax2.plot([0,2],[0.01,0.01],marker='o',ms=10,c='k')
    ax2.plot([0,1],[0.01,-0.0],marker='o',ms=10,c='k')
    ax2.plot([1,2],[-0.0,0.01],marker='o',ms=10,c='k')

    #plt.plot([0,2],[-0.01,-0.01],marker='o',ms=10,c='k')
    ax2.plot([0,1],[-0.01,-0.0],marker='o',ms=10,c='k')
    ax2.plot([1,2],[-0.0,-0.01],marker='o',ms=10,c='k')
    
    #plt.plot([0,2],[0.01,0.01],marker='o',ms=10,c='k')
    x0=4
    ax2.text(x0-0.1,0.004,r'$J_1^\prime$')
    ax2.text(x0+1.8,0.004,r'$J_2^\prime$')
    ax2.text(x0-0.1,-0.004,r'$J_4^\prime$')
    ax2.text(x0+1.8,-0.004,r'$J_3^\prime$')
    ax2.text(x0+0.8,-0.013,r'$J_5^\prime$')
    ax2.plot([x0+0,x0+1],[0.01,-0.0],marker='o',ms=10,c='k')
    ax2.plot([x0+1,x0+2],[-0.0,0.01],marker='o',ms=10,c='k')
    ax2.plot([x0+0,x0+2],[-0.01,-0.01],marker='o',ms=10,c='k')
    ax2.plot([x0+0,x0+1],[-0.01,-0.0],marker='o',ms=10,c='k')
    ax2.plot([x0+1,x0+2],[-0.0,-0.01],marker='o',ms=10,c='k')
    ax2.set_ylim(-0.02,0.02)
    ax2.set_axis_off()
    ax2.set_title('X-transform')
    
    return 0

def test():

    PF1,PF2 = mp.matrix(2, 1) 
    J1,J2 = mp.randmatrix(2,1)+1j*mp.randmatrix(2,1) 
    j1,j2 = mp.exp(-J1),mp.exp(-J2)
    basis = spin_basis(3)
    for i in range(len(basis)):
        Z1,Z2,Z3 = basis[i]
        PF1 += mp.exp(J1*Z1*Z2+J2*Z2*Z3)
    
    jp,dF = L_transform(j1,j2)   
    Jp = -mp.log(jp)
    basis = spin_basis(2)
    for i in range(len(basis)):
        Z1,Z2 = basis[i]
        PF2 += mp.exp(dF+Jp*Z1*Z2)
     
    print("--------------------------------------")
    print("L-transform test:")
    print("Original:    ", PF1)
    print("Renormalized:", PF2)
    print("--------------------------------------")
    
    PF1,PF2 = mp.matrix(2, 1) 
    J0,J1,J2,J3 = mp.randmatrix(4,1)+1j*mp.randmatrix(4,1) 
    j0,j1,j2,j3 = mp.exp(-J0),mp.exp(-J1),mp.exp(-J2),mp.exp(-J3) 
    j1p,j2p,j3p,dF = Y_transform(j0,j1,j2,j3)
    J1p,J2p,J3p = -mp.log(j1p),-mp.log(j2p),-mp.log(j3p)
    basis = spin_basis(4)
    for i in range(len(basis)):
        Z1,Z2,Z3,Z4 = basis[i]
        PF1 += mp.exp(J0*Z1*Z2+J1*Z1*Z3+J2*Z2*Z3+J3*Z3*Z4)
        PF2 += mp.exp(dF+J1p*Z1*Z3+J2p*Z2*Z3+J3p*Z3*Z4)
        
    print("--------------------------------------")
    print("Y-transform test:")
    print("Original:    ", PF1)
    print("Renormalized:", PF2)
    print("--------------------------------------")
    
    PF1,PF2 = mp.matrix(2, 1) 
    J0,J1,J2 = mp.randmatrix(3,1)+1j*mp.randmatrix(3,1) 
    j0,j1,j2 = mp.exp(-J0),mp.exp(-J1),mp.exp(-J2)
    j1p,j2p,dF = D_transform(j0,j1,j2)
    J1p,J2p = -mp.log(j1p),-mp.log(j2p)
    basis = spin_basis(3)
    for i in range(len(basis)):
        Z1,Z2,Z3 = basis[i]
        PF1 += mp.exp(J0*Z1*Z2+J1*Z1*Z3+J2*Z2*Z3)
        PF2 += mp.exp(dF+J1p*Z1*Z3+J2p*Z2*Z3)
        
    print("--------------------------------------")
    print("D-transform test:")
    print("Original:    ", PF1)
    print("Renormalized:", PF2)
    print("--------------------------------------")
    
    #------------------------------------------------------------------------------
    
    PF1,PF2 = mp.matrix(2, 1)   
    J0,J1,J2,J3,J4 = mp.randmatrix(5,1)+1j*mp.randmatrix(5,1)
    J1=0
    j0,j1,j2,j3,j4 = mp.exp(-J0),mp.exp(-J1),mp.exp(-J2),mp.exp(-J3),mp.exp(-J4)
    j1p,j2p,j3p,j4p,j5p,dF = X_transform(j0,j1,j2,j3,j4)
    J1p,J2p,J3p,J4p,J5p = -mp.log(j1p),-mp.log(j2p),-mp.log(j3p),-mp.log(j4p),-mp.log(j5p)
    basis = spin_basis(5)
    for i in range(len(basis)):
        Z1,Z2,Z3,Z4,Z5 = basis[i]
        PF1 += mp.exp(J0*Z1*Z2+J1*Z1*Z3+J2*Z2*Z3+J3*Z3*Z5+J4*Z3*Z4)
        PF2 += mp.exp(dF+J1p*Z1*Z3+J2p*Z2*Z3+J3p*Z3*Z5+J4p*Z3*Z4+J5p*Z4*Z5)
    
    print("--------------------------------------")
    print("X-transform test:")
    print("Original:    ", PF1)
    print("Renormalized:", PF2)
    print("--------------------------------------")
    
    return 0

#test()
#show_configuration()




