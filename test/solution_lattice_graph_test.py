#------------------------------------------------------------------------------
# Partition function computation using approach described in
# http://dx.doi.org/10.1103/PhysRevLett.97.227205
#------------------------------------------------------------------------------

from __future__ import division 
import numpy as np
from numpy import logical_and as AND
from numpy import logical_or as OR
from numpy import logical_xor as XOR
from numpy import logical_not as NOT
import matplotlib.pyplot as plt
from transforms4 import Y_transform, D_transform, X_transform, L_transform
from transforms4 import log_partition_function

#------------------------------------------------------------------------------
# mpmath: free (BSD licensed) Python library for real and complex floating-point 
# arithmetic with arbitrary precision. http://mpmath.org/

from mpmath import mp

# setting digit precision
mp.dps = 50
mexp = np.vectorize(mp.exp)

#------------------------------------------------------------------------------

def create_simple_lattice(L,T,kind = 'rnd'):
    
    #--------------------------------------------------------------------------
    # forming adjacency matrix
    basis_x = np.tile(np.arange(L),T)
    basis_y = np.repeat(np.arange(T),L)
    X,Y = np.meshgrid(basis_x,basis_y)
    adj_matrix  = AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0)
    adj_matrix += AND(np.abs(X-X.T)==0,np.abs(Y-Y.T)==1)
    adj_matrix += np.triu((X-X.T)*(Y-Y.T)==1*(2*(Y%2)-1))
    adj_matrix = adj_matrix+adj_matrix.T
    
    #--------------------------------------------------------------------------
    # setting couplings matrix
    if kind == 'rnd':
        R = np.random.normal(0,1,size = (L*T,L*T))
        J_matrix = np.float_(adj_matrix)*(R+R.T)+0j
    if kind == 'frm':
        J_matrix = -1.0*np.float_(adj_matrix)+0j
    if kind == 'afm':
        J_matrix = +1.0*np.float_(adj_matrix)+0j
        
    #--------------------------------------------------------------------------
    # transformation to j_i = exp(-J_i), in mpc format 
    E = mp.mpc(0)
    j_matrix = mp.matrix(np.exp(-J_matrix).tolist())
    
    return E,j_matrix

def create_lattice(L,T):

    #--------------------------------------------------------------------------
    # forming adjacency matrix
    basis_x = np.tile(np.arange(L),T)
    basis_y = np.repeat(np.arange(T),L)
    X,Y = np.meshgrid(basis_x,basis_y)
    adj_matrix  = AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0)
    adj_matrix += AND(np.abs(X-X.T)==0,np.abs(Y-Y.T)==1)
    adj_matrix += np.triu((X-X.T)*(Y-Y.T)==2*(Y%2)-1)
    adj_matrix = adj_matrix+adj_matrix.T
    
    # -- different parts of the lattice ---
    
    right_edge  = OR(AND(X==L-1,X.T==L-1),AND(AND(OR(X==L-1,X.T==L-1),np.abs(Y-Y.T)==0),OR((Y%2)==0,(Y.T)%2==0)))
    left_edge   = OR(AND(X==0,X.T==0),AND(AND(OR(X==0,X.T==0),np.abs(Y-Y.T)==0),OR((Y%2)==1,(Y.T)%2==1)))
    top_edge    = AND((X-X.T)==1,AND(Y==T-1,Y.T==T-1))
    bottom_edge = AND((X-X.T)==1,AND(Y==0,Y.T==0))
    if L%2==1:
        bottom_edge[L-2,L-1],bottom_edge[L-1,L-2] = False,False
    inkline_lines = AND(adj_matrix,np.abs(Y-Y.T)==1)
    horizon_lines = AND(adj_matrix,np.abs(Y-Y.T)==0)
    if L%2==1:
        spec_bd_lines = np.triu(AND(inkline_lines,AND(X==0,Y%2==0)))+np.triu(AND(inkline_lines,AND(np.abs(X.T-X)==1,AND(X==L-1,Y.T%2==1))))
        spec_bd_lines = spec_bd_lines + spec_bd_lines.T
    if L%2==0:
        spec_bd_lines = np.triu(AND(inkline_lines,AND(X==L-1,Y%2==1)))+np.triu(AND(inkline_lines,AND(np.abs(X.T-X)==1,AND(X==0,Y%2==1))))
        spec_bd_lines = spec_bd_lines + spec_bd_lines.T
    
    
    x0 = int(L/2)-1
    central_link = np.zeros([T*L,T*L],bool)
    central_link[x0,x0+1],central_link[x0+1,x0] = True,True

    #--------------------------------------------------------------------------
    # setting couplings matrix
    
    # qudit dimension
    q=2
    # effective temperature
    beta = np.log((q**2+1)/q)
    # default zero
    zero = 1e-8        
    # projecting parameters                 
    G1,G2,G3 = 5,5,5
    E = mp.mpc(0)
    
    if L%2==1:
        edge_exclusion = right_edge
    if L%2==0:
        edge_exclusion = left_edge

    # ------ setting edges -------
    
    edges_1 = np.zeros([L*T,L*T],complex)
    edges_1[XOR(inkline_lines,AND(inkline_lines,edge_exclusion))] = +beta*0.5
    E += -beta*0.5*np.count_nonzero(edges_1)/2
    
    edges_2 = np.zeros([L*T,L*T],complex)
    edges_2[AND(XOR(horizon_lines,AND(horizon_lines,edge_exclusion)),NOT(top_edge))] = -beta*G1
    E += -beta*G1*np.count_nonzero(edges_2)/2
#    
    edges_3 = np.zeros([L*T,L*T],complex)
    edges_3[XOR(spec_bd_lines,XOR(inkline_lines,AND(inkline_lines,edge_exclusion)))] = +beta*G1
#
    edges_4 = np.zeros([L*T,L*T],complex)
    edges_4[spec_bd_lines] = +beta*G3
    E += -beta*G3*np.count_nonzero(edges_4)/2    
# 
    edges_5 = np.zeros([L*T,L*T],complex)
    edges_5[central_link] = -beta*G2
    E += -beta*G2*np.count_nonzero(edges_5)/2    
#   
    edges_6 = np.zeros([L*T,L*T],complex)
    edges_6[XOR(bottom_edge,central_link)] = +beta*G2
    E += -beta*G2*np.count_nonzero(edges_6)/2    
#
    J_matrix = zero*np.float_(adj_matrix)+edges_1+edges_2+edges_3+edges_4+edges_5+edges_6#+edges_5+edges_6
    print J_matrix[edges_6!=0]
#    E += -(int(T/2)+L%2)*mp.log(2)
#    print 'x'
#    
    plt.subplot(2,3,1)
    show_lattice(adj_matrix,L,T,col='0.9')
    show_lattice(edges_1!=0,L,T,col='k')
    plt.subplot(2,3,2)
    show_lattice(adj_matrix,L,T,col='0.9')
    show_lattice(edges_2!=0,L,T,col='b')
    plt.subplot(2,3,3)
    show_lattice(adj_matrix,L,T,col='0.9')
    show_lattice(edges_3!=0,L,T,col='r')
    plt.subplot(2,3,4)
    show_lattice(adj_matrix,L,T,col='0.9')
    show_lattice(edges_4!=0,L,T,col='g')
    plt.subplot(2,3,5)
    show_lattice(adj_matrix,L,T,col='0.9')
    show_lattice(edges_5!=0,L,T,col='orange')
    show_lattice(edges_6!=0,L,T,col='cyan')
    plt.subplot(2,3,6)
    show_lattice(np.abs(J_matrix)>zero,L,T,col='0.')
#    
    log_partition_function(E,J_matrix,L,T)
    
    j_matrix = mp.matrix(mexp(-J_matrix).tolist())
    
    return E,j_matrix

def show_lattice(adj_matrix ,L,T,col='k'):
    basis_x = np.tile(np.arange(L),T)
    basis_y = np.repeat(np.arange(T),L)
    X,Y = np.meshgrid(basis_x,basis_y)
    s=1
    for i in range(len(adj_matrix)):
        for j in range(i+1,len(adj_matrix)):
            if adj_matrix[i,j]:
                plt.plot([basis_x[i]-0.5*s*(basis_y[i]%2),basis_x[j]-0.5*s*(basis_y[j]%2)],[basis_y[i],basis_y[j]],c=col)  
    plt.axis('off')    
    return 0

def kill_right(E,j_matrix,x,y):
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

def kill_left(E,j_matrix,x,y):
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

def kill_cell(E,j_matrix,x,y):
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
        E,j_matrix = kill_left(E,j_matrix,xm,y)
        xm = xm-1
        if xm<0:
            break
    return E,j_matrix

def kill_vertical(E,j_matrix,x,y):
    y0,y1 = y,y+1
    j0 = j_matrix[y0*L+x,y1*L+x]
    j_matrix[y0*L+x,y1*L+x],j_matrix[y1*L+x,y0*L+x] = 1,1
    dF = mp.log(j0+1/j0)-mp.log(2)
    E += dF
    return E,j_matrix

def kill_horizontal(E,j_matrix,x,y):
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
    
def remove_layer_right(E,j_matrix):
    #E = mp.mpc(0)
    print log_partition_function(E,j_matrix,L,T).real
    #show_lattice(np.log(j_matrix)!=0,L,T)
    func_choice = [kill_right,kill_left]
    count_choice = [np.flip(np.arange(L-1)),np.arange(L-1)]
    for y1 in range(T-1):
        func,count = func_choice[y1%2],count_choice[y1%2]
        for y2 in np.flip(np.arange(y1+1)):
            for x in count:
                E,j_matrix = func(E,j_matrix,x,y2)
    for y in np.flip(np.arange(T-1)):
        for x in np.flip(np.arange(L-1)):
            E,j_matrix = kill_cell(E,j_matrix,x,y)
        E,j_matrix = kill_vertical(E,j_matrix,0,y)
    for x in np.flip(np.arange(L-1)):
        E,j_matrix = kill_horizontal(E,j_matrix,x,0)
    logPF = E.real+len(j_matrix)*mp.log(2)
    return logPF

def cellular_automaton_method_old(Lph,Tph,q=2):
    s = (int(Lph/2)+1)%2
    P = np.ones(Lph+1,float)
    for t in range(Tph):
        P1,P2 = np.roll(P,+1),np.roll(P,-1)
        P[(t+s)%2::2] = q/(q**2+1)*(P1[(t+s)%2::2]+P2[(t+s)%2::2])
        P[0],P[-1] = 1,1
    S = -np.log(P[int(Lph/2)])
    return S

Lph = 6
Tph = 2

L,T = int(Lph/2)+(Lph+1)%2,Tph+1

#E,j_matrix = create_simple_lattice(L,T)
E,j_matrix = create_lattice(L,T)
#show_lattice(j_matrix,L,T,col='k')
remove_layer_right(E,j_matrix)

print 'CA:',cellular_automaton_method_old(Lph,Tph)