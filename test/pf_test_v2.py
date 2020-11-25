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
import itertools
import matplotlib.pyplot as plt
from pf_calculation import partition_function_optimized
import time as tmf

#------------------------------------------------------------------------------
# mpmath: free (BSD licensed) Python library for real and complex floating-point 
# arithmetic with arbitrary precision. http://mpmath.org/

from mpmath import mp

# setting digit precision
mp.dps = 400
mexp = np.vectorize(mp.exp)

#------------------------------------------------------------------------------

#def spin_basis(num_sites):
#    full_basis = np.array(list(itertools.product([-1, 1], repeat=num_sites))) 
#    return full_basis

N=10
Lph = N
Tph = N

def add_measurements(meas,E,J_matrix,L,T,G1):

    q=2
    beta = np.log((q**2+1)/q)
    #---
    K = 4*np.cosh(beta/2+2*beta*G1)*np.cosh(beta/2)
    E1 = -1/(2*beta)*np.log(K)
    c = -0.5-G1+1/(2*beta)*np.log(np.cosh(beta/2+2*beta*G1)/np.cosh(beta/2))
    E2 = -1/beta*np.log(np.exp(beta*(1+G1))+2*np.exp(+beta*G1)+np.exp(-beta*(1+3*G1)))
    
    K_matrix = 0*J_matrix
    for j in range(Tph):
        for x in np.arange(1-L%2,Lph+1-L%2):
            indx1,indx2 = (j+j%2)*L+int(x/2),(j+1-j%2)*L+int(x/2)+x%2
            if meas[j,x-1+L%2]:
                K_matrix[indx1,indx2],K_matrix[indx2,indx1] = 1,1
                if j%2==1 and ((x!=Lph and x!=1) or L%2==1):
                    K_matrix[indx2-x%2,indx2+1-x%2],K_matrix[indx2+1-x%2,indx2-x%2] = 1,1
                if j%2==0 and ((x!=Lph-1 and x!=0) or L%2==0):
                    K_matrix[indx1+x%2,indx1-1+x%2],K_matrix[indx1-1+x%2,indx1+x%2] = 1,1

    J_matrix[K_matrix==1]=0
    for j in range(T-1):
        for i in range(L):
            indx1,indx2,indx3 = j*L+i,(j+1)*L+i+1-j%2,j*L+i+1
            if K_matrix[indx1,indx3]==1:
                #J_matrix[indx1,indx3],K_matrix[indx3,indx1] = 0,0
                if K_matrix[indx2,indx3]==0 and K_matrix[indx1,indx2]==1:
                    #J_matrix[indx1,indx2],J_matrix[indx2,indx1] = 0,0
                    J_matrix[indx2,indx3],J_matrix[indx3,indx2] = c,c
                    E+=E1
                if K_matrix[indx1,indx2]==0 and K_matrix[indx2,indx3]==1:
                    J_matrix[indx1,indx2],J_matrix[indx2,indx1] = c,c
                    #J_matrix[indx2,indx3],J_matrix[indx3,indx2] = 0,0
                    E+=E1
                if K_matrix[indx1,indx2]==1 and K_matrix[indx2,indx3]==1:
                    #J_matrix[indx1,indx2],J_matrix[indx2,indx1] = 0,0
                    #J_matrix[indx2,indx3],J_matrix[indx3,indx2] = 0,0
                    E+=E2
                
    #show_lattice(K_matrix==1,L,T,col='r')
    return E,J_matrix,K_matrix

def show_lattice(adj_matrix ,L,T,col='k',ls='-',label=''):
    basis_x = np.tile(np.arange(L),T)
    basis_y = np.repeat(np.arange(T),L)
    X,Y = np.meshgrid(basis_x,basis_y)
    s=1
    for i in range(len(adj_matrix)):
        for j in range(i+1,len(adj_matrix)):
            if adj_matrix[i,j]:
                plt.plot([basis_x[i]-0.5*s*(basis_y[i]%2),basis_x[j]-0.5*s*(basis_y[j]%2)],[-basis_y[i],-basis_y[j]],c=col,ls=ls,markersize=6,marker='o',mfc='w',mec='k',label=label)  
    plt.axis('off')    
    return 0

def create_lattice2(Lph,Tph):
    
    q=2
    beta = np.log((q**2+1)/q)    
    L,T = int(Lph/2)+(Lph+1)%2,Tph+1
    
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
    top_edge    = AND(Y==T-1,AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0))
    bottom_edge = AND(Y==0  ,AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0))
    
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
              
    G1,G2,G3 = 10+1e-3,10+1e-4,10+1e-5
    E = mp.mpc(0)
    
    if L%2==1:
        edge_exclusion = right_edge
    if L%2==0:
        edge_exclusion = left_edge

    # ------ setting edges -------
    
    edges_1 = np.zeros([L*T,L*T],complex)
    edges_1[XOR(inkline_lines,AND(inkline_lines,edge_exclusion))] = -0.5
    E += 0.5*np.count_nonzero(edges_1)/2
    
    edges_2 = np.zeros([L*T,L*T],complex)
    edges_2[AND(XOR(horizon_lines,AND(horizon_lines,edge_exclusion)),NOT(top_edge))] = G1
    E += G1*np.count_nonzero(edges_2)/2
    
    edges_3 = np.zeros([L*T,L*T],complex)
    edges_3[XOR(spec_bd_lines,XOR(inkline_lines,AND(inkline_lines,edge_exclusion)))] = -G1

    edges_4 = np.zeros([L*T,L*T],complex)
    edges_4[spec_bd_lines] = -G3  
 
    J_matrix = edges_1 + edges_2 + edges_3 + edges_4
    
    nu=0.15
    meas = np.random.choice([True,False],p=(nu,1-nu),size=[Tph,Lph])
    E,J_matrix,K_matrix = add_measurements(meas,E,J_matrix,L,T,G1)
    
    E += G3*np.count_nonzero(AND(edges_4,NOT(K_matrix)))/2  
    #print np.count_nonzero(AND(edges_4,NOT(K_matrix)))/2 
    
    edges_5 = np.zeros([L*T,L*T],complex)
    edges_5[XOR(bottom_edge,central_link)] = -G2
    E += G2*np.count_nonzero(edges_5)/2 
    
    edges_6 = np.zeros([L*T,L*T],complex)
    edges_6[central_link] = G2
    E += G2*np.count_nonzero(edges_6)/2  

    edges_7 = np.zeros([L*T,L*T],complex)
    edges_7[central_link] = -G2
    E += G2*np.count_nonzero(edges_7)/2  

    J_matrix1 = J_matrix + edges_5 + edges_6#+1e-10*np.int_(adj_matrix)
    J_matrix2 = J_matrix + edges_5 + edges_7#+1e-10*np.int_(adj_matrix)
    
    #show_lattice(adj_matrix,L,T,col='0.9')
    #show_lattice(np.abs(J_matrix)!=0,L,T,col='0.1')

    c = -0.5-G1+1/(2*beta)*np.log(np.cosh(beta/2+2*beta*G1)/np.cosh(beta/2))
    
    J_matrix_display = J_matrix1.copy()
    show_lattice(J_matrix_display==-0.5-G1,L,T,col='k',label=r'$-1/2-\Gamma_1$')
    show_lattice(J_matrix_display==G1,L,T,col='b',label=r'$\Gamma_1$')
    show_lattice(J_matrix_display==c,L,T,col='r',label=r'$J_R$')
    show_lattice(J_matrix_display==-0.5-G3,L,T,col='brown',label=r'$-1/2-\Gamma_3$')
    show_lattice(J_matrix_display== G1-G2,L,T,col='cyan',label=r'$\Gamma_1-\Gamma_2$')
    show_lattice(J_matrix_display==-G2,L,T,col='green',label=r'$-\Gamma_2$')
    show_lattice(J_matrix_display== G1+G2,L,T,col='violet',label=r'$\Gamma_1+\Gamma_2$')
    show_lattice(J_matrix_display==+G2,L,T,col='violet',label=r'$\Gamma_2$')
    show_lattice(AND(K_matrix,NOT(edges_5 + edges_6)!=0),L,T,col='k',ls='--',label=r'$0$')
    plt.legend(loc='bottom',markerscale=None,ncol=3)
    plt.ylim(-T-1,0.1)
    
    J_matrix1 += 1e-20*np.int_(adj_matrix)
    J_matrix2 += 1e-20*np.int_(adj_matrix)
    
    #print 'start'
    #print "Partition function  :", partition_function_method(E,J_matrix1,L,T,ctype = 'direct').real-partition_function_method(E,J_matrix2,L,T,ctype = 'direct').real
    #t0 = tmf.clock()
    C = partition_function_method(E,J_matrix1,L,T,ctype = 'efficient').real-partition_function_method(E,J_matrix2,L,T,ctype = 'efficient').real
    print "Partition function:", mp.nstr(C, 8)
    if nu>0.:
        print "Cellular automaton:",cellular_automaton_method2(L=Lph,tm=Tph,measured=meas,smp_std=1000000)
    if nu==0.:
        print "Cellular automaton:",cellular_automaton_method(Lph,Tph,q=2)
    
    return 0

def create_lattice(Lph,Tph):

    L,T = int(Lph/2)+(Lph+1)%2,Tph+1
    
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
    top_edge    = AND(Y==T-1,AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0))
    bottom_edge = AND(Y==0  ,AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0))
    
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
              
    G1,G2,G3 = 10,10,10
    E = mp.mpc(0)
    
    if L%2==1:
        edge_exclusion = right_edge
    if L%2==0:
        edge_exclusion = left_edge

    # ------ setting edges -------
    
    edges_1 = np.zeros([L*T,L*T],complex)
    edges_1[XOR(inkline_lines,AND(inkline_lines,edge_exclusion))] = -0.5
    E += 0.5*np.count_nonzero(edges_1)/2
    
    edges_2 = np.zeros([L*T,L*T],complex)
    edges_2[AND(XOR(horizon_lines,AND(horizon_lines,edge_exclusion)),NOT(top_edge))] = G1
    E += G1*np.count_nonzero(edges_2)/2
    
    edges_3 = np.zeros([L*T,L*T],complex)
    edges_3[XOR(spec_bd_lines,XOR(inkline_lines,AND(inkline_lines,edge_exclusion)))] = -G1

    edges_4 = np.zeros([L*T,L*T],complex)
    edges_4[spec_bd_lines] = -G3
    E += G3*np.count_nonzero(edges_4)/2    
 
    edges_5 = np.zeros([L*T,L*T],complex)
    edges_5[central_link] = G2
    E += G2*np.count_nonzero(edges_5)/2  

    edges_6 = np.zeros([L*T,L*T],complex)
    edges_6[XOR(bottom_edge,central_link)] = -G2
    E += G2*np.count_nonzero(edges_6)/2    

    J_matrix = edges_1 + edges_2 + edges_3 + edges_4 + edges_5 + edges_6+1e-10*np.int_(adj_matrix)

    #print "Partition function  :", partition_function_method(E,J_matrix,L,T,ctype = 'direct').real
    print "Partition function 2:", partition_function_method(E,J_matrix,L,T,ctype = 'efficient').real
    print "Cellular automaton  :", cellular_automaton_method(Lph,Tph)
    
     
    return 0

#def log_pf(beta,J_matrix):
#    basis = spin_basis(len(J_matrix))
#    PF = mp.mpc(0)
#    for i in range(len(basis)):
#        v = basis[i]
#        PF += mp.exp(-beta*0.5*np.dot(v,np.dot(J_matrix,v)))
#    return mp.log(PF)

def partition_function_method(E,J_matrix,L,T,ctype = 'direct'):
    q=2
    beta = mp.log((q**2+1)/q)
    if ctype=='direct':
        logPF = log_pf(beta,J_matrix)
    if ctype=='efficient':
        J_matrix = -beta*mp.matrix(J_matrix.tolist())
        j_matrix = (-J_matrix).apply(mp.exp)
        logPF = partition_function_optimized(mp.mpc(0),j_matrix,L,T)
    return -logPF+beta*E+(1+int(T/2)+L%2)*mp.log(2)

def cellular_automaton_method(Lph,Tph,q=2):
    s = (int(Lph/2)+1)%2
    P = np.ones(Lph+1,float)
    for t in range(Tph):
        P1,P2 = np.roll(P,+1),np.roll(P,-1)
        P[(t+s)%2::2] = q/(q**2+1)*(P1[(t+s)%2::2]+P2[(t+s)%2::2])
        P[0],P[-1] = 1,1
    S = -np.log(P[int(Lph/2)])
    return S

def unitary_update(P,Omega,t,s,smp,L):
    walls = XOR(Omega,np.roll(Omega,+1))
    walls[(t+s)%2::2] = False
    walls[::L] = False
    supp = np.zeros(smp*L,bool)
    supp[walls] = np.random.randint(2,size=np.count_nonzero(walls))>0
    supp  = XOR(supp,np.roll(supp,-1))
    walls = XOR(walls,supp)
    Omega = XOR(Omega,walls)
    P += np.count_nonzero(walls.reshape(smp,L),axis=1)
    return P,Omega

def cellular_automaton_method2(L,tm,measured,smp_std,q=2):
    s = (int(L/2)+1)%2
    beta2 = np.log((q**2+1)/(2*q))
    exp_sumA,exp_sum0 = np.zeros(tm+1,float),np.zeros(tm+1,float)
    exp_sumA[0],exp_sum0[0] = smp_std,smp_std
    P0,PA = np.zeros(smp_std,int),np.zeros(smp_std,int)
    Omega0 = np.zeros(smp_std*L,bool)
    OmegaA = np.repeat(True,2*smp_std)
    OmegaA[::2] = False
    OmegaA = np.repeat(OmegaA,int(L/2))
    for t in range(tm):
        C = np.tile(measured[t],smp_std)
        K = np.random.randint(2,size=np.count_nonzero(C))>0
        C[C] = K
        Omega0[C],OmegaA[C] = NOT(Omega0[C]),NOT(OmegaA[C])
        P0,Omega0 = unitary_update(P0,Omega0,t,s,smp_std,L)
        PA,OmegaA = unitary_update(PA,OmegaA,t,s,smp_std,L)
        exp_sumA[t+1] = np.sum(np.exp(-beta2*PA))
        exp_sum0[t+1] = np.sum(np.exp(-beta2*P0))
    RE = -np.log(exp_sumA)+np.log(exp_sum0)
    return RE[-1]

#create_lattice(Lph,Tph)
create_lattice2(Lph,Tph)
#create_simple_lattice(Lph,Tph,kind = 'rnd')
