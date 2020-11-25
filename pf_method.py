#------------------------------------------------------------------------------
# This method estimates the annealed average Renyi entropy by evaluating the 
# partition function for the effective spi model, see details arXiv:2004.06736
#------------------------------------------------------------------------------

import numpy as np
from numpy import logical_and as AND
from numpy import logical_or as OR
from numpy import logical_xor as XOR
from numpy import logical_not as NOT
import matplotlib.pyplot as plt
from partition_function import log_partition_function

#------------------------------------------------------------------------------
# mpmath: free (BSD licensed) Python library for real and complex floating-point 
# arithmetic with arbitrary precision. http://mpmath.org/

from mpmath import mp

#------------------------------------------------------------------------------

def entropy_pf(Lph,Tph,meas,G,prec=15,q=2,dps=200):
    
    # setting digit precision
    mp.dps = dps
    
    # =============================================================================
    # Evaluation of the entropy using partition function
    # Lph,Tph -- physical size (number of qubits) and time (circuit depth)
    # G -- (large) parameter controling the gap between relevant and irrelevant states
    # prec -- number of digits in the answer
    # q -- qudit dimension
    # =============================================================================
    
    # -- define effective inverse temperature
    beta = mp.log((q**2+1)/q)
    # -- dimension of the effective lattice
    L,T = int(Lph/2)+(Lph+1)%2,Tph+1
    # -- corresponding couplings for the numerator and denominator 
    J_matrix1,J_matrix2 = generate_lattice_couplings(L,T,meas,G,beta)
    # -- including the temperature
    J_matrix1 = -beta*mp.matrix(J_matrix1.tolist())
    J_matrix2 = -beta*mp.matrix(J_matrix2.tolist())
    # -- evaluation of the entropy
    entropy = -log_partition_function(J_matrix1,L,T).real\
              +log_partition_function(J_matrix2,L,T).real
    
    return mp.nstr(entropy,prec)


def generate_lattice_couplings(L,T,meas,G,beta,q=2,plot=False,ax=[]):
    
    # =============================================================================
    # Generating couplings for the effective lattice    
    # L,T -- dimensions of the lattice 
    # meas -- boolean matrix of measurement occurence
    # G -- (large) parameter controling the gap between relevant and irrelevant states  
    # q -- qudit dimension 
    # plot -- if True, the lattice couplings would be visualized
    # =============================================================================
    
    # 01 -- parameters
    
    # 1.01 gaps
    G1,G2,G3 = G+1e-3,G+1e-4,G+1e-5
    # 1.02 effective temperature
    beta = np.log((q**2+1)/q)
    # 1.03 renormalize coupling
    J_eff = -0.5-G1+1/(2*beta)*np.log(np.cosh(beta/2+2*beta*G1)/np.cosh(beta/2))
    
    # 02 -- x and y coordinates of the lattice vertices
    basis_x = np.tile(np.arange(L),T)
    basis_y = np.repeat(np.arange(T),L)
    X,Y = np.meshgrid(basis_x,basis_y)
    
    # 03 -- full adjacency matrix for the lattice
    adj_matrix  = AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0)
    adj_matrix += AND(np.abs(X-X.T)==0,np.abs(Y-Y.T)==1)
    adj_matrix += np.triu((X-X.T)*(Y-Y.T)==2*(Y%2)-1)
    adj_matrix = adj_matrix+adj_matrix.T
    
    # 04 -- top/bottom horizontal couplings
    top_edge    = AND(Y==T-1,AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0))
    bottom_edge = AND(Y==0  ,AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0))
    if L%2==1:
        bottom_edge[L-2,L-1],bottom_edge[L-1,L-2] = False,False
        
    # 05 -- angle couplings     
    inkline_lines = AND(adj_matrix,np.abs(Y-Y.T)==1)
    
    # 06 -- horizontal couplings
    horizon_lines = AND(adj_matrix,np.abs(Y-Y.T)==0)
    
    # 07 -- special boundary edges
    if L%2==1:
        spec_bd_lines = np.triu(AND(inkline_lines,AND(X==0,Y%2==0)))+np.triu(AND(inkline_lines,AND(np.abs(X.T-X)==1,AND(X==L-1,Y.T%2==1))))
        spec_bd_lines = spec_bd_lines + spec_bd_lines.T
    if L%2==0:
        spec_bd_lines = np.triu(AND(inkline_lines,AND(X==L-1,Y%2==1)))+np.triu(AND(inkline_lines,AND(np.abs(X.T-X)==1,AND(X==0,Y%2==1))))
        spec_bd_lines = spec_bd_lines + spec_bd_lines.T
    
    # 08 -- central link separating subsystems A and B
    x0 = int(L/2)-1
    central_link = np.zeros([T*L,T*L],bool)
    central_link[x0,x0+1],central_link[x0+1,x0] = True,True

    # 09 -- edges that should be excluded from the lattice
    right_edge  = OR(AND(X==L-1,X.T==L-1),AND(AND(OR(X==L-1,X.T==L-1),np.abs(Y-Y.T)==0),OR((Y%2)==0,(Y.T)%2==0)))
    left_edge   = OR(AND(X==0,X.T==0),AND(AND(OR(X==0,X.T==0),np.abs(Y-Y.T)==0),OR((Y%2)==1,(Y.T)%2==1)))
    if L%2==1:
        edge_exclusion = right_edge
    if L%2==0:
        edge_exclusion = left_edge

    # 10 -- lattice couplings ----------------------------------------------------
    
    # 10.1 -- base square lattice
    edges_1 = np.zeros([L*T,L*T],complex)
    edges_1[XOR(inkline_lines,AND(inkline_lines,edge_exclusion))] = -0.5
    
    # 10.2 -- projector prohibiting '\lambda'-configurations
    
    # 10.2.1 horizontal edges
    edges_2 = np.zeros([L*T,L*T],complex)
    edges_2[AND(XOR(horizon_lines,AND(horizon_lines,edge_exclusion)),NOT(top_edge))] = G1
    # 10.2.2 angle edges
    edges_3 = np.zeros([L*T,L*T],complex)
    edges_3[XOR(spec_bd_lines,XOR(inkline_lines,AND(inkline_lines,edge_exclusion)))] = -G1

    # 10.3 -- boundary conditions
    edges_4 = np.zeros([L*T,L*T],complex)
    edges_4[spec_bd_lines] = -G3  
 
    # 10.4 -- combining couplings and applying measurements 
    J_matrix = edges_1 + edges_2 + edges_3 + edges_4
    J_matrix,K_matrix = add_measurements(meas,J_matrix,L,T,G1,J_eff)
    
    # 10.5 -- final state boundary conditions
    edges_5 = np.zeros([L*T,L*T],complex)
    edges_5[XOR(bottom_edge,central_link)] = -G2
    
    # 10.5.1 -- numerator: division on two halves
    edges_6 = np.zeros([L*T,L*T],complex)
    edges_6[central_link] = G2
    # 10.5.2 -- denominator: no division
    edges_7 = np.zeros([L*T,L*T],complex)
    edges_7[central_link] = -G2

    # 10.6 final setting the couplings
    J_matrix1 = J_matrix + edges_5 + edges_6
    J_matrix2 = J_matrix + edges_5 + edges_7

    # 11 -- plotting the lattice (optional)
    if plot:
        J_matrix_display = J_matrix1.copy()
        show_lattice(ax,J_matrix_display==-0.5-G1,L,T,col='k',label=r'$-1/2-\Gamma_1$')
        show_lattice(ax,J_matrix_display==G1,L,T,col='b',label=r'$\Gamma_1$')
        show_lattice(ax,J_matrix_display==J_eff,L,T,col='r',label=r'$J_R$')
        show_lattice(ax,J_matrix_display==-0.5-G3,L,T,col='brown',label=r'$-1/2-\Gamma_3$')
        show_lattice(ax,J_matrix_display== G1-G2,L,T,col='cyan',label=r'$\Gamma_1-\Gamma_2$')
        show_lattice(ax,J_matrix_display==-G2,L,T,col='green',label=r'$-\Gamma_2$')
        show_lattice(ax,J_matrix_display== G1+G2,L,T,col='violet',label=r'$\Gamma_1+\Gamma_2$')
        show_lattice(ax,J_matrix_display==+G2,L,T,col='violet',label=r'$\Gamma_2$')
        show_lattice(ax,AND(K_matrix,NOT(edges_5 + edges_6)!=0),L,T,col='k',ls='--',label=r'$0$')
        plt.legend(loc=(0.005*L,-0.1),ncol=3,numpoints=None)
        plt.ylim(-T-1,0.1)
    
    # 12 -- adding regularization
    J_matrix1 += 1e-10*np.int_(adj_matrix)
    J_matrix2 += 1e-10*np.int_(adj_matrix)
    
    return J_matrix1,J_matrix2

def add_measurements(meas,J_matrix,L,T,G1,J_eff):
    
    Lph,Tph = len(meas.T),len(meas)
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
                if K_matrix[indx2,indx3]==0 and K_matrix[indx1,indx2]==1:
                    J_matrix[indx2,indx3],J_matrix[indx3,indx2] = J_eff,J_eff
                if K_matrix[indx1,indx2]==0 and K_matrix[indx2,indx3]==1:
                    J_matrix[indx1,indx2],J_matrix[indx2,indx1] = J_eff,J_eff
                    
    return J_matrix,K_matrix

# =============================================================================
# Constructing the lattice for Ising model using the adjacency matrix
# adj_matrix -- adjacency matrix for the lattice
# L - size of the lattice (function of number of qudits)
# T - depth of the lattie (function of the circuit depth)
# =============================================================================
    
def show_lattice(ax,adj_matrix,L,T,col='k',ls='-',label=''):
    basis_x = np.tile(np.arange(L),T)
    basis_y = np.repeat(np.arange(T),L)
    X,Y = np.meshgrid(basis_x,basis_y)
    s,m_indc=1,1
    for i in range(len(adj_matrix)):
        for j in range(i+1,len(adj_matrix)):
            if adj_matrix[i,j]:
               if m_indc==1:
                   ax.plot([0,0],[0,0],c=col,ls=ls,markersize=None,label=label)
                   m_indc=0
               ax.plot([basis_x[i]-0.5*s*(basis_y[i]%2),basis_x[j]-0.5*s*(basis_y[j]%2)],[-basis_y[i],-basis_y[j]],c=col
                      ,ls=ls,markersize=6,marker='o',mfc='w',mec='k')
               
                   
    ax.axis('off')    
    return 0

def visualize_lattice(ax,meas):
    Tph,Lph  = np.shape(meas)#.shape()
    L,T = int(Lph/2)+(Lph+1)%2,Tph+1
    generate_lattice_couplings(L,T,meas,G=1,beta=1,q=2,plot=True,ax=ax)