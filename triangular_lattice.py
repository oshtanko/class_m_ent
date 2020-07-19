from __future__ import division  
import numpy as np
from numpy import logical_and as AND
from numpy import logical_or as OR
import itertools
import matplotlib.pyplot as plt

#=====================================================================================================================================

def add(a,B):
    if len(B)==0:
        B = a
    else:
        B = np.vstack((B,a))  
    return B

def angle(n):
    n = n-1e-12
    n = n/np.sqrt(np.sum(n**2))
    phi = 2*np.pi*(n[1]<0)+np.arccos(n[0])*np.sign(n[1])
    return phi%(2*np.pi)

def trnglr_lattice(L,T):
    #-------------------------------------------------------------------------
    basisx = np.tile(np.hstack((np.arange(L),np.arange(L-1))),int(T/2))
    basisy = np.repeat(np.arange(T),np.tile([L,L-1],int(T/2)))
    #--------------------------------------------------------------------------
    X,Y = np.meshgrid(basisx,basisy)
    Adjmatrix =  AND(X-X.T==1,Y-Y.T==0)
    Adjmatrix += AND(X-X.T==0,np.abs(Y-Y.T)==1)
    Adjmatrix += (X-X.T)*(Y-Y.T)==1-2*(Y%2)
    Adjmatrix = np.triu(Adjmatrix)
    Adjmatrix += Adjmatrix.T
    return basisx,basisy,Adjmatrix

def list_edges(Adjmatrix):
    S = len(Adjmatrix)
    edges = []
    for i in range(S):
        for j in range(i+1,S):
            if Adjmatrix[i][j]!=0:
                edges = add([i,j],edges)
    E = len(edges)
    database = np.zeros([E,7],int)
    for i in range(E):
        database[i][0] = i
        database[i][1:3] = edges[i]
        #---------------------------------------
        vertex = edges[i][0]
        NN = np.arange(S)[Adjmatrix[vertex]]
        NNidx, NNang = np.zeros(len(NN),int),np.zeros(len(NN))
        for s in range(len(NN)):
            NNidx[s] = np.arange(E)[OR(AND(edges.T[0]==vertex,edges.T[1]==NN[s]),AND(edges.T[1]==vertex,edges.T[0]==NN[s]))][0]
            n = np.array([basisx[NN[s]]-basisx[vertex],basisy[NN[s]]-basisy[vertex]])
            NNang[s] = angle(n)
        NNidx = NNidx[np.argsort(NNang)]
        indx0 = np.arange(len(NNidx))[NNidx==i][0]
        database[i][3],database[i][4] = NNidx[(indx0+1)%len(NNidx)],NNidx[(indx0-1)%len(NNidx)]
        #---------------------------------------
        vertex = edges[i][1]
        NN = np.arange(S)[Adjmatrix[vertex]]
        NNidx, NNang = np.zeros(len(NN),int),np.zeros(len(NN))
        for s in range(len(NN)):
            NNidx[s] = np.arange(E)[OR(AND(edges.T[0]==vertex,edges.T[1]==NN[s]),AND(edges.T[1]==vertex,edges.T[0]==NN[s]))][0]
            n = np.array([basisx[NN[s]]-basisx[vertex],basisy[NN[s]]-basisy[vertex]])
            NNang[s] = angle(n)
        NNidx = NNidx[np.argsort(NNang)]
        indx0 = np.arange(len(NNidx))[NNidx==i][0]
        database[i][5],database[i][6] = NNidx[(indx0+1)%len(NNidx)],NNidx[(indx0-1)%len(NNidx)]
    return database

def visualize(database,basisx,basisy):
    for i in range(len(Adjmatrix)):
        for j in range(i+1,len(Adjmatrix)):
            if Adjmatrix[i][j]!=0:
                plt.plot([basisx[i]+0.5*(basisy[i]%2),basisx[j]+0.5*(basisy[j]%2)],[basisy[i],basisy[j]],c='k')
    plt.axis('off')
    for i in range(len(Adjmatrix)):
        plt.text(basisx[i]+0.5*(basisy[i]%2)-0.05,basisy[i]+0.1,str(i))
    for k in range(len(database)):
        i,j = database[k][1:3]
        plt.text(0.5*(basisx[i]+basisx[j])+0.25*(basisy[j]%2)+0.25*(basisy[i]%2)-0.025,0.5*(basisy[i]+basisy[j])-0.025,str(k),bbox={'facecolor': 'white', 'alpha': 1, 'pad': 1})
    
#=====================================================================================================================================
# generate random couplings from adjacency matrix
def random_couplings(Adjmatrix):
    Jmatrix = np.zeros([len(Adjmatrix),len(Adjmatrix)])
    for i in range(len(Adjmatrix)):
        for j in range(i+1,len(Adjmatrix)):
            if Adjmatrix[i][j]!=0:
                Jmatrix[i,j] = 0.5*np.random.normal(0,1)
                Jmatrix[j,i] = Jmatrix[i,j]
    return Jmatrix

# basis of classical spins
def spin_basis(num_sites):
    full_basis = np.array(list(itertools.product([-1,1], repeat=num_sites))) 
    return full_basis

def partition_function(Jmatrix):
    basis = spin_basis(len(Jmatrix))
    Z = 0
    for si in range(len(basis)):
        vec = basis[si]
        Z += np.exp(-np.dot(vec,np.dot(Jmatrix,vec)))
    return Z

#======================================================================================================================================

#length and time for the lattice 
L,T=4,2  
# x-positions of vertices, y-positions of vertices, adjacency matrix
basisx,basisy,Adjmatrix = trnglr_lattice(L,T)
# database of the edges in the format:
# [index,vertex1,vertex2,c1,cc1,c2,cc2]
database = list_edges(Adjmatrix)
# visualize the lattice with labels for edges and vertices
visualize(database,basisx,basisy)
# -- partition function using direct calculation
Jmatrix = random_couplings(Adjmatrix)
Z = partition_function(Jmatrix)


        

