from __future__ import division 
import sys
import numpy as np
from numpy import logical_and as AND
from numpy import logical_or as OR
from numpy import logical_xor as XOR
#from numpy import logical_xor as XOR
from numpy import logical_not as NOT
import itertools
import matplotlib.pyplot as plt
from transforms import L_transform, X_transform, spin_basis, log_partition_function


def visualize(basisx,basisy,adjmatrix,c='k',ls="-"):
    for i in range(len(adjmatrix)):
        for j in range(i+1,len(adjmatrix)):
            if adjmatrix[i][j]:#==-0.5-G1
                plt.plot([basisx[i]+0.5*(basisy[i]%2),basisx[j]+0.5*(basisy[j]%2)],[basisy[i],basisy[j]],c=c,ls=ls)
    plt.axis('off')
    for i in range(len(adjmatrix)):
        plt.text(basisx[i]+0.5*(basisy[i]%2)-0.05,basisy[i]+0.1,'x'+str(basisx[i])+'y'+str(basisy[i]))
#    for i in range(len(jmatrix)):
#        for j in range(i+1,len(jmatrix)):
#            if adjmatrix[i][j]:
#                plt.text(0.5*(basisx[i]+basisx[j])+0.25*(basisy[j]%2)+0.25*(basisy[i]%2)-0.025,0.5*(basisy[i]+basisy[j])-0.025,jmatrix[i,j],bbox={'facecolor': 'white', 'alpha': 1, 'pad': 1})
    return 0

def allOR(A):
    res = A[0]
    for i in range(1,len(A)):
        res = OR(res,A[i])
    return res

def symmetrize(matrix):
    matrix = np.triu(matrix)
    return matrix+matrix.T


def trnglr_lattice(L,Lph,T,p,plot=False):
    #-------------------------------------------------------------------------
    basisx = np.hstack((np.tile(np.hstack((np.arange(L),np.arange(-p,L-1+p))),int(T/2)),np.arange(L)))
    basisy = np.hstack((np.repeat(np.arange(T),np.tile([L,L-1+2*p],int(T/2))),np.repeat(T,L)))
    #--------------------------------------------------------------------------
    X,Y = np.meshgrid(basisx,basisy)
    Rset = OR(AND(AND(X-X.T==1,np.abs(Y-Y.T)==1),Y%2==1),AND(AND(X-X.T==0,np.abs(Y-Y.T)==1),Y%2==0))
    Lset = OR(AND(AND(X-X.T==-1,np.abs(Y-Y.T)==1),Y%2==0),AND(AND(X-X.T==0,np.abs(Y-Y.T)==1),Y%2==1))

    Rset = np.triu(Rset)
    Rset += Rset.T
    #---
    Lset = np.triu(Lset)
    Lset += Lset.T
    #---
    Hset = symmetrize(np.dot(np.tril(Lset),np.triu(Rset)))
    
    jmatrix = np.int_(OR(OR(Rset,Lset),Hset))
    if plot:
        visualize(basisx,basisy,jmatrix>0,c='b')
    
    return basisx,basisy,jmatrix,Lset,Rset,Hset

#-------------------------------------------------------------------------------------------------------------------

def move_row(E,jmatrix,y):
    if y>1:
        y0,y1,y2=y,y-1,y-2
        s=y%2
        for x in range(L-1-s):
            #--
            J0 = jmatrix[y0*L-int(y0/2)+x,y0*L-int(y0/2)+x+1]
            J1 = jmatrix[y0*L-int(y0/2)+x,y1*L-int(y1/2)+x+s]
            J2 = jmatrix[y0*L-int(y0/2)+x+1,y1*L-int(y1/2)+x+s]
            J3 = jmatrix[y2*L-int(y2/2)+x+1,y1*L-int(y1/2)+x+s]
            J4 = jmatrix[y2*L-int(y2/2)+x,y1*L-int(y1/2)+x+s]
            J5 = jmatrix[y2*L-int(y2/2)+x,y2*L-int(y2/2)+x+1]
            #--
            J1p,J2p,J3p,J4p,J5p,dF = X_transform(J0,J1,J2,J3,J4)
            #--
            jmatrix[y0*L-int(y0/2)+x,y0*L-int(y0/2)+x+1],jmatrix[y0*L-int(y0/2)+x+1,y0*L-int(y0/2)+x]=0,0
            jmatrix[y0*L-int(y0/2)+x,y1*L-int(y1/2)+x+s],jmatrix[y1*L-int(y1/2)+x+s,y0*L-int(y0/2)+x]=J1p,J1p
            jmatrix[y0*L-int(y0/2)+x+1,y1*L-int(y1/2)+x+s],jmatrix[y1*L-int(y1/2)+x+s,y0*L-int(y0/2)+x+1]=J2p,J2p
            jmatrix[y2*L-int(y2/2)+x+1,y1*L-int(y1/2)+x+s],jmatrix[y1*L-int(y1/2)+x+s,y2*L-int(y2/2)+x+1]=J3p,J3p
            jmatrix[y2*L-int(y2/2)+x,y1*L-int(y1/2)+x+s],jmatrix[y1*L-int(y1/2)+x+s,y2*L-int(y2/2)+x]=J4p,J4p
            jmatrix[y2*L-int(y2/2)+x,y2*L-int(y2/2)+x+1],jmatrix[y2*L-int(y2/2)+x+1,y2*L-int(y2/2)+x]=J5+J5p,J5+J5p
            #--
            E += dF
            print 'remove link '+str(y)+', position '+str(x)+':',E+log_partition_function(jmatrix)
            sys.exit() 
    return E,jmatrix

def cut_teeth(E,jmatrix,y):
    y0,y1 = y,y+1
    s=y%2
    for x in range(L-1-s):
        #--
        J1 = jmatrix[y1*L-int(y1/2)+x+s,y0*L-int(y0/2)+x+0]
        J2 = jmatrix[y1*L-int(y1/2)+x+s,y0*L-int(y0/2)+x+1]
        J0 = jmatrix[y0*L-int(y0/2)+x,y0*L-int(y0/2)+x+1]
        #--
        Jp,dF = L_transform(J2,J1)
        #--
        jmatrix[y1*L-int(y1/2)+x+s,y0*L-int(y0/2)+x+0],jmatrix[y0*L-int(y0/2)+x+0,y1*L-int(y1/2)+x+s] = 0,0
        jmatrix[y1*L-int(y1/2)+x+s,y0*L-int(y0/2)+x+1],jmatrix[y0*L-int(y0/2)+x+1,y1*L-int(y1/2)+x+s] = 0,0
        jmatrix[y0*L-int(y0/2)+x,y0*L-int(y0/2)+x+1],jmatrix[y0*L-int(y0/2)+x+1,y0*L-int(y0/2)+x] = J0+Jp, J0+Jp
        E += dF-np.log(2)
        #--
    if s==1:
        J1,J2 = jmatrix[y1*L-int(y1/2),y0*L-int(y0/2)],jmatrix[y1*L-int(y1/2)+L-1,y0*L-int(y0/2)+L-2]
        jmatrix[y1*L-int(y1/2),y0*L-int(y0/2)],jmatrix[y0*L-int(y0/2)+0,y1*L-int(y1/2)]=0,0
        jmatrix[y1*L-int(y1/2)+L-1,y0*L-int(y0/2)+L-2],jmatrix[y0*L-int(y0/2)+L-2+0,y1*L-int(y1/2)+L-1]=0,0
        E += np.log(np.cosh(J1+0j)+0j)+np.log(np.cosh(J2+0j)+0j)
    return E,jmatrix

def reduce_length(E,jmatrix):
    for x in np.arange(L-2):
        #--
        J1 = jmatrix[x,x+1]
        J2 = jmatrix[x,L+x]
        J0 = jmatrix[x+1,L+x]
        #--
        Jp,dF = L_transform(J1,J2)
        #--
        jmatrix[x,x+1],jmatrix[x+1,x] = 0,0
        jmatrix[x,L+x],jmatrix[L+x,x] = 0,0
        jmatrix[x+1,L+x],jmatrix[L+x,x+1] = J0+Jp,J0+Jp
        #--
        E += dF-np.log(2)
        #--
        J1 = jmatrix[x+1,L+x]
        J2 = jmatrix[L+x,L+x+1]
        J0 = jmatrix[x+1,L+x+1]
        #--
        Jp,dF = L_transform(J1,J2)
        #--
        jmatrix[x+1,L+x],jmatrix[L+x,x+1] = 0,0
        jmatrix[L+x,L+x+1],jmatrix[L+x+1,L+x] = 0,0
        jmatrix[x+1,L+x+1],jmatrix[L+x+1,x+1] = J0+Jp,J0+Jp
        #--
        E += dF-np.log(2)
    #--
    x0 = L-2
    J1 = jmatrix[x0,x0+1]
    J2 = jmatrix[x0,L+x0]
    J0 = jmatrix[x0+1,L+x0]
    #--
    Jp,dF = L_transform(J1,J2)
    #--
    jmatrix[x0,x0+1],jmatrix[x0+1,x0] = 0,0
    jmatrix[x0,L+x0],jmatrix[L+x0,x0] = 0,0
    jmatrix[x0+1,L+x0],jmatrix[L+x0,x0+1] = J0+Jp,J0+Jp
    #--
    E += dF-np.log(2)
    return E,jmatrix

#-------------------------------------------------------------------------------------------------------------------
    
def reduction_algorithm(basisx,basisy,Lset,Rset,Hset,jmatrix):
    X,Y = np.meshgrid(basisx,basisy)
    E=0
    R = np.random.normal(0,1,size = (len(jmatrix),len(jmatrix)))
    jmatrix = jmatrix*(R+R.T)+0j
    #--------------------------------------------------------------------------
    plt.subplot(1,2,1)
    visualize(basisx,basisy,np.abs(jmatrix)!=0,c='b')
    print 'initial:      ',log_partition_function(jmatrix) 
    for i in np.arange(2,T):
        for j in np.flip(np.arange(2+i%2,i+1,2)):
            E,jmatrix = move_row(E,jmatrix,j)  
    for i in np.flip(np.arange(2,T)):
        E,jmatrix = cut_teeth(E,jmatrix,i)
        print 'flatten row '+str(i)+':', E+log_partition_function(jmatrix)
        for j in np.flip(np.arange(2+i%2,i+1,2)):
            E,jmatrix = move_row(E,jmatrix,j)
    E,jmatrix = cut_teeth(E,jmatrix,1)
    print 'flatten row 1:', E+log_partition_function(jmatrix)
    E,jmatrix = reduce_length(E,jmatrix)
    print '2-row reduct :', E+log_partition_function(jmatrix)
    plt.subplot(1,2,2)
    visualize(basisx,basisy,np.abs(jmatrix)!=0,c='b')
    K = np.sum(jmatrix)/2
    E += np.log(4*np.cosh(K+0j)+0j)+(len(jmatrix)-2)*np.log(2)
    print 'Result:       ', E
    return 0

plt.subplot(1,2,1)
#Lph,T = 2,8
Lph,T = 6,4
L = int(Lph/2)+(int(Lph/2)%2)
p = 1-(int(Lph/2)%2)
basisx,basisy,jmatrix,Lset,Rset,Hset = trnglr_lattice(L,Lph,T,p,plot=False)
plt.title('p='+str(p))
trnglr_lattice(L,Lph,T,p,plot=True)
#reduction_algorithm(basisx,basisy,Lset,Rset,Hset,jmatrix)

#plt.subplot(1,2,2)
#Lph,T = 8,4
#L = int(Lph/2)+(int(Lph/2)%2)
#p = 1-(int(Lph/2)%2)
#basisx,basisy,jmatrix,Lset,Rset,Hset = trnglr_lattice(L,Lph,T,p,plot=True)
#plt.title('p='+str(p))