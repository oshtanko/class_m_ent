from __future__ import division  
import numpy as np
from basis import spin_basis

L=16
S,R,G = 2**int(L/4),2**int(L/2),2**int(3*L/4)

print 'A ==========================================================================='

basis = spin_basis(L)
basis1 = spin_basis(int(L/4))
basis2 = spin_basis(int(3*L/4))
A = np.zeros([2**int(L/4),2**int(3*L/4)],int)
for i in range(2**int(L/4)):
    v1 = basis1[i]
    for j in range(2**int(3*L/4)):
        v2 = basis2[j]
        v = np.zeros(L,int)
        v[:int(L/4)] = v1
        v[int(L/4):] = v2
        k = np.arange(2**L)[np.sum(np.abs(basis-v),axis=1)==0]
        A[i,j] = k
#print A

indx = np.arange(2**L).reshape(S,G)
print np.sum(np.abs(indx-A))

print 'B ==========================================================================='

basis = spin_basis(L)
basis1 = spin_basis(int(L/4))
basis2 = spin_basis(int(3*L/4))
A = np.zeros([2**int(L/4),2**int(3*L/4)],int)
for i in range(2**int(L/4)):
    v1 = basis1[i]
    for j in range(2**int(3*L/4)):
        v2 = basis2[j]
        v = np.zeros(L,int)
        v[:int(L/4)] = v2[:int(L/4)]
        v[int(L/4):int(L/2)] = v1
        v[int(L/2):] = v2[int(L/4):]
        k = np.arange(2**L)[np.sum(np.abs(basis-v),axis=1)==0]
        A[i,j] = k
#print A

indx = np.tile(np.tile(np.arange(R),S)+np.repeat(np.arange(S)*G,R),S).reshape(S,G)+\
       np.tile(np.arange(S)*R,G).reshape(G,S).T
print np.sum(np.abs(indx-A))

print 'C ==========================================================================='

basis = spin_basis(L)
basis1 = spin_basis(int(L/4))
basis2 = spin_basis(int(3*L/4))
A = np.zeros([2**int(L/4),2**int(3*L/4)],int)
for i in range(2**int(L/4)):
    v1 = basis1[i]
    for j in range(2**int(3*L/4)):
        v2 = basis2[j]
        v = np.zeros(L,int)
        v[:int(L/2)] = v2[:int(L/2)]
        v[int(L/2):int(3*L/4)] = v1
        v[int(3*L/4):] = v2[int(L/2):]
        k = np.arange(2**L)[np.sum(np.abs(basis-v),axis=1)==0]
        A[i,j] = k

a = np.kron(np.ones(R,int),np.arange(S))+np.kron(np.arange(R)*R,np.ones(S,int))
indx = np.tile(a,S).reshape(S,G)+np.tile(np.arange(S)*S,G).reshape(G,S).T
print np.sum(np.abs(indx-A))

print 'D ==========================================================================='

basis = spin_basis(L)
basis1 = spin_basis(int(L/4))
basis2 = spin_basis(int(3*L/4))
A = np.zeros([2**int(L/4),2**int(3*L/4)],int)
for i in range(2**int(L/4)):
    v1 = basis1[i]
    for j in range(2**int(3*L/4)):
        v2 = basis2[j]
        v = np.zeros(L,int)
        v[:int(3*L/4)] = v2
        v[int(3*L/4):] = v1
        k = np.arange(2**L)[np.sum(np.abs(basis-v),axis=1)==0]
        A[i,j] = k

indx = np.arange(2**L).reshape(G,S).T
print np.sum(np.abs(indx-A)) 


print 'A + B ==========================================================================='

basis = spin_basis(L)
basis1 = spin_basis(int(L/2))
basis2 = spin_basis(int(L/2))
A = np.zeros([2**int(L/2),2**int(L/2)],int)
for i in range(2**int(L/2)):
    v1 = basis1[i]
    for j in range(2**int(L/2)):
        v2 = basis2[j]
        v = np.zeros(L,int)
        v[:int(L/2)] = v1
        v[int(L/2):] = v2
        k = np.arange(2**L)[np.sum(np.abs(basis-v),axis=1)==0]
        A[i,j] = k

indx = np.arange(2**L).reshape(2**int(L/2),2**int(L/2))
print np.sum(np.abs(indx-A))

print 'B + C =============================================================================='

basis = spin_basis(L)
basis1 = spin_basis(int(L/2))
basis2 = spin_basis(int(L/2))
A = np.zeros([2**int(L/2),2**int(L/2)],int)
for i in range(2**int(L/2)):
    v1 = basis1[i]
    for j in range(2**int(L/2)):
        v2 = basis2[j]
        v = np.zeros(L,int)
        v[:int(L/4)] = v1[:int(L/4)]
        v[int(L/4):int(3*L/4)] = v2
        v[int(3*L/4):] = v1[int(L/4):]
        k = np.arange(2**L)[np.sum(np.abs(basis-v),axis=1)==0]
        A[i,j] = k

indx = np.tile(np.arange(R)*S,R).reshape(R,R)+\
       np.tile(np.tile(np.arange(S),S)+np.repeat(np.arange(S)*G,S),R).reshape(R,R).T
print np.sum(np.abs(indx-A))

print 'A + C =============================================================================='

basis = spin_basis(L)
basis1 = spin_basis(int(L/2))
basis2 = spin_basis(int(L/2))
A = np.zeros([2**int(L/2),2**int(L/2)],int)
for i in range(2**int(L/2)):
    v1 = basis1[i]
    for j in range(2**int(L/2)):
        v2 = basis2[j]
        v = np.zeros(L,int)
        v[:int(L/4)] = v1[:int(L/4)]
        v[int(L/4):int(L/2)] = v2[:int(L/4)]
        v[int(L/2):int(3*L/4)] = v1[int(L/4):]
        v[int(3*L/4):] = v2[int(L/4):]
        k = np.arange(2**L)[np.sum(np.abs(basis-v),axis=1)==0]
        A[i,j] = k
#print A

a = np.kron(np.ones(S,int),np.arange(S))+np.kron(np.arange(S)*R,np.ones(S,int))
b = np.tile(a,R).reshape(R,R)
indx = b+b.T*S
print np.sum(np.abs(indx-A))
