# -- This code exploits the Monte Carlo method described in paper
# arXiv:2004.06736

import numpy as np
from numpy import logical_xor as XOR
from numpy import logical_not as NOT

#==============================================================================
# Computing partition function using the Monte Carlo method
# Lph -- physical size of the system (number of qudits)
# Tph -- physicl time (circuit depth)
# meas -- position of the measurements
# smp -- number of Monte Carlo samples
# q = 2 --local dimension
#==============================================================================

def entropy_mc(Lph,Tph,meas,smp,q=2):
    if np.count_nonzero(meas)==0:
        S = mc_method_unitary(Lph,Tph,q=2)
    if np.count_nonzero(meas)>0:
        S = mc_method_meas(Lph,Tph,meas,smp,q=2)
    return S

#==============================================================================
# Monte Carlo result in absence of measurements (deterministic)
# uses random walk formalism to obtain purities
# ------------------------
# Lph -- physical size of the system (number of qudits)
# Tph -- physicl time (circuit depth)
# q = 2 --local dimension
#==============================================================================
    
def mc_method_unitary(Lph,Tph,q=2):
    # parity of the inital layer
    s = (int(Lph/2)+1)%2
    # purity array
    P = np.ones(Lph+1,float)
    # random walk algorithm
    for t in range(Tph):
        P1,P2 = np.roll(P,+1),np.roll(P,-1)
        P[(t+s)%2::2] = q/(q**2+1)*(P1[(t+s)%2::2]+P2[(t+s)%2::2])
        P[0],P[-1] = 1,1
    # taking logarithm to obtain annealed-average entropy
    S = -np.log(P[int(Lph/2)])
    return S

#==============================================================================
# Monte Carlo result in the presence of measurements (probabilistic)
# uses cellular automaton formalism to evaluate the entropy
# ------------------------
# Lph -- physical size of the system (number of qudits)
# Tph -- physicl time (circuit depth)
# meas -- configuration of errors
# smp -- number of Monte Carlo samples
# q = 2 --local dimension
#==============================================================================
    
def mc_method_meas(Lph,Tph,meas,smp,q=2):
    s = (int(Lph/2)+1)%2
    # effective inverse temperature (renormalized)
    beta2 = np.log((q**2+1)/(2*q))
    # array of exponents for numerator and denominator
    exp_sumA,exp_sum0 = np.zeros(Tph+1,float),np.zeros(Tph+1,float)
    # initial conditions
    exp_sumA[0],exp_sum0[0] = smp,smp
    # initial conditions
    P0,PA = np.zeros(smp,int),np.zeros(smp,int)
    # initial conditions: denominator configuration
    Omega0 = np.zeros(smp*Lph,bool)
    # initial conditions: numerator configuration
    OmegaA = np.repeat(True,2*smp)
    OmegaA[::2] = False
    OmegaA = np.repeat(OmegaA,int(Lph/2))
    # time evolution
    for t in range(Tph):
        # adding measurements
        C = np.tile(meas[t],smp)
        K = np.random.randint(2,size=np.count_nonzero(C))>0
        C[C] = K
        # updating configurations
        Omega0[C],OmegaA[C] = NOT(Omega0[C]),NOT(OmegaA[C])
        P0,Omega0 = unitary_update(P0,Omega0,t,s,smp,Lph)
        PA,OmegaA = unitary_update(PA,OmegaA,t,s,smp,Lph)
        # storing the information in array of exponentials
        exp_sumA[t+1] = np.sum(np.exp(-beta2*PA))
        exp_sum0[t+1] = np.sum(np.exp(-beta2*P0))
    # computing Renyi entropy
    RE = -np.log(exp_sumA)+np.log(exp_sum0)
    # taking last value
    S = RE[-1]
    return S

#==============================================================================
# Unitary update submodule of mc_method_meas:
# implements unitary update rule for cellular automaton
# ------------------------
# Lph -- physical size of the system (number of qudits)
# Tph -- physicl time (circuit depth)
# meas -- configuration of errors
# smp -- number of Monte Carlo samples
# q = 2 --local dimension
#==============================================================================
      
def unitary_update(P,Omega,t,s,smp,Lph):
    walls = XOR(Omega,np.roll(Omega,+1))
    walls[(t+s)%2::2] = False
    walls[::Lph] = False
    supp = np.zeros(smp*Lph,bool)
    supp[walls] = np.random.randint(2,size=np.count_nonzero(walls))>0
    supp  = XOR(supp,np.roll(supp,-1))
    walls = XOR(walls,supp)
    Omega = XOR(Omega,walls)
    P += np.count_nonzero(walls.reshape(smp,Lph),axis=1)
    return P,Omega