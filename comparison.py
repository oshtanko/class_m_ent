from __future__ import division 
import numpy as np
from exact_num_lib.simulation import plot_brickwork_ruc_uavg
from pf_method import entropy_pf#,generate_lattice_couplings
import matplotlib.pyplot as plt
#from numpy import logical_not as NOT

L=10
tmax=L
p=0.
meas = np.random.choice([True,False],p=(p,1-p),size=[tmax,L])#np.random.choice([True,False],p=(p,1-p),size=L*tmax).reshape(tmax,L)
plot_brickwork_ruc_uavg(L,tmax,meas)

entropy = np.zeros(tmax)
tmax = 10
for ti in range(tmax):
    print(ti)
    entropy[ti] = entropy_pf(Lph=L,Tph=ti,meas=meas,G=10)
    print(entropy[ti])
plt.plot(entropy,c='r')