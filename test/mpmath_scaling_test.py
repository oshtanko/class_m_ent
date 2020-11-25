import numpy as np
import time as tmf
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# mpmath: free (BSD licensed) Python library for real and complex floating-point 
# arithmetic with arbitrary precision. http://mpmath.org/

from mpmath import mp

#------------------------------------------------------------------------------

dps = np.arange(10,2000,10)
T = np.zeros(len(dps))
samples = 1000
for si in range(samples):
    for i in range(len(dps)):
        mp.dps = dps[i]
        number1,number2 = mp.rand(),mp.rand()
        t0 = tmf.clock()
        C = mp.sqrt(number1)*mp.sqrt(number2)
        T[i] += (tmf.clock()-t0)/samples

plt.plot(dps,T)