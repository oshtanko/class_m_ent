# Generating panel (d) Fig. 2 in the main text: collapse of the order parameter curves

import matplotlib.pyplot as plt
from scaling import plot_entropy_scaling

plt.figure(figsize=(8, 4))
ax1,ax2 = plt.subplot(121),plt.subplot(122)

# system sizes
Ns = [8,16,24,32,40,48]

# number of samples: take 0 only if already evaluated and stored
samples = 0

#plotting
p0_2,nu_2 = plot_entropy_scaling(ax1=ax1,ax2=ax2,Ns = Ns,samples=0,overwrite = False)

# values of critical point
print('Critical point:',p0_2)
print('Critical explonent:',nu_2)
