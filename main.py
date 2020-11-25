import numpy as np
import matplotlib.pyplot as plt
from precision_estimate import plot_entropy_gap_dependence,plot_error_gap_dependence,plot_error_precision_dependence
from pf_method import visualize_lattice
from scaling import plot_entropy_scaling

p=0.5
Tph,Lph = 10,10
meas = np.random.choice([True,False],p=(p,1-p),size=(Tph,Lph))
plt.figure(figsize=(5,5))
ax  = plt.subplot()
visualize_lattice(ax,meas)

plt.figure(figsize=(8, 4))
ax1,ax2 = plt.subplot(121),plt.subplot(122)
p0_2,nu_2 = plot_entropy_scaling(ax1=ax1,ax2=ax2,Ns = [8,16,24,32,40,48],samples=0)
print(p0_2,nu_2)

plt.figure(figsize=(10, 4))
ax1,ax2,ax3 = plt.subplot(131),plt.subplot(132),plt.subplot(133)

plot_entropy_gap_dependence(ax=ax1,nu=0.,system_size = [16,24,32,40])
a0 = plot_error_gap_dependence(ax=ax2,nu=0.,system_size = [16,24,32,40])
plot_error_precision_dependence(ax=ax3,nu=0.,system_size = [16,24,32,40],a0=a0)  

plt.tight_layout()
plt.show()