# -- Generating Supplement Figure S1 -- the effective lattice
import numpy as np
import matplotlib.pyplot as plt
from pf_method import visualize_lattice

# probability of measurement
p=0.5

# physical time and size (depth of the circuit and number of qubits)
Tph,Lph = 10,10

# generate measurement positions
meas = np.random.choice([True,False],p=(p,1-p),size=(Tph,Lph))

# plot the figure
plt.figure(figsize=(5,5))
ax  = plt.subplot()
visualize_lattice(ax,meas)
