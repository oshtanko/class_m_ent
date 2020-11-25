# planar_ising
Code associated with the paper: "Classical Models of Entanglement in Monitored Random Circuits," Oles Shtanko, Yaroslav A. Kharkov, Luis Pedro García-Pintos, Alexey V. Gorshkov (2020) arXiv:2004.06736.

# Description

This code contains the algorithms to compute circuit-average Renyi entropy using (a) Monte Carlo method and (b) partition function method (for corresponding planar Ising model). Dependencies: the code requires installed packages Numpy >= 1.19.2, Scipy >=0.19.1, and Matplotlib >=3.3.2.

## Lattice structure for Ising model

Input arguments:

 ```Lph```  -- physical length (number of qubits) 
 
 ```Tph```  -- physical time (circuit depth)
 
 ```p```     --  measurement probability 
 
 
This command

```python3 generate_lattice_plot.py Lph=10 Tph=10 p=0.5```

saves the figure below to ```/planar_ising/figures/lattice.png```


![lattice](https://user-images.githubusercontent.com/35434445/100168282-f5436300-2e8e-11eb-8a9e-394e57539748.png)

The resulting lattice shows different couplings for the corresponding effective classical Ising model. The lattice size can be modified in the file using parameters. The program automatically recalculates corresponding lattice dimensions and marks the specific couplings. The arrangement measurements can be manually accessed by changing the boolean array ```meas``` .

## Collapse plot

This command 

```python3 generate_collapse_plot.py```

saves the figure below to ```/planar_ising/figures/collapse.png``` and outputs the values of the critical parameters.

![collapse](https://user-images.githubusercontent.com/35434445/100168407-2fad0000-2e8f-11eb-9eaa-f7ba5dc3fb8e.png)

The left panel shows the dependence of Renyi entropy as function of measurement probability $p$. The curves collapse (on the right) is used to derive the critical point $p_c$ and the critical exponent $\nu$. The evaluated critical parameters are printed upon running the file. More samples can be generated and stored upon setting the parameter ```samples``` to any float number. The parameter ```overwrite```, if ```True```, resets the data files and starts sampling from the beginning.

## Precision plot

This command 

```python3 generate_precision_plot.py```

saves the figure below to ```/planar_ising/figures/precision.png```

![precision](https://user-images.githubusercontent.com/35434445/100168588-8c101f80-2e8f-11eb-8008-7b8c80f2cb4b.png)

The leftmost plot demonstrates how quickly the value of entropy saturates as the gap parameter ```G``` ($\Gamma$) increases saturating to the value at the limit $\Gamma\to\infty$ as required by Eq. (S.63) of the Supplement. The circuit is of size ```Lph``` = L and depth ```Tph``` = L. The center plot illustrates the error's decay with an increase of ```G``` by comparing it to the Monte Carlo simulation result. The dashed lines show the asymptotics of the exponential error suppression. The rightmost graph illustrates the error caused by insufficient digital precision of ```G```. For this plot, it follows that the digital precision should grow at least as O(L). The parameter ```overwrite```, if ```True```, resets the data files and starts sampling from scratch.

# References:

Shtanko, O., Kharkov, Y.A., García-Pintos, L.P. and Gorshkov, A.V. Classical models of entanglement in monitored random circuits. https://arxiv.org/abs/2004.06736

Loh, Y.L. and Carlson, E.W. Efficient algorithm for random-bond Ising models in 2d. Phys. Rev. Lett., 97(22), p.227205 (2006)

