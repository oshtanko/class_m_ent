# planar_ising
Code associated with the paper: "Classical Models of Entanglement in Monitored Random Circuits," Oles Shtanko, Yaroslav A. Kharkov, Luis Pedro García-Pintos, Alexey V. Gorshkov (2020) arXiv:2004.06736 (in preparation for resubmission).

# Description

This code is used to replicate the results of the paper. The code implements algorithms to compute annealed average Renyi entropy using (a) Monte Carlo method and (b) partition function method (for corresponding planar Ising model). Dependencies: the code requires installed packages Numpy >= 1.19.2, Scipy >=0.19.1, and Matplotlib >=3.3.2.

## Lattice structure

The structure of Ising triangular lattice generated by running the file: /planar_ising/generate_lattice_plot.py

The result (for a random measurement configuration) is illustrated below:

![lattice](https://user-images.githubusercontent.com/35434445/100168282-f5436300-2e8e-11eb-8a9e-394e57539748.png)

The resulting lattice shows different couplings for the corresponding classical Ising model. The lattice size can be modified in the file using parameters Lph (physical length/number of qubits) and Tph (physical time/circuit depth), and measurement rate $p$. The program automatically recalculates corresponding lattice dimensions and marks the specific couplings. The arrangement measurements can be manually accessed by changing the boolean array "meas."

## Collapse plot

The collapse plot including critical point and critical exponent generated by running the file: /planar_ising/generate_collapse_plot.py

The result is demonstrated below:

![collapse](https://user-images.githubusercontent.com/35434445/100168407-2fad0000-2e8f-11eb-9eaa-f7ba5dc3fb8e.png)

The order parameter system-size scaling can be used to derive the critical point $p_c$ and the critical exponent $\nu$. The evaluated critical parameters are printed upon running the file. More samples can be generated and stored upon setting the parameter "samples" to any float number. The parameter "overwrite", if True, resets the data files and starts sampling from scratch.

## Precision plot

The error analysis plot is generated by running the file: /planar_ising/generate_precision_plot.py

The result is demonstrated below:

![precision](https://user-images.githubusercontent.com/35434445/100168588-8c101f80-2e8f-11eb-8008-7b8c80f2cb4b.png)

The first plot demonstrates how quickly the value of entropy saturates as parameter $\Gamma$ increases in the limit in Eq. (S.63) of the Supplement. The circuit is of size Lph = L and depth Tph = L. The second plot illustrates the error's decay with an increase of $Gamma$ by comparing it to the Monte Carlo simulation result. The dashed line shows the asymptotics of the exponential error suppression. The third graph illustrates the error caused by insufficient digital precision of $\Gamma$. For this plot, it follows that the digital precision should grow at least as O(L). The parameter "overwrite", if True, resets the data files and starts sampling from scratch.

# References:

Shtanko, O., Kharkov, Y.A., García-Pintos, L.P. and Gorshkov, A.V. Classical models of entanglement in monitored random circuits. https://arxiv.org/abs/2004.06736

