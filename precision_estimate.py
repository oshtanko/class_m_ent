#from __future__ import division 
import numpy as np
from pf_method import entropy_pf
from mc_method import entropy_mc
import matplotlib.pyplot as plt
from storage.files import savedata,loaddata,exist
from filters.savitzky_golay import savitzky_golay

# apply Savitzky-Golay filter iteratively
def repetitive_sgolay(A,rep):
    for i in range(rep):
        A = savitzky_golay(A,3,1)
    return A

#==============================================================================
# Dependence of the entropy evaluated by partition function
# as function of gap
#-----------------------------------------
# nu -- measurement rate
# system_size -- array of system sizes
# q = 2 -- local dimension
# Gmax = 10 -- maximum value of the gap
# Gres = 10 -- gap dependnece resolution
# rewrite = False -- rewrite the file
#==============================================================================
    
def plot_entropy_gap_dependence(ax,nu,system_size,q=2,Gmax=10,Gres=10,overwrite = False):
    Sn = len(system_size)
    # array of gap parameter
    Gm = np.linspace(0,Gmax,Gres) 
    # effective inverse temperature
    beta = np.log((q**2+1)/q)
    markers = ["s","*","D","o"]
    size = [5,8,5,5]
    for ni in range(Sn):
        N = system_size[ni]
        # setting physical size and circuit depth (physical time) equal to N
        Lph,Tph = N,N
        # choice of measurements pattern
        meas = np.random.choice([True,False],p=(nu,1-nu),size=[Tph,Lph])
        # setting filename
        filename = 'pf_calculation_L'+str(Lph)+'_T'+str(Tph)+'_nu'+str(nu)+'_Gmax'+str(Gmax)+'_Gres'+str(Gres)
        if not exist(filename) or overwrite:
            # entropy as a function of gap parameter
            entropy1 = np.zeros(len(Gm))
            for gi in range(len(Gm)):
                print(gi)
                G = Gm[gi]
                # computing entropy-gap dependence by evaluating partition function
                entropy1[gi] = entropy_pf(Lph,Tph,meas,G=G)
            # computing entropy using Monte Carlo method
            entropy0 = entropy_mc(Lph,Tph,meas,smp=int(1e6))
            # store the information into the file
            data = np.empty(2,np.ndarray) 
            data[0] = entropy1
            data[1] = entropy0
            savedata(data,filename)
        # load data from file   
        entropy1,entropy0 = loaddata(filename)
        # plotting the entropy-gap dependence
        #plt.subplot(1,3,1)
        ax.plot(beta*Gm,entropy1,marker=markers[ni],markersize=size[ni],label = 'L='+str(N))
        ax.plot([beta*Gm[0],beta*Gm[-1]],[entropy0,entropy0],c='k',ls="--")
        ax.set_xlabel(r'Gap parameter  $\beta\Gamma$')
        ax.set_ylabel('Outcome (entropy)')
        ax.legend(loc = 'lower right')

#==============================================================================
# Dependence of the entropy ERROR evaluated by partition function method
# as function of gap
#-----------------------------------------
# nu -- measurement rate
# system_size -- array of system sizes
# q = 2 -- local dimension
# Gmax = 10 -- maximum value of the gap
# Gres = 10 -- gap dependnece resolution
# rewrite = False -- rewrite the file
#==============================================================================
        
def plot_error_gap_dependence(ax,nu,system_size,q=2,Gmax=10,Gres=10,overwrite = False):   
    Sn = len(system_size)
    # array of gap parameter
    Gm = np.linspace(0,Gmax,Gres) 
    # effective inverse temperature
    beta = np.log((q**2+1)/q)
    error_slope = np.array([])
    markers = ["s","*","D","o"]
    size = [5,8,5,5]       
    for ni in range(Sn):
        # setting physical size and circuit depth (physical time) equal to N
        Lph,Tph = system_size[ni],system_size[ni]
        # choice of measurements pattern
        meas = np.random.choice([True,False],p=(nu,1-nu),size=[Tph,Lph])
        # setting filename
        filename = 'pf_calculation_L'+str(Lph)+'_T'+str(Tph)+'_nu'+str(nu)+'_Gmax'+str(Gmax)+'_Gres'+str(Gres)
        if not exist(filename) or overwrite:
            # entropy as a function of gap parameter
            entropy = np.zeros(len(Gm))
            for gi in range(len(Gm)):
                print(gi)
                G = Gm[gi]
                # computing entropy-gap dependence by evaluating partition function
                entropy[gi] = entropy_pf(Lph,Tph,meas,G=G)
            # computing entropy using Monte Carlo method
            entropy0 = entropy_mc(Lph,Tph,meas,smp=int(1e6))
            # store the information into the file
            data = np.empty(2,np.ndarray) 
            data[0] = entropy
            data[1] = entropy0
            savedata(data,filename)
        # load data from file   
        entropy,entropy0 = loaddata(filename)
        # plotting the entropy-gap dependence
        #plt.subplot(1,3,2)
        # computing relative error
        error = np.abs(entropy-entropy0)/entropy0
        ax.plot(beta*Gm,error,marker=markers[ni],markersize=size[ni],zorder=1,label = 'L='+str(system_size[ni]))
        # add error slope
        error_slope = np.append(error_slope,np.polyfit(beta*Gm[-5:],np.log(error[-5:]),1)[1])
    # derive the error asymptotics
    a1,a0 = np.polyfit(system_size,error_slope,1)
    for ni in range(Sn):
        N = system_size[ni]
        # plot asymptotics
        ax.plot(beta*Gm,np.exp(-2*beta*Gm+a1*N-a0),ls=":",c='k',zorder=0)
        ax.set_yscale('log')
        ax.set_ylim(1e-7,10)
        ax.set_xlabel(r'Gap parameter $\beta\Gamma$')
        ax.set_ylabel('Relative error $\epsilon$') 
        ax.legend(loc = 'lower left')
    return a0  

#==============================================================================
# Dependence of the entropy ERROR evaluated by partition function method
# as function of digital resolution
#-----------------------------------------
# nu -- measurement rate
# system_size -- array of system sizes
# q = 2 -- local dimension
# dps_min = 10 -- minimum digital resolution
# dps_max = 10 -- maximum digital resolution
# dres = 10 -- resolution
# rewrite = False -- rewrite the file
#==============================================================================
    
def plot_error_precision_dependence(ax,nu,system_size,a0,q=2,dps_min = 10,dps_max = 200, overwrite = False):
    Sn = len(system_size)
    # array of digital precisions used in evaluation
    DPS = np.arange(dps_min,dps_max+1,1)
    # effective inverse temperature
    beta = np.log((q**2+1)/q)
    # curves color
    col = ['b','orange','green','red']
    for ni in range(Sn):
        N = system_size[ni]
        # setting physical size and circuit depth (physical time) equal to N
        Lph,Tph = N,N
        # setting optimal asymptotics for the gap parameter
        G = 10+a0*(N-16)/(2*beta)
        # choice of measurements pattern
        meas = np.random.choice([True,False],p=(nu,1-nu),size=[Tph,Lph])
        # setting filename
        filename = 'pf_prec_calculation_L'+str(Lph)+'_T'+str(Tph)+'_nu'+str(nu)+'_dps'+str(dps_max)
        if not exist(filename) or overwrite:
            # computing entropy using Monte Carlo method
            entropy0 = entropy_mc(Lph,Tph,meas,smp=int(1e6))
            # computing entropy-gap dependence by evaluating partition function
            entropy1 = np.zeros(len(DPS))
            for di in range(len(DPS)):
                print(di)
                # setting digital precision
                dps = DPS[di]
                # evaluating
                entropy1[di] = entropy_pf(Lph,Tph,meas,G=G,dps=dps)
            # store the information into the file
            data = np.empty(2,np.ndarray)
            data[1],data[0] = entropy0,entropy1
            savedata(data,filename)
        # load data from file
        entropy1,entropy0 = loaddata(filename)
        # computing relative error between monte carlo entropy
        # and the partition funcion entropy
        error = np.abs(entropy1-entropy0)/entropy0
        # computing the average of the error
        error_av = np.exp(repetitive_sgolay(np.log(error),rep=20))
        # plot actual error (noisy curve)
        plt.plot(DPS,error,label = 'L='+str(N),alpha=0.2,c=col[ni])
        # plot averaged error (smoothened curve)
        ax.plot(DPS,error_av,c=col[ni],lw=2)#,label = 'L='+str(N))#,c='b')
        ax.set_xlabel('Precision (digits)')
        ax.set_ylabel('Relative error $\epsilon$')
        ax.set_yscale('log')
        ax.legend()
