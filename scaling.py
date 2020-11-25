#from __future__ import division 
import numpy as np
from pf_method import entropy_pf
import matplotlib.pyplot as plt
from storage.files import savedata,loaddata,exist
from numpy import logical_and as AND
from filters.savitzky_golay import savitzky_golay
from scipy.optimize import minimize as fmin

#==============================================================================
# The cost fucntion used in collapsing data to derive scaling
# F -- the critical parameters (critical point and crtical exponents)
# X -- set of parameters (measurement rate)
# Y -- set of functions (entropy)
# dY -- precision of fucntion (entropy)
# Ls -- list of system sizes for each point in X,Y
#==============================================================================

def cost_function(F,X,Y,dY,Ls):
    x = (X-F[0])*Ls**(1/F[1])
    indx = np.argsort(x)
    x,y,d = x[indx],Y[indx],dY[indx]
    x1,x2 = np.roll(x,-1),np.roll(x,+1)
    y1,y2 = np.roll(y,-1),np.roll(y,+1)
    d1,d2 = np.roll(d,-1),np.roll(d,+1)
    Dy = d*d+((x2-x)/(x2-x1)*d1)**2+((x1-x)/(x2-x1)*d2)**2
    vy = ((x2-x)*y1-(x1-x)*y2)/(x2-x1)
    w = (y-vy)**2/Dy
    return np.sum(w[1:-1])/(len(w)-2)

#==============================================================================
# function performing optimal collaps and deriving the critical exponents
# ax -- the canva for plotting
# X -- set of parameters (measurement rate)
# Y -- set of functions (entropy)
# dY -- precision of fucntion (entropy)
# Ls -- list of system sizes for each point in X,Y
# returns critical parameters p0_2 and nu_2
#==============================================================================
    
def collapse(ax,X,Y,dY,LS):
    p0_2,nu_2 = fmin(fun = cost_function,x0 = [0.15,1.8], args = (X,Y,dY,LS), method = 'Nelder-Mead',tol=1e-10).x
    #p0_2,nu_2 = 0.17,1.5
    ax.scatter((X-p0_2)*LS**(1/nu_2),Y,s=2)
    return p0_2,nu_2

#==============================================================================
# Applies Savitzky-Golay filter iteratively
#==============================================================================
    
def repetitive_sgolay(A,rep):
    for i in range(rep):
        A = savitzky_golay(A,3,1)
    return A

#==============================================================================
# Function computing the scaling of the order parameter (entropy) for given
# system sizes
# Ns -- systems size
# samples -- number of sumples to add
#==============================================================================
    
def plot_entropy_scaling(ax1,ax2,Ns,samples,overwrite=False):
    markers = ["s","*","D","o",'o','o']
    size = [5,8,5,5,5,5]
    X,Y,LS = [],[],[]
    for ni in range(len(Ns)):#,16,24]: 
        N = Ns[ni]
        s=ni
        q=2
        beta = np.log((q**2+1)/q)
        Lph,Tph = N,3*N
        Nu=np.arange(0,0.5,0.05)
        #--------------------------------------------------------------------------
        filename = 'pf_transition'+str(N)
        if not exist(filename) or overwrite:
            data = np.empty(2,np.ndarray)
            data = 0,np.zeros(len(Nu))
            savedata(data,filename)
        #--------------------------------------------------------------------------
        for si in range(samples):
            print(si)
            S_pf = np.zeros(len(Nu))
            for ni in range(len(Nu)):
                nu = Nu[ni]
                G = 10+0.1852*(N-16)/(2*beta)
                dps = 5*Tph
                meas = np.random.choice([True,False],p=(nu,1-nu),size=[Tph,Lph])
                S_pf[ni] = entropy_pf(Lph,Tph,meas,G=G,dps=dps)
            smp,SPF = loaddata(filename)
            SPF = (SPF*smp+S_pf)/(smp+1)
            smp += 1 
            data = np.empty(2,np.ndarray)
            data = smp,SPF
            savedata(data,filename)
        #--------------------------------------------------------------------------
        smp,SPF = loaddata(filename)       
        #entropy,entropy0 = loaddata(filename)
        ax1.plot(Nu,SPF,marker=markers[s],markersize=size[s],label = 'L='+str(N))
        ax1.plot([Nu[0],Nu[-1]],[0,0],ls="--",c='k')
        #plt.plot(Nu,S_mc,marker=markers[s],markersize=size[s],ls="--")
        ax1.set_xlabel(r'Meas. rate')
        ax1.set_ylabel('Entropy')
        ax1.legend(loc = 'upper right')
        #--------------------------------------------------------------------------
        X = np.hstack((X,Nu))
        Y = np.hstack((Y,SPF))
        LS = np.hstack((LS,N*np.ones(len(Nu))))
    p0_2,nu_2 = collapse(ax2,X,Y,0.1+0*Y,LS)
    return p0_2,nu_2
