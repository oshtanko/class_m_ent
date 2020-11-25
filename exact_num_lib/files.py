import os.path
import numpy as np

#------- save the array into file ------- 
def savedata(M,filename):
    np.save('data/'+filename+'.npy', M)
    return 0
#------- load the array from the file ------- 
def loaddata(filename):
    M = np.load('data/'+filename+'.npy',allow_pickle=True)
    return M
#-------- check if the file exists ------- 
def exist(filename):
    return os.path.isfile('data/'+filename+'.npy')

def display_time(T):
    hrs = int(T/3600)
    T2 = T-3600*hrs
    mns = int(T/60)
    sec = int(T2-60*mns)
    print 'single-sample time:',hrs,'h',mns,'m',sec,'s'
    return 0