import numpy as np
import pylab as pl
from config import *

def sporadicSampling(N, indata, square=False):
    '''Created a sporadically sampled subset of the original Mx3 data cube
    Input: 
    N: (int) minimum number of points per lightcurve
    square: (bool) if True each object in the dataset will have exactly N observations. if False N is an upper limit to the number of observations

    '''
    # set for reproducibility
    np.random.seed(123)

    #index of arrays created as uniform sampling of the indeces of each object
    subSampled = (np.random.rand(indata.shape[0], N) *
                  len(indata[0][0])).astype(int)
    subSampled.sort(axis=1)
    
    #identify duplicate indeces
    duplicates = np.diff(subSampled, axis=1) == 0
    #replace duplicate indeces if the array has to be square (exactly N observations per row)
    # otherwise just drop duplicates
    if square:
        while duplicates.sum() > 0:
            subSampled[:,1:][duplicates] += 1
            duplicates = np.diff(subSampled, axis=1) == 0
        
    newind = [subSampled[i][a] for i,a in
     enumerate(np.concatenate([np.atleast_2d(
         np.array([True]*duplicates.shape[0])).T, ~duplicates], axis=1))]
    return newind,  np.array([[indata[i][0][ni], indata[i][1][ni],
                   indata[i][2][ni]] for i,ni in enumerate(newind)])

if __name__ == '__main__':
    #testing on 5 objects from KeplerSampleFullQ.npy
    kobjects5 = np.array(np.load(DATAPATH + "/KeplerSampleFullQ.npy",
                        encoding='latin1')[:5])
    npoints = len(kobjects5[0][0])
    kobjects5 = np.array([k[j] for k in kobjects5
                      for j in range(3)])
    kobjects5 = kobjects5.reshape(5, 3, npoints)
    ax = pl.figure().add_subplot(111)
    for i in range(len(kobjects5)):
        pl.errorbar(kobjects5[i][0], kobjects5[i][1], yerr=kobjects5[i][2],
                fmt='--')

    
    #print(sporadicSampling(10, kobjects5, square=False)[2])
    for k,a in enumerate(sporadicSampling(10, kobjects5, square=True)[0]):
        #print (kobjects5[k][:,a].shape)
        #print(k,kobjects5[k][0,a], kobjects5[k][1,a],
        #            kobjects5[k][2,a])
        ax.errorbar(kobjects5[k][0,a], kobjects5[k][1,a],
                    yerr=kobjects5[k][2,a], fmt='o')

    for k,a in enumerate(sporadicSampling(10, kobjects5, square=False)[1]):
        #print(k,a)
        ax.errorbar(a[0], a[1],
                    yerr=a[2], fmt='x')
    pl.show()
        

    ax.set_xlabel("time")
    ax.set_ylabel("flux")
    pl.show()
    
