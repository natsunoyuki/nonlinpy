import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def recon_sp(tauI, dm, data):
    """
    This function reconstructs the attractor of a time series through 
    time delay embedding
    
    tauI: number of time delay steps to use
    dm: number of embedding dimensions
    data: original 1 dimensional time series data
    """
    Nmax = len(data) - (dm - 1) * tauI
    v = np.zeros([Nmax, dm])
    
    for j in range(Nmax):
        for i in range(dm):
            datindex = j + i * tauI
            v[j, i] = data[datindex]
            
    return v

def cao_coefficients(tauI, dm, xlist):
    """
    This function calculates the Cao Coefficients for
    determining the embedding dimension
    L Cao, Physica D 110 (1997) 43-50
    
    tauI: time delay
    dm: number of embedding dimensions
    xlist: original time series data
    """
    recM0 = recon_sp(tauI, dm, xlist)
    recM1 = recon_sp(tauI, dm + 1, xlist)
    nmax = len(recM0) - tauI
    LL = np.zeros([nmax, nmax])
    
    def nor(x, i, j):
        return linalg.norm(x[i] - x[j])
    
    for i in range(nmax):
        for j in range(nmax):
            if i < j:
                LL[i, j]=nor(recM0, i, j)
            else:
                LL[i, j]=0
                
    LL = LL + np.transpose(LL)
    X = np.empty((nmax, nmax, 2))
    
    for i in range(nmax):
        for j in range(nmax):
            for k in range(2):
                X[i, j] = np.array([LL[i, j], j + 1])

    def getKey(item):
        return item[0]

    N = np.zeros([nmax, 2])
    for i in range(nmax):
        nnl = sorted(X[i], key=getKey) #this is a list not an array
        k = 0
        while nnl[k][0] == 0:
            k = k + 1
            N[i,:] = nnl[k]
            
    aim = 0
    absX = 0
    
    for i in range(nmax):
        distM0 = N[i, 0]
        distM1 = linalg.norm(recM1[i] - recM1[int(N[i,1]-1)])
        aim = aim + distM1 / distM0
        absX = absX + abs(xlist[i+dm*tauI]-xlist[int(N[i,1]-1+dm*tauI)])
        
    caoE = aim / nmax
    caoEast = absX / nmax
    return caoE, caoEast

def calc_cao_coeff(data, tau = 50, length = 10):
    """
    data: original time series
    tau: time delay used
    length: number of dimensions to test for
    """
    CE = np.zeros(length)
    CEAST = np.zeros(length)
    print("Beginning calculation for tau =", tau)
    
    for dm in range(length):
        caoE, caoEast = cao_coefficients(tau, dm + 1, data)
        print(caoE, caoEast)
        CE[dm] = caoE
        CEAST[dm] = caoEast

    E1D = CE[1:] / CE[:-1]
    E2D = CEAST[1:] / CEAST[:-1]
     
    DM = np.arange(1, dm + 1, 1)
    plt.plot(DM,E1D,'-o'); plt.plot(DM,E2D,'-o'); plt.grid('on')
    plt.legend(['E1D','E2D'])
    plt.xlabel('Embedding Dimension')
    plt.ylabel('E1 and E2')
    plt.show()