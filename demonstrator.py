import numpy as np
import time
import matplotlib.pyplot as plt

from reconstructor import recon_sp
from measure import calc_corr_dim, mean_attractor_radius, nearest_neighbours, Stau
from attractor import logistic_map_attractor

# This script contains several demonstrators on how to perform nonlinear
# time series analysis

def corr_dim_demonstration():
    """
    This function demonstrates how to calculate the correlation dimension
    using the logistic map attractor as an example
    """
    STARTTIME = time.time()
    
    data = logistic_map_attractor(N = 5000, A = 4, data_0 = np.pi / 10.0)
    data = data[-1000:]
    
    recsp = recon_sp(tauI = 1, dm = 3, data = data) 
    
    r, C = calc_corr_dim(recsp, r0 = 0.01, r1 = np.exp(-1), Nr = 10)
    
    print("Time taken for computation: {:.3f} s".format(time.time() - STARTTIME)) 
    
    m = np.polyfit(np.log(r), np.log(C), 1)
    print("Estimated correlation dimension: {}".format(m[0]))
    
    plt.figure(figsize=(10,5))
    plt.plot(np.log(r), np.log(C), '-o')
    plt.xlabel('log(r)')
    plt.ylabel('log(C)')
    plt.show()
    
    return r, C
    
def lyapunov_demonstration():
    """
    This function demonstrates how to calculate the maximal Lyapunov exponent
    using the logistic map attractor as an example
    """
    STARTTIME = time.time()
    
    # create logistic map attractor data
    data = logistic_map_attractor(N = 5000, A = 4, data_0 = np.pi / 10.0)
    data = data[1000:]

    # time delayed reconstruction
    recsp = recon_sp(tauI = 1, dm = 3, data = data) 
    # columns correspond to dimensions,
    # rows correspond to time, so each row is a state vector with respect to t
    [nmax, dd] = np.shape(recsp)

    D = mean_attractor_radius(nmax, recsp)
    print("Mean attractor radius: {}".format(D))

    print("Calculating Nearest Neighbors...")
    EPNb = nearest_neighbours(recsp, tauS = 1, tauL = 20, eps_tol = 0.001, nst = 0, ned = 999, theiler_window = 0)

    print("Calculating S...")
    x, y = Stau(recsp, EPNb, tauS = 1, tauL = 20, nst = 0, ned = 999)

    m = np.polyfit(x[0: 10], y[0: 10], 1)
    print("Estimated maximal Lyapunov exponent: {}".format(m[0]))

    print("Time taken for computation: {:.3f} s".format(time.time() - STARTTIME))

    plt.figure(figsize=(15, 5))
    plt.plot(x, y, '-o')
    plt.xlabel('Tau')
    plt.ylabel('S(Tau)')
    plt.grid('on')
    plt.show()
    
    return x, y