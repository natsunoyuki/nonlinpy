import numpy as np
from scipy import linalg

# This file contains the functions used in nonlinear time series analysis
# to perform measurements on the reconstructed attractor

def mean_attractor_radius(nmax, recsp):
    """
    Calculates the mean reconstructed attractor radius
    
    Inputs
    ------
    nmax: int
        number of elements to use
    recsp: np.array
        reconstructed attractor
    """
    D = np.zeros(nmax)
    
    for count in range(nmax):
        D[count] = linalg.norm(recsp[count, :])
        
    return np.mean(D)

def MD_neighbour(i, tahead, recsp):
    """
    This function calculates the nearest neighbors for some particular point along
    the trajectory of the attractor.
    each column in recsp must correspond to one state, each row must
    correspond to one time step!!!!!!!!!
    Note that in the output;
    column 0 of the output corresponds to distances
    column 1 of the output corresponds to index of distances
    
    Inputs
    ------
    i: int
        Euclidean distance in N dimensional reconstructed space 
    tahead: int
        number of timesteps ahead
    recsp: np.array
        the reconstructed attractor   
        
    Returns
    -------
    Q1: np.array
        Q1[:,0]: nearest neighbour distances
        Q1[:,1]: index of the distances in Q1[:,0]
    """
    nmax = len(recsp) - tahead
    Q1 = np.zeros([0, 2])
    
    for j in range(i):
        Q1 = np.vstack([Q1, [linalg.norm(recsp[i, :] - recsp[j, :]), j]])
        
    for j in range(i + 1, nmax):
        Q1 = np.vstack([Q1, [linalg.norm(recsp[i, :] - recsp[j, :]), j]])
        
    p = np.argsort(Q1[:, 0])
    Q1[:, 0] = Q1[p, 0]
    Q1[:, 1] = Q1[p, 1]
    return Q1

def nearest_neighbours(recsp, tauS = 1, tauL = 50, eps_tol = 0.001, nst = 0, ned = 999, theiler_window = 0):
    """
    returns list of nearest neighbours for each point in the attractor
    
    Inputs
    ------
    recsp: np.array
        reconstructed attractor
    tauS: int
        initial relative time
    tauL: int
        final relative time
    eps_tol: float
        tolerance level to define nearest neighbours
    nst: int
        starting index for testing, PYTHON INDEX BEGINS AT 0!
    ned: int
        ending index for testing, -1 because of 0 indexing!
    theiler_window: int
        use a Theiler window to prevent autocorrelation error
        
    Returns
    -------
    EPNb: np.array
        list of nearest neighbours for each point in the reconstructed attractor
    """
    N = range(nst, ned + 1, 1)
    EPNb = [] # list to hold the results

    for k in range(len(N)):
        NbTbl = MD_neighbour(N[k], tauL, recsp)
        [ndt, b] = np.shape(NbTbl)
        epTbl = np.zeros([0, 2])
        
        for kk in range(ndt):
            dist = NbTbl[kk, 0]
            theiler = NbTbl[kk, 1]
            if dist < eps_tol and abs(theiler - N[k]) >= theiler_window:
                epTbl = np.vstack([epTbl, NbTbl[kk, :]])
                
        epList = epTbl[:, 1]
        epList = np.hstack([N[k], epList])
        EPNb.append(epList)
        #print("Time step: {}. No. of NN: {}".format(k, len(epList) - 1))
        
    return np.array(EPNb)

def Stau(recsp, EPNb, tauS = 1, tauL = 50, nst = 0, ned = 999):   
    """
    This function calculates the stretching factor S(Tau) as proposed by
    H. Kantz, 'A robust method to estimate the maximal Lyapunov exponent of a time series'
    Physics Letters A, Volume 185, Issue 1, Pages 77-87, 1994.
    The gradient of S vs. Tau provides the maximal Lyapunov exponent for an orbit.
    
    Inputs
    ------
    recsp: np.array
        reconstructed attractor
    EPBn: np.array
        list of nearest neighbours
    tauS: int
        starting time delay
    tauL: int
        ending time delay
    nst: int
        start index
    ned: int
        end index
    
    Returns
    -------
    x: np.array
        time delay
    y: np.array
        stretching factor. The linear portion of dlog(y)/dlog(x) gives the estimate of the  maximal Lyapunov exponent
    """
    N = range(nst, ned + 1, 1)
    StauTbl = np.zeros([0, 2])
    
    for tad in range(tauS, tauL + 1):
        #print("Now at step: {}".format(tad))
        KDTbl = []
        ntcount = 0
        
        for it in range(len(N)):
            mnear = len(EPNb[it])
            if mnear < 2:
                continue
            
            aheadList = np.ones(mnear) * tad
            epNext = EPNb[it] + aheadList
            KantzDist = 0
            
            for k in range(1, mnear):
                KantzDist = KantzDist + linalg.norm(recsp[int(epNext[0]), :] - recsp[int(epNext[k]), :])
                
            KantzDist = np.log(KantzDist / (mnear - 1))
            KDTbl = np.hstack([KDTbl, KantzDist])
            ntcount = ntcount + 1
            
        StauList = [tad, np.sum(KDTbl) / ntcount]
        StauTbl = np.vstack([StauTbl, StauList]) 
        
    print("    Total points used: {}".format(ntcount))
    print("    Points without nearest neighbours: {}".format(len(N)-ntcount))
    
    x = StauTbl[:, 0] #time delay
    y = StauTbl[:, 1] #stretching factor
    return x, y

def grassberger_procaccia(r, recsp):
    """
    This code shows how to calculate the correlation dimension of a nonlinear system based on the
    equations proposed by:
    Peter Grassberger and Itamar Procaccia (1983). 
    "Measuring the Strangeness of Strange Attractors". Physica D: Nonlinear Phenomena. 9 (1‒2): 189‒208.
    
    Input
    -----
    r: float
        distance in N dimensional space
    recsp: np.array
        reconstructed attractor
    
    Returns
    -------
    float
        Correlation value of Peter Grassberger and Itamar Procaccia (1983).
    """
    [nmax, dd] = np.shape(recsp)
    LL = np.zeros([nmax,nmax])
    
    for i in range(nmax):
        for j in range(nmax):
            LL[i, j] = linalg.norm(recsp[i, :] - recsp[j, :])
            
    NN = np.zeros(nmax)
    
    for i in range(nmax):
        nnl = LL[i, :] 
        nnl = nnl[i:nmax] 
        nnl = np.sort(nnl)
        k = nmax - i - 1
        while nnl[k] > r:
            k = k - 1
            
        NN[i] = k
        
    return np.sum(NN) * 2 / nmax ** 2

def calc_corr_dim(recsp, r0, r1, Nr):
    """
    Using a given range of distances, calculate the correlation dimension
    using the Grassberger Procaccia algorithm.
    
    Inputs
    ------
    recsp: np.array
        reconstructed attractor
    r0: float
        starting distance in N dimensional space
    r1: float
        ending distance in N dimensional space
    Nr: int
        number of distances to test between r0 and r1
        
    Returns
    -------
    r: np.array
        np.array of distances used
    C: np.array
        np.array of correlation values of Peter Grassberger and Itamar Procaccia (1983).
        The linear portion of dlog(C)/dlog(r) gives an estimate of the correlation dimension.
    """
    r = np.linspace(r0, r1, Nr)
    C = np.zeros(len(r))
    
    for q in range(len(r)):
        C[q] = grassberger_procaccia(r[q], recsp)  
        print("Step {} / {} complete...".format(q, len(r)))
        
    return r, C

