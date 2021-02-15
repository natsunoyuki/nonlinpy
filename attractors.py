import numpy as np

def logistic_map_attractor(N = 5000, A = 4, data_0 = np.pi / 10.0):
    """
    Create the chaotic logistic map attractor
    
    Inputs
    ------
    N: int
        number of points to generate
    A: float
        Logistic map factor
    data_0: float
        initial conditions
        
    Returns
    -------
    data: np.array
        1D logistic map values
    """
    data = np.zeros(N)
    data[0] = data_0
    
    for i in range(N - 1):
        data[i + 1] = A * data[i] * (1 - data[i])    
        
    return data

def rossler_attractor(x, t, a, b, c):
    """
    This function creates the Rossler attractor.
    
    Inputs
    ------
    x: np.array
        Rossler attractor ODE x values.
    t: float
        time step value
    a: float
        Rossler attractor parameter a 
    b: float
        Rossler attractor parameter b 
    c: float
        Rossler attractor parameter c 
    Returns
    -------
    xdot: np.array
        Rossler attractor ODE xdot values.
    """    
    xdot = np.zeros(len(x))
    xdot[0] = -x[1] - x[2]  # x
    xdot[1] = x[0] + a * x[1]  # y
    xdot[2] = b + x[2] * (x[0] - c)  # z
    return xdot

def lorenz_attractor(x, t, rho, sigma, beta):
    """
    This function creates the Lorenz attractor.
    
    Inputs
    ------
    x: np.array
        Lorenz attractor ODE x values.
    t: float
        time step value.
    rho: float
        Lorenz attractor parameter rho.
    sigma: float
        Lorenz attractor parameter sigma.
    beta: float
        Lorenz attractor parameter beta.
        
    Returns
    -------
    xdot: np.array
        Lorenz attractor ODE xdot values.
    """        
    xdot = np.zeros(len(x))
    xdot[0] = sigma * (x[1] - x[0])
    xdot[1] = x[0] * (rho - x[2]) - x[1]
    xdot[2] = x[1] * x[0] - beta * x[2]
    return xdot

def julian_tremor_model(x, t, p1):
    """
    This function creates the Julian tremor model.
    
    Inputs
    ------
    x: np.array
        Julian tremor model ODE x values.
    t: float
        time step value.
    p1: float
        Julian tremor model parameter p1.
        
    Returns
    -------
    xdot: np.array
        Julian tremor model ODE xdot values.
    """  
    k = 600*10**6
    M = (3*10**5) * 0
    rho = 2500
    eta = 50
    p2 = 0.1*10**6 
    #p1 = 19*10**6
    h0 = 1
    L = 10
    A = (10**7)*1
    xdot = np.zeros(len(x))
    effectm = M+rho*L**3/12/x[1]
    damping = A+L**3/12/x[1]*(12*eta/x[1]**2-rho/2*x[2]/x[1])
    kcoeff = k*(x[1]-h0)
    Lcoeff = L*(p1+p2)/2-L*rho*x[0]**2/2
    xdot[0] = (p1-p2)/(rho*L)-(12*eta*x[0])/(rho*x[1]**2)
    xdot[1] = x[2]    
    xdot[2] = (Lcoeff-kcoeff-damping*x[2])/effectm
    return xdot
