import numpy as np
from scipy import integrate

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

def rossler_attractor():
    """
    This function creates the Rossler attractor
    
    Returns:
    --------
    a: np.array
        3D Rossler attractor values
    """    
    def xdot_fun(x, t):
        a = 0.2
        b = 0.4
        c = 5.7
        xdot = np.zeros(len(x))
        xdot[0] = -x[1] - x[2]  # x
        xdot[1] = x[0] + a * x[1]  # y
        xdot[2] = b + x[2] * (x[0] - c)  # z
        return xdot

    dt = np.pi / 100.0
    t = np.arange(0, dt * (1048576 + 1000), dt)
    x0 = np.array([10, 0, 0])
    a = integrate.odeint(xdot_fun, x0, t)
    
    return a

def rossler_attractor_x():
    """
    This function creates the x axis of the Rossler attractor
    
    Returns
    -------
    X: np.array
        x axis values of the Rossler attractor
    """

    a = rossler_attractor()    

    X = a[:,0]
    X = X[1000000:]
    
    return X
