import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

def dx4(x, dt):
    """
    This function calulates 1st order central derivative 
    with 4th order errors, hence the name dx4
    Inputs:
    x: np.array
        array to be differentiated
    dt: float
        time step size
    Outputs:
    dx: np.array
        array of differentiated values of x
    """
    dx = np.zeros(len(x) - 4)
    for i in range(2, len(x) - 2):
        dx[i-2] = (-x[i+2] + 8 * x[i+1] - 8 * x[i-1] + x[i-2]) / (dt * 12)
    return dx

def make_dx4(X, dt):
    """
    This is the wrapper function to apply dx4() to a matrix instead of an array
    of nonlinear signals
    Input:
    X: np.array
        matrix (multi-dimensional array) to be differentiated. 
        Rows: time steps and columns: features
    dt: float
        time step size
    Output:
    dX: np.array
        matrix (multi-dimensional array) of differentiated X
    """
    n, m = np.shape(X)
    dX = np.zeros([n-4, m])
    for i in range(m):
        dX[:, i] = dx4(X[:, i], dt)
    return dX

def least_squares(d, G):
    """
    Linear inversion using least squares to get the Penrose pseudo inverse
    for over determined problems.
    For a linear system given by: |d> = G |m>
    |m_est> = Gg |d>
    Minimize the error E = <e|e> where |e> = |d> - G|m>
    Gg = [G.T G]**-1 G.T
    
    Input:
    d: np.array
        inversion data matrix
    G: np.array
        inversion kernel matrix
    Output:
    m: np.array
        inversion model matrix
    """    
    m = np.dot(np.linalg.inv(np.dot(G.T, G)), G.T)
    m = np.dot(m, d)
    return m

def sparse_representation(D, G, m, la = 0.1):
    """
    Sparse Representation Algorithm (Brunton et al. PNAS 2016) to remove 
    extremely small model parameters.
    
    Input:
    D: np.array
        inversion data matrix
    G: np.array
        inversion kernel matrix   
    m: np.array
        inversion model matrix
    la: float
        threshold limit for expansion coefficient values. 0.1 by default
    Output:
    m: np.array
        modified inversion model matrix
    """
    for k in range(10):
        smallinds = abs(m) < la
        m[smallinds] = 0
        for ind in range(np.shape(D)[1]):
            biginds = np.logical_not(smallinds[:, ind])
            M = least_squares(D[:, ind], G[:, biginds])
            m[biginds, ind] = M
    return m

def calculate_m(D, G):
    """
    Perform the linear inversion, as well as calculate the sparse representation of m.
    Input:
    D: np.array
        inversion data matrix
    G: np.array
        inversion kernel matrix
    Output:
    m: np.array
        inversion model matrix
    """
    m = least_squares(D, G)
    m = sparse_representation(D, G, m)   
    return m

def make_G(X, degree):
    """
    Make inversion kernel G using the raw data matrix X.
    We tailor the length in order to account for the lost of 2 rows at the start
    and end of the time series due to the 1st order central differentiation dx4()
    Also, we use power series of polynomials of X to construct the kernel G.
    Input:
    X: raw data matrix
    degree: polynomial degree to be constructed
    Output:
    G: inversion kernel matrix consisting of polynomials of X
    """
    G = X[2:-2]
    polyfeats = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=True)
    G = polyfeats.fit_transform(G)
    return G  
    
def nonlinear_inversion(X, dt, degree):
    """
    This is the main driver function to perform linear inversion of nonlinear signals
    to obtain the ODE model.
    Inputs:
    X: np.array
        the nonlinear signal to be analyzed.
    dt: float
        time step size
    degree: int
        the degree of the inversion kernel matrix
    Outputs:
    D: np.array
        inversion data matrix
    G: np.array
        inversion kernel matrix
    m: np.array
        inversion model matrix
    P: np.array
        inversion predictions of D
    """
    # First of all perform linear inversion of the nonlinear signal
    # Perform the order differentiation
    D = make_dx4(X, dt)
    # Build the kernel for linear inversion/regression
    G = make_G(X, degree)
    # Perform the linear inversion, as well as the sparse representation
    m = calculate_m(D, G)  
    
    # Make the predictions using the model
    P = np.dot(G, m)
    
    return D, G, m, P

def demo():
    """
    This is a demo on how to use the functions above to perform power series
    inversion of a nonlinear system! The Lorenz system is used as the example
    signal.
    """
    # Create nonlinear signal to be inverted for the model parameters
    def lorenz_attractor(x, t, s = 10, B = 8.0 / 3.0, r = 28):
        V = np.zeros(len(x))
        V[0] = s * (x[1] - x[0])
        V[1] = r * x[0] - x[1] - x[0] * x[2]
        V[2] = x[0] * x[1] - B * x[2]
        return V
    dt = 0.01
    t = np.arange(0, 1000, dt)
    x0 = np.array([1, 2, 3])
    a = integrate.odeint(lorenz_attractor, x0, t)
    a = a[-int(len(a) / 10):]
    
    # Perform the nonlinear signal inversion for the ODE parameters!
    D, G, m, P = nonlinear_inversion(a, dt, 2)
    print("Estimated nonlinear ODE parameters:")
    print(m)
    
    # Plot the predictions vs the actual data:
    PX = P[:, 0]
    PY = P[:, 1]   
    PZ = P[:, 2]
    dx = D[:,0] 
    dy = D[:,1]
    dz = D[:,2]
    
    plt.figure(figsize = (10, 5))    
    plt.subplot(3, 1, 1)
    plt.plot(dx)
    plt.plot(PX)
    plt.ylabel(str(np.corrcoef(dx, PX)[0, 1])[:5])
    plt.subplot(3, 1, 2)
    plt.plot(dy)
    plt.plot(PY)
    plt.ylabel(str(np.corrcoef(dy, PY)[0, 1])[:5])
    plt.subplot(3, 1, 3)
    plt.plot(dz)
    plt.plot(PZ)
    plt.ylabel(str(np.corrcoef(dz, PZ)[0, 1])[:5])
    plt.show()
