import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import optimize
from scipy import integrate

def make_orbit_diagram(xdot_fun, x0, t, R):
    """
    Makes the orbit diagram for some 3D ODE system.

    Inputs
    ------
    xdot_fun: function
        function containing the 3D ODE system (set of 3 ODEs)
    x0: np.array
        np.array of initial conditions
    t: np.array
        np.array of time steps to integrate over
    R: np.array
        np.array of ODE parameters to vary in order to create the orbit diagram

    Returns
    -------
    X, Y, Z: list
        lists of steady state solutions for all 3 dimensions of the ODE system
        for each value of R
    """

    # Each element in X, Y and Z is a set of steady state solutions to the 3D 
    # ODE system for each value of the parameter R
    X = []
    Y = []
    Z = []

    for i in range(len(R)):
        # integrate the ODE and get the steady state solution for each R
        r = R[i]
        a = integrate.odeint(xdot_fun, x0, t, args = (r,)) 
        
        # discard the first half of the results to remove transient solutions
        start_index = int(len(t) * 0.5)
        x = a[start_index:, 0]
        y = a[start_index:, 1]
        z = a[start_index:, 2]

        X_max = []
        Y_max = []
        Z_max = []
        X_min = []
        Y_min = []
        Z_min = []

        # as we evaluate the local maximum and minimum using 5 points, 
        # we leave out the first and last 2 elements
        for count in range(2, len(z)-2, 1):
            # for each index, check if x[count] is the local maximum/minimum, 
            # and update the previous and current values for all 3 variables
            if x[count-2] < x[count-1] < x[count] > x[count+1] > x[count+2]:
                X_max.append(x[count])
            if x[count-2] > x[count-1] > x[count] < x[count+1] < x[count+2]:
                X_min.append(x[count])
            if y[count-2] < y[count-1] < y[count] > y[count+1] > y[count+2]:
                Y_max.append(y[count])
            if y[count-2] > y[count-1] > y[count] < y[count+1] < y[count+2]:
                Y_min.append(y[count])
            if z[count-2] < z[count-1] < z[count] > z[count+1] > z[count+2]:
                Z_max.append(z[count])
            if z[count-2] > z[count-1] > z[count] < z[count+1] < z[count+2]:
                Z_min.append(z[count])

        X_max = X_max + X_min
        Y_max = Y_max + Y_min
        Z_max = Z_max + Z_min

        X.append(X_max)
        Y.append(Y_max) 
        Z.append(Z_max)

    return X, Y, Z

def plot_orbit_diagram(R, Z):
    """
    Plots the orbit diagram made by make_orbit_diagram().

    Inputs
    ------
    R: np.array
        np.array of parameter values which was varied
    Z: np.array
        one of the 3 dimensions of the system to plot.
    """
    plt.figure(figsize = (10, 10))
    for i in range(len(R)):
        plt.plot(R[i] * np.ones(len(Z[i])), Z[i], 'k.', markersize=1)
    plt.grid(True)
    plt.xlabel("p1")
    plt.ylabel("Z")
    plt.show()

def demo():
    """
    Demonstrates how the use the functions defined above to create an ODE's 
    orbit diagram
    """
    def xdot_fun(x, t, p1):
        """
        xdot_fun() of ODEs to integrate numerically. In this demonstration we
        use the Julian volcanic tremor model.
        Volcanic tremor: Nonlinear excitation by fluid flow
        Bruce R. Julian
        Journal of Geophysical Research Solid Earth
        Volume 99, Issue B6
        10 June 1994
        Pages 11859-11877

        Inputs
        ------
        x: np.array
            np.array of [x, y, z] of ODE variables
        t: float
            time step
        p1: float
            varying ODE parameter

        Returns
        -------
        xdot: np.array
            np.array of [x_dot, y_dot, z_dot] of ODE variables
        """
        k = 600*10**6
        M = (3*10**5) * 0
        rho = 2500
        eta = 50
        p2 = 0.1*10**6 
        #p1 = 14*10**6 # the parameter p1 is now controlled externally!
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

    # range of parameter values to vary:
    dR = 10**5 / 2
    R = np.arange(10*10**6, 30*10**6 + dR, dR)
    # time steps to try:
    t = np.arange(0, 20, 0.001)
    # initial conditions:
    x0 = np.array([0, 1.02, 0])

    X, Y, Z = make_orbit_diagram(xdot_fun, x0, t, R)
    plot_orbit_diagram(R, Y)
