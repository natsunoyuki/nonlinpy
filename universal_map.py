import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def universal_map(x, y, z):
    """
    Creates the universal map for the ODE system in x, y and z.

    Inputs
    ------
    x, y, z: np.array
        np.array of time series from a nonlinear ODE.

    Returns
    -------
    X0, Y0, Z0: np.array
        universal map corresponding to x, y and z.
    """
    X0 = []
    Y0 = []
    Z0 = []
    
    previous_local_max_x = x[0]
    local_max_x = x[0]
    previous_local_max_y = y[0]
    local_max_y = y[0]
    previous_local_max_z = z[0]
    local_max_z = z[0]
    
    for t in range(2, len(z)):
        if x[t] < x[t-1] and x[t-1] > x[t-2]:
            previous_local_max_x = local_max_x
            local_max_x = x[t-1]
            X0.append(local_max_x)
        if y[t] < y[t-1] and y[t-1] > y[t-2]:
            previous_local_max_y = local_max_y
            local_max_y = y[t-1]
            Y0.append(local_max_y)
        if z[t] < z[t-1] and z[t-1] > z[t-2]:
            previous_local_max_z = local_max_z
            local_max_z = z[t-1]
            Z0.append(local_max_z)

    return np.array(X0), np.array(Y0), np.array(Z0)

def plot_universal_map(X0, Y0, Z0):
    plt.figure(figsize = (15, 5))
    plt.subplot(1,3,1)
    plt.plot(X0[0:len(X0)-1], X0[1:len(X0)], 'k.')
    plt.xlabel('xmax_n')
    plt.ylabel('xmax_n+1')
    plt.grid(True)
    plt.subplot(1,3,2)
    plt.plot(Y0[0:len(Y0)-1], Y0[1:len(Y0)], 'k.')
    plt.xlabel('ymax_n')
    plt.ylabel('ymax_n+1')
    plt.grid(True)
    plt.subplot(1,3,3)
    plt.plot(Z0[0:len(Z0)-1], Z0[1:len(Z0)], 'k.')
    plt.xlabel('zmax_n')
    plt.ylabel('zmax_n+1')
    plt.grid(True)
    plt.show()

def demo():
    def xdot_fun(x, t, a, b, c):
        xdot = np.zeros(len(x))
        xdot[0] = -x[1] - x[2] # x
        xdot[1] = x[0] + a * x[1] # y
        xdot[2] = b + x[2] * (x[0] - c) # z
        return xdot

    a = 0.398
    b = 2.0
    c = 4.0
    dt = 0.01
    t = np.arange(0, 2000, dt)
    x0 = np.array([-1.0, 0, 0])
    v = integrate.odeint(xdot_fun, x0, t, args = (a, b, c))
    [n, m] = np.shape(v)
    x = v[int(n/2):, 0]
    y = v[int(n/2):, 1]
    z = v[int(n/2):, 2]

    X0, Y0, Z0 = universal_map(x, y, z)
    plot_universal_map(X0, Y0, Z0)