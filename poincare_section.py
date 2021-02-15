import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def poincare_section(x, y, z, plane):
    """
    This script computes the Poincare section in the x-y plane (z = constant by convention)
    moving from under to above the x-y plane (along z = constant by convention)
    
    Inputs
    ------
    x, y, z: np.array
        np.array of time series from a nonlinear ODE.
    plane: float
        value of the plane (perpendicular to the z-axis by convention) of the Poincare map.
    
    Returns
    -------
    x_surface, y_surface
    """
    assert len(x) == len(y) == len(z)

    old_distance = 0
    k = 0
    x_surface = []
    y_surface = []
    
    for i in range(len(z)):
        new_distance = z[i]
        if new_distance >= plane and old_distance < plane:
            total_distance = new_distance - old_distance
            x_surface = np.hstack([x_surface, x[i-1] - old_distance / total_distance * (x[i] - x[i-1])])
            y_surface = np.hstack([y_surface, y[i-1] - old_distance / total_distance * (y[i] - y[i-1])])
            k = k + 1
        old_distance = new_distance
        
    return x_surface, y_surface

def plot_poincare_section(x_surface, y_surface):
    plt.figure(figsize = (15, 5))    
    plt.plot(x_surface, y_surface, '.')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

def demo():
    def xdot_fun(x, t, rho, sigma, beta):
        xdot = np.zeros(len(x))
        xdot[0] = sigma * (x[1] - x[0])
        xdot[1] = x[0] * (rho - x[2]) - x[1]
        xdot[2] = x[0] * x[1] - beta * x[2]
        return xdot

    a = 28.0
    b = 10.0
    c = 8.0 / 3.0
    dt = 0.01
    t = np.arange(0, 200, dt)
    x0 = np.array([1.0, 0, 0])
    v = integrate.odeint(xdot_fun, x0, t, args = (a, b, c))
    [n, m] = np.shape(v)
    x = v[int(n*0.7):, 0]
    y = v[int(n*0.7):, 1]
    z = v[int(n*0.7):, 2]

    x_s, y_s = poincare_section(x, y, z, 25)
    plot_poincare_section(x_s, y_s)