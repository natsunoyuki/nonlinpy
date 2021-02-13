import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time

# this script demonstrates the concept of observability of a nonlinear system

def observability(x, y, z, J):
    """
    Inputs
    ------
    x, y, z: np.array
        np.array of time series from a nonlinear ODE.
    J: np.array
        ODE Jacobian matrix corresponding to x, y and z.
        
    Returns
    -------
    dx, dy, dz: np.array
        observability of x, y and z.
    """
    assert len(x) == len(y) == len(z) == len(J)
    Cx = np.array([1, 0, 0])
    Cy = np.array([0, 1, 0])
    Cz = np.array([0, 0, 1])

    dx = np.zeros(len(x))
    dy = np.zeros(len(y))
    dz = np.zeros(len(z))

    for i in range(len(x)):
        Ai = J[i]
        Qx = np.vstack([Cx, np.dot(Cx, Ai), np.dot(Cx, np.dot(Ai, Ai))])
        Qy = np.vstack([Cy, np.dot(Cy, Ai), np.dot(Cy, np.dot(Ai, Ai))])
        Qz = np.vstack([Cz, np.dot(Cz, Ai), np.dot(Cz, np.dot(Ai, Ai))])
        QQx = np.dot(Qx, np.transpose(Qx))
        QQy = np.dot(Qy, np.transpose(Qy))
        QQz = np.dot(Qz, np.transpose(Qz))
        [ux,vx] = np.linalg.eig(QQx)
        [uy,vy] = np.linalg.eig(QQy)
        [uz,vz] = np.linalg.eig(QQz)
        ux = np.real(ux)
        uy = np.real(uy)
        uz = np.real(uz)
        dx[i] = np.abs(np.min(ux)) / np.abs(np.max(ux))
        dy[i] = np.abs(np.min(uy)) / np.abs(np.max(uy))
        dz[i] = np.abs(np.min(uz)) / np.abs(np.max(uz))

    return dx, dy, dz

def plot_obs(dx, dy, dz):
    plt.figure(figsize = (15, 5))
    plt.subplot(3,1,1)
    plt.plot(dx[0:int(len(dx)/2)])
    plt.subplot(3,1,2)
    plt.plot(dy[0:int(len(dy)/2)])
    plt.subplot(3,1,3)
    plt.plot(dz[0:int(len(dz)/2)])
    plt.show()

def demo():
    # use Rossler system as an example
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
    
    J = []
    for i in range(len(x)):
        J.append(np.array([[0, -1, -1], [1, a, 0], [z[i], 0, x[i] - c]]))
    
    dx, dy, dz = observability(x, y, z, J)
    plot_obs(dx, dy, dz)