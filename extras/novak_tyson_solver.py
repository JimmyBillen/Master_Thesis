import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

"""
They are in both ways linear, x(y) linear
but for the other direction it is NOT possible, having a 1/x correlation (harder for ReLU?)
"""



Kd = 1
Et = 1
k2 = 1
p=4
k1 = kdx = kdy = 0.05

Km = 0.1 * Kd

Ki = 2 / Kd

S = 1

print( k2 * Et / Kd)
ksy = 1


def x_dot(x_val, y_val):
    return k1*S*(Kd**p/(Kd**p+y_val**p))-x_val*kdx

def y_dot(xval, yval):
    return ksy*xval - kdy*yval - (k2*Et)*(yval)/(Km+yval+Ki*yval**2)

def compute_novak_dynamics() -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
    """
    Compute the dynamics of the FitzHugh-Nagumo model using Euler's method.

    Returns:
    time (array): Array of time values.
    v_values (array): Array of membrane potential values.
    w_values (array): Array of recovery variable values.

    Notes:
    - The FitzHugh-Nagumo model describes the dynamics of excitable media.
    - It consists of two coupled differential equations for the membrane potential (v)
      and the recovery variable (w).
    - The equations are integrated using Euler's method.
    - The integration parameters depend on the value of TAU (want same points per period).
    """
    # Initial conditions 
    x0 = 0.65178  # Initial value of v
    y0 = 0.65178  # Initial value of w

    # Kpd = 1
    # ksy=1
    # k2=1
    # Et=1
    # Km=1
    # Ki=1

    # Time parameters
    t0=0
    t_end = 1000
    num_steps = 150000


    # Create arrays to store the values
    time = np.linspace(t0, t_end, num_steps + 1) # +1 to work as expected
    h = (t_end - t0) / num_steps
    x_values = np.zeros(num_steps + 1)
    y_values = np.zeros(num_steps + 1)

    # Initialize the values at t0
    x_values[0] = x0
    y_values[0] = y0

    # Implement Euler's method
    for i in range(1, num_steps + 1):
        x_values[i] = x_values[i - 1] + h * x_dot(x_values[i - 1], y_values[i - 1])
        y_values[i] = y_values[i - 1] + h * y_dot(x_values[i - 1], y_values[i - 1])
    
    return time, x_values, y_values

def plot_timeseries():
    # Plot the results
    time, v_values, w_values = compute_novak_dynamics()

    plt.figure(figsize=(10, 5))
    plt.plot(time, v_values, label='v(t)')
    plt.plot(time, w_values, label='w(t)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(rf'Novak-Tsyon')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    plot_timeseries()