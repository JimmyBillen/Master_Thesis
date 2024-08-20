import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

"""
Deze heb ik niet getoond maar beide is x(y) richting linear
maar andere richting is het 1keer NIET mogelijk, en andere keer een 1/x connectie (moeilijker voor ReLU?)

"""

# constants
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

# Define the functions
def x_dot(x_val, y_val):
    return k1 * S * (Kd**p / (Kd**p + y_val**p)) - x_val * kdx

def y_dot(x_val, y_val):
    return ksy * x_val - kdy * y_val - (k2 * Et) * (y_val) / (Km + y_val + Ki * y_val**2)


def compute_novak_dynamics(num_steps=15000) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
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
    t_end = 365.85
    num_steps = num_steps #standard 15000

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
    plt.title(rf'Novak-Tyson')
    plt.grid()
    plt.show()

def plot_timeseries_compare():
    """
    Compare euler convergence for 15k and 150k timesteps. Visually the same (seen with zooming).
    """
    ax, fig = plt.subplots(figsize=(10,5))
    # Plot the results
    time, v_values, w_values = compute_novak_dynamics(15000)
    plt.plot(time, v_values, label='v(t) 15k', color='C0', alpha=0.5)
    plt.plot(time, w_values, label='w(t) 15k', color='C1', alpha=0.5)

    time_2, v_values_2, w_values_2 = compute_novak_dynamics(150000)
    plt.plot(time, v_values, label='v(t) 150k', linestyle='--', color='C3')
    plt.plot(time, w_values, label='w(t) 150k', linestyle='--', color='C4')

    # plt.plot(time, v_values, label='v(t)')
    # plt.plot(time, w_values, label='w(t)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(rf'Novak-Tyson')
    # plt.grid()
    plt.show()

def derivative_plotter():
    """Analysis shows that difference in derivative methods:
    finite difference method: 1) forward, 2)center, 3)backwards only differ 
    by 0.06 at the 'delta' place, compared to the height of 1.3 of the delta,
    so accuracy wise does not matter which one you take!
    Even though accuracy wise center method might be better, we have chosen
    forward method throughout the thesis (except the last point, uses backward)
    """

    from derivate import calculate_derivatives

    time, x_t_data, y_t_data = compute_novak_dynamics() # assigning v->v, w->v see heads-up above.
    u_dot_t_data = np.array(calculate_derivatives(time, y_t_data))
    v_dot_t_data = np.array(calculate_derivatives(time, x_t_data))
    print(len(time), len(x_t_data), len(u_dot_t_data), len(v_dot_t_data))

    plt.plot(time, x_t_data, label=r"$x(t)$", color='C0')
    plt.plot(time, y_t_data, label=r"$y(t)$", color='C1')
    plt.plot(time, v_dot_t_data, label=r"$x'(t)$", color='C2')
    # plt.plot(time, u_dot_t_data, label=r"$u'(t)$")
    # plt.title(f"Time Series of $u,v$ and the derivatives of tau={TAU}")
    plt.hlines(y=0, xmin=min(time), xmax=max(time), colors='black', label='y=0')
    # plt.ylim(-2, 1.5)
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    # plot_timeseries_compare()
    plot_timeseries()
    # derivative_plotter()