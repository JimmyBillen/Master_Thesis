# FitzHugh-Nagumo (t: in function of time), ODE of wiki and solved using Euler's method
# Goal to plot v(t) and w(t)

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

R = 0.1
I = 10
tau = 100
A = 0.7
B = 0.8
NUM_OF_POINTS = 15000


# # Constants
# from settings import R, I, tau, A, B, NUM_OF_POINTS
# R = 0.1
# I = 10
# tau = 7.5/1/10/100
# A = 0.7
# B = 0.8

def sample_subarray(original_array, new_size):
    # Ensure the new size is less than the original array size
    if new_size >= len(original_array):
        raise ValueError("New size must be less than the original array size")

    # Calculate the size of the inner segment to sample
    inner_size = new_size - 2  # Exclude the first and last elements
    total_size = len(original_array) - 2  # Exclude the first and last elements
    
    # Generate indices for the inner segment
    inner_indices = np.linspace(1, total_size, inner_size, dtype=int)
    
    # Construct the sampled subarray
    sampled_array = [original_array[0]]  # Start with the first element
    sampled_array.extend(original_array[i] for i in inner_indices)  # Add the sampled points
    sampled_array.append(original_array[-1])  # End with the last element
    
    return np.array(sampled_array)


# Define the ODEs (https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model)
def v_dot(v, w, I, R):
    """
    The differential equation describing the voltage of the FitzHugh-Nagumo model.
    """
    return v - (v**3) / 3 - w + R * I

def w_dot(v, w, A, B, tau):
    """
    The differential equation describing the relaxation of the FitzHugh-Nagumo model.
    """
    return (v + A - B * w) / tau

def compute_fitzhugh_nagumo_dynamics(tau) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
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
    - The integration parameters depend on the value of tau (want same points per period).
    """
    # Initial conditions 
    v0 = 1.0  # Initial value of v
    w0 = 2.0  # Initial value of w

    # Time parameters
    if tau == 7.5:
        t0 = 0.0  # Initial time
        t_end = 150.0  # End time
        num_steps = 15000

    if tau == 1:
        t0 = 0.0
        t_end = 65.5
        num_steps = 15000

    if tau == 2:
        t0 = 0.0
        t_end = 72.27
        num_steps = 15000
    
    if tau == 5:
        t0 = 0.0
        t_end = 116.6
        num_steps = 15000
        # nog niks mee gedaan

    if tau == 6:
        t0 = 0.0
        t_end = 131.25
        num_steps = 15000

    if tau == 10:
        t0 = 0.0
        t_end = 187.0
        num_steps = 15000

    if tau == 20:
        t0 = 0.0
        t_end = 318.8
        num_steps = 15000  

    if tau == 25:
        t0 = 0.0
        t_end = 382.1
        num_steps = 15000  

    if tau == 40:
        t0 = 0.0
        t_end = 567.2
        num_steps = 15000

    if tau == 50:
        t0 = 0.0
        t_end = 688.2
        num_steps = 15000

    if tau == 60:
        t0 = 0.0
        t_end = 807.9
        num_steps = 15000

    if tau == 80:
        t0 = 0.0
        t_end = 1044.8
        num_steps = 15000

    if tau == 100:
        t0 = 0.0
        t_end = 1279.0
        num_steps = 15000
    

    # Create arrays to store the values
    time = np.linspace(t0, t_end, num_steps + 1) # +1 to work as expected
    h = (t_end - t0) / num_steps
    v_values = np.zeros(num_steps + 1)
    w_values = np.zeros(num_steps + 1)

    # Initialize the values at t0
    v_values[0] = v0
    w_values[0] = w0

    # Implement Euler's method
    for i in range(1, num_steps + 1):
        v_values[i] = v_values[i - 1] + h * v_dot(v_values[i - 1], w_values[i - 1], I, R)
        w_values[i] = w_values[i - 1] + h * w_dot(v_values[i - 1], w_values[i - 1], A, B, tau)
    

    return time, v_values, w_values

def plot_timeseries():
    # Plot the results
    time, v_values, w_values = compute_fitzhugh_nagumo_dynamics()

    plt.figure(figsize=(10, 5))
    plt.plot(time, v_values, label='v(t)')
    plt.plot(time, w_values, label='w(t)')
    print('for v, difference in time between maxima', find_maxima_differences(time, v_values) ,'\n')
    print('for w, difference in time between maxima', find_maxima_differences(time, w_values))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(rf'FitzHugh-Nagumo in Function of Time for $\tau$={tau}')
    plt.grid()
    plt.show()
    print(len(time), len(v_values), len(w_values))

def find_local_maxima(x, y, variable=""):
    local_maxima_index = []
    maxima_x_value = []
    local_maxima_value = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            local_maxima_index.append(i)
            maxima_x_value.append(x[i])
            local_maxima_value.append(y[i])
    print('local maxima happens for t-values:', maxima_x_value)

    local_minima_value = []
    for j in range(1, len(y) - 1):
        if y[j] < y[j-1] and y[j] < y[j+1]:
            local_minima_value.append(y[j])
    print('Maxima values are', local_maxima_value, local_minima_value)
    print(variable, 'boundary, option')

    return local_maxima_index

def find_maxima_differences(x, y, variable='v'):
    """"""
    local_maxima_indices = find_local_maxima(x, y, variable)
    differences = []
    for i in range(len(local_maxima_indices) - 1):
        diff = x[local_maxima_indices[i + 1]] - x[local_maxima_indices[i]]
        differences.append(diff)
    return differences

def find_boundary_nullclines():
    # vdot nullcline (cubic): take lowest v-value and high v-value
    time, v, w = compute_fitzhugh_nagumo_dynamics()

    # option 1
    low_opt1_v, high_opt1_v = inverse_boundary_nullclines(time, w, v)

    # option 2
    low_opt2_w, high_opt2_w = calc_boundary_nullclines(time, w)

    # option 3
    low_opt3_v, high_opt3_v = calc_boundary_nullclines(time, v)

    # option 4:
    # solving w=v-v^3/3 + RI => boundary v^2=1 => v=+-1, filling back in w: +-1-+1/3+R*I
    low_opt4_w = -1 + (1/3) + R * I
    high_opt4_w = 1 - (1/3) + R * I

    boundary_nullclines = {"option_1": [low_opt1_v, high_opt1_v],
                           "option_2": [low_opt2_w, high_opt2_w],
                           "option_3": [low_opt3_v, high_opt3_v],
                           "option_4": [low_opt4_w, high_opt4_w]}
    print(boundary_nullclines)
    return boundary_nullclines

def calc_boundary_nullclines(time, y):
    """Calculates the boundary of the nullcline
    
    for v: ifo v
    for w: ifo w
    """
    local_maxima_value = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            local_maxima_value.append(y[i])

    local_minima_value = []
    for j in range(1, len(y) - 1):
        if y[j] < y[j-1] and y[j] < y[j+1]:
            local_minima_value.append(y[j])

    # take second-last value
    low_limit = local_minima_value[-2]
    high_limit = local_maxima_value[-2]
    return low_limit, high_limit

def inverse_boundary_nullclines(time, y1, y2):
    """Calculates the boundary of the nullcline
    
    for v: ifo w (!)
    for w: ifo v (!)
    """
    local_maxima_value = []
    for i in range(1, len(y1) - 1):
        if y1[i] > y1[i - 1] and y1[i] > y1[i + 1]:
            local_maxima_value.append(y2[i])

    local_minima_value = []
    for j in range(1, len(y1) - 1):
        if y1[j] < y1[j-1] and y1[j] < y1[j+1]:
            local_minima_value.append(y2[j])

    # take second-last value
    low_limit = local_minima_value[-2]
    high_limit = local_maxima_value[-2]
    return low_limit, high_limit

# calculating derivatives using finite difference method
def forward_difference(x_values, y_values, begin=0, end=None):
    if end is None:
        end = len(x_values)-1
    derivatives = [(y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i]) for i in range(begin, end)]
    return derivatives

def centered_difference(x_values, y_values, begin=1, end=None):
    if end is None:
        end=len(x_values)-1
    derivatives = [(y_values[i + 1] - y_values[i - 1]) / (x_values[i + 1] - x_values[i - 1]) for i in range(begin, end)]
    return derivatives

def backward_difference(x_values, y_values, begin=1, end=None):
    if end is None:
        end=len(x_values)
    derivatives = [(y_values[i] - y_values[i - 1]) / (x_values[i] - x_values[i - 1]) for i in range(begin, end)]
    return derivatives

def calculate_derivatives(values, h):
    forward_deriv = forward_difference(values, h, begin=0, end = len(values)-1)
    backward_deriv = backward_difference(values, h, begin=len(values)-1, end=len(values))

    return forward_deriv + backward_deriv


def derivative_plotter():
    """Analysis shows that difference in derivative methods:
    finite difference method: 1) forward, 2)center, 3)backwards only differ 
    by 0.06 at the 'delta' place, compared to the height of 1.3 of the delta,
    so accuracy wise does not matter which one you take!
    We have chosen forward method throughout the thesis (except the last point, uses backward)
    """

    plt.figure(figsize=(2,2))
    time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics(7.5) # assigning v->v, w->v see heads-up above.
    u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
    v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))
    print(len(time), len(v_t_data), len(u_dot_t_data), len(v_dot_t_data))

    # plt.plot(time, v_t_data, label=r"$v(t)$", color='C0')
    # plt.plot(time, u_t_data, label=r"$u(t)$", color='C1')
    x_axis = np.linspace(0,100, len(time))
    plt.plot(x_axis, v_dot_t_data, label=r"$\tau=7.5$", color='C2', linestyle='dashed')

    time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics(100) # assigning v->v, w->v see heads-up above.
    u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
    v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))
    print(len(time), len(v_t_data), len(u_dot_t_data), len(v_dot_t_data))
    x_axis = np.linspace(0,100, len(time))
    plt.plot(x_axis, v_dot_t_data, label=r"$\tau=100$", color='C2')



    # plt.plot(time, u_dot_t_data, label=r"$u'(t)$")
    # plt.title(f"Time Series of $u,v$ and the derivatives of tau={tau}")
    plt.hlines(y=0, xmin=min(x_axis), xmax=max(x_axis), colors='black', label='y=0', alpha=0.6, zorder=0, linestyle=':')
    plt.xlim(20, 37)
    plt.xticks([])
    plt.xlabel('Time')
    plt.ylabel("Amount in arb. units", labelpad=-2)
    plt.legend(loc='lower left', borderpad=0.4)

    plt.subplots_adjust(top=0.99,
bottom=0.11,
left=0.234,
right=0.985,
hspace=0.2,
wspace=0.2)
    
    import matplotlib as mpl
    mpl.rc("savefig", dpi=300)
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\VaryingTimeScaleSeparation\DerivativeTimeseries.png')

    plt.show()

    # plt.plot(time, v_dot_t_data, label=r"$v(t)$")
    # from FitzHugh_Nagumo_ps import nullcline_vdot
    # plt.plot(time, nullcline_vdot(v_t_data), label='Nullcline vdot', color='C4')
    # plt.plot(time, v_t_data, label=r"$v(t)$", alpha=0.7, color='C0')
    # plt.plot(time, u_t_data, label=r"$u(t)$", alpha=0.7, color='C1')
    # plt.plot(time, v_dot_t_data, label=r"$v'(t)$", color='C2')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    derivative_plotter()
    # plot_timeseries()
    # find_boundary_nullclines()
    # changing_num_points_plotter()