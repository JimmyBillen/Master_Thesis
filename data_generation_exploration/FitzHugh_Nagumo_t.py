# FitzHugh-Nagumo (t: in function of time), ODE of wiki and solved using Euler's method
# Goal to plot v(t) and w(t)

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# Constants
from .. import settings
from settings import R, I, TAU, A, B, NUM_OF_POINTS
# R = 0.1
# I = 10
# TAU = 7.5/1/10/100
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

def w_dot(v, w, A, B, TAU):
    """
    The differential equation describing the relaxation of the FitzHugh-Nagumo model.
    """
    return (v + A - B * w) / TAU

def compute_fitzhugh_nagumo_dynamics(num_points=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
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
    v0 = 1.0  # Initial value of v
    w0 = 2.0  # Initial value of w

    # Time parameters
    if TAU == 7.5:
        t0 = 0.0  # Initial time
        t_end = 150.0  # End time
        num_steps = 15000

    if TAU == 1:
        t0 = 0.0
        t_end = 65.5
        num_steps = 15000

    if TAU == 2:
        t0 = 0.0
        t_end = 72.27
        num_steps = 15000
    
    if TAU == 5:
        t0 = 0.0
        t_end = 116.6
        num_steps = 15000
        # nog niks mee gedaan

    if TAU == 6:
        t0 = 0.0
        t_end = 131.25
        num_steps = 15000

    if TAU == 10:
        t0 = 0.0
        t_end = 187.0
        num_steps = 15000

    if TAU == 20:
        t0 = 0.0
        t_end = 318.8
        num_steps = 15000  

    if TAU == 25:
        t0 = 0.0
        t_end = 382.1
        num_steps = 15000  

    if TAU == 40:
        t0 = 0.0
        t_end = 567.2
        num_steps = 15000

    if TAU == 50:
        t0 = 0.0
        t_end = 688.2
        num_steps = 15000

    if TAU == 60:
        t0 = 0.0
        t_end = 807.9
        num_steps = 15000

    if TAU == 80:
        t0 = 0.0
        t_end = 1044.8
        num_steps = 15000

    if TAU == 100:
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
        w_values[i] = w_values[i - 1] + h * w_dot(v_values[i - 1], w_values[i - 1], A, B, TAU)
    
    if num_points is None:
        sampled_time = sample_subarray(time, NUM_OF_POINTS)
        sampled_v_values = sample_subarray(v_values, NUM_OF_POINTS)
        sampled_w_values = sample_subarray(w_values, NUM_OF_POINTS)
    else:
        input(f"Are you sure you don't want to use global NUM_OF_POINTS={NUM_OF_POINTS}?, if 'no': cancel by CONTROL+C")
        sampled_time = sample_subarray(time, num_points)
        sampled_v_values = sample_subarray(v_values, num_points)
        sampled_w_values = sample_subarray(w_values, num_points)

    return sampled_time, sampled_v_values, sampled_w_values

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
    plt.title(rf'FitzHugh-Nagumo in Function of Time for $\tau$={TAU}')
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
    print("Boundary of option 1 are synthetically chosen, not unbiased")
    # option 2
    low_opt2_w, high_opt2_w = calc_boundary_nullclines(time, w)

    # option 3
    low_opt3_v, high_opt3_v = calc_boundary_nullclines(time, v)
    # low_opt1_v, high_opt1_v = low_opt3_v, high_opt3_v
    # input('change asap in _t file')
    # option 4:
    # solving w=v-v^3/3 + RI => boundary v^2=1 => v=+-1, filling back in w: +-1-+1/3+R*I
    print('boundary for option 4 are for tau=7.5')
    low_opt4_w = 0.0954 
    high_opt4_w = 1.8481

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


def derivative_plotter():
    """Analysis shows that difference in derivative methods:
    finite difference method: 1) forward, 2)center, 3)backwards only differ 
    by 0.06 at the 'delta' place, compared to the height of 1.3 of the delta,
    so accuracy wise does not matter which one you take!
    We have chosen forward method throughout the thesis (except the last point, uses backward)
    """

    from create_NN_FHN import calculate_derivatives

    time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics() # assigning v->v, w->v see heads-up above.
    u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
    v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))
    print(len(time), len(v_t_data), len(u_dot_t_data), len(v_dot_t_data))

    plt.plot(time, v_t_data, label=r"$v(t)$", color='C0')
    plt.plot(time, u_t_data, label=r"$u(t)$", color='C1')
    plt.plot(time, v_dot_t_data, label=r"$v'(t)$", color='C2')
    # plt.plot(time, u_dot_t_data, label=r"$u'(t)$")
    plt.title(f"Time Series of $u,v$ and the derivatives of tau={TAU}")
    plt.hlines(y=0, xmin=min(time), xmax=max(time), colors='black', label='y=0')
    # plt.ylim(-2, 1.5)
    plt.legend(loc='upper right')
    plt.show()

    # plt.plot(time, v_dot_t_data, label=r"$v(t)$")
    from FitzHugh_Nagumo_ps import nullcline_vdot
    plt.plot(time, nullcline_vdot(v_t_data), label='Nullcline vdot', color='C4')
    plt.plot(time, v_t_data, label=r"$v(t)$", alpha=0.7, color='C0')
    plt.plot(time, u_t_data, label=r"$u(t)$", alpha=0.7, color='C1')
    plt.plot(time, v_dot_t_data, label=r"$v'(t)$", color='C2')
    plt.legend()
    plt.show()

def changing_num_points_plotter():
    """plots only the derivatives of u and v"""

    from create_NN_FHN import calculate_derivatives

    # # Define base colors
    # v_base_color = np.array([1, 0, 0])  # Red for v
    # u_base_color = np.array([0, 1, 0])  # Green for u

    # num_of_points_deriv = [500, 700, 1_000, 5000, 15_000]
    # for i, num_points in enumerate(num_of_points_deriv):
    #     time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics(num_points) # assigning v->v, w->v

    #     u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
    #     v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

    #     # print(len(time), len(v_t_data), len(u_dot_t_data), len(v_dot_t_data))

    #     alpha=0.6
    #     v_color = v_base_color * (1 - 0.1 * i)  # Slightly vary color for v
    #     u_color = u_base_color * (1 - 0.1 * i)  # Slightly vary color for u
    #     plt.plot(time, u_t_data, label=rf"$u(t)$ {num_points}", color=np.append(u_base_color, alpha))
    #     plt.plot(time, v_t_data, label=rf"$v(t)$ {num_points}", color=np.append(v_color, alpha))

    #     # plt.plot(time, u_dot_t_data, label=r"$u'(t)$")
    # plt.title(rf"Time Series of $v$ and $u$ and the derivatives at tau={TAU}")
    # plt.hlines(y=0, xmin=min(time), xmax=max(time), colors='black', label='y=0')
    #     # plt.ylim(-2, 1.5)
    # plt.legend(loc='upper right')
    # plt.show()

    # num_of_points_deriv = [500, 700, 1_000, 5000, 15_000]
    # for i, num_points in enumerate(num_of_points_deriv):
    #     time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics(num_points) # assigning v->v, w->v

    #     u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
    #     v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

    #     # print(len(time), len(v_t_data), len(u_dot_t_data), len(v_dot_t_data))

    #     # plt.plot(time, u_t_data, label=rf"$u(t)$ {num_points}")
    #     plt.plot(time, v_t_data, label=rf"$v(t)$ {num_points}")

    #     # plt.plot(time, u_dot_t_data, label=r"$u'(t)$")
    # plt.title(rf"Time Series of $v$ and $u$ and the derivatives at tau={TAU}")
    #     # plt.ylim(-2, 1.5)
    # plt.legend(loc='upper right')
    # plt.show()

    # time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics() # assigning v->v, w->v see heads-up above.

    fig, ax = plt.subplots(1,2)
    fig.set_figheight(3)
    fig.set_figwidth(6)


    num_of_points_deriv = [700, 800, 900, 5000, 15_000]
    markersizes=[5,4,3,2,1]
    for markersize, num_points in zip(markersizes,num_of_points_deriv):
        time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics(num_points) # assigning v->v, w->v

        u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
        v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

        # print(len(time), len(v_t_data), len(u_dot_t_data), len(v_dot_t_data))

        ax[0].plot(time, v_dot_t_data)
    
        # plt.plot(time, u_dot_t_data, label=r"$u'(t)$")
    ax[0].hlines(y=0, xmin=min(time), xmax=max(time), colors='black', label='y=0', alpha=0.6, linestyle='dashed')


    num_of_points_deriv = [700, 800, 900, 5000, 15_000]
    markersizes=[5,4,3,2,1]
    for markersize, num_points in zip(markersizes,num_of_points_deriv):
        time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics(num_points) # assigning v->v, w->v

        u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
        v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

        # print(len(time), len(v_t_data), len(u_dot_t_data), len(v_dot_t_data))

        ax[1].plot(time, v_dot_t_data, label=rf"{num_points}", marker="D", markersize=markersize, alpha=0.8)
        # plt.plot(time, u_dot_t_data, label=r"$u'(t)$")
    ax[1].set_xlim(525, 537)
    ax[1].set_ylim(-0.07, 1.45)
    # plt.title(f"Derivative of v in time ")
    ax[1].hlines(y=0, xmin=min(time), xmax=max(time), colors='black', label='y=0', alpha=0.6, linestyle='dashed')
        # plt.ylim(-2, 1.5)
    ax[1].legend(loc='upper left')

    # ax[0].set_yticklabels([])
    ax[0].set_xticks([0, 1000])
    ax[0].set_yticks([-1.5, 0, 1.5])

    ax[1].set_xticks([526, 536])
    ax[1].set_yticks([0, 1.4])

    # ax[1].set_yticklabels([])
    # ax[1].set_xticklabels([])

    # ax[0].set_xticks([])
    # ax[0].set_yticks([])


    ax[0].set_xlabel("Time", labelpad=-10)
    ax[1].set_xlabel("Time", labelpad=-10)
    ax[0].set_ylabel(r"$v$'", labelpad=-10)


    plt.tight_layout()
    plt.subplots_adjust(top=0.979,
bottom=0.089,
left=0.087,
right=0.99,
hspace=0.2,
wspace=0.195)
    
    import matplotlib as mpl
    mpl.rc("savefig", dpi=300)
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\Time_resolution\changing derivative.png')

    plt.show()

    # num_of_points_deriv = [500, 700, 1_000, 5000, 15_000]
    # for num_points in num_of_points_deriv:
    #     time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics(num_points) # assigning v->v, w->v

    #     u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
    #     v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

    #     plt.plot(time, u_dot_t_data, label=rf"$u'(t)$ {num_points}")

    # # plt.plot(time, u_dot_t_data, label=r"$u'(t)$")
    # plt.title(f"$u'$ Derivative Time Series at tau={TAU}")
    # plt.hlines(y=0, xmin=min(time), xmax=max(time), colors='black', label='y=0')
    # # plt.ylim(-2, 1.5)
    # plt.legend(loc='upper right')
    # plt.show()

if __name__ == '__main__':
    # derivative_plotter()
    # plot_timeseries()
    find_boundary_nullclines()
    # changing_num_points_plotter()