# FitzHugh-Nagumo (t: in function of time), ODE of wiki and solved using Euler's method
# Goal to plot v(t) and w(t)

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# Constants
R = 0.1
I = 10
TAU = 7.5
A = 0.7
B = 0.8
NUM_OF_POINTS = 15000

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
def v_dot(v, w):
    """
    The differential equation describing the voltage of the FitzHugh-Nagumo model.
    """
    return v - (v**3) / 3 - w + R * I

def w_dot(v, w):
    """
    The differential equation describing the relaxation of the FitzHugh-Nagumo model.
    """
    return (v + A - B * w) / TAU

def nullcline_vdot(v):
    """
    Calculate the cubic nullcline function w(v).

    Parameters:
    - v: Input value.

    Returns:
    - Result of the nullcline function.
    """
    return v - (1/3)*v**3 + R * I

def nullcline_wdot(v):
    """Calculates the linear nullcline function w(v)
    
    Parameters:
    - v: Input value.

    Returns:
    - Result of the nullcline function.    
    """
    return (v + A) / B

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
        t_end = 50.0  # End time
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
        v_values[i] = v_values[i - 1] + h * v_dot(v_values[i - 1], w_values[i - 1])
        w_values[i] = w_values[i - 1] + h * w_dot(v_values[i - 1], w_values[i - 1])

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

    mean_val = np.min(time)
    std_dev = np.max(time) - np.min(time)
    time = (time - mean_val) / std_dev    

    mean_val = np.min(v_values)
    std_dev = np.max(v_values) - np.min(v_values)
    v_values = (v_values - mean_val) / std_dev    

    mean_val = np.min(w_values)
    std_dev = np.max(w_values) - np.min(w_values)
    w_values = (w_values - mean_val) / std_dev    

    v_values = v_values*0.6+0.1
    w_values = w_values*0.7+0.16

    plt.figure(figsize=(2.5, 2.5))
    plt.plot(time, v_values, label=r'$x_1$', color='red', linestyle='dashed')
    plt.plot(time, w_values, label=r'$x_2$', color='red')
    plt.xlabel('Time', loc='right', fontsize=12)
    plt.ylabel(r'$x$', loc='top', rotation=0, fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=[0.53, 1.04], ncols=2, frameon=True, fontsize=12)
    plt.title('Time Series')
    # plt.grid()
    
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['right'].set_color('none')

    # Add arrows at the end of x and y axes
    plt.gca().spines['left'].set_position(('data', 0))
    plt.gca().spines['bottom'].set_position(('data', 0))

    plt.gca().spines['left'].set_bounds(0, 1)
    plt.gca().spines['bottom'].set_bounds(0, 1)

    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

    plt.gca().plot(1, 0, ">k", transform=plt.gca().get_xaxis_transform(), clip_on=False)
    plt.gca().plot(0, 1, "^k", transform=plt.gca().get_yaxis_transform(), clip_on=False)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Remove ticks
    plt.xticks([])
    plt.yticks([])

    import matplotlib as mpl
    mpl.rc("savefig", dpi=300)
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Introduction\Timeseries.png')

    # Show the plot
    plt.show()


def plot_phase_space_hidden_nullcline():
    time, v_values, w_values = compute_fitzhugh_nagumo_dynamics()



    mean_val = np.min(time)
    std_dev = np.max(time) - np.min(time)
    time = (time - mean_val) / std_dev    

    mean_val = np.min(v_values)
    std_dev = np.max(v_values) - np.min(v_values)
    v_values = (v_values - mean_val) / std_dev    

    mean_val = np.min(w_values)
    std_dev = np.max(w_values) - np.min(w_values)
    w_values = (w_values - mean_val) / std_dev    

    v_values = v_values*0.7+0.1
    w_values = w_values*0.76+0.05

    plt.figure(figsize=(2.5, 2.5))
    plt.plot(v_values, w_values, label=r'$Trajectory$', color='red')


    x = np.linspace(-2.5, 3, 10000)
    mean_val = np.min(x)
    std_dev = np.max(x) - np.min(x)
    x_mean = (x - mean_val) / std_dev    

    scale = 0.35
    # plt.plot(x_mean, scale*nullcline_vdot(x), label=r'nullcline 1', color='lime', alpha=0.5, linestyle=':')
    # plt.plot(x_mean, scale*nullcline_wdot(x), label='nullcline 2', color='cyan', alpha=0.5, linestyle=':')
    plt.plot(x_mean, scale*nullcline_vdot(x), color='black', alpha=0.1)
    plt.plot(x_mean, scale*nullcline_wdot(x), color='black', alpha=0.1)
    plt.scatter([0.528884], [0.485130], color='black', alpha=0.1, edgecolors='none')
    # plt.plot(x_mean, scale*nullcline_vdot(x), label=r'nullcline 1', color='grey', alpha=0.5, linestyle=':')
    # plt.plot(x_mean, scale*nullcline_wdot(x), label='nullcline 2', color='grey', alpha=0.5, linestyle=':')


    plt.plot()

    plt.xlabel(r'$x_1$', loc='right', fontsize=14)
    plt.ylabel(r'$x_2$', loc='top', rotation=0, fontsize=14)
    # plt.legend(loc='upper center', ncols=2, frameon=True)
    plt.legend(loc='upper center', bbox_to_anchor=[0.53, 1.04], ncols=2, frameon=True, fontsize=10)

    plt.title('Phase Space')
    # plt.grid()
    
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['right'].set_color('none')

    # Add arrows at the end of x and y axes
    plt.gca().spines['left'].set_position(('data', 0))
    plt.gca().spines['bottom'].set_position(('data', 0))

    xmin, xmax = 0, 1
    ymin, ymax = 0, 1

    plt.gca().spines['left'].set_bounds(0, 1)
    plt.gca().spines['bottom'].set_bounds(0, 1)

    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

    plt.gca().plot(1, 0, ">k", transform=plt.gca().get_xaxis_transform(), clip_on=False)
    plt.gca().plot(0, 1, "^k", transform=plt.gca().get_yaxis_transform(), clip_on=False)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # Remove ticks
    plt.xticks([])
    plt.yticks([])

    import matplotlib as mpl
    mpl.rc("savefig", dpi=300)
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Introduction\HiddenPS.png')

    # Show the plot
    plt.show()

def plot_phase_space_revealed_nullcline():
    time, v_values, w_values = compute_fitzhugh_nagumo_dynamics()



    mean_val = np.min(time)
    std_dev = np.max(time) - np.min(time)
    time = (time - mean_val) / std_dev    

    mean_val = np.min(v_values)
    std_dev = np.max(v_values) - np.min(v_values)
    v_values = (v_values - mean_val) / std_dev    

    mean_val = np.min(w_values)
    std_dev = np.max(w_values) - np.min(w_values)
    w_values = (w_values - mean_val) / std_dev    

    v_values = v_values*0.7+0.1
    w_values = w_values*0.76+0.05

    plt.figure(figsize=(2.5, 2.5))
    plt.plot(v_values, w_values, label=r'$Trajectory$', color='red')


    x = np.linspace(-2.5, 3, 10000)
    mean_val = np.min(x)
    std_dev = np.max(x) - np.min(x)
    x_mean = (x - mean_val) / std_dev    

    scale = 0.35
    # plt.plot(x_mean, scale*nullcline_vdot(x), label=r'nullcline 1', color='lime', alpha=0.5, linestyle=':')
    # plt.plot(x_mean, scale*nullcline_wdot(x), label='nullcline 2', color='cyan', alpha=0.5, linestyle=':')

    plt.plot(x_mean, scale*nullcline_wdot(x), color='cyan', label=r'Nullcline $x_2$',alpha=1)
    plt.plot(x_mean, scale*nullcline_vdot(x), color='lime', label=r'Nullcline $x_1$', alpha=1)
    plt.scatter([0.528884], [0.485130], color='black', label='Fixed Point',alpha=1, edgecolors='none', zorder=10)
    # plt.plot(x_mean, scale*nullcline_vdot(x), label=r'nullcline 1', color='grey', alpha=0.5, linestyle=':')
    # plt.plot(x_mean, scale*nullcline_wdot(x), label='nullcline 2', color='grey', alpha=0.5, linestyle=':')


    plt.xlabel(r'$x_1$', loc='right', fontsize=14)
    plt.ylabel(r'$x_2$', loc='top', rotation=0, fontsize=14)
    # plt.legend(loc='upper center', bbox_to_anchor=[0.55, 1.39],ncols=2, frameon=True, columnspacing=0.4, fontsize=10)
    # plt.legend(loc='upper center', bbox_to_anchor=[0.53, 1.04], ncols=2, frameon=True, fontsize=10, columnspacing=0.4)

    # plt.title('Phase Space', pad=50)
    plt.title('Phase Space')

    # plt.grid()
    
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['right'].set_color('none')

    # Add arrows at the end of x and y axes
    plt.gca().spines['left'].set_position(('data', 0))
    plt.gca().spines['bottom'].set_position(('data', 0))

    xmin, xmax = 0, 1
    ymin, ymax = 0, 1

    plt.gca().spines['left'].set_bounds(0, 1)
    plt.gca().spines['bottom'].set_bounds(0, 1)

    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

    plt.gca().plot(1, 0, ">k", transform=plt.gca().get_xaxis_transform(), clip_on=False)
    plt.gca().plot(0, 1, "^k", transform=plt.gca().get_yaxis_transform(), clip_on=False)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # Remove ticks
    plt.xticks([])
    plt.yticks([])

    plt.subplots_adjust(top=0.715,
bottom=0.11,
left=0.125,
right=0.9,
hspace=0.2,
wspace=0.2)
    plt.subplots_adjust(top=0.88,
bottom=0.11,
left=0.125,
right=0.9,
hspace=0.2,
wspace=0.2)

    import matplotlib as mpl
    mpl.rc("savefig", dpi=300)
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Introduction\RevealedPS2.png')

    # Show the plot
    plt.show()



if __name__ == '__main__':
    # plot_timeseries()
    # plot_phase_space_revealed_nullcline()
    plot_phase_space_hidden_nullcline()