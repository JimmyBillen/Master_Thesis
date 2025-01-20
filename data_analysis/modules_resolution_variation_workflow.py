# This file contains all the functions needed to create the FitzHugh-Nagumo data, train neural networks, and evaluate and store
# their results. This script is used to easily change the number of data points for the above mentioned algorithm.

# The program is executed when the script is run as a standalone program.
# It is controlled by the following block at the end: if __name__ == '__main__': here the parameters can be tuned in the code.
# General nullcline error, validation error and pearson correlation coefficient data analysis is performed in 
# the script 'performance_vs_data_size_analysis.py'.

import sys
sys.path.append('../../Master_Thesis') # needed to import settings

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from keras.models import Model
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import save_model, load_model
from keras import utils
import uuid
import time

from ast import literal_eval
import seaborn as sns
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols

from scipy.stats import f_oneway, shapiro, levene, kruskal, mannwhitneyu

import math
import matplotlib as mpl
import pickle

from datetime import datetime

import json

from settings import R, I, TAU, A, B

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
        sampled_time = sample_subarray(time, NUM_POINTS)
        sampled_v_values = sample_subarray(v_values, NUM_POINTS)
        sampled_w_values = sample_subarray(w_values, NUM_POINTS)
    else:
        input(f"Are you sure you don't want to use global NUM_POINTS={NUM_POINTS}?, if 'no': cancel by CONTROL+C")
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


def derivative_plotter():
    """Analysis shows that difference in derivative methods:
    finite difference method: 1) forward, 2)center, 3)backwards only differ 
    by 0.06 at the 'delta' place, compared to the height of 1.3 of the delta,
    so accuracy wise does not matter which one you take!
    We have chosen forward method throughout the thesis (except the last point, uses backward)
    """

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

def pure_derivative_plotter():
    """plots only the derivatives of u and v"""

    # time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics() # assigning v->v, w->v see heads-up above.

    num_of_points_deriv = [1_000, 5_000, 10_000, 15_000]
    for num_points in num_of_points_deriv:
        time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics(num_points) # assigning v->v, w->v

        u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
        v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

        # print(len(time), len(v_t_data), len(u_dot_t_data), len(v_dot_t_data))

        plt.plot(time, v_dot_t_data, label=rf"$v'(t)$ {num_points}")
        # plt.plot(time, u_dot_t_data, label=r"$u'(t)$")
    plt.title(f"Time Series of $v'$ and the derivatives at tau={TAU}, numpoints={NUM_POINTS}")
    plt.hlines(y=0, xmin=min(time), xmax=max(time), colors='black', label='y=0')
        # plt.ylim(-2, 1.5)
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(time, u_dot_t_data, label=r"$u'(t)$", color='C2')
    # plt.plot(time, u_dot_t_data, label=r"$u'(t)$")
    plt.title(f"Time Series of $u'$ and the derivatives at tau={TAU}, numpoints={NUM_POINTS}")
    plt.hlines(y=0, xmin=min(time), xmax=max(time), colors='black', label='y=0')
    # plt.ylim(-2, 1.5)
    plt.legend(loc='upper right')
    plt.show()

def normalization_with_mean_std(data, mean_std):
    return ( data - mean_std[0] ) / mean_std[1]

def reverse_z_score_normalization(standardized_data, mean_std):
    """
    Reverse the Z-score normalization (standardization) operation.

    Parameters:
    - standardized_data: NumPy array or list, the Z-score normalized data.
    - mean_val: float, the mean value used for standardization.
    - std_dev: float, the standard deviation used for standardization.

    Returns:
    - original_data: NumPy array, the original data before normalization.
    """
    reversed_data = standardized_data * mean_std[1] + mean_std[0]
    return reversed_data

# functions for this file
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

def nullcline_wdot_inverse(w):
    """
    Calculate the inverse of the nullcline function v(w).

    Parameters:
    - w: Input value.

    Returns:
    - Result of the inverse nullcline function.
    """
    return B * w - A

def calculate_mean_squared_error(real_data: np.ndarray, generated_data: np.ndarray):
    """
    Calculate the Mean Squared Error (MSE) between real_data and generated_data.

    Parameters:
    - real_data: Array of real data.
    - generated_data: Array of generated data.

    Returns:
    - MSE value.
    """
    # boundbox: leftunder [-0.6235, 0.09566] [0.773182, 1.842041]
    if generated_data.shape!=real_data.shape:
        assert ValueError(f'The shapes of {generated_data} and {real_data} are not the same.')

    return np.sum( np.square(generated_data - real_data)) / len(real_data)

def limit_cycle(tau=None):
    'limit cycle from time-series data.'
    _, v_values, w_values = compute_fitzhugh_nagumo_dynamics()
    return v_values, w_values

def nullcline_and_boundary(option, amount_of_points):
    nullclines_per_option = find_boundary_nullclines()
    if option == 'option_1':
        bound_nullcline = nullclines_per_option['option_1']
        q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline), amount_of_points)
        nullcline = nullcline_wdot(q)
    if option == 'option_2':
        bound_nullcline = nullclines_per_option['option_2']
        q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline), amount_of_points)
        nullcline = nullcline_wdot_inverse(q)
    if option == 'option_3':
        bound_nullcline = nullclines_per_option['option_3']
        q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline), amount_of_points)
        nullcline = nullcline_vdot(q)
    if option == 'option_4':
        bound_nullcline = nullclines_per_option=['option_4']
        q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline), amount_of_points)
        nullcline = np.zeros(len(q))    # just give zero, don't trust MSE values
        print("MSE values of option_4 cannot be trusted")
    return q, nullcline

def plot_limit_cycle(u_nullcline=True, y_ifo_x=True, model: Model=None, with_neural_network=False, mean_std=None, plot=True):
    """ Plots the limit cycle with the model
    
    This is the old version before optimisation, new version is 'plot_limit_cycle_with_model' """
    # Plot Limit Cycle
    x_lc, y_lc = limit_cycle()
    if plot:
        plt.plot(x_lc, y_lc, 'r-', label=f'Limit Cycle')

    # Nullclines
        # vdot
    v = np.linspace(-2.5, 2.5, 1000)
    if plot:
        # plt.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$w=v - (1/3)*v**3 + R * I$"+r" ,$\dot{v}=0$ Nullcline")
        plt.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$\dot{x}=0$ Nullcline")

        # wdot
    v = np.linspace(-2.5, 2.5, 1000)
    if plot:
        # plt.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$w=(v + A) / B$"+r" ,$\dot{w}=0$ Nullcline")
        plt.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$\dot{y}=0$ Nullcline")

    #     # Plotting a dot where the nullclines intersect
    dots = [0.409]
    dots_null = [(i + A) / B for i in dots]
    if plot:
        plt.plot(dots, dots_null, 'bo', label='Fixed Point')

    # Plotting neural network
    if with_neural_network:
    # for the neural network it is best to only consider values IN the limit cycle, so choose v \in [v_min, v_max] (see q)

        add_dots_to_figures = False
        # bepalen welke normalized u_dot=0 of v_dot=0 te geven aan de Neural Network

        # this can be optimized
        if y_ifo_x and u_nullcline: # option 1
            bound_nullcline = [-0.6235, 0.773182]
            q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline))
            nullcline_data = nullcline_wdot(q)
        elif (not y_ifo_x) and u_nullcline: # option 2
            bound_nullcline = [0.09566, 1.842041]
            q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline))
            nullcline_data = nullcline_wdot_inverse(q)
        elif y_ifo_x and (not u_nullcline): # option 3
            bound_nullcline = [-1.840664, 1.8958]
            q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline))
            nullcline_data = nullcline_vdot(q)
        else:
            q = np.linspace(np.min(y_lc), np.max(y_lc), 1000)
            # de inverse is geen functie dus kan niet gedaan worden

        # You have to give the Neural Network Normalized data as well.
        if y_ifo_x: # option 1 or 3
            # bound_nullcline = [-0.6235, 0.773182]
            # q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline))
            # q = np.linspace(np.min(x_lc), np.max(x_lc), 1000)
            # zet q om naar normalized with v
            normalized_x = normalization_with_mean_std(q, mean_std["v_t_data_norm"]) # de v is input
        else: # option 2 or 4
            # bound_nullcline = [0.09566, 1.842041]
            # q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline))
            # q = np.linspace(np.min(y_lc), np.max(y_lc), 1000)
            # zet q om naar normalized with u
            normalized_x = normalization_with_mean_std(q, mean_std["u_t_data_norm"]) # de u is input
        if u_nullcline: # option 1 or 2
            # zet np.zeros(len(v)) om naar normalized with udot
            normalized_dot = normalization_with_mean_std(np.zeros(len(q)), mean_std["u_dot_t_data_norm"]) #
        else:   # option 3 or 4
            # zet np.zeros(len(v)) om naar normalized with vdot
            normalized_dot = normalization_with_mean_std(np.zeros(len(q)), mean_std["v_dot_t_data_norm"])        


        x_predict = np.column_stack((normalized_x, normalized_dot)) # de v is gewoon een variabele tussen -2.5 en 2.5, verwijst niet naar 
        predictions_norm = model.predict(x_predict)
        # the predictions are (len,1) shape while normally we work with (len,) shape. Use predictions.reshape(-1)
        if y_ifo_x:
            predictions = reverse_z_score_normalization(predictions_norm, mean_std["u_t_data_norm"]) # de u is output
            if plot: # sometimes we don't want plot but just want MSE
                plt.plot(q, predictions, label = 'prediction')
                if add_dots_to_figures:
                    v_subset = v[::25]
                    predictions_subset = predictions[::25]
                    plt.scatter(v_subset, predictions_subset, s=5, alpha=0.5, color='purple', label='Prediction coord')
        else:
            predictions = reverse_z_score_normalization(predictions_norm, mean_std["v_t_data_norm"])
            if plot:
                plt.plot(predictions, q, label = 'prediction')
                if add_dots_to_figures:
                    v_subset = v[::25]
                    predictions_subset = predictions[::25]
                    plt.scatter(predictions_subset, v_subset, s=5, alpha=0.5, color='purple', label='Prediction coord')
        predictions = predictions.reshape(-1) # hierna opnieuw -1??
        mse = calculate_mean_squared_error(nullcline_data, predictions.reshape(-1))
        return mse
    if plot:
        plt.xlabel('v (voltage)')
        plt.ylabel('w (recovery variable)')
        plt.title('Phase Space of FitzHugh-Nagumo Model:\n Limit Cycle and Nullclines')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.show()
        plt.clf()

    return None

def plot_density_line_limit_cycle():
    """not as nice as heatmap"""
    x_lc, y_lc = limit_cycle()
    distances = np.sqrt(np.diff(x_lc)**2+np.diff(y_lc)**2)
    normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

    for i in range(len(x_lc) - 1):
        plt.plot([x_lc[i], x_lc[i+1]], [y_lc[i], y_lc[i+1]], color=(1-normalized_distances[i], 0, 0))

    # plt.plot(x_lc, y_lc, 'r-', label=f'Limit Cycle', alpha=density_normalized)
    plt.show()

def plot_heatmap_line_limit_cycle():
    import seaborn as sns
    x, y = limit_cycle()
    heatmap_resolution = 200
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=heatmap_resolution)
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap.T, cmap='viridis', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', vmin=0, vmax=100)
    plt.xlabel('X')
    plt.colorbar(label='Density')
    plt.ylabel('Y')
    plt.title(f'Density Heatmap of Curve in Phase Space for Tau={TAU}')
    plt.show()

def remake_dataframe(tau, num_of_points):
    print("Rebuilding...")
    columns = ['run', 'normalization_method', 'activation_function', 'learning_rate', 'nodes', 'layers', 'epoch', 'loss', 'validation', 'modelname', 'option', 'mean_std']
    # als je verder zou gaan met een bepaalde epoch, zorg er dan voor dat die dezelfde run number krijgt

    # Create an empty DataFrame
    df = pd.DataFrame(columns=columns)

    # save in right folder
    relative_folder_path = 'Master-Thesis'

    if not os.path.exists(relative_folder_path):
        assert False, 'This relative folder path "Master-Thesis" does not exist.'

    name_file = f"FHN_NN_loss_and_model_{tau}_{num_of_points}.csv"
    output_path = os.path.join(relative_folder_path, name_file)

    df.to_csv(output_path, index=False)
    print("Done")

def to_list(val):
    return list(pd.eval(val))

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

# normalization_options
def normalization(data: np.ndarray, normalization_method: str='z-score'):
    """
    Perform Z-score normalization (standardization) on the given data.

    Parameters:
    - data: NumPy array or list, the data to be normalized.

    Returns:
    - standardized_data: NumPy array, the Z-score normalized data.
    """
    data = np.array(data)
    if normalization_method == 'z-score':
        mean_val = np.mean(data)
        std_dev = np.std(data)
    elif normalization_method == 'min-max':
        mean_val = np.min(data)
        std_dev = np.max(data) - np.min(data)
    elif normalization_method == 'no-norm':
        mean_val = 0
        std_dev = 1
    else:
        raise ValueError("Invalid normalization method. Please choose 'z-score' or 'min-max' or 'no-norm'.")

    standardized_data = (data - mean_val) / std_dev    
    return standardized_data, mean_val, std_dev

def normalization_with_mean_std(data, mean_std):   # used when we want to normalize data again (see FitzHugh_Nagumo_ps.py)
    """
    Perform normalization of data with already known mean and standard deviation:
    Typically used when we want to feed data into a model, first we need to normalize this data.
    """
    return ( data - mean_std[0] ) / mean_std[1]

def reverse_normalization(standardized_data, mean_std):
    """
    Reverse the normalization (standardization) operation.

    Parameters:
    - standardized_data: NumPy array or list, the Z-score normalized data.
    - mean_val: float, the mean value used for standardization.
    - std_dev: float, the standard deviation used for standardization.

    Returns:
    - original_data: NumPy array, the original data before normalization.
    """
    reversed_data = standardized_data * mean_std[1] + mean_std[0]
    return reversed_data

def split_train_validation_data_seed(data_1, data_2, data_3, data_4, validation_ratio=0.2, seed=0):
    """
    Splits all the data randomly into training and validation data.

    Parameters:
    - data_1, data_2, data_3, data_4: The different data we want to shuffle in the same way.
    - validation_ratio: The ratio of validation data compared to the total amount of data.

    Returns:
    train_1, val_1, train_2, val_2, train_3, val_3, train_4, val_4
    """
    num_samples = len(data_1)
    num_validation_samples = int(num_samples * validation_ratio)

    # Introduce Random Number Generator
    rng = np.random.default_rng(seed)

    # Randomly shuffle the data and labels
    indices = np.arange(num_samples).astype(int)
    rng.shuffle(indices)
    data_1 = data_1[indices]
    data_2 = data_2[indices]
    data_3 = data_3[indices]
    data_4 = data_4[indices]

    # Split the data and labels
    val_1 = data_1[:num_validation_samples]
    val_2 = data_2[:num_validation_samples]
    val_3 = data_3[:num_validation_samples]
    val_4 = data_4[:num_validation_samples]
    train_1 = data_1[num_validation_samples:]
    train_2 = data_2[num_validation_samples:]
    train_3 = data_3[num_validation_samples:]
    train_4 = data_4[num_validation_samples:]

    return train_1, val_1, train_2, val_2, train_3, val_3, train_4, val_4

def nullcline_choice(train_u, val_u, train_v, val_v,
              train_u_dot, val_u_dot, train_v_dot, val_v_dot,
              u_nullcline: True, u_ifo_v: True):
    """Chooses from all the data the correct ones for training for the specific nullcline we want to remake. 
    
    Eg. When training option_1 (so u_dot=0 and u(v)) we would like to have as input: udot and v

    Parameters:
    -u_nullcline: Bool which is True if we want to train the NN to reproduce the udot=0 nullcline.
    -u_ifo_v: Bool which is True if we want to calculate the nullcine in as u(v), False for v(u).

    Returns:
        tuple:
        column stack of x_train and x_dot_train,
        column stack of x_validation and x_dot_validation,
        y_train, 
        y_validation
    """
    if u_nullcline:
        x_dot_train = train_u_dot
        x_dot_val = val_u_dot
    else:
        x_dot_train = train_v_dot
        x_dot_val = val_v_dot
    
    if u_ifo_v: # u = f(v)
        x_train = train_v
        x_val = val_v
        y_train = train_u
        y_val = val_u
    else:       # v = g(u) 
        x_train = train_u
        x_val = val_u
        y_train = train_v
        y_val = val_v

    alpha_betadot_data_train = np.column_stack((x_train, x_dot_train))
    alpha_betadot_data_val = np.column_stack((x_val, x_dot_val))

    return alpha_betadot_data_train, alpha_betadot_data_val, y_train, y_val

def train_and_save_losses(model, X_train, X_val, y_train, y_val, save, epochs, nodes, layers, learning_rate, normalization_method, activation_function, option, mean_std, batchsize=32):
    """
    Plots the loss of error for validation and training data.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize, validation_data=(X_val, y_val), verbose=0)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    if save:
        save_data_in_dataframe(train_loss, val_loss, nodes, layers, learning_rate, normalization_method, activation_function, model, option, mean_std)
    return model


def make_dataframe(train_loss, validation_loss, nodes, layers, learning_rate, normalization, activation_function, modelname, option, mean_std):
    length_data = len(train_loss)
    epochs = np.arange(length_data)
    new_data = {'epoch': epochs,
            'normalization_method': [normalization] * length_data,
            'activation_function': [activation_function] * length_data,
            'nodes': [nodes] * length_data,
            'layers': [layers] * length_data,
            'learning_rate': [learning_rate] * length_data,
            'loss': train_loss,
            'validation': validation_loss,
            'modelname': [modelname] * length_data,
            'option': [option] * length_data,
            'mean_std': [mean_std] * length_data
            }
    # we still need to add the 'run' column, this is code in 'concatenating_dataframes' function
    df_new_data = pd.DataFrame(new_data)
    return df_new_data


def save_kerasmodel(model):
    """
    Saves the Neural Network Kerasmodel in a file whose name is a unique ID.
    """
    unique_model_name = uuid.uuid4().hex    # https://stackoverflow.com/questions/2961509/python-how-to-create-a-unique-file-name/44992275#44992275

    # saving in right file
    absolute_path = os.path.dirname(__file__)
    relative_path = "saved_NN_models"
    folder_path = os.path.join(absolute_path, relative_path)
    full_path = os.path.join(folder_path, unique_model_name + '.h5')
    # save_model(model, unique_model_name+'.h5') # alternative: time
    save_model(model, full_path)
    # print("This model is saved with identifier:", unique_model_name) # this print statement is bad, because when printing (and want to break in between, the )
    return unique_model_name

def save_data_in_dataframe(train_loss, validation_loss, nodes, layers, learning_rate, normalization_method, activation_function, model, option, mean_std):
    """
    Saves the newly made data in the already existing dataframe. It also saves the model that was used in another file.
    """
    # load df from right folder
    absolute_folder_path = os.path.dirname(__file__)
    begin_name_file = "FHN_NN_loss_and_model"
    name_file_add_on = f"_{TAU}_{NUM_POINTS}"
    name_file_extension = ".csv"
    name_file = begin_name_file + name_file_add_on + name_file_extension
    output_path = os.path.join(absolute_folder_path, name_file)
    df = pd.read_csv(output_path, converters={'nodes': to_list}) # converters needed such that list returns as list, not as string (List objects have a string representation, allowing them to be stored as . csv files. Loading the . csv will then yield that string representation.)

    modelname = save_kerasmodel(model)
    new_df = make_dataframe(train_loss, validation_loss, nodes, layers, learning_rate, normalization_method, activation_function, modelname, option, mean_std)

    concatenated_df = concatenate_dataframes(df, new_df, normalization_method, activation_function, nodes, layers, option, learning_rate)

    concatenated_df.to_csv(output_path, index=False)

def new_highest_run_calculator(df):
    """Calculates the 'run' value for in the new dataframe. It looks at what the run number was previously and adds one. If there is no other run it starts at zero."""
    if df.empty:
        highest_run = 0
    else:
        highest_run = max(df["run"]) + 1
    return highest_run


def concatenate_dataframes(existing_df, new_df, normalization_method, activation_function, nodes, layers, option, learning_rate):
    """
    First we add the 'run' column to the newly made dataframe, later we concatenate the exisiting dataframe with the newly made.
    """
    # Here we check if those sets of parameters were chosen already before, this is used to determine the amount of 'runs' with these parameters
    # check same values
    sub_df = existing_df.loc[(existing_df["normalization_method"]==normalization_method)
                        & (existing_df["activation_function"]==activation_function)
                        & (existing_df["layers"]==layers)
                        & (existing_df["option"]==option)
                        & (existing_df["learning_rate"]==learning_rate)
                        ]       # https://www.statology.org/pandas-select-rows-based-on-column-values/
    # to compare lists we use apply
    print('1', sub_df)
    sub_df = sub_df[sub_df["nodes"].apply(lambda x: x==nodes)]
    print('2', sub_df)
    
    # print("DEBUGGING", nodes, type(nodes), sub_df["nodes"], type(sub_df["nodes"]), sub_df["nodes"])
    # sub_df = sub_df[sub_df["nodes"].apply(lambda x: np.array_equal(x, nodes))]

    # check the 'run' number
    new_highest_run = new_highest_run_calculator(sub_df)
    num_rows = new_df.shape[0]
    new_df["run"] = [new_highest_run] * num_rows

    # Now our new dataframe is fully constructed, we concatenate with already existing.
    concatenated_df = pd.concat([existing_df, new_df], ignore_index=True)

    return concatenated_df


def create_neural_network_and_save(num_layers=1, nodes_per_layer=None, activation_function: str='relu', learning_rate=0.01, option: str='option_1', normalization_method: str='z-score', save: bool=True, epochs: int=100, seed=None, batchsize=32):
    '''
    Program that creates the neural network from FHN data and saves it.

    num_layers:
        Number of hidden layers that is wanted: minimum is 1
    --------
    option: 
        option_1: u_dot=0 and u(v), option 2: u_dot=0 and v(u), option 3: v_dot=0 and u(v), option 4: v_dot=0 and v(u)
    '''
    # Check if input of nodes and layers is correct
    if nodes_per_layer is None: # list is mutable object: https://stackoverflow.com/questions/50501777/why-does-tensorflow-use-none-as-the-default-activation , https://medium.com/@inexturesolutions/top-common-python-programming-mistakes-and-how-to-fix-them-90f0a8bcce43 (mistake 1)
        nodes_per_layer = [10]
    elif type(nodes_per_layer) == int: # if same #nodes for all layers
        nodes_per_layer = np.ones(num_layers) * nodes_per_layer
    if (num_layers < 1) or (type(num_layers) != int):
        assert False, "Please make sure the number of layers is an integer greater than zero."
    if len(nodes_per_layer) != num_layers:
        assert False, f"Please make sure the number of nodes per (hidden)layer (={len(nodes_per_layer)}) are equal to the amount of (hidden) layers (={num_layers})."
    if activation_function != 'relu' and activation_function != 'tanh' and activation_function != 'sigmoid':
        assert False, "Please choose as activation function between 'relu', 'tanh' or 'sigmoid'."
    if seed is None:
        assert False, 'Please use seed, seed is None'

    print("Using:\n",
          f"Number of layers = {num_layers}\n",
          f"Nodes per layer {nodes_per_layer}\n",
          f"Normalization method {normalization_method}\n",
          f"Activation function {activation_function}\n",
          f"Learning rate {learning_rate}\n",
          f"Option {option}\n",
          f"Epochs {epochs}\n"
          )

    # creating the data of the FHN system used for training and validating
    time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics() # assigning v->v, w->v see heads-up above.
    u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
    v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

    # normalizing the data
    u_t_data_norm, mean_u, std_u = normalization(u_t_data, normalization_method)  # mean_u, and std_u equal x_min and (x_max - x_min) respectively when doing min-max normalization
    v_t_data_norm, mean_v, std_v = normalization(v_t_data, normalization_method) 
    u_dot_t_data_norm, mean_u_dot, std_u_dot = normalization(u_dot_t_data, normalization_method)
    v_dot_t_data_norm, mean_v_dot, std_v_dot  = normalization(v_dot_t_data, normalization_method)
    mean_std = {"u_t_data_norm":[mean_u, std_u], "v_t_data_norm":[mean_v, std_v], "u_dot_t_data_norm": [mean_u_dot, std_u_dot], "v_dot_t_data_norm": [mean_v_dot, std_v_dot]}
    print(f"1) {len(u_dot_t_data), len(u_dot_t_data_norm)}")
    # Creating Neural Network (no training yet)
    # Step : Seed selection
    utils.set_random_seed(seed)

    # Step 2: Build the Modelstructure
    model = Sequential()
    model.add(Dense(nodes_per_layer[0], input_dim=2, activation=activation_function)) # 1 input dimension
    for i in range(1, num_layers):
        model.add(Dense(nodes_per_layer[i], activation=activation_function))
    model.add(Dense(1, activation='linear'))

    # Step 3: Compile the model (choose optimizer..)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate)) # default optimizer Adam, learning rate 0.01 common starting point

    # Step 4: Separate training and validation data
    train_u, val_u, train_v, val_v, train_u_dot, val_u_dot, train_v_dot, val_v_dot = split_train_validation_data_seed(u_t_data_norm, v_t_data_norm, u_dot_t_data_norm, v_dot_t_data_norm, validation_ratio=0.2, seed=seed)
    print(f"2) {len(train_u), len(val_u), len(train_u_dot), len(val_u_dot)}")

    # Step 5: Train Model & Plot
    if option == 'option_1':
        u_nullcline = True  # u_dot = 0
        u_ifo_v = True  # u = f(v)
    elif option == 'option_2':
        u_nullcline = True  # u_dot = 0
        u_ifo_v = False  # v = g(u)
    elif option == 'option_3':
        u_nullcline = False  # v_dot = 0
        u_ifo_v = True  # u(v)
    elif option == 'option_4':
        u_nullcline = False  # v_dot = 0
        u_ifo_v = False  # v(u)        
    
    X_train, X_val, y_train, y_val = nullcline_choice(train_u, val_u, train_v, val_v,
                                        train_u_dot, val_u_dot, train_v_dot, val_v_dot,
                                        u_nullcline, u_ifo_v)

    # now that is chosen with which nullcine and in which way u(x) or v(x) we want to train, the neural network is trained and saved:
    train_and_save_losses(model, X_train, X_val, y_train, y_val, save, epochs, nodes_per_layer, num_layers, learning_rate, normalization_method, activation_function, option, mean_std, batchsize)

def count_everything(df=None):
    """Counts the amount of runs of every simulation based on option, layers, nodes, lr, epoch, normalization method and activation function."""
    # upload pd
    if df is None:
        absolute_path = os.path.dirname(__file__)
        relative_path = f"FHN_NN_loss_and_model_{TAU}.csv"
        csv_name = os.path.join(absolute_path, relative_path)
        df = pd.read_csv(csv_name) # literal eval returns [2,2] as list not as str
    last_occurrences = df.drop_duplicates('modelname', keep='last').reset_index(drop=True)
    df_dropped = last_occurrences.drop(columns=['mean_std', 'modelname', 'validation', 'loss',])
    count = df_dropped.groupby(["option", "layers", "nodes", "learning_rate", "epoch", "normalization_method", "activation_function"]).count()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(count)

def adjust_run_number():
    """Adjusts the 'run' column for rows with normalization_method='no-norm', 
    activation_function='relu', learning_rate=0.01, and layers=2 by subtracting 1.
    
    (Was necessary at the time)
    """
    absolute_path = os.path.dirname(__file__)
    relative_path = "FHN_NN_loss_and_model.csv"
    csv_name = os.path.join(absolute_path, relative_path)
    df = pd.read_csv(csv_name, converters={"nodes": literal_eval}) # literal eval returns [2,2] as list not as str

    df.loc[(df['normalization_method'] == 'no-norm') &
           (df['activation_function'] == 'relu') &
           (df['learning_rate'] == 0.010) &
           (df['option'] == 'option_1') &
           (df['layers'] == 2), 'run'] += -1

    # use this to watch it more closely (don't save it with this!)
    # df = df[(df['normalization_method'] == 'no-norm') &
    #        (df['activation_function'] == 'relu') &
    #        (df['learning_rate'] == 0.01) &
    #        (df['option'] == 'option_1') &
    #        (df['nodes'].apply(lambda x: x == nodes)) &
    #        (df['layers'] == 2)]

    # print(df)

    df.to_csv(csv_name, index=False)
    print("Done")
  
def df_select_max_epoch(df_selection, max_epochs):
    """From a df_selection tries to choose only the values with right amount of epochs (not more)
    
    Input
    ------
    df_selection:
      Dataframe containing all runs to go up to max_epochs (or higher) where norm, activ, lr, layers, nodes, option have already been selected
    max_epochs:
      Epoch limit that model was trained on
      
    Returns
    -------
    pandas.dataframe
      Returns the dataframe with the selected models (and the epochs going from 0 to max_epochs)
    """
    modelnames = df_selection["modelname"].unique()
    modelnames_selection = []
    for modelname in modelnames:
        count_modelname = df_selection["modelname"].value_counts()[modelname]
        if count_modelname == max_epochs + 1:
            modelnames_selection.append(modelname)
    df_selection_modelname = df_selection[(df_selection["modelname"].isin(modelnames_selection))]
    assert len(modelnames_selection) != 0, "No models found"
    return df_selection_modelname

def remove_rows(normalization_method, activation_function, learning_rate, nodes: str, layers, option, max_epochs):
   
   """ Removes ONE (the last) row with above specifications
   Pas op, omdat count everything moeilijk doet met lists moet type van nodes een str zijn
   """
   absolute_path = os.path.dirname(__file__)
   relative_path = "FHN_NN_loss_and_model.csv"
   csv_name = os.path.join(absolute_path, relative_path)
   df = pd.read_csv(csv_name) # literal eval returns [2,2] as list not as str

     # makes pre-selection of everything to parameters EXCEPT epoch
   new_df=df[(df["normalization_method"] == normalization_method) &
          (df["activation_function"] == activation_function) &
          (df["learning_rate"] == learning_rate) &
          (df["layers"] == layers) &
          (df['nodes'] ==  nodes) &  # moet aangepast worden als we beginnen spelen met layers
          (df['nodes'].apply(lambda x: x == nodes)) &  # ofwel deze ofwel die hierboven
          (df["option"] == option)
          ]
    
    # removes all the cases that go further than max_epochs
   df_selection = df_select_max_epoch(new_df, max_epochs)

    # shows all run_values
   run_values = df_selection['run'].unique()

   assert run_values.size > 0, 'no run values that work'

    # only want the max:
   max_run = run_values.max()

  #  take the corresponding modelname
   modelname = df_selection[df['run'] == max_run]['modelname'].tolist()[0]

  # search for index of this modelname and remove it from model
   part_df = df[(df['modelname'] == modelname)]
   print(part_df.index)
   new_df = df.drop(part_df.index)

   new_df.reset_index(drop=True, inplace=True) # zodat alle indices normaal doen

  # save model
   absolute_path = os.path.dirname(__file__)
   relative_path = "FHN_NN_loss_and_model.csv"
   csv_name = os.path.join(absolute_path, relative_path)

   print('wordt niet gesaved')
  #  new_df.to_csv(csv_name, index=False) # soms uitzetten voor veiligheid

   count_everything(new_df)

def does_data_exist(df: pd.DataFrame, normalization_method, activation_function, learning_rate, nodes, layers, max_epoch, option, average) -> list:
    """Checks if there is enough data with the given specifications"""
    # 'normalization_method', 'activation_function', 'learning_rate', 'nodes', 'layers', 'epoch'
    conditions = {'normalization_method': normalization_method,
                  'activation_function': activation_function,
                  'learning_rate': learning_rate,
                  'nodes': nodes,
                  'layers': layers,
                  'epoch': max_epoch,
                  'option': option}
    condition = df.apply(lambda row: all(row[key] == value for key, value in conditions.items()), axis=1)
    # condition = df.apply(lambda row:
    #                      all(
    #                      (print(f"Key: {key}, Value in DataFrame: {row[key]}, Value in conditions: {value}", row[key]==value, type(row[key]), type(value)),
    #                       row[key] == value)
    #                      for key, value in conditions.items()
    #                      ),
    #                      axis=1
    #                      )

    if any(condition):
        print("There exists a row with the specified conditions")
        matching_rows = df.loc[condition]
        unique_scalars_count = matching_rows['run'].nunique()
        assert average <= unique_scalars_count, f"There are only {unique_scalars_count} simulations that exist, and we need {average}"
        unique_scalars = matching_rows['run'].unique()
        run_values = unique_scalars[:average]
        return run_values # is a list
    else:
        assert False, "No row with these specifications exists"

def retrieve_validation_data(df: pd.DataFrame, run, normalization_method, activation_function, learning_rate, nodes, layers, max_epoch, option):
    # zoek naar RUN_value waar does_data_exist() positive is, en neem dan vanaf epoch 0/1 tot max_epoch
    filtered_df = df[(df['run'] == run) &
                     (df['normalization_method'] == normalization_method) &
                     (df['activation_function'] == activation_function) &
                     (df['learning_rate'] == learning_rate) &
                     (df['nodes'].apply(lambda x: x == nodes)) &
                     (df['layers'] == layers) &
                     (df['epoch']).between(0,max_epoch) &
                     (df['option'] == option)
                     ]
    validation_func = filtered_df['validation'].tolist()
    epochs_selected = filtered_df['epoch'].tolist()

    assert validation_func, f"The list is empty, something went wrong {validation_func}, {normalization_method},{activation_function}"
    return epochs_selected, validation_func


def select_and_concat_first_average(df, all_normalization_methods, all_activation_functions, average):
    """
    Selects the first #Average occurrences of specific values in specified columns and concatenates them into one DataFrame.
    """
    new_df = pd.DataFrame()
    for norm_method in all_normalization_methods:
        for activ_func in all_activation_functions:
            first_average_occurences = df[(df["normalization_method"] == norm_method) & (df["activation_function"] == activ_func)].head(average)
            new_df = pd.concat([new_df, first_average_occurences], ignore_index=True)

    return new_df

def validation_data_compare_fast(normalization_method, activation_function, learning_rate, nodes, layers, max_epochs, option, average=1):
    # Load DataFrame
    absolute_path = os.path.dirname(__file__)
    relative_path = f"FHN_NN_loss_and_model_{TAU}.csv"
    csv_name = os.path.join(absolute_path, relative_path)
    df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}) # literal eval returns [2,2] as list not as str

    # Select data based on specified parameters
    df_selection = select_data_from_df(df, learning_rate, nodes, layers, option, max_epochs, normalization_methods=normalization_method, activation_functions=activation_function)
    # Only select last epoch (validation at max_epochs)
    # df_selection = df_selection[(df_selection["epoch"] == max_epochs)].reset_index(drop=True)

    return None

def validation_data_compare(normalization_method, activation_function, learning_rate, nodes, layers, max_epochs, option, average=1):
    """
    Compares the loss function for different combinations of input parameters.
    
    TO BE OPTIMIZED: DOING IT WITH THE RUN SEEMS OUTDATED: DOING IT WITH THE MODELNAME SEEMS BETTER

    max_epochs:
        The number of epochs you want to plot, (remember, n run epochs correspond to having epochs from 0 - (n-1) )
    average:
        amount of numbers to be plotted (does not average!)
    """
    parameters = []
    plot_dictionary = {
        "normalization_method": False,
        "activation_function": False,
        "learning_rate": False,
        "nodes": False,
        "layers": False,
        "max_epochs": False
    }

    if type(normalization_method) is list:
        parameters.append(normalization_method)
        plot_dictionary["normalization_method"]=True
        all_normalization_methods = normalization_method.copy()
    if type(activation_function) is list:
        parameters.append(activation_function)
        plot_dictionary["activation_function"]=True
        all_activation_functions = activation_function.copy()
    if type(learning_rate) is list:
        parameters.append(activation_function)
        plot_dictionary["learning_rate"]=True
    if type(nodes[0]) is list: #checks if its made of lists of lists
        parameters.append(nodes)
        plot_dictionary["nodes"]=True
    if type(layers) is list:
        parameters.append(layers)
        plot_dictionary["layers"]
    if type(max_epochs) is list:
        parameters.append(max_epochs)
        plot_dictionary["max_epochs"]=True
    
    assert len(parameters)==2, f"There are too many/few things you want to compare, can only compare two and have {parameters}"

    read_time = time.time()
    # upload pd
    absolute_path = os.path.dirname(__file__)
    relative_path = f"FHN_NN_loss_and_model_{TAU}.csv"
    csv_name = os.path.join(absolute_path, relative_path)
    df = pd.read_csv(csv_name, converters={"nodes": literal_eval}) # literal eval returns [2,2] as list not as str
    print('opening df', time.time()-read_time)

    first_check_time = time.time()
    # lenparameters0: norm, lenparameters1: activation
    run_values = [[None] * len(parameters[0]) for _ in range(len(parameters[1]))] # use run_values[#act][#norm]
    run_list = []
    # checks which parameters have to be on x- and y-axis
    for n,i in enumerate(parameters[0]): # norm
        for m,j in enumerate(parameters[1]): # act 
            count = 0
            new_val = [i, j]

            if plot_dictionary["normalization_method"]:
                normalization_method=new_val[count]
                count += 1
            if plot_dictionary["activation_function"]:
                activation_function=new_val[count]
                count+=1
            if plot_dictionary["learning_rate"]:
                learning_rate=new_val[count]
                count+=1
            if plot_dictionary['nodes']:
                nodes=new_val[count]
                count+=1
            if plot_dictionary['layers']:
                layers=new_val[count]
                count+=1
            if plot_dictionary['max_epochs']:
                max_epochs=new_val[count]
                count+=1

            run = does_data_exist(df, normalization_method, activation_function, learning_rate, nodes, layers, max_epochs, option, average)
            run_values[m][n] = run
            run_list.append(run)
    # extract data from csv
    end_validation_data = [[None] * len(parameters[0]) for _ in range(len(parameters[1]))]   # to avoid deep/shallow copy
    std_dev_loss_functions = [[None] * len(parameters[0]) for _ in range(len(parameters[1]))]

    min_loss_tot = np.infty
    max_loss_tot = -np.infty

    print('first check time', time.time()-first_check_time)
    second_check_time = time.time()

    # retrieves data
    for n, i in enumerate(parameters[0]):
        for m, j in enumerate(parameters[1]):
            count = 0
            new_val = [i, j]

            if plot_dictionary["normalization_method"]:
                normalization_method=new_val[count]
                count += 1
            if plot_dictionary["activation_function"]:
                activation_function=new_val[count]
                count+=1
            if plot_dictionary["learning_rate"]:
                learning_rate=new_val[count]
                count+=1
            if plot_dictionary['nodes']:
                nodes=new_val[count]
                count+=1
            if plot_dictionary['layers']:
                layers=new_val[count]
                count+=1
            if plot_dictionary['max_epochs']:
                max_epochs=new_val[count]
                count+=1

            validation_func_same_param = []
            for run in run_values[m][n]:
                epochs, validation_func =  retrieve_validation_data(df, run, normalization_method, activation_function, learning_rate, nodes, layers, max_epochs, option)
                validation_func_same_param.append(validation_func)

            validation_func_data_same_param = np.array(validation_func_same_param)
            mean_values = np.mean(validation_func_data_same_param, axis=0)
            std_dev_values = np.std(validation_func_data_same_param, axis=0)

            end_validation_data[m][n] = mean_values
            std_dev_loss_functions[m][n] = std_dev_values
            min_loss_tot = min(min_loss_tot, min(mean_values-std_dev_values))
            max_loss_tot = max(max_loss_tot, max(mean_values+std_dev_values))
    print('second_check_time', time.time() - second_check_time)
    # now all the data has been taken we can start with image processing
    
    # selection on dataframe (without mentioning epoch)
    df_plot = df[ # dit kan mss misgaan als run 0 niet wordt gekozen? NAKIJKEN
                    (df['normalization_method'].isin(all_normalization_methods)) &
                    (df['activation_function'].isin(all_activation_functions)) &
                    (df['learning_rate'] == learning_rate) &
                    (df['nodes'].apply(lambda x: x == nodes)) &
                    (df['layers'] == layers) &
                    (df['option'] == option)
                    ]
    # now select runs that have been trained up to max_epochs (not further)
    df_plot = df_select_max_epoch(df_plot, max_epochs)
    # select max_epoch
    df_plot = df_plot[(df_plot["epoch"] == max_epochs)]

    # Takes first #Average (does not do any averaging)
    df_plot = select_and_concat_first_average(df_plot, all_normalization_methods, all_activation_functions, average)

    # need logaritm for boxplot Anova
    df_plot['log_validation'] = np.log10(df_plot['validation'])

    # 1) ALL
    # Two-way ANOVA
    statistic_bool = True
    if statistic_bool:
        model = ols('log_validation ~ C(normalization_method) + C(activation_function) + C(normalization_method):C(activation_function)', data=df_plot).fit()
        anova_table = sm.stats.anova_lm(model, type=2)

    plt.figure(figsize=(6.4, 6))

    ax = sns.boxplot(data=df_plot, x="normalization_method", y="validation", hue="activation_function", palette='pastel', log_scale=True)
    sns.stripplot(data=df_plot, x="normalization_method", y="validation", hue="activation_function", dodge=True, palette='tab10')
    plt.yscale('log')
    if statistic_bool:
        plt.text(0.5, 0.1, f'Factor1 p-value: {anova_table["PR(>F)"].iloc[0]:.4f}', ha='center', transform=plt.gca().transAxes) #gca (get current axes), transAxes: zorgt ervoor dat coordinaat linksonder (0,0) en rechtsboven (1,1)
        plt.text(0.5, 0.05, f'Factor2 p-value: {anova_table["PR(>F)"].iloc[1]:.4f}', ha='center', transform=plt.gca().transAxes)
        plt.text(0.5, 0.0, f'Interaction p-value: {anova_table["PR(>F)"].iloc[2]:.4f}', ha='center', transform=plt.gca().transAxes)

    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles[3:6], labels[3:6], loc='upper right')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[3:6], labels[3:6], loc='upper right')

    plt.title(f"Validation: {average} simulations, {option}.\n lr: {learning_rate}, nodes: {nodes}, layers: {layers}, max epochs {max_epochs}\n")
    plt.show()

    # 2) Together Normalization Method
    # Anova-test
    p_value = None # such that don't take other p-value from previous
    if statistic_bool:
        group_data = [df_plot[(df_plot["normalization_method"] == norm_method)]["log_validation"] for norm_method in all_normalization_methods]
        f_statistic, p_value = f_oneway(*group_data)
        print(f"p-value, {p_value}")
        p_value = round(p_value, 4)

    plt.figure(figsize=(6.4, 6))

    sns.boxplot(data=df_plot, x="normalization_method", y="validation", hue="normalization_method", log_scale=True)
    sns.stripplot(data=df_plot, x="normalization_method", y="validation", hue="normalization_method", palette='tab10')
    plt.yscale('log')
    plt.title(f"Validation: {average} simulations, {option}.\n lr: {learning_rate}, nodes: {nodes}, layers: {layers}, max epochs {max_epochs}\n p-value: {p_value}")
    plt.show()

    # 3) Together Activation Function
    p_value = None
    if statistic_bool:
        group_data = [df_plot[(df_plot["activation_function"] == activ_function)]["log_validation"] for activ_function in all_activation_functions]
        f_statistic, p_value = f_oneway(*group_data)
        print(f"F-statistic, {f_statistic}")
        print(f"p-value, {p_value}")
        p_value = round(p_value, 4)

    plt.figure(figsize=(6.4, 6))

    sns.boxplot(data=df_plot, x="activation_function", y="validation", hue="activation_function", palette='pastel', log_scale=True)
    sns.stripplot(data=df_plot, x="activation_function", y="validation", hue="activation_function", palette='tab10')
    plt.yscale('log')
    plt.title(f"Validation: {average} simulations, {option}.\n lr: {learning_rate}, nodes: {nodes}, layers: {layers}, max epochs {max_epochs}\n p-value: {p_value}")
    plt.legend(loc='upper right')
    plt.show()

    # 4) Plot validation data in function of epoch
    fig, axs = plt.subplots(len(parameters[0]), len(parameters[1]), squeeze=False)
    for n, i in enumerate(parameters[0]): # over normalization
        for m, j in enumerate(parameters[1]): # over activation
            axs[n, m].plot(epochs, end_validation_data[m][n], color='b')
            axs[n, m].fill_between(epochs, end_validation_data[m][n]-std_dev_loss_functions[m][n], end_validation_data[m][n]+std_dev_loss_functions[m][n], color='grey', alpha=0.4)
            # axs[n, m].errorbar(epochs, loss_functions[n][m], yerr=std_dev_loss_functions[n][m], color='orange', capsize=0.1, alpha=0.6)
            axs[n, m].set_title(str(i) + "," + str(j))
            axs[n, m].set_yscale('log')
            axs[n, m].set_ylim([min_loss_tot, max_loss_tot])
            axs[n, m].set_xlabel("Epoch")
            axs[n,m].set_ylabel("Validation Loss")
    fig.suptitle(f"Validation Loss: Averaged over {average} in {option}.\n lr: {learning_rate}, nodes: {nodes}, layers: {layers}, max epochs {max_epochs}")

    plt.tight_layout()
    plt.show()


def plot_loss_and_validation_loss_one_model():
    """
    Plots the loss and validation for one specific model.
    """
    absolute_path = os.path.dirname(__file__)
    relative_path = f"FHN_NN_loss_and_model_{TAU}.csv"
    csv_name = os.path.join(absolute_path, relative_path)
    df = pd.read_csv(csv_name, converters={"nodes": literal_eval}) # literal eval returns [2,2] as list not as str

    df_plot=df[df['modelname'] == "7222292822d84716a9e279e716db87d3"].copy()

    df_plot['log_validation'] = np.log10(df_plot['validation'])
    df_plot['log_loss'] = np.log10(df_plot['loss'])

    plt.plot(df_plot['epoch'], df_plot['log_validation'], label='validation', alpha=0.5)
    plt.plot(df_plot['epoch'], df_plot['log_loss'], label='loss', alpha=0.5)

    plt.legend()
    plt.show()

def select_amount_modelnames(df, amount):
    """Selects the first 'amount' rows"""
    modelnames = df['modelname'].unique()
    modelnames_avg = modelnames[:amount]
    return modelnames_avg

def retrieve_loss_and_validation_from_modelname(df, modelname):
    df_select=df[df['modelname'] == modelname].copy()
    return df_select['epoch'], df_select['validation'], df_select['loss']

def select_data_from_df(df, learning_rate, nodes, layers, option, max_epochs, normalization_methods=None, activation_functions=None):
    """"
    From the dataframe with all the data the data with given specifications is selected.
    It has also been made sure that only the data trained up to the max epoch has been selected. 
    """
    # select all these data
    if normalization_methods is None and activation_functions is None:
        df_selection = df[
                    (df['learning_rate'] == learning_rate) &
                    (df['nodes'].apply(lambda x: x == nodes)) &
                    (df['layers'] == layers) &
                    (df['option'] == option)
                    ]
    else:
        df_selection = df[
                    (df['learning_rate'] == learning_rate) &
                    (df['nodes'].apply(lambda x: x == nodes)) &
                    (df['layers'] == layers) &
                    (df['option'] == option) &
                    (df['normalization_method'].isin(normalization_methods)) &
                    (df['activation_function'].isin(activation_functions))
                    ]
    # select data with maximal epoch of training = max_epochs
    df_selection = df_select_max_epoch(df_selection, max_epochs)
    return df_selection

def plot_loss_and_validation_loss_param(normalization_method, activation_function, learning_rate, nodes, layers, max_epochs, option, average, df=None):
    """"
    MODIFIED IN modules.py!

    Plots the averaged + std training and validation error for specific combination of normalization method and activation function
    """
    # Load DataFrame
    if df is None:
        absolute_path = os.path.dirname(__file__)
        relative_path = f"FHN_NN_loss_and_model_{TAU}_{NUM_POINTS}.csv"
        csv_name = os.path.join(absolute_path, relative_path)
        df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}) # literal eval returns [2,2] as list not as str

    # Select data based on specified parameters
    df_selection = select_data_from_df(df, learning_rate, nodes, layers, option, max_epochs, [normalization_method], [activation_function])
    # select first 'average' rows of df
    modelnames = select_amount_modelnames(df_selection, average)

    validations_per_modelname = []
    loss_per_modelname = []
    for modelname in modelnames:
        epochs, validation_data, loss_data =  retrieve_loss_and_validation_from_modelname(df_selection, modelname)
        validations_per_modelname.append(validation_data)
        loss_per_modelname.append(loss_data)


    validations_per_modelname = np.array(validations_per_modelname)
    loss_per_modelname = np.array(loss_per_modelname)

    log_validations_per_modelname = np.log10(validations_per_modelname)
    log_loss_per_modelname = np.log10(loss_per_modelname)

    log_mean_values_validation = np.mean(log_validations_per_modelname, axis=0)
    log_std_dev_validation = np.std(log_validations_per_modelname, axis=0)
    log_mean_values_loss = np.mean(log_loss_per_modelname, axis=0)
    log_std_dev_loss = np.std(log_loss_per_modelname, axis=0)

    return log_mean_values_validation[-1]
    plt.fill_between(epochs, log_mean_values_validation-log_std_dev_validation, log_mean_values_validation+log_std_dev_validation, color='orange', alpha=0.4)
    plt.fill_between(epochs, log_mean_values_loss-log_std_dev_loss, log_mean_values_loss+log_std_dev_loss, color='blue', alpha=0.4)
    plt.plot(epochs, log_mean_values_validation, label='mean validation', color='orange')
    plt.plot(epochs, log_mean_values_loss, label='mean loss', color='blue')


    # mean_values_validation = np.mean(validations_per_modelname, axis=0)
    # std_dev_validation = np.std(validations_per_modelname, axis=0)
    # mean_values_loss = np.mean(loss_per_modelname, axis=0)
    # std_dev_loss = np.std(loss_per_modelname, axis=0)

    # plt.fill_between(epochs, mean_values_validation-std_dev_validation, mean_values_validation+std_dev_validation, color='orange', alpha=0.4)
    # plt.fill_between(epochs, mean_values_loss-std_dev_loss, mean_values_loss+std_dev_loss, color='blue', alpha=0.4)
    # plt.plot(epochs, mean_values_validation, label='mean validation', color='orange')
    # plt.plot(epochs, mean_values_loss, label='mean loss', color='blue')
    plt.title(f"Loss and Validation: Tau{TAU}, averaged:{average} in {option}.\n lr: {learning_rate}, nodes: {nodes}, max epochs {max_epochs}\n {activation_function},{normalization_method}")
    plt.xlabel('epoch')
    plt.ylabel(fr'$\log10($loss$)$')
    # plt.yscale('log')
    plt.legend(loc='upper right')
    # plt.ylim(10**-5, 10**-1)
    plt.show()
    return df

def pearson_correlation(x, y):
    """
    Compute Pearson correlation coefficient between two arrays x and y.
    """
    # Calculate means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate covariance and standard deviations
    covariance = np.sum((x - mean_x) * (y - mean_y))
    std_dev_x = np.sqrt(np.sum((x - mean_x)**2))
    std_dev_y = np.sqrt(np.sum((y - mean_y)**2))
    
    # Calculate Pearson correlation coefficient
    pearson_corr = covariance / (std_dev_x * std_dev_y)
    
    return pearson_corr

def retrieve_modelname_meanstd(df: pd.DataFrame, run, normalization_method, activation_function, learning_rate, nodes, layers, max_epoch, option):
    """
    Give parameters and retrieves modelname and mean_std:dictionary
    """
    filtered_df = df[(df['run'] == run) &
                    (df['normalization_method'] == normalization_method) &
                    (df['activation_function'] == activation_function) &
                    (df['learning_rate'] == learning_rate) &
                    (df['nodes'].apply(lambda x: x == nodes)) &
                    (df['layers'] == layers) &
                    (df['epoch'] == max_epoch) &
                    (df['option'] == option)
                    ]
    assert len(filtered_df)==1, "Something went wrong, dataframe is too big"
    NN_modelname = filtered_df['modelname'].iloc[0]
    mean_std = filtered_df["mean_std"].iloc[0]
    return NN_modelname, mean_std

def retrieve_model_from_name(unique_modelname) -> Model:
    """Give the modelname and returns the keras.Model"""
    absolute_path = os.path.dirname(__file__)
    relative_path = "saved_NN_models"
    folder_path = os.path.join(absolute_path, relative_path)
    full_path = os.path.join(folder_path, unique_modelname + '.h5')
    if not os.path.exists(full_path):
        assert False, f"The model with name {unique_modelname} cannot be found in path {full_path}"
    loaded_model = load_model(full_path)
    return loaded_model

def normalize_axis_values(axis_value, all_mean_std, option):
    """We have values of the x/or/y axis of the phase space and returns the normalized versions.
    
    This is needed because the neural network model only takes in normalized inputs.
    """
    if option == 'option_1': # nullcine is udot/wdot = 0
        # axis value in this case is the x-axis (v-axis)
        mean_std = all_mean_std["v_t_data_norm"]
        normalized_axis_values = normalization_with_mean_std(axis_value, mean_std)

        # nullcline of option 1, udot/wdot = 0, so we have to fill in zeros (but has to be normalized first for the model)
        mean_std = all_mean_std["u_dot_t_data_norm"]
        normalized_dot = normalization_with_mean_std(np.zeros(len(axis_value)), mean_std)

        # The mean std that will be used later for reversing the normalization
        reverse_norm_mean_std = all_mean_std["u_t_data_norm"]

    if option == 'option_2':
        # axis value in this case is the y-axis (w-axis / u-axis)
        mean_std = all_mean_std["u_t_data_norm"]
        normalized_axis_values = normalization_with_mean_std(axis_value, mean_std)

        # nullcine of option 2, udot/wdot = 0, so we have to fill in zeros (but has to be normalized first for the model)
        mean_std = all_mean_std["u_dot_t_data_norm"]
        normalized_dot = normalization_with_mean_std(np.zeros(len(axis_value)), mean_std)

        # The mean std that will be used later for reversing the normalization
        reverse_norm_mean_std = all_mean_std["v_t_data_norm"]

    if option == 'option_3':
        # axis value in this case is the x-axis (v-axis)
        mean_std = all_mean_std["v_t_data_norm"]
        normalized_axis_values = normalization_with_mean_std(axis_value, mean_std)

        # nullcine of option 3, vdot = 0, so we have to fill in zeros (but has to be normalized first for the model)
        mean_std = all_mean_std["v_dot_t_data_norm"]
        normalized_dot = normalization_with_mean_std(np.zeros(len(axis_value)), mean_std)

        # The mean std that will be used later for reversing the normalization
        reverse_norm_mean_std = all_mean_std["u_t_data_norm"]

    if option == 'option_4':
        # just give some result so program is generalizable, do not trust said values
        normalized_axis_values = axis_value
        normalized_dot = np.zeros(len(axis_value))

        reverse_norm_mean_std = [0,1]


    input_prediction = np.column_stack((normalized_axis_values, normalized_dot))
 
    return input_prediction, reverse_norm_mean_std

def calculate_MSE_data_from_modelname(modelname, option, mean_std):
    """Calculates the MSE from the modelname 
    
    Function is used in plot_all_MSE_vs_VAL and plot_MSE_VS_VALDATION_data
    """
    all_mean_std = mean_std

    model = retrieve_model_from_name(modelname)
    
    # load data of nullclines in phasespace
    amount_of_points = 500
    axis_values, nullcline_values = nullcline_and_boundary(option, amount_of_points)

    # Predict normalized data 
    input_prediction, reverse_norm_mean_std = normalize_axis_values(axis_values, all_mean_std, option)
    prediction_output_normalized = model.predict(input_prediction)
    # Reverse normalize to 'normal' data
    prediction_output_column = reverse_normalization(prediction_output_normalized, reverse_norm_mean_std)
    prediction_output = prediction_output_column.reshape(-1)

    mse_val = calculate_mean_squared_error(nullcline_values, prediction_output)

    if option == 'option_4':
        # use y-region to predict x values, fill these x values in nullcline vdot and calculate difference with y-region values.
        vdot_nullcline_for_predicted_v = nullcline_vdot(prediction_output)
        mse_val = calculate_mean_squared_error(vdot_nullcline_for_predicted_v, axis_values)
    return mse_val

def plot_lc_from_model(df):
    """
    df is a dataframe with one row containing everything
    """
    modelname = df['modelname']
    option = df['option']
    mean_std = df['mean_std']
    model = retrieve_model_from_name(modelname)

    # load data of nullclines in phasespace
    amount_of_points = 500
    axis_values_for_nullcline, nullcline_values = nullcline_and_boundary(option, amount_of_points)

    # Predict normalized data 
    input_prediction, reverse_norm_mean_std = normalize_axis_values(axis_values_for_nullcline, mean_std, option)
    prediction_output_normalized = model.predict(input_prediction)
    # Reverse normalize to 'normal' data
    prediction_output_column = reverse_normalization(prediction_output_normalized, reverse_norm_mean_std)
    prediction_output = prediction_output_column.reshape(-1)
    
    # plot normal LC
    x_lc, y_lc = limit_cycle()
    plt.plot(x_lc, y_lc, 'r-', label=f'LC = {0}')
    # Plot Nullcines
    # vdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$w=v - (1/3)*v**3 + R * I$"+r" ,$\dot{v}=0$ nullcline")
    # wdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$w=(v + A) / B$"+r" ,$\dot{w}=0$ nullcline")

    plt.plot(axis_values_for_nullcline, prediction_output, label = 'prediction')
    plt.xlabel('v (voltage)')
    plt.ylabel('w (recovery variable)')
    plt.title(f"Phase Space: Limit Cycle and Cubic Nullcline with Prediction\n{option}, lr{df['learning_rate']}, {df['nodes']}, epoch {df['epoch']}\n {df['normalization_method']}, {df['activation_function']}")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

def study_model(df):
    """Takes a dataframe and returns three models from it.
    
    Chosen models are best, middle best and worst MSE performing
    df:
        Dataframe
    """
    df_sorted = df.sort_values(by=['MSE'], ascending=True)
    print(df)

    # select three models
    indexes = [0, (len(df) - 1 )//2, len(df) - 1]
    df_selection = df_sorted.iloc[indexes].reset_index(drop=True)

    print("MSE are", df_sorted["MSE"].iloc[indexes[0]], df_sorted["MSE"].iloc[indexes[1]], df_sorted["MSE"].iloc[indexes[2]])

    for index in range(len(df_selection)):
        plot_lc_from_model(df_selection.iloc[index])

    df_selection['Index'] = df_selection.index
    return df_selection

def check_if_model_last_epoch(df, modelnames, max_epoch):
    """Checks if the model we want to use to calculate the MSE is the trained model (so in the right epoch)
    
    In the CSV file where the data is saved the saved model only corresponds to the last epoch of that training.
    This function will check if we are using this model at the right epoch.
    """
    # Kijk of alle epochs worden genomen
    # df_selection[modelname in modelnames] == df_selection[modelname in modelnames & epochs <= max_epoch]
    assert df[(df["modelname"].isin(modelnames))].shape == df[(df["modelname"].isin(modelnames)) 
                                                        & df["epoch"].isin(range(0, max_epoch+1))].shape, f"The chosen epochs are not the last epochs"

def select_first_instances(df, amount):
    """Assuming varying normalization and activation"""
    normalization_methods = ['no-norm', 'z-score', 'min-max']
    activation_functions = ['relu', 'tanh', 'sigmoid']

    all_indices_keep = []
    for normalization in normalization_methods:
        for activation in activation_functions:
            indices_to_keep = df.index[(df['normalization_method'] == normalization) & (df['activation_function'] == activation)][:amount]
            all_indices_keep.extend(indices_to_keep)
    
    df = df.iloc[all_indices_keep].reset_index(drop=True)

    return df

def select_data_from_df(df, learning_rate, nodes, layers, option, max_epochs, normalization_methods=None, activation_functions=None):
    """"
    From the dataframe with all the data the data with given specifications is selected.
    It has also been made sure that only the data trained up to the max epoch has been selected. 
    """
    # select all these data
    if normalization_methods is None and activation_functions is None:
        df_selection = df[
                    (df['learning_rate'] == learning_rate) &
                    (df['nodes'].apply(lambda x: x == nodes)) &
                    (df['layers'] == layers) &
                    (df['option'] == option)
                    ]
    else:
        df_selection = df[
                    (df['learning_rate'] == learning_rate) &
                    (df['nodes'].apply(lambda x: x == nodes)) &
                    (df['layers'] == layers) &
                    (df['option'] == option) &
                    (df['normalization_method'].isin(normalization_methods)) &
                    (df['activation_function'].isin(activation_functions))
                    ]
    # select data with maximal epoch of training = max_epochs
    df_selection = df_select_max_epoch(df_selection, max_epochs)
    return df_selection

# => statistics <=

def check_Anova(data_group, alpha = 0.05):
    """Checks Shapiro-Wilk test and Levene's test
    
    To apply an Anova test (comparing at least 3 levels compare of 1 independent variable) we must have independent observations (which we assume).
    We must also have Normality of Distributed variables (checked by Shapiro-Wilk) and we must have Homogeneity of Variance (checked by Levene's test).
    """
    anova_check = True

    # Shapiro test
    for data in data_group:
        statistic, p_value = shapiro(data)
        if p_value < alpha:  # alpha standard is 0.05
            print(f"Data does NOT satisfy Shapiros test, p-value: {p_value}")
            anova_check = False
        else: 
            print(f"Data DOES satisfy Shapiros test, p-value: {p_value}")

    # Levene's test
    statistic, p_value = levene(*data_group)
    print(p_value, "levene")
    if p_value < alpha:
        print(f"Data does NOT satisfy Levene's test, p-value: {p_value}")
        anova_check = False
    else:
        print(f"Data DOES satisfy Levene's test, p-value: {p_value}")

    if anova_check == False:
        # Kruskal-Wallis test
        is_kruskal_wallis_significant: bool
        statistic, p_value = kruskal(*data_group, nan_policy='raise')
        if p_value < alpha:
            print(f"Data does NOT satisfy Kruskal-Walis test, p-value: {p_value}")
            # anova_check = False 
            is_kruskal_wallis_significant = False
        else:
            print(f"Data DOES satisfy Kruskal-Walis test, p-value: {p_value}")
            is_kruskal_wallis_significant = True
        
        # Mann-Whitney U test
        if is_kruskal_wallis_significant is False:
            all_pairs = [(data_left, data_right) for n, data_left in enumerate(data_group) for data_right in data_group[n + 1:]]
            for index, (data_i, data_j) in enumerate(all_pairs):
                statistic, p_value = mannwhitneyu(data_i, data_j)
                if p_value < alpha:
                    print(f"Data does NOT satisfy Mann-Whitney U test, p-value: {p_value}, index {index}")
                else:
                    print(f"Data DOES satisfy Mann-Whitney U test, p-value: {p_value}, index {index}")
    # If all tests were succesful we can go further with the Anova test
    if anova_check:
        print("Anova-assumptions are satisfied")
    return anova_check

def check_Anova_one_norm_two_act(data_group, alpha = 0.05):
    """Checks Shapiro-Wilk test and Levene's test
    
    To apply an Anova test (comparing at least 3 levels compare of 1 independent variable) we must have independent observations (which we assume).
    We must also have Normality of Distributed variables (checked by Shapiro-Wilk) and we must have Homogeneity of Variance (checked by Levene's test).
    """
    anova_check = True

    # Shapiro test
    for data in data_group:
        statistic, p_value = shapiro(data)
        if p_value < alpha:  # alpha standard is 0.05
            print(f"Data does NOT satisfy Shapiros test, p-value: {p_value}")
            anova_check = False
        else: 
            print(f"Data DOES satisfy Shapiros test, p-value: {p_value}")

    # Levene's test
    statistic, p_value = levene(*data_group)
    print(p_value, "levene")
    if p_value < alpha:
        print(f"Data does NOT satisfy Levene's test, p-value: {p_value}")
        anova_check = False
    else:
        print(f"Data DOES satisfy Levene's test, p-value: {p_value}")

    if anova_check:
        print("Anova-assumptions are satisfied")
        f_statistic, p_value = f_oneway(*data_group)
        print(f"F-statistic, {f_statistic}")
        print(f"p-value, {p_value}")
        p_value = round(p_value, 4)
        return {'Anova': p_value}

    # Kruskal-Wallis test
    is_kruskal_wallis_significant: bool
    statistic, p_value = kruskal(*data_group, nan_policy='raise')
    if p_value < alpha:
        print(f"Data does NOT satisfy Kruskal-Walis test, p-value: {p_value}")
        anova_check = False
        is_kruskal_wallis_significant = False
        p_value = round(p_value, 4)
        return {'Kruskal Wallis': p_value}
    else:
        print(f"Data DOES satisfy Kruskal-Walis test, p-value: {p_value}")
        is_kruskal_wallis_significant = True
        p_value = round(p_value, 4)
        return {'Kruskal Wallis': p_value}
    return {'Statistic': None}

#  => Saving <=

# Extra functions of saving

def retrieve_MSE_data_from_param_and_average(normalization_method, activation_function, learning_rate, nodes, layers, max_epochs, option, average, df=None):
    """Retrieve Mean Squared Error (MSE) data for given parameters and average count.

    OUTDATED: Uses 'run' method instead of modelname
    
    This function retrieves MSE data for a specified combination of parameters and the specified number of averages.
    It returns a list containing the MSE values. Can be used to make a boxplot.
    
    Args:
        normalization_method (str): The normalization method used.
        activation_function (str): The activation function used.
        learning_rate (float): The learning rate used.
        nodes (list): List of node counts for each layer.
        layers (int): Number of layers in the neural network.
        max_epochs (int): Maximum number of epochs for training.
        option (str): Additional option for data retrieval.
        average (int): Number of averages to retrieve.
        df (DataFrame, optional): DataFrame containing the data. If None, loads the default DataFrame from a CSV file.
    
    Returns:
        list: A list containing the MSE values for the specified parameters and averages.
    
    Note:
        This function does not compute the average MSE. It retrieves all MSE values for the specified averages,
        allowing the average to be computed outside of the function.
        (Function not used extensively; plotting functionality needs to be added.)
    """
    # Load Dataframe if not provided
    if df is None:
        absolute_path = os.path.dirname(__file__)
        relative_path = f"FHN_NN_loss_and_model_{TAU}.csv"
        csv_name = os.path.join(absolute_path, relative_path)
        df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}) # literal eval returns [2,2] as list not as str

    # Check if data exists for the specified parameters and averages
    run_values = does_data_exist(df, normalization_method, activation_function, learning_rate, nodes, layers, max_epochs, option, average)  # returns list of run values
    
    mse_values = []
    for run in run_values:
        # Retrieve model(name) and mean_std
        modelname, all_mean_std = retrieve_modelname_meanstd(df, run, normalization_method, activation_function, learning_rate, nodes, layers, max_epochs, option)
        
        # Check if model belongs to last epoch
        check_if_model_last_epoch(df, [modelname], max_epochs)

        model = retrieve_model_from_name(modelname)
        
        # Load data of nullclines in phasespace
        amount_of_points = 500
        axis_values, nullcline_values = nullcline_and_boundary(option, amount_of_points)

        # Predict normalized data 
        input_prediction, reverse_norm_mean_std = normalize_axis_values(axis_values, all_mean_std, option)
        prediction_output_normalized = model.predict(input_prediction)
        # Reverse normalize to 'normal' data
        prediction_output_column = reverse_normalization(prediction_output_normalized, reverse_norm_mean_std)
        prediction_output = prediction_output_column.reshape(-1)

        mse_val = calculate_mean_squared_error(nullcline_values, prediction_output)
        mse_values.append(mse_val)
    return mse_values

def save_val_mse_df(df: pd.DataFrame, name):
    """
    Saves the dataframe which includes a column named 'MSE'.

    Note:
        This function is not used on its own.
    """
    absolute_path = os.path.dirname(__file__)
    relative_path = f"VAL_vs_MSE_{TAU}_{NUM_POINTS}"
    folder_path = os.path.join(absolute_path, relative_path)
    relative_path = f"{name}.csv"
    csv_name = os.path.join(folder_path, relative_path)

    df.to_csv(csv_name, index=False)

# Main functions of saving

def save_all_MSE_vs_VAL(learning_rate, nodes, layers, max_epochs, option, amount_per_parameter, save):
    """
    Save Mean Squared Error (MSE) vs. Validation (VAL) data to a CSV file.
    
    This function selects data from a DataFrame based on specified parameters and saves the calculated MSE 
    alongside the rest of the dataframe in the CSV file. The saved file can be later opened for data processing 
    using the 'open_csv_and_plot' function.
    
    Args:
        learning_rate (float): The learning rate used for training.
        nodes (list): List of node counts for each layer in the neural network.
        layers (int): Number of layers in the neural network.
        max_epochs (int): Maximum number of epochs for training.
        option (str): Additional option for data selection.
        amount_per_parameter (int): Number of instances to consider per parameter combination.
        save (bool): Whether to save the data to a CSV file.
    
    Returns:
        None
    
    Note:
        This function prints the number of models found and checks if the expected amount of data is obtained
        before saving. It also performs checks to ensure that the maximum epoch of training belongs to the model.
    """
    # Load DataFrame
    absolute_path = os.path.dirname(__file__)
    relative_path = f"FHN_NN_loss_and_model_{TAU}.csv"
    csv_name = os.path.join(absolute_path, relative_path)
    df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}) # literal eval returns [2,2] as list not as str

    # Select data based on specified parameters
    df_selection = select_data_from_df(df, learning_rate, nodes, layers, option, max_epochs)
    # Only select last epoch (validation at max_epochs)
    df_selection = df_selection[(df_selection["epoch"] == max_epochs)].reset_index(drop=True)

    # Consider a fixed amount before plotting: amount_per_param * amount_of_param (amount_of_param = 9=3x3)
    df_selection = select_first_instances(df_selection, amount_per_parameter) # important to do this before sorting so that it stays 'random'

    sorted_df = df_selection.sort_values(by='validation').reset_index(drop=True)
    modelnames = sorted_df['modelname'].values # returns numpy array

    # Check if the amount is correct
    print(len(modelnames), "Models have been found.")
    assert len(modelnames) == amount_per_parameter * 9, 'something went wrong, not saving amount wanted'
    
    # Create DataFrame for plotting
    df_for_plot = sorted_df
    df_for_plot['MSE'] = pd.Series(dtype=object)
    for modelname in modelnames:
        mean_std = sorted_df.loc[sorted_df['modelname'] == modelname, 'mean_std'].iloc[0] # takes one mean_std value (all same for same model) and makes dict from it
        mse_value = calculate_MSE_data_from_modelname(modelname, option, mean_std)
        df_for_plot.loc[sorted_df['modelname'] == modelname, 'MSE'] = mse_value

    save_name = f"VAL_VS_MSE_{option}_lr{learning_rate}_epochs{max_epochs}_total{len(modelnames)}_{nodes}_layers{layers}"
    if save:
        save_val_mse_df(df_for_plot, save_name)

def save_one_norm_two_act_MSE_vs_VAL(learning_rate, nodes, layers, max_epochs, option, normalization_method: list, activation_functions: list, amount_per_parameter, save):
    """
    Save Mean Squared Error (MSE) vs. Validation (VAL) data to a CSV file for one normalization method and two activation functions.
    
    This function selects data from a DataFrame based on specified parameters, including one normalization method and two activation functions.
    It saves the calculated MSE alongside the DataFrame a CSV file. The saved file can be later opened for data processing 
    using the 'open_csv_and_plot' function.
    
    Args:
        learning_rate (float): The learning rate used for training.
        nodes (list): List of node counts for each layer in the neural network.
        layers (int): Number of layers in the neural network.
        max_epochs (int): Maximum number of epochs for training.
        option (str): Additional option for data selection.
        normalization_method (list): List of normalization methods to consider.
        activation_functions (list): List of activation functions to consider.
        amount_per_parameter (int): Number of instances to consider per parameter combination.
        save (bool): Whether to save the data to a CSV file.
    
    Returns:
        None
    
    Note:
        This function prints the number of models found and checks if the expected amount of data is obtained
        before saving. It also performs checks to ensure that the maximum epoch for calculating MSE belongs to each model.
    """
    # Load DataFrame
    absolute_path = os.path.dirname(__file__)
    relative_path = f"FHN_NN_loss_and_model_{TAU}_{NUM_POINTS}.csv"
    csv_name = os.path.join(absolute_path, relative_path)
    df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}) # literal eval returns [2,2] as list not as str

    # Select data based on specified parameters
    df_selection = select_data_from_df(df, learning_rate, nodes, layers, option, max_epochs, normalization_method, activation_functions)
    # Only select last epoch (validation at max_epochs)
    df_selection = df_selection[(df_selection["epoch"] == max_epochs)].reset_index(drop=True)

    # Consider a fixed amount before plotting: amount_per_param * amount_of_param (amount_of_param = 9=3x3)
    df_selection = select_first_instances(df_selection, amount_per_parameter) # Important to do this before sorting so that it stays 'random'

    sorted_df = df_selection.sort_values(by='validation').reset_index(drop=True)
    modelnames = sorted_df['modelname'].values # Returns numpy array

    # Check if the amount is correct
    print(len(modelnames), "Models have been found.")
    assert len(modelnames) == amount_per_parameter * (len(normalization_method)*len(activation_functions)), f'something went wrong, not saving amount wanted {sorted_df}'

    # Create DataFrame for plotting
    df_for_plot = sorted_df
    df_for_plot['MSE'] = pd.Series(dtype=object)
    for modelname in modelnames:
        mean_std = sorted_df.loc[sorted_df['modelname'] == modelname, 'mean_std'].iloc[0] # takes one mean_std value (all same for same model) and makes dict from it
        mse_value = calculate_MSE_data_from_modelname(modelname, option, mean_std)
        df_for_plot.loc[sorted_df['modelname'] == modelname, 'MSE'] = mse_value

    save_name = f"VAL_VS_MSE_{option}_{normalization_method}_{activation_functions}_lr{learning_rate}_epochs{max_epochs}_total{len(modelnames)}_{nodes}_layers{layers}"
    if save:
        save_val_mse_df(df_for_plot, save_name)
    

# => Plotting <=

# Extra functions for plotting
# (used by open_csv_and_plot_all)

def plot_seaborn_validation_mse(df, plot_title, study_limit_cycle, hue_order=None, style_order=None):
    """
    Create a scatterplot using Seaborn to visualize the relationship between validation and MSE.

    This function generates a scatterplot to visualize the relationship between validation values and Mean Squared Error (MSE)
    using Seaborn library. It also provides an option to overlay specific data points for detailed analysis.

    Args:
        df (DataFrame): The DataFrame containing the data to be plotted.
        plot_title (str): Title for the plot.
        study_limit_cycle (bool): Flag to indicate whether to include additional data points for detailed study.
        hue_order (list, optional): List specifying the order of hue (color) categories. Default is None.
        style_order (list, optional): List specifying the order of style categories (marker shapes or line styles). Default is None.

    Returns:
        None

    Note:
        - The 'hue_order' parameter determines the order of colors for different normalization methods.
        - The 'style_order' parameter determines the order of marker shapes or line styles for different activation functions.
        - When 'study_limit_cycle' is True, additional data points are overlaid and highlighted on the plot for which have been studied more in detail.
    """
    # Set figure size
    plt.figure(figsize=(13,6))
    
    if hue_order is None:
        hue_order = ['no-norm', 'z-score', 'min-max']
    if style_order is None:
        style_order = ['relu', 'tanh', 'sigmoid']
    # hue_order = ['min-max']                    
    # style_order = ['relu', 'sigmoid']         
     
    # Create scatterplot
    sns.scatterplot(data=df, x="validation", y="MSE", hue="normalization_method", hue_order=hue_order, style='activation_function', style_order=style_order)

    # Add identity line
    plt.plot(df['validation'], df['validation'], label='Identity')

    # Overlay additional data points if requested
    if study_limit_cycle:
        # Hardcoded model names for additional data points: must be done manual (chosen in file 'plot_NN_ps.py')
        
        #lr0_01 500 epoch 2 layer [8,8]
        # modelnames = ['2cdbd6e254c84cf18ee4cd692ff169db', '12136245d29447eb9dd7ff4e357808a5', '4aa7f465e242460eb4d3edd19f6712a0', '5f7fde69ff9d4d50856a1c26d8bcd942'] #lr0_01_500epoch 2 layer [8,8]
        
        #lr0.005 1000 epoch 2 layer [8,8]
        # modelnames = ['2b79bed8546649ab91589958d992cbae', '36da3fdf6aa04f04a9527df5b5bea88a', '429930bf12ab472e87a384d44ca7d5db', 'a0d044c5fedd4d779d82e8b7adbf8343'] #lr0.005_1000epoch 2layer [8,8]
        # if df['option'][0] == ...df_plot['learning_rate'][0], df_plot['nodes'][0], df_plot['layers'][0], df_plot['epoch'][0]

        # lr 0.01, 500 epoch, 16 layers, each 8 nodes
        modelnames = ['003654a309db440296f8993d8176b6d2', 'ab3b8d2e3677475e9562d94a14e53116', '61966c1986c24475b3d411f4ebf6913d', '85477d64407e4cc785f967ff888cfd0c']

        df_selection_lc = df[(df['modelname'].isin(modelnames))]
        sns.scatterplot(data=df_selection_lc, x='validation', y='MSE', marker='^', hue='modelname', alpha=0.7, s=100, palette='dark')

    plt.title(plot_title)
    plt.xlabel("Validation")
    plt.ylabel("Mean Squared Error")
    
    plt.xlim(df['validation'].min(), df['validation'].max())
    plt.ylim(df['MSE'].min(), df['MSE'].max())
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout(rect=[0, 0, 0.7, 1])
    # plt.subplots_adjust(left=None, bottom=None, right=1, top=None, wspace=None, hspace=None)
    plt.legend(loc="upper right", bbox_to_anchor=(1.44,1))
    plt.show()

def plot_seaborn_jointplot_validation_mse(df, plot_title, df_selection_lc, study_limit_cycle):
    """
    Jointplot seaborn of validation vs. mse

    Not used on its own
    """
    # using seaborn
    # plt.figure(figsize=(10,6))
    # sns.relplot(data=df, x="validation", y="MSE", hue="normalization_method", style='activation_function', height=6, aspect=1.4)
    hue_order = ['no-norm', 'z-score', 'min-max']
    # style_order = ['relu', 'tanh', 'sigmoid']
    # sns.scatterplot(data=df, x="validation", y="MSE", hue="normalization_method", hue_order=hue_order, style='activation_function', style_order=style_order)
    if study_limit_cycle:
        sns.scatterplot(data=df_selection_lc, x='validation', y='MSE', marker='^', hue='Index', alpha=0.7, s=100, palette='dark')
    
    sns.jointplot(data=df, x="log_validation", y="log_MSE", hue='normalization_method', hue_order=hue_order, kind='kde')
    plt.title(plot_title)
    plt.xlabel("Validation")
    plt.ylabel("Mean Squared Error")

    sns.jointplot(data=df, x="log_validation", y="log_MSE", hue='activation_function', hue_order=['relu', 'tanh', 'sigmoid'] , kind='kde')
    plt.title(plot_title)
    plt.xlabel("Validation")
    plt.ylabel("Mean Squared Error")

    # cooler version of thing below
    sns.jointplot(data=df, x="log_validation", y="log_MSE", kind='kde', fill=True, cmap='mako', tresh=0, levels=100)
    plt.title(plot_title)
    plt.xlabel("Validation")
    plt.ylabel("Mean Squared Error")

    sns.jointplot(data=df, x="log_validation", y="log_MSE", kind='kde')
    plt.title(plot_title)
    plt.xlabel("Validation")
    plt.ylabel("Mean Squared Error")

    sns.jointplot(data=df, x="log_validation", y="log_MSE", hue = 'normalization_method')
    plt.title(plot_title)
    plt.xlabel("Validation")
    plt.ylabel("Mean Squared Error")

    g = sns.jointplot(data=df, x="log_validation", y="log_MSE", hue = 'normalization_method')
    g.plot_joint(sns.kdeplot, color='r', zorder=0, levels=5)
    plt.title(plot_title)
    plt.xlabel("Validation")
    plt.ylabel("Mean Squared Error")

    sns.jointplot(x="log_validation", y="log_MSE", data=df,
                    kind="reg", truncate=False)
    plt.title(plot_title)
    plt.xlabel("Validation")
    plt.ylabel("Mean Squared Error")

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # above are shown from last (under) to first (above)
    plt.show()

def boxplot_mse(df_plot, total_amount):
    """
    Generate boxplots to visualize the Mean Squared Error (MSE) across different parameters.

    This function creates boxplots to visualize the distribution of MSE values across different combinations
    of normalization methods and activation functions.

    Args:
        df_plot (DataFrame): DataFrame containing the data to be plotted.
        total_amount (int): Total number of simulations.

    Returns:
        None
    """

    # ANOVA (ANalysis Of VAriance)
    # H0 hypothesis: the group means are the same, if p<0.05 (random empirical) than hypothesis wrong: not the same
    # so if p < 0.05, then means not the same: so the VARIABLE HAS an effect on the mean, so VARIABLE has a significant effect
    
    # 1) All - Two-Way ANOVA Test
    all_normalization_methods= ['no-norm', 'z-score', 'min-max']
    all_activation_functions = ['relu', 'tanh', 'sigmoid' ]

    model = ols('log_MSE ~ C(normalization_method) + C(activation_function) + C(normalization_method):C(activation_function)', data=df_plot).fit()
    anova_table = sm.stats.anova_lm(model, type=2)
    print(anova_table)
    plt.figure(figsize=(6.4, 6))
    
    hue_order = ['relu', 'tanh', 'sigmoid']
    x_order = ['no-norm', 'z-score', 'min-max']
    ax = sns.boxplot(data=df_plot, x="normalization_method", y="MSE", hue="activation_function", hue_order=hue_order, order=x_order, palette='pastel', log_scale=True)
    # ax = sns.boxplot(data=df_plot, x="normalization_method", y="MSE", hue="activation_function", hue_order=hue_order, order=x_order, palette='pastel', log_scale=True)
    sns.stripplot(data=df_plot, x="normalization_method", y="MSE", hue="activation_function", hue_order=hue_order, order=x_order, dodge=True, palette='tab10')
    # plt.yscale('log')
    plt.title(f"MSE: {total_amount//9} simulations, {df_plot['option'][0]}.\n lr: {df_plot['learning_rate'][0]}, nodes: {df_plot['nodes'][0]}, layers: {df_plot['layers'][0]}, max epochs {df_plot['epoch'][0]}")

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[3:6], labels[3:6], loc='upper right')

    plt.text(0.5, 0.1, f'Factor1 p-value: {anova_table["PR(>F)"].iloc[0]:.4f}', ha='center', transform=plt.gca().transAxes) #gca (get current axes), transAxes: zorgt ervoor dat coordinaat linksonder (0,0) en rechtsboven (1,1)
    plt.text(0.5, 0.05, f'Factor2 p-value: {anova_table["PR(>F)"].iloc[1]:.4f}', ha='center', transform=plt.gca().transAxes)
    plt.text(0.5, 0.0, f'Interaction p-value: {anova_table["PR(>F)"].iloc[2]:.4f}', ha='center', transform=plt.gca().transAxes)

    plt.show()

    # 2) Together Normalization Method
    # Check if (one-way) Anova-test is allowed:
    group_data = [df_plot[(df_plot["normalization_method"] == norm_method)]["log_MSE"] for norm_method in all_normalization_methods]
    anova_allowed = check_Anova(group_data, alpha = 0.05)
    # Anova-test
    if anova_allowed:
        group_data = [df_plot[(df_plot["normalization_method"] == norm_method)]["log_MSE"] for norm_method in all_normalization_methods]
        f_statistic, p_value = f_oneway(*group_data)
        print(f"F-statistic, {f_statistic}")
        print(f"p-value, {p_value}")
        p_value = round(p_value, 4)
    else:
        p_value = None

    plt.figure(figsize=(6.4, 6))

    hue_order=['no-norm', 'z-score', 'min-max']
    x_order = ['no-norm', 'z-score', 'min-max']
    sns.boxplot(data=df_plot, x="normalization_method", y="MSE", hue="normalization_method", hue_order=hue_order, order=x_order, log_scale=True)
    sns.stripplot(data=df_plot, x="normalization_method", y="MSE", hue="normalization_method", hue_order=hue_order, order=x_order, palette='tab10')
    plt.yscale('log')
    plt.title(f"MSE: {total_amount//9} simulations, {df_plot['option'][0]}.\n lr: {df_plot['learning_rate'][0]}, nodes: {df_plot['nodes'][0]}, layers: {df_plot['layers'][0]}, max epochs {df_plot['epoch'][0]}\n p-value: {p_value}")
    plt.show()

    # 2b) Without relu to show interaction effect better (only used for specific case)
    if False:
        group_data = [df_plot[(df_plot["normalization_method"] == norm_method)]["log_MSE"] for norm_method in all_normalization_methods]
        anova_allowed = check_Anova(group_data, alpha = 0.05)
        # Anova-test
        if anova_allowed:
            group_data = [df_plot[(df_plot["normalization_method"] == norm_method)]["log_MSE"] for norm_method in all_normalization_methods]
            f_statistic, p_value = f_oneway(*group_data)
            print(f"F-statistic, {f_statistic}")
            print(f"p-value, {p_value}")
            p_value = round(p_value, 4)
        else:
            p_value = None

        plt.figure(figsize=(6.4, 6))

        hue_order=['no-norm', 'z-score', 'min-max']
        x_order = ['no-norm', 'z-score', 'min-max']
        df_subplot = df_plot[df_plot['activation_function'].isin(['tanh', 'sigmoid'])]
        sns.boxplot(data=df_subplot, x="normalization_method", y="MSE", hue="normalization_method", hue_order=hue_order, order=x_order, log_scale=True)
        sns.stripplot(data=df_subplot, x="normalization_method", y="MSE", hue="normalization_method", hue_order=hue_order, order=x_order, palette='tab10')
        plt.yscale('log')
        plt.title(f"MSE: {total_amount//9} simulations, tanh and sigmoid\n{df_plot['option'][0]}, lr: {df_plot['learning_rate'][0]}, nodes: {df_plot['nodes'][0]}, layers: {df_plot['layers'][0]}.\nMax epochs {df_plot['epoch'][0]} p-value: {p_value}")
        plt.show()

    # 3) Together Activation Function
    # Check if (one-way) Anova-test is allowed
    group_data = [df_plot[(df_plot["activation_function"] == activ_function)]["log_MSE"] for activ_function in all_activation_functions]
    anova_allowed = check_Anova(group_data, alpha = 0.05)
    # Anova-test
    if anova_allowed:
        group_data = [df_plot[(df_plot["activation_function"] == activ_function)]["log_MSE"] for activ_function in all_activation_functions]
        f_statistic, p_value = f_oneway(*group_data)
        print(f"F-statistic, {f_statistic}")
        print(f"p-value, {p_value}")
        p_value = round(p_value, 4)
    else:
        p_value = None

    plt.figure(figsize=(6.4, 6))

    hue_order = ['relu', 'tanh', 'sigmoid']
    # hue_order = ['relu', 'sigmoid']
    sns.boxplot(data=df_plot, x="activation_function", y="MSE", hue="activation_function", hue_order=hue_order, order=hue_order, palette = 'pastel', log_scale=True)
    sns.stripplot(data=df_plot, x="activation_function", y="MSE", hue="activation_function", hue_order=hue_order, order=hue_order, palette='tab10')
    plt.yscale('log')
    plt.title(f"MSE: {total_amount//9} simulations, {df_plot['option'][0]}.\n lr: {df_plot['learning_rate'][0]}, nodes: {df_plot['nodes'][0]}, layers: {df_plot['layers'][0]}, max epochs {df_plot['epoch'][0]}\n p-value: {p_value}")

    plt.show()

def boxplot_mse_one_norm_two_act(df_plot, total_amount):
    """
    Generate boxplots to visualize the Mean Squared Error (MSE) across different combinations of one normalization method and two activation functions.

    This function creates boxplots to visualize the distribution of MSE values across different combinations
    of one normalization method and two activation functions.

    Args:
        df_plot (DataFrame): DataFrame containing the data to be plotted.
        total_amount (int): Total number of simulations.
        statistic_bool (bool, optional): Whether to perform statistical analysis. Defaults to False.

    Returns:
        None
    """

    # ANOVA (ANalysis Of VAriance)
    # H0 hypothesis: the group means are the same, if p<0.05 (random empirical) than hypothesis wrong: not the same
    # so if p < 0.05, then means not the same: so the VARIABLE HAS an effect on the mean, so VARIABLE has a significant effect
    
    statistic_bool = False
    # 1) All - Two-Way ANOVA Test
    all_normalization_methods= ['min-max']
    all_activation_functions = ['relu', 'sigmoid']

    if statistic_bool:
        model = ols('log_MSE ~ C(normalization_method) + C(activation_function) + C(normalization_method):C(activation_function)', data=df_plot).fit()
        anova_table = sm.stats.anova_lm(model, type=2)
        print(anova_table)
    plt.figure(figsize=(6.4, 6))
    
    # hue_order = ['relu', 'tanh', 'sigmoid']
    # x_order = ['no-norm', 'z-score', 'min-max']
    hue_order = ['relu', 'sigmoid']
    x_order = ['min-max']
    ax = sns.boxplot(data=df_plot, x="normalization_method", y="MSE", hue="activation_function", hue_order=hue_order, order=x_order, palette='pastel', log_scale=True)
    sns.stripplot(data=df_plot, x="normalization_method", y="MSE", hue="activation_function", hue_order=hue_order, order=x_order, dodge=True, palette='tab10')
    plt.yscale('log')
    plt.title(f"MSE: {total_amount//9} simulations, {df_plot['option'][0]}.\n lr: {df_plot['learning_rate'][0]}, nodes: {df_plot['nodes'][0]}, layers: {df_plot['layers'][0]}, max epochs {df_plot['epoch'][0]}")

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[2:4], labels[2:4], loc='upper right')

    if statistic_bool:
        plt.text(0.5, 0.1, f'Factor1 p-value: {anova_table["PR(>F)"].iloc[0]:.4f}', ha='center', transform=plt.gca().transAxes) #gca (get current axes), transAxes: zorgt ervoor dat coordinaat linksonder (0,0) en rechtsboven (1,1)
        plt.text(0.5, 0.05, f'Factor2 p-value: {anova_table["PR(>F)"].iloc[1]:.4f}', ha='center', transform=plt.gca().transAxes)
        plt.text(0.5, 0.0, f'Interaction p-value: {anova_table["PR(>F)"].iloc[2]:.4f}', ha='center', transform=plt.gca().transAxes)

    plt.show()

    # 2) Together Activation Function
    # Check if (one-way) Anova-test is allowed
    p_value = None
    group_data = [df_plot[(df_plot["activation_function"] == activ_function)]["log_MSE"] for activ_function in all_activation_functions]
    statistic_p_value = check_Anova_one_norm_two_act(group_data, alpha = 0.05)

    plt.figure(figsize=(6.4, 6))

    # hue_order = ['relu', 'tanh', 'sigmoid']
    hue_order = ['relu', 'sigmoid']
    sns.boxplot(data=df_plot, x="activation_function", y="MSE", hue="activation_function", hue_order=hue_order, order=hue_order, palette = 'pastel', log_scale=True)
    sns.stripplot(data=df_plot, x="activation_function", y="MSE", hue="activation_function", hue_order=hue_order, order=hue_order, palette='tab10')
    plt.yscale('log')
    plt.title(f"MSE: {total_amount//9} simulations, {df_plot['option'][0]}.\n lr: {df_plot['learning_rate'][0]}, nodes: {df_plot['nodes'][0]}, layers: {df_plot['layers'][0]}, max epochs {df_plot['epoch'][0]}\n{statistic_p_value}, 'min-max'.")

    plt.show()

def plot_seaborn_validation_mse_one_norm_two_act(df, plot_title):
    """
    Create a scatterplot using Seaborn to visualize the relationship between validation and MSE for one normalization method and two activation functions.

    This function generates a scatterplot to visualize the relationship between validation values and Mean Squared Error (MSE)
    using Seaborn library. It also provides an option to overlay specific data points for detailed analysis.

    Args:
        df (DataFrame): The DataFrame containing the data to be plotted.
        plot_title (str): Title for the plot.

    Returns:
        None

    Note:
        Not used on its own, used in 'big_MSE_VAL_for_one_norm_two_act
    """
    # using seaborn
    plt.figure(figsize=(13,6))
    hue_order = ['relu', 'sigmoid']
    sns.scatterplot(data=df, x="validation", y="MSE", hue="activation_function", hue_order=hue_order) # , style='activation_function, style_order=style_order)
    plt.plot(df['validation'], df['validation'], label='Identity')
    
    plt.title(plot_title)
    plt.xlabel("Validation")
    plt.ylabel("Mean Squared Error")
    
    print("boundaries of validation:", df['validation'].min(), df['validation'].max())
    print("boundaries of MSE", df['MSE'].min(), df['MSE'].max())
    plt.xlim(df['validation'].min(), df['validation'].max())
    plt.ylim(df['MSE'].min(), df['MSE'].max())
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout(rect=[0, 0, 0.7, 1])
    # plt.subplots_adjust(left=None, bottom=None, right=1, top=None, wspace=None, hspace=None)
    plt.legend(loc="upper right", bbox_to_anchor=(1.44,1))
    plt.show()

    df['log_validation'] = np.log10(df['validation'])
    df['log_MSE'] = np.log10(df['MSE'])

    sns.jointplot(x="log_validation", y="log_MSE", data=df, hue='activation_function', hue_order=['relu', 'sigmoid'])
    plt.show()

def concatenate_values(row):
    """used in big_MSE_for_one_norm_two_activation"""
    first_value = str(row['nodes'][0])
    return first_value + row['activation_function']

# (used by three_by_three_plot)
def open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount, normalization_methods=None, activation_functions=None):
    """
    Open a saved CSV file containing MSE data and return the DataFrame.

    This function opens a CSV file containing Mean Squared Error (MSE) data and returns the DataFrame.
    If specific normalization methods and activation functions are provided, it opens the corresponding CSV file.
    If no normalization methods and activation functions are provided, it opens the CSV file containing all combinations.

    Args:
        option (str): The option used for training.
        learning_rate (float): The learning rate used for training.
        max_epochs (int): The maximum number of epochs used for training.
        nodes (int): The number of nodes in the neural network.
        layers (int): The number of layers in the neural network.
        amount (int): The total amount of simulations.
        normalization_methods (list, optional): A list of normalization methods. Defaults to None.
        activation_functions (list, optional): A list of activation functions. Defaults to None.

    Returns:
        DataFrame: DataFrame containing the MSE data.

    Note:
        Used by the function 'three_by_three_plot', 'plot_validation_vs_mse_one_norm_two_act', 'Val_vs_MSE_node_norm_act_plot', 'big_MSE_for_one_norm_two_activation', 'big_MSE_vs_VAL_for_one_norm_two_act'
    """
    if normalization_methods is None and activation_functions is None:
        save_name = f"VAL_VS_MSE_{option}_lr{learning_rate}_epochs{max_epochs}_total{amount*9}_{nodes}_layers{layers}"
    else:
        save_name = f"VAL_VS_MSE_{option}_{normalization_methods}_{activation_functions}_lr{learning_rate}_epochs{max_epochs}_total{amount*len(normalization_methods)*len(activation_functions)}_{nodes}_layers{layers}"
    
    # Open CSV file
    # absolute_path = os.path.dirname(__file__)
    # folder_path = os.path.join(absolute_path, f"VAL_vs_MSE_{TAU}_{NUM_POINTS}")
    # csv_name = os.path.join(folder_path, f"{save_name}.csv")

    # Try the first folder path
    try:
        absolute_path = os.path.dirname(__file__)
        folder_path = os.path.join(absolute_path, f"VAL_vs_MSE_{TAU}_{NUM_POINTS}")
        csv_name = os.path.join(folder_path, f"{save_name}.csv")
        
        # Check if the folder and file exist, raise an exception if not
        if not os.path.exists(folder_path):
            input(f"Het is misgegaan, de file: {folder_path} wordt niet gevonden, heel zeker dat we verder moeten gaan met andere file?")
            raise FileNotFoundError(f"Directory {folder_path} does not exist.")
        if not os.path.isfile(csv_name):
            input(f"Het is misgegaan, de CSV: {csv_name} wordt niet gevonden, heel zeker dat we verder moeten gaan met andere file?")
            raise FileNotFoundError(f"File {csv_name} does not exist.")

    except (FileNotFoundError, NotADirectoryError) as e:
        # Try the second folder path
        folder_path = os.path.join(absolute_path, f"VAL_vs_MSE_{TAU}")
        csv_name = os.path.join(folder_path, f"{save_name}.csv")
        
        # Check if the folder and file exist, raise an exception if not
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory {folder_path} does not exist.")
        if not os.path.isfile(csv_name):
            raise FileNotFoundError(f"File {csv_name} does not exist.")

    df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval})


    return df

def scatterplot_setup_provider(df_plot, ax):
    """
    Plot the validation on the x-axis and the MSE on the y-axis using provided DataFrame.

    This function prepares and plots a scatterplot of validation versus MSE using the given DataFrame.
    It calculates the logarithm of the MSE and validation values for better visualization.
    The Pearson correlation coefficient (PCC) is computed and displayed in the plot title.

    Args:
        df_plot (DataFrame): The DataFrame containing the data to be plotted.
        ax (matplotlib.axes.Axes): The axes object to plot on.

    Returns:
        None
    
    Note:
        Not used on its own, used in 'three_by_three_plot'
    """

    df_plot.loc[:, 'log_MSE'] = np.log10(df_plot['MSE'])
    df_plot.loc[:, 'log_validation'] = np.log10(df_plot['validation'])

    pearson_corr_coefficient = round(pearson_correlation(df_plot['log_validation'], df_plot['log_MSE']),4)
    plot_title = f'PCC:{pearson_corr_coefficient}'

    sns.scatterplot(data = df_plot, x='validation', y='MSE', ax=ax)
    ax.set_xlabel("Validation")
    ax.set_ylabel("Mean Squared Error")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(plot_title, fontsize=10)
    return pearson_corr_coefficient

# Main functions for plotting

# (3 normalization combination 3 activation)
def open_csv_and_plot_all(option, learning_rate, max_epochs, nodes, layers, total, study_lc: bool=False, normalization_methods=None, activation_functions=None):
    """
    Opens a saved DataFrame with MSE data and plots the Validation vs. MSE, and boxplots of the MSE.

    This function reads the CSV file containing MSE data, computes the logarithm of MSE and validation values for better visualization,
    calculates the Pearson correlation coefficient, and then plots the data using seaborn. It also provides options to study specific models
    and generates different types of plots based on the input parameters, depending if normalization and activation are specifically provided.

    Args:
        option (str): The option used for the simulation.
        learning_rate (float): The learning rate used for the simulation.
        max_epochs (int): The maximum number of epochs used for the simulation.
        nodes (int): The number of nodes used in the neural network.
        layers (int): The number of layers in the neural network.
        total (int): The total number of simulations.
        study_lc (bool, optional): Whether to study limit cycle models. Defaults to False.
        normalization_methods (list, optional): List of normalization methods. Defaults to None.
        activation_functions (list, optional): List of activation functions. Defaults to None.

    Returns:
        None
    """
    if normalization_methods is None and activation_functions is None:
        save_name = f"VAL_VS_MSE_{option}_lr{learning_rate}_epochs{max_epochs}_total{total}_{nodes}_layers{layers}"
    else:
        save_name = f"VAL_VS_MSE_{option}_{normalization_methods}_{activation_functions}_lr{learning_rate}_epochs{max_epochs}_total{total}_{nodes}_layers{layers}"

    # Open CSV File
    absolute_path = os.path.dirname(__file__)
    relative_path = f"VAL_vs_MSE_{TAU}"
    folder_path = os.path.join(absolute_path, relative_path)
    relative_path = f"{save_name}.csv"
    csv_name = os.path.join(folder_path, relative_path)
    df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval})

    # retrieve log_validation and log_MSE values (can only do pearson on normal distribution: after log)
    df.loc[:, 'log_MSE'] = np.log10(df['MSE'])
    df.loc[:, 'log_validation'] = np.log10(df['validation'])
    log_validation_values = df['log_validation'].values
    log_mse_values = df['log_MSE'].values

    # calculate pearson correlation coefficient
    pearson_corr_coefficient = round(pearson_correlation(log_validation_values, log_mse_values),4)
    
    # look at some results (best, worst, middle) (this code and below is a bit too automated: best to choose own)
    df_study = pd.DataFrame()
    if study_lc:
        df_study = study_model(df)

    # Prepare for plotting
    plot_title = f" Validation vs MSE,\n lr {learning_rate}, nodes {nodes}, layers {layers}, {option}, {max_epochs} max epochs, Amount: {total}\n Pearson Correlation Coefficient {pearson_corr_coefficient}"

    # Scatterplot
    plot_seaborn_validation_mse(df, plot_title, study_lc)
    
    # Joint_plot (optional for further study)
    # plot_seaborn_jointplot_validation_mse(df, plot_title, df_study, study_lc)

    # Boxplots
    if normalization_methods is None and activation_functions is None:
        boxplot_mse(df, total)
    else:
        boxplot_mse_one_norm_two_act(df, total)

def three_by_three_plot(learning_rate = 0.005, nodes = [8,8], layers=2, max_epochs=999, option='option_3', amount=40):
    """
    Plots a three by three grid of 'Validation vs MSE' using already saved data for different normalization and activation combinations.

    This function generates a three by three grid of scatterplots, where each plot represents the relationship between
    validation and mean squared error (MSE) for a specific combination of normalization method and activation function.
    The data for each plot is retrieved from the saved DataFrame obtained through 'open_csv_and_return_all' function.

    Args:
        learning_rate (float, optional): The learning rate used for the simulation. Defaults to 0.005.
        nodes (list, optional): List of node configurations for the neural network. Defaults to [8,8].
        layers (int, optional): Number of layers in the neural network. Defaults to 2.
        max_epochs (int, optional): Maximum number of epochs used for the simulation. Defaults to 999.
        option (str, optional): The option used for the simulation. Defaults to 'option_3'.
        amount (int, optional): The total number of simulations. Defaults to 40.

    Returns:
        None

    Note:
        Is used on itself, but needed to use 'save_all_MSE_VS_VAL' to save the data first.
    """
    normalization_methods = ['no-norm', 'z-score', 'min-max']
    activation_functions = ['relu', 'tanh', 'sigmoid']

    fig, axs = plt.subplots(3, 3, figsize=(8,6))

    min_mse = float('inf')
    max_mse = -float('inf')
    min_validation = float('inf')
    max_validation = -float('inf')

    df_plot = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount)

    for i, norm_method in enumerate(normalization_methods):
        for j, activ_func in enumerate(activation_functions):
            df_selection = df_plot[(df_plot['normalization_method'] == norm_method) & (df_plot['activation_function'] == activ_func)]

            min_mse = min(min_mse, min(df_selection['MSE']))
            max_mse = max(max_mse, max(df_selection['MSE']))
            min_validation = min(min_validation, min(df_selection['validation']))
            max_validation = max(max_validation, max(df_selection['validation']))

            scatterplot_setup_provider(df_selection, axs[i,j])

    for ax in axs.flat: # axs.flat: to iterate over axes
        ax.set_xlim(min_validation, max_validation)  
        ax.set_ylim(min_mse, max_mse)

    fig.text(0.1, 0.01, 'relu', ha='left', fontsize=12, color='maroon')
    fig.text(0.45, 0.01, 'tanh', ha='center', fontsize=12, color='maroon')
    fig.text(0.8, 0.01, 'sigmoid', ha='right', fontsize=12, color='maroon')

    fig.text(0.975, 0.85, 'no-norm', va='center', rotation='vertical', fontsize=12, color='maroon')
    fig.text(0.975, 0.5, 'z-score', va='center', rotation='vertical', fontsize=12, color='maroon')
    fig.text(0.975, 0.2, 'min-max', va='center', rotation='vertical', fontsize=12, color='maroon')

    plot_title = f" Validation vs MSE Tau {TAU},\n lr {learning_rate}, nodes {nodes}, layers {layers}, {option}, {max_epochs} max epochs, Amount: {amount}"
    fig.suptitle(plot_title)

    plt.tight_layout()
    plt.show()

# (1 normalization combination two activation)
def plot_validation_vs_mse_one_norm_two_act(learning_rate = 0.005, nodes = [8,8], layers=2, max_epochs=999, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid']):
    """
    plt.show MODIFIED IN modules.py

    Plots a two by one grid of 'Validation vs MSE' using already saved data for a single normalization method and two activation functions.

    This function generates a two by one grid of scatterplots, where each plot represents the relationship between
    validation and mean squared error (MSE) for a specific activation function, using a single normalization method.
    The data for each plot is retrieved from the saved DataFrame obtained through 'open_csv_and_return_all' function.

    Args:
        learning_rate (float, optional): The learning rate used for the simulation. Defaults to 0.005.
        nodes (list, optional): List of node configurations for the neural network. Defaults to [8,8].
        layers (int, optional): Number of layers in the neural network. Defaults to 2.
        max_epochs (int, optional): Maximum number of epochs used for the simulation. Defaults to 999.
        option (str, optional): The option used for the simulation. Defaults to 'option_3'.
        amount (int, optional): The total number of simulations. Defaults to 40.
        normalization_methods (list, optional): List of normalization methods. Defaults to ['min-max'].
        activation_functions (list, optional): List of activation functions. Defaults to ['relu', 'sigmoid'].

    Returns:
        None

    Note:
        This function is used on itself but requires the data to be saved first using 'save_one_norm_two_act_MSE_vs_VAL'.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10,5), squeeze=False)

    min_mse = float('inf')
    max_mse = -float('inf')
    min_validation = float('inf')
    max_validation = -float('inf')

    df_plot = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount, normalization_methods, activation_functions)

    for i, norm_method in enumerate(normalization_methods):
        for j, activ_func in enumerate(activation_functions):
            df_selection = df_plot[(df_plot['normalization_method'] == norm_method) & (df_plot['activation_function'] == activ_func)].copy()

            min_mse = min(min_mse, min(df_selection['MSE']))
            max_mse = max(max_mse, max(df_selection['MSE']))
            min_validation = min(min_validation, min(df_selection['validation']))
            max_validation = max(max_validation, max(df_selection['validation']))

            pearson_correlation_coefficient = scatterplot_setup_provider(df_selection, axs[i,j])

    for ax in axs.flat: # axs.flat: to iterate over axes
        ax.set_xlim(min_validation, max_validation)  
        ax.set_ylim(min_mse, max_mse)
        print("Customized limits can be employed here")
        # print("Customized Limits For TAU100")
        # ax.set_xlim(9*(10**(-7)), 0.2)
        # ax.set_ylim((10**(-6)),1) 

    fig.text(0.1, 0.01, 'relu', ha='left', fontsize=12, color='maroon')
    fig.text(0.6, 0.01, 'sigmoid', ha='right', fontsize=12, color='maroon')

    fig.text(0.985, 0.5, 'min-max', va='center', rotation='vertical', fontsize=12, color='maroon')

    plot_title = f" Validation vs MSE,\n lr {learning_rate}, nodes {nodes}, layers {layers}, {option}, {max_epochs} max epochs, Amount: {amount}"
    fig.suptitle(plot_title)

    plt.tight_layout()
    plt.clf()
    return pearson_correlation_coefficient
    plt.show()

def big_MSE_for_one_norm_two_activation(option, learning_rate, max_epochs, amount, normalization_methods, activation_functions, plot_option):
    """
    Performs detailed boxplot analysis and plots the distribution of Mean Squared Error (MSE) for one normalization method and two activation functions.

    This function generates boxplots to visualize the distribution of MSE for different configurations of nodes and layers, considering one normalization method and two activation functions.

    Args:
        option (str): The option used for the simulation.
        learning_rate (float): The learning rate used for the simulation.
        max_epochs (int): Maximum number of epochs used for the simulation.
        amount (int): The total number of simulations.
        normalization_methods (list): List of normalization methods to be analyzed.
        activation_functions (list): List of activation functions to be analyzed.
        plot_option (str): Plotting option, '0' for default coloring, '1' for custom coloring.

    Returns:
        None

    Note:
        This function is used on itself, but it requires the data to be saved first using 'save_one_norm_two_act_MSE_vs_VAL'.
    """
    
    df_together = pd.DataFrame()
    all_nodes_list = [[[4]*2, [8]*2, [16]*2], [[4]*4, [8]*4, [16]*4], [[4]*8, [8]*8, [16]*8], [[4]*16, [8]*16, [16]*16]]
    layers_list = [2, 4, 8, 16]

    plt.subplots(figsize=(12,6))

    if plot_option == '0':
        for nodes_list, layers in zip(all_nodes_list, layers_list):
            for nodes in nodes_list:
                print('at nodes, layers', nodes, layers)
                df_plot = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount, normalization_methods, activation_functions)
                # select the interesting rows (so everything except option, lr, max_epochs, nodes, layers, amount, normalization_method, activation_functions) Maak van de nodes ipv [X,X,X] gewoon X
                df_new = df_plot.copy()
                df_new.loc[:, 'nodes_per_layer'] = df_new['nodes'].iloc[0][0] # from first row takes first element
                df_together = pd.concat([df_together, df_new], ignore_index=True)
    
        hue_order = [4, 8, 16]
        x_order = [2, 4, 8, 16]

        sns.boxplot(data=df_together, x="layers", y="MSE", hue="nodes_per_layer", hue_order=hue_order, order=x_order, palette = 'pastel', log_scale=True)
        sns.stripplot(data=df_together, x="layers", y="MSE", hue="nodes_per_layer", hue_order=hue_order, order=x_order, dodge=True, palette='tab10')

    if plot_option == '1':
        for nodes_list, layers in zip(all_nodes_list, layers_list):
            for nodes in nodes_list:
                print('at nodes, layers', nodes, layers)
                df_plot = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount, normalization_methods, activation_functions)
                # select the interesting rows (so everything except option, lr, max_epochs, nodes, layers, amount, normalization_method, activation_functions) Maak van de nodes ipv [X,X,X] gewoon X
                df_new = df_plot.copy()
                # df_new['nodes_per_layer'] =  df_new['nodes'].iloc[0][0].astype(str) + df_new['activation_function'] # eerste rij, neem van die lijst eerste element
                df_new['nodes_per_layer'] = df_new.apply(concatenate_values, axis=1)
                df_together = pd.concat([df_together, df_new], ignore_index=True)
        
        hue_colors = {'4relu': 'royalblue', '4sigmoid': 'lightskyblue', '8relu': 'orange', '8sigmoid': 'moccasin','16relu': 'seagreen', '16sigmoid': 'palegreen'}
        hue_order = ['4relu', '4sigmoid', '8relu', '8sigmoid', '16relu', '16sigmoid']
        x_order = [2, 4, 8, 16]
        
        ax = sns.boxplot(data=df_together, x="layers", y="MSE", hue="nodes_per_layer", order=x_order, palette = hue_colors, hue_order=hue_order, log_scale=True)
        sns.stripplot(data=df_together, x="layers", y="MSE", hue="nodes_per_layer", order=x_order, dodge=True, palette=hue_colors, size=4, hue_order=hue_order)
        
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[6:12], labels[6:12], bbox_to_anchor=(1.0, 1.0))

        plt.subplots_adjust(right=0.86)
    
    plt.yscale('log')
    plt.title('Mean Squared Error Distribution for lr 0.01, 500 epochs')
    plt.show()

def big_MSE_vs_VAL_for_one_norm_two_act(option, learning_rate, max_epochs, amount, normalization_methods, activation_functions, plot_option):
    """
    Performs detailed analysis and plots Validation vs MSE for one normalization method and two activation functions.

    This function generates plots to visualize the relationship between Validation and Mean Squared Error (MSE) for different configurations of nodes and layers, considering one normalization method and two activation functions.

    Args:
        option (str): The option used for the simulation.
        learning_rate (float): The learning rate used for the simulation.
        max_epochs (int): Maximum number of epochs used for the simulation.
        amount (int): The total number of simulations.
        normalization_methods (list): List of normalization methods to be analyzed.
        activation_functions (list): List of activation functions to be analyzed.
        plot_option (str): Plotting option, '0' for default coloring.

    Returns:
        None

    Note:
        This function relies on previously saved data and requires the 'save_one_norm_two_act_MSE_vs_VAL' function to be executed first.
        Difference with 'plot_validation_vs_mse_one_norm_two_act' is that this plots everything together on one plot
    """

    df_together = pd.DataFrame()
    
    all_nodes_list = [[[4]*2, [8]*2, [16]*2], [[4]*4, [8]*4, [16]*4], [[4]*8, [8]*8, [16]*8], [[8]*16, [16]*16]]
    layers_list = [2, 4, 8, 16]

    if plot_option == '0':
        for nodes_list, layers in zip(all_nodes_list, layers_list):
            for nodes in nodes_list:
                print('at nodes, layers', nodes, layers)
                df_plot = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount, normalization_methods, activation_functions)
                # select the interesting rows (so everything except option, lr, max_epochs, nodes, layers, amount, normalization_method, activation_functions) Maak van de nodes ipv [X,X,X] gewoon X
                df_new = df_plot.copy()
                df_together = pd.concat([df_together, df_new], ignore_index=True)

    plot_title = 'Validation vs. MSE'
    plot_seaborn_validation_mse_one_norm_two_act(df_together, plot_title)

# (one norm, one activation)
def Val_vs_MSE_node_norm_act_plot(learning_rate = 0.005, nodes = [8,8], layers=2, max_epochs=999, option='option_3', amount=40, normalization_method='min-max', activation_function='relu', remove_biggest_validation_outlier=False, search_activation_functions=['relu', 'sigmoid']):
    """
    Plots a Validation vs MSE scatter plot for specific node configuration, normalization method, and activation function, using already saved data.

    This function generates a scatter plot representing the relationship between validation and mean squared error (MSE)
    for a specific node configuration, normalization method, and activation function. The data is retrieved from the
    saved DataFrame obtained through 'open_csv_and_return_all' function.

    Args:
        learning_rate (float, optional): The learning rate used for the simulation. Defaults to 0.005.
        nodes (list, optional): List of node configurations for the neural network. Defaults to [8,8].
        layers (int, optional): Number of layers in the neural network. Defaults to 2.
        max_epochs (int, optional): Maximum number of epochs used for the simulation. Defaults to 999.
        option (str, optional): The option used for the simulation. Defaults to 'option_3'.
        amount (int, optional): The total number of simulations. Defaults to 40.
        normalization_method (str, optional): The normalization method to be plotted. Defaults to 'min-max'.
        activation_function (str, optional): The activation function to be plotted. Defaults to 'relu'.
        remove_biggest_validation_outlier (bool, optional): Whether to remove the data point with the highest validation value. Defaults to False.

    Returns:
        None

    Note:
        This function is used on itself but requires the data to be saved first using 'save_one_norm_two_act_MSE_vs_VAL'.
    """
    df_plot = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount, normalization_methods=['min-max'], activation_functions=search_activation_functions)

    df_selection = df_plot[(df_plot['normalization_method'] == normalization_method) & (df_plot['activation_function'] == activation_function)].copy()

    if remove_biggest_validation_outlier:
        df_selection = df_selection.loc[df_selection['validation'] != df_selection['validation'].max()]

    df_selection.loc[:, 'log_MSE'] = np.log10(df_selection['MSE'])
    df_selection.loc[:, 'log_validation'] = np.log10(df_selection['validation'])

    pearson_corr_coefficient = round(pearson_correlation(df_selection['log_validation'], df_selection['log_MSE']),4)
    plot_title = f" Validation vs MSE,\n lr {learning_rate}, nodes {nodes}, layers {layers}, {option}, {max_epochs} max epochs, Amount: {amount}\n {normalization_method}, {activation_function}, PCC:{pearson_corr_coefficient}"

    sns.scatterplot(data = df_selection, x='validation', y='MSE')

    print("limits can be customized")
    # plt.xlim(10**(-5), 10**(-3))  
    # plt.ylim(10**(-3), 10**0)
    
    plt.xlabel("Validation")
    plt.ylabel("Mean Squared Error")
    plt.xscale('log')
    plt.yscale('log')
    plt.title(plot_title, fontsize=10)

    plt.tight_layout()
    plt.show()

def specific_MSE_vs_VAL_for_one_norm_one_act(option, learning_rate, max_epochs, amount, normalization_methods, activation_functions, plot_option):
    """
    => plot_validation_vs_mse_one_norm_two_act

    Performs detailed analysis and plots Validation vs MSE for one normalization method and two activation functions.

    This function generates plots to visualize the relationship between Validation and Mean Squared Error (MSE) for different configurations of nodes and layers, considering one normalization method and two activation functions.

    Args:
        option (str): The option used for the simulation.
        learning_rate (float): The learning rate used for the simulation.
        max_epochs (int): Maximum number of epochs used for the simulation.
        amount (int): The total number of simulations.
        normalization_methods (list): List of normalization methods to be analyzed.
        activation_functions (list): List of activation functions to be analyzed.
        plot_option (str): Plotting option, '0' for default coloring.

    Returns:
        None

    Note:
        This function relies on previously saved data and requires the 'save_one_norm_two_act_MSE_vs_VAL' function to be executed first.
        Difference with 'plot_validation_vs_mse_one_norm_two_act' is that this plots everything together on one plot
    """

    df_together = pd.DataFrame()
    
    all_nodes_list = [[[4]*2, [8]*2, [16]*2]]
    layers_list = [2]

    if plot_option == '0':
        for nodes_list, layers in zip(all_nodes_list, layers_list):
            for nodes in nodes_list:
                print('at nodes, layers', nodes, layers)
                df_plot = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount, normalization_methods, activation_functions)
                # select the interesting rows (so everything except option, lr, max_epochs, nodes, layers, amount, normalization_method, activation_functions) Maak van de nodes ipv [X,X,X] gewoon X
                df_new = df_plot.copy()
                df_together = pd.concat([df_together, df_new], ignore_index=True)

    plot_title = 'Validation vs. MSE'
    plot_seaborn_validation_mse_one_norm_two_act(df_together, plot_title)

def specific_MSE_for_one_norm_two_activation(option, learning_rate, max_epochs, amount, normalization_methods, activation_functions, plot_option):
    """
    Performs detailed boxplot analysis and plots the distribution of Mean Squared Error (MSE) for one normalization method and two activation functions.
    Using same amount of layers (2) but different nodes, seperating the activation functions (relu-sigmoid)

    This function generates boxplots to visualize the distribution of MSE for different configurations of nodes and layers, considering one normalization method and two activation functions.

    Args:
        option (str): The option used for the simulation.
        learning_rate (float): The learning rate used for the simulation.
        max_epochs (int): Maximum number of epochs used for the simulation.
        amount (int): The total number of simulations.
        normalization_methods (list): List of normalization methods to be analyzed.
        activation_functions (list): List of activation functions to be analyzed.
        plot_option (str): Plotting option, '0' for default coloring, '1' for custom coloring.

    Returns:
        None

    Note:
        This function is used on itself, but it requires the data to be saved first using 'save_one_norm_two_act_MSE_vs_VAL'.
    """
    
    df_together = pd.DataFrame()
    all_nodes_list = [[[4]*2, [8]*2, [16]*2]]
    layers_list = [2]

    plt.subplots(figsize=(12,6))

    if plot_option == '0':
        for nodes_list, layers in zip(all_nodes_list, layers_list):
            for nodes in nodes_list:
                print('at nodes, layers', nodes, layers)
                df_plot = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount, normalization_methods, activation_functions)
                # select the interesting rows (so everything except option, lr, max_epochs, nodes, layers, amount, normalization_method, activation_functions) Maak van de nodes ipv [X,X,X] gewoon X
                df_new = df_plot.copy()
                df_new.loc[:, 'nodes_per_layer'] = df_new['nodes'].iloc[0][0] # from first row takes first element
                df_together = pd.concat([df_together, df_new], ignore_index=True)
    
        hue_order = [4, 8, 16]
        x_order = [2, 4, 8, 16]
        print("PAS OP MET LOG_SCALE=TRUE, MSS?")
        sns.boxplot(data=df_together, x="layers", y="MSE", hue="nodes_per_layer", hue_order=hue_order, order=x_order, palette = 'pastel', log_scale='log')
        sns.stripplot(data=df_together, x="layers", y="MSE", hue="nodes_per_layer", hue_order=hue_order, order=x_order, dodge=True, palette='tab10')

    
    if plot_option == '1':
        for nodes_list, layers in zip(all_nodes_list, layers_list):
            for nodes in nodes_list:
                print('at nodes, layers', nodes, layers)
                df_plot = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount, normalization_methods, activation_functions)
                # select the interesting rows (so everything except option, lr, max_epochs, nodes, layers, amount, normalization_method, activation_functions) Maak van de nodes ipv [X,X,X] gewoon X
                df_new = df_plot.copy()
                # df_new['nodes_per_layer'] =  df_new['nodes'].iloc[0][0].astype(str) + df_new['activation_function'] # eerste rij, neem van die lijst eerste element
                df_new['nodes_per_layer'] = df_new.apply(concatenate_values, axis=1)
                df_together = pd.concat([df_together, df_new], ignore_index=True)
        
        hue_colors = {'4relu': 'royalblue', '4sigmoid': 'lightskyblue', '8relu': 'orange', '8sigmoid': 'moccasin','16relu': 'seagreen', '16sigmoid': 'palegreen'}
        hue_order = ['4relu', '4sigmoid', '8relu', '8sigmoid', '16relu', '16sigmoid']
        x_order = layers_list
        ax = sns.boxplot(data=df_together, x="layers", y="MSE", hue="nodes_per_layer", order=x_order, palette = hue_colors, hue_order=hue_order, log_scale=True)
        sns.stripplot(data=df_together, x="layers", y="MSE", hue="nodes_per_layer", order=x_order, dodge=True, palette=hue_colors, size=4, hue_order=hue_order)
        
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[6:12], labels[6:12], bbox_to_anchor=(1.0, 1.0))

        plt.subplots_adjust(right=0.86)


        print("Median is 4relu, 8relu, 16relu", [df_together[df_together["nodes_per_layer"]=="4relu"]["MSE"].median(),
                                               df_together[df_together["nodes_per_layer"]=="8relu"]["MSE"].median(),
                                               df_together[df_together["nodes_per_layer"]=="16relu"]["MSE"].median()])

    
    plt.yscale('log')
    plt.ylim(0.000001,1)
    plt.title(f'MSE Distribution for lr {learning_rate}, 500 {max_epochs}\nTAU{TAU} {option} {normalization_methods} {activation_functions}')
    plt.show()



def plot_validation_vs_mse_one_norm_one_act_one_layer_three_nodes(learning_rate = 0.005, max_epochs=999, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid']):
    """
    # Plots a two by one grid of 'Validation vs MSE' using already saved data for a single normalization method and two activation functions.

    # This function generates a two by one grid of scatterplots, where each plot represents the relationship between
    # validation and mean squared error (MSE) for a specific activation function, using a single normalization method.
    # The data for each plot is retrieved from the saved DataFrame obtained through 'open_csv_and_return_all' function.

    Args:
        learning_rate (float, optional): The learning rate used for the simulation. Defaults to 0.005.
        nodes (list, optional): List of node configurations for the neural network. Defaults to [8,8].
        layers (int, optional): Number of layers in the neural network. Defaults to 2.
        max_epochs (int, optional): Maximum number of epochs used for the simulation. Defaults to 999.
        option (str, optional): The option used for the simulation. Defaults to 'option_3'.
        amount (int, optional): The total number of simulations. Defaults to 40.
        normalization_methods (list, optional): List of normalization methods. Defaults to ['min-max'].
        activation_functions (list, optional): List of activation functions. Defaults to ['relu', 'sigmoid'].

    Returns:
        None

    Note:
        This function is used on itself but requires the data to be saved first using 'save_one_norm_two_act_MSE_vs_VAL'.
    """

    min_mse = float('inf')
    max_mse = -float('inf')
    min_validation = float('inf')
    max_validation = -float('inf')

    layers = 2
    nodeslist = [[4,4], [8,8], [16,16]]
    fig, axs = plt.subplots(1, len(nodeslist), figsize=(10,5), squeeze=False)


    for i, nodes in enumerate(nodeslist):
        df_plot = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount, normalization_methods, activation_functions)
        df_selection = df_plot[(df_plot['normalization_method'] == normalization_methods[0]) & (df_plot['activation_function'] == activation_functions[0])].copy()

        min_mse = min(min_mse, min(df_selection['MSE']))
        max_mse = max(max_mse, max(df_selection['MSE']))
        min_validation = min(min_validation, min(df_selection['validation']))
        max_validation = max(max_validation, max(df_selection['validation']))

        scatterplot_setup_provider(df_selection, axs[0,i])

    for ax in axs.flat: # axs.flat: to iterate over axes
        # ax.set_xlim(min_validation, max_validation)  
        # ax.set_ylim(min_mse, max_mse)
        ax.set_xlim(0.000001, 1)        # average over TAU=1,7.5,20,100
        ax.set_ylim(0.000001, 1)

    fig.text(0.1, 0.01, '[4,4]', ha='left', fontsize=12, color='maroon')
    fig.text(0.45, 0.01, '[8,8]', ha='right', fontsize=12, color='maroon')
    fig.text(0.8, 0.01, '[16,16]', ha='right', fontsize=12, color='maroon')

    fig.text(0.985, 0.5, 'min-max', va='center', rotation='vertical', fontsize=12, color='maroon')

    plot_title = f" Validation vs MSE, Tau:{TAU} 'min-max' 'relu'\n lr {learning_rate}, layers {layers}, {option}, {max_epochs} max epochs, Amount: {amount}"
    fig.suptitle(plot_title)

    plt.tight_layout()
    plt.show()

def on_pick(event, df):
    ind = event.ind[0]  # Index of the selected point
    selected_point = df.iloc[ind]
    print(f"Modelname of point is {selected_point['modelname']}, {selected_point['normalization_method']}, {selected_point['activation_function']}, MSE {selected_point['MSE']}")


def plot_seaborn_validation_mse(df, plot_title):
    """
    Plot a seaborn scatterplot of validation versus MSE, with interactive modelname display on pick events.

    Parameters:
    - df (DataFrame): DataFrame containing validation and MSE values.
    - plot_title (str): Title of the plot.

    Returns:
    - None

    Notes:
    - This function is not intended for standalone use; it is utilized within 'search_modelname_of_point'.

    Example:
    >>> plot_seaborn_validation_mse(data_frame, f'Validation vs MSE,\n lr {learning_rate}, nodes {nodes}, layers {layers}, {option}, {max_epochs} max epochs, Amount: {total}')
    """
    # 
    fig, ax = plt.subplots()
    ax.scatter(df["validation"], df["MSE"], picker=True)
    plt.plot(df['validation'], df['validation'], label='Identity')

    plt.title(plot_title)
    plt.xlabel("Validation")
    plt.ylabel("Mean Squared Error")
    
    plt.xlim(df['validation'].min(), df['validation'].max())
    plt.ylim(df['MSE'].min(), df['MSE'].max())
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.legend(loc="upper right", bbox_to_anchor=(1.23,1))

    # Attach names to each point for retrieval
    # for i, txt in enumerate(df['modelname']):
    #     ax.annotate(txt, (df['validation'][i], df['MSE'][i]))

    fig.canvas.mpl_connect('pick_event', lambda event: on_pick(event, df))
    
    plt.show()


def search_modelname_of_point(option, learning_rate, max_epochs, nodes, layers, total, normalization_methods=None, activation_functions=None):
    """
    Plots the validation vs MSE plot for specific learning rate, epoch, and nodes where full analysis has been conducted.
    Clicking on the plot allows retrieval of the modelnames.

    The model should first be saved by functions in the 'Nullcline_MSE_plot.py' file.

    These model names can be utilized by 'plot_lc_from_modelname' to plot the nullclines in the phase space.

    Parameters:
    - option (str): Option used in the analysis.
    - learning_rate (float): Learning rate used.
    - max_epochs (int): Maximum number of epochs.
    - nodes (int): Number of nodes.
    - layers (int): Number of layers.
    - total (int): Total count.

    Returns:
    - None
    """
    
    if normalization_methods is None and activation_functions is None:
        save_name = f"VAL_VS_MSE_{option}_lr{learning_rate}_epochs{max_epochs}_total{total}_{nodes}_layers{layers}"
    else:
        save_name = f"VAL_VS_MSE_{option}_{normalization_methods}_{activation_functions}_lr{learning_rate}_epochs{max_epochs}_total{total}_{nodes}_layers{layers}"

    # Open CSV
    absolute_path = os.path.dirname(__file__)
    relative_path = f"VAL_vs_MSE_{TAU}"
    folder_path = os.path.join(absolute_path, relative_path)
    relative_path = f"{save_name}.csv"
    csv_name = os.path.join(folder_path, relative_path)
    df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}, engine='c')

    plot_title = f"Validation vs MSE,\n lr {learning_rate}, nodes {nodes}, layers {layers}, {option}, {max_epochs} max epochs, Amount: {total}"


    # Retrieve log_validation and log_MSE values (can only do pearson on normal distribution: after log)
    df['log_MSE'] = np.log10(df['MSE'])
    df['log_validation'] = np.log10(df['validation'])
    log_validation_values = df['log_validation'].values
    log_mse_values = df['log_MSE'].values

    plot_seaborn_validation_mse(df, plot_title)


def plot_val_vs_MSE_extract_modelname_tau(option, learning_rate, max_epochs, amount, normalization_methods, activation_functions):
    ''''extract the modelname of the whole tau in min-max sigmoid'''
    df_together = pd.DataFrame()
    
    all_nodes_list = [[[4]*2, [8]*2, [16]*2], [[4]*4, [8]*4, [16]*4], [[4]*8, [8]*8, [16]*8], [[8]*16, [16]*16]]
    layers_list = [2, 4, 8, 16]

    for nodes_list, layers in zip(all_nodes_list, layers_list):
        for nodes in nodes_list:
            print('at nodes, layers', nodes, layers)
            df_plot = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount, normalization_methods, activation_functions)
            # select the interesting rows (so everything except option, lr, max_epochs, nodes, layers, amount, normalization_method, activation_functions) Maak van de nodes ipv [X,X,X] gewoon X
            df_new = df_plot.copy()
            df_together = pd.concat([df_together, df_new], ignore_index=True)
    df_together['log_MSE'] = np.log10(df_together['MSE'])
    df_together['log_validation'] = np.log10(df_together['validation'])
    log_validation_values = df_together['log_validation'].values
    log_mse_values = df_together['log_MSE'].values

    plot_title='One Tau'
    plot_seaborn_validation_mse(df_together, plot_title)



def search_5_best_5_worst_modelnames(option, learning_rate, max_epochs, nodes, layers, normalization_method, activation_function):
    """
    Retrieves the names of the 5 best and 5 worst models based on validation performance.

    Parameters:
    - option (str): Option used in the analysis.
    - learning_rate (float): Learning rate used.
    - max_epochs (int): Maximum number of epochs.
    - nodes (int): Number of nodes.
    - layers (int): Number of layers.
    - normalization_method (str): Method used for normalization.
    - activation_function (str): Activation function used.

    Returns:
    - dict: A dictionary containing the names of the 5 best and 5 worst models under 'best models' and 'worst models' keys respectively.
    """

    df = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount=40, normalization_methods=[normalization_method], activation_functions=[activation_function])

    df_selection = df[(df['normalization_method'] == normalization_method) & (df['activation_function'] == activation_function)].copy()

    df_sorted = df_selection.sort_values(by=['validation'], ascending=True)

    cut_off = 5
    best_models = df_sorted.iloc[:5]['modelname'].tolist()
    print("Mse of 5 best models", df_sorted.iloc[:5]['MSE'].tolist())
    worst_models = df_sorted.tail(cut_off)['modelname'].tolist()

    return {'best models': best_models, 'worst models': worst_models}, df_sorted


def retrieve_model_from_name(unique_modelname) -> Model:
    """Give the modelname and returns the keras.Model
    
    Parameters:
    - unique_modelname (str): Modelname

    Returns:
    - None
    """
    absolute_path = os.path.dirname(__file__)
    relative_path = "saved_NN_models"
    folder_path = os.path.join(absolute_path, relative_path)
    full_path = os.path.join(folder_path, unique_modelname + '.h5')
    if not os.path.exists(full_path):
        assert False, f"The model with name {unique_modelname} cannot be found in path {full_path}"
    # legacy_optimizer = tf.keras.optimizers.legacy.Adam
    # loaded_model = tf.keras.models.load_model(full_path, custom_objects={'SGD': legacy_optimizer})

    # loaded_model = keras.saving.load_model("model.keras")


    # loaded_model = tf.saved_model.load(full_path)
    
    loaded_model = load_model(full_path)
    return loaded_model

def normalize_axis_values(axis_value, all_mean_std, option):
    """We have values of the x/or/y axis of the phase space and returns the normalized versions.
    
    This is needed because the neural network model only takes in normalized inputs.
    """
    if option == 'option_1': # nullcine is udot/wdot = 0
        # axis value in this case is the x-axis (v-axis)
        mean_std = all_mean_std["v_t_data_norm"]
        normalized_axis_values = normalization_with_mean_std(axis_value, mean_std)

        # nullcline of option 1, udot/wdot = 0, so we have to fill in zeros (but has to be normalized first for the model)
        mean_std = all_mean_std["u_dot_t_data_norm"]
        normalized_dot = normalization_with_mean_std(np.zeros(len(axis_value)), mean_std)

        # The mean std that will be used later for reversing the normalization
        reverse_norm_mean_std = all_mean_std["u_t_data_norm"]

    if option == 'option_2':
        # axis value in this case is the y-axis (w-axis / u-axis)
        mean_std = all_mean_std["u_t_data_norm"]
        normalized_axis_values = normalization_with_mean_std(axis_value, mean_std)

        # nullcine of option 2, udot/wdot = 0, so we have to fill in zeros (but has to be normalized first for the model)
        mean_std = all_mean_std["u_dot_t_data_norm"]
        normalized_dot = normalization_with_mean_std(np.zeros(len(axis_value)), mean_std)

        # The mean std that will be used later for reversing the normalization
        reverse_norm_mean_std = all_mean_std["v_t_data_norm"]

    if option == 'option_3':
        # axis value in this case is the x-axis (v-axis)
        mean_std = all_mean_std["v_t_data_norm"]
        normalized_axis_values = normalization_with_mean_std(axis_value, mean_std)

        # nullcine of option 3, vdot = 0, so we have to fill in zeros (but has to be normalized first for the model)
        mean_std = all_mean_std["v_dot_t_data_norm"]
        normalized_dot = normalization_with_mean_std(np.zeros(len(axis_value)), mean_std)

        # The mean std that will be used later for reversing the normalization
        reverse_norm_mean_std = all_mean_std["u_t_data_norm"]

    if option == 'option_4':
        # just give some result so program is generalizable, do not trust said values
        normalized_axis_values = axis_value
        normalized_dot = np.zeros(len(axis_value))

        reverse_norm_mean_std = [0,1]

    input_prediction = np.column_stack((normalized_axis_values, normalized_dot))

    return input_prediction, reverse_norm_mean_std

def plot_lc_from_modelname(modelname, title_extra='', plot_bool=True, df=None):
    """
    Plots the nullcline on the phase space.

    Parameters:
    - modelname (str): Name of the model.
    - title_extra (str): Additional information to add to the title, ex. 'low val, high MSE'.
    - plot_bool (bool): Boolean indicating whether to plot the nullcline or not.
    - df (DataFrame): DataFrame containing model information.

    Returns:
    - tuple: Tuple containing axis values for nullcline, prediction output, and DataFrame.

    Notes:
    - If df is not provided, the function reads the data from a default CSV file.
    """
    
    if df is None:
        start_time = time.time()

        absolute_path = os.path.dirname(__file__)
        relative_path = f"FHN_NN_loss_and_model_{TAU}_{NUM_POINTS}.csv"
        csv_name = os.path.join(absolute_path, relative_path)
        df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}, engine='c') # literal eval returns [2,2] as list not as str
        
        end_time = time.time()
        print('took seconds:', end_time - start_time)

    option = df[(df['modelname'] == modelname)]['option'].iloc[0]
    mean_std = df[(df['modelname'] == modelname)]['mean_std'].iloc[0]
    learning_rate = df[(df['modelname'] == modelname)]['learning_rate'].iloc[0]
    nodes = df[(df['modelname'] == modelname)]['nodes'].iloc[0]
    layers = df[(df['modelname'] == modelname)]['layers'].iloc[0]
    max_epochs = df[(df['modelname'] == modelname)]['epoch'].iloc[-1]
    normalization_method = df[(df['modelname'] == modelname)]['normalization_method'].iloc[0]
    activation_function = df[(df['modelname'] == modelname)]['activation_function'].iloc[0]

    print(option, mean_std)
    model = retrieve_model_from_name(modelname)

    # load data of nullclines in phasespace
    amount_of_points = 500
    axis_values_for_nullcline, exact_nullcline_values = nullcline_and_boundary(option, amount_of_points)

    # Predict normalized data 
    input_prediction, reverse_norm_mean_std = normalize_axis_values(axis_values_for_nullcline, mean_std, option)
    prediction_output_normalized = model.predict(input_prediction)
    # Reverse normalize to 'normal' data
    prediction_output_column = reverse_normalization(prediction_output_normalized, reverse_norm_mean_std)
    prediction_output = prediction_output_column.reshape(-1)
    
    if plot_bool:
        # plot normal LC
        x_lc, y_lc = limit_cycle()
        plt.plot(x_lc, y_lc, 'r-', label=f'Trajectory')
        # Plot Nullcines
        # vdot
        v = np.linspace(-2.5, 2.5, 1000)
        plt.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$\dot{v}=0$ nullcline") #$w=v - (1/3)*v**3 + R * I$"+r" ,
        # wdot
        v = np.linspace(-2.5, 2.5, 1000)
        plt.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$\dot{w}=0$ nullcline") #$w=(v + A) / B$"+r" ,

        if option == 'option_1' or option == 'option_3':
            plt.plot(axis_values_for_nullcline, prediction_output, label = 'prediction')
        if option == 'option_2' or option == 'option_4':
            plt.plot(prediction_output, axis_values_for_nullcline, label = 'prediction')
        plt.xlabel('v (voltage)')
        plt.ylabel('w (recovery variable)')
        plt.title(f"Phase Space: Limit Cycle and Cubic Nullcline with Prediction\n{option}, lr{learning_rate}, {nodes}, epoch {max_epochs}\n {normalization_method}, {activation_function}\n{title_extra}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return (axis_values_for_nullcline, prediction_output, df)


def average_lc_from_modelnames(modelnames:list, performance='',df=None, *args):
    """
    Computes the average prediction from a list of model names and plots it along with standard deviation.
    Only for option3
    
    Parameters:
    - modelnames (list): List of model names.
    - performance (str): Description of the performance.

    Returns:
    - None

    Notes:
    - This function is used by 'search_modelname_of_point' plot_best_worst_avg_param.

    """

    all_predictions = np.zeros((len(modelnames),), dtype=object)
    # df = None
    for i, modelname in enumerate(modelnames):
        (axis_value, all_predictions[i], df) =  plot_lc_from_modelname(modelname, title_extra='', plot_bool=False, df = df)
    
    mean_prediction = np.mean(all_predictions, axis=0)

    axis_values, nullcline_values = nullcline_and_boundary("option_3", len(mean_prediction))
    MSE_calculated = calculate_mean_squared_error(nullcline_values, mean_prediction)
    if args[-1] == 'no plot':
        return axis_value, mean_prediction, df
    std_dev_prediction = np.std(all_predictions, axis=0)

    plt.figure(figsize=(3, 2))

    if TAU == 7.5:
        plt.xlim(-2.05, 2.05)
        plt.ylim(-0.05, 2.2)
        # plt.ylim(-0.05, 2.5) # if legend

    if TAU == 100:
        plt.ylim(-0.05,2.4)
        plt.xlim(-2.3, 2.2)

    plt.plot(axis_value, mean_prediction, color='b', label='Mean', zorder=5, alpha=0.7)
    plt.fill_between(axis_value, mean_prediction-std_dev_prediction, mean_prediction+std_dev_prediction, color='grey', alpha=0.7, label="Std", zorder=0)

    # Now the plotting the limit cycle together with the (real) nullclines
    x_lc, y_lc = limit_cycle()
    plt.plot(x_lc, y_lc, 'r-', label=f'Trajectory')
    # Plot Nullcines
    # vdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$\dot{v}=0$") # r"$w=v - (1/3)*v**3 + R * I$"
    # wdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$\dot{w}=0$") # r"$w=(v + A) / B$"
    
    plt.xlabel(r'$v$ (voltage)')
    plt.ylabel(r'$w$ (recovery variable)')
    print(f'Phase Space: Limit Cycle and Cubic Nullcline with Prediction\nAverage of 5 {performance}\n{args}\nMSE of mean: {"{:.2e}".format(MSE_calculated)}')
    # plt.title(f'Phase Space: Limit Cycle and Cubic Nullcline with Prediction\nAverage of 5 {performance}\n{args}\nMSE of mean: {"{:.2e}".format(MSE_calculated)}')

    # plt.title(f"Phase Space: Limit Cycle and Cubic Nullcline with Prediction\n{option}, lr{learning_rate}, {nodes}, epoch {max_epochs}\n {normalization_method}, {activation_function}\n{title_extra}")
    # plt.grid(True)
    print("Legend can be added still")
    # plt.legend(frameon=False, loc='upper left', ncols=2, labelspacing=0.2, columnspacing=0.5,bbox_to_anchor=[-0.02, 1.04], handlelength=1.4)
    plt.tight_layout()

    mpl.rc("savefig", dpi=300)

    current_time = datetime.now()
    filename_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\PCCs\{args}, {performance}, {"{:.2e}".format(MSE_calculated)}_{TAU}_{filename_time}.png')

    # plt.show()
    plt.subplots(figsize=(3,1))

    # plt.plot(axis_value, mean_prediction, color='b', label='Mean')
    plt.plot(axis_value, np.array(mean_prediction)-np.array(nullcline_vdot(axis_value)), '--', color = "gray") # r"$w=v - (1/3)*v**3 + R * I$"
    plt.fill_between(axis_value, mean_prediction-std_dev_prediction-np.array(nullcline_vdot(axis_value)), mean_prediction+std_dev_prediction-np.array(nullcline_vdot(axis_value)), color='grey', alpha=0.4, label="Std")

    ymin, ymax = -0.3, 0.3
    yticks = np.linspace(ymin, ymax, 7)
    plt.ylim(ymin, ymax)
    plt.xticks()
    # plt.yticks(yticks)
    # print(yticks)
    plt.axhline(0, color='black',linewidth=0.5, zorder=0)

    plt.ylabel("Error")

    plt.tight_layout()
    plt.subplots_adjust(top=0.939,
bottom=0.228,
left=0.234,
right=0.945,
hspace=0.2,
wspace=0.2)


    mpl.rc("savefig", dpi=300)
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\PCCs\ERROR_{args}, {performance}, {"{:.2e}".format(MSE_calculated)}_{TAU}_{filename_time}.png')

    # plt.show()


    # seperately
    print('Possibility to plot each of the 5 separately here >>>')
    '''
    plt.figure(figsize=(3, 3))
    for i,v in enumerate(modelnames):
        plt.plot(axis_value, all_predictions[i], color='b', alpha=0.5)
        # np.savetxt('othernullcline_x', axis_value)
        # np.savetxt('othernullcline_y', all_predictions[i])

    # Now the plotting the limit cycle together with the (real) nullclines
    x_lc, y_lc = limit_cycle()
    plt.plot(x_lc, y_lc, 'r-', label=f'LC')
    plt.plot(axis_value, mean_prediction, color='b', label='Mean', zorder=5, alpha=0.7)

    # Plot Nullcines
    # vdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$w=v - (1/3)*v**3 + R * I$"+r" ,$\dot{v}=0$ nullcline")
    # wdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$w=(v + A) / B$"+r" ,$\dot{w}=0$ nullcline")
    
    plt.xlim(-2.1, 2.01)
    plt.ylim(0.25,2.04)

    plt.xlabel('v (voltage)')
    plt.ylabel('w (recovery variable)')
    # plt.title(f'Phase Space: Limit Cycle and Cubic Nullcline with Prediction\nAverage of 5 {performance}\n{args}\nMSE of mean: {"{:.2e}".format(MSE_calculated)}')
    print(f'Phase Space: Limit Cycle and Cubic Nullcline with Prediction\nAverage of 5 {performance}\n{args}\nMSE of mean: {"{:.2e}".format(MSE_calculated)}')
    # plt.title(f"Phase Space: Limit Cycle and Cubic Nullcline with Prediction\n{option}, lr{learning_rate}, {nodes}, epoch {max_epochs}\n {normalization_method}, {activation_function}\n{title_extra}")
    plt.grid(True)
    # plt.legend()
    plt.show()
    '''
    return axis_value, mean_prediction, df



def plot_best_worst_avg_param(option, learning_rate, max_epochs, nodes, layers, normalization_method, activation_function, df=None):
    """
    Plots the phase space with averaged predictions showing deviation from the best and worst performing models.

    Parameters:
    - option (str): Option used in the analysis.
    - learning_rate (float): Learning rate used.
    - max_epochs (int): Maximum number of epochs.
    - nodes (int): Number of nodes.
    - layers (int): Number of layers.
    - normalization_method (str): Method used for normalization.
    - activation_function (str): Activation function used.

    Returns:
    - None

    Notes:
    - Averages predictions with deviation from the best and worst performing models using 'search_5_best_5_worst_modelnames'.
    """

    best_worst_modelnames, _ = search_5_best_5_worst_modelnames(option, learning_rate, max_epochs, nodes, layers, normalization_method, activation_function)

    for performance_modelname, modelnames in best_worst_modelnames.items():
        _, _, df= average_lc_from_modelnames(modelnames, performance_modelname, df, option, nodes, learning_rate, max_epochs, normalization_method, activation_function)
        print("Break here stops showing worst guess")
        break
    return df

def plot_best_avg_param(option, learning_rate, max_epochs, nodes, layers, normalization_method, activation_function, df=None):
    """
    Plots the phase space with averaged predictions showing deviation from ONLY the 5 best performing models.

    Parameters:
    - option (str): Option used in the analysis.
    - learning_rate (float): Learning rate used.
    - max_epochs (int): Maximum number of epochs.
    - nodes (int): Number of nodes.
    - layers (int): Number of layers.
    - normalization_method (str): Method used for normalization.
    - activation_function (str): Activation function used.

    Returns:
    - None

    Notes:
    - Averages predictions with deviation from the best and worst performing models using 'search_5_best_5_worst_modelnames'.
    """

    best_worst_modelnames_dict, _ = search_5_best_5_worst_modelnames(option, learning_rate, max_epochs, nodes, layers, normalization_method, activation_function)
    best_modelnames = best_worst_modelnames_dict['best models']
    performance_modelname = 'best' # for in the title, this function plots the best modelnames
    axis_value, mean_prediction, _ = average_lc_from_modelnames(best_modelnames, performance_modelname, df ,option, nodes, learning_rate, max_epochs, normalization_method, activation_function)
    return axis_value, mean_prediction

def fitting_hyperparam1_to_avg_hyperparam2(option, learning_rate, max_epochs, nodes, layers, normalization_method, activation_function_1, activation_function_2, df=None):
    """Take the 5 best (lowest validation error) networks with hyperpameter2, average the nullcline prediction
    and search for the network that best fits to this average. Calculate its MSE as well. 

    Only difference in hyperparameter allowed right now is the activation function. 
    
    >>> Example:
    Using Benchmark ReLU for average and applying Sigmoid for more detail
    """
    axis_value , mean_prediction = plot_best_avg_param(option, learning_rate, max_epochs, nodes, layers, normalization_method, activation_function_2, df)
    # select from df the hyperparam2
    df = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount=40, normalization_methods=[normalization_method], activation_functions=['relu', 'sigmoid']) # made it work, save seperately otherwise
    df_selection = df[(df['normalization_method'] == normalization_method) & (df['activation_function'] == activation_function_1)].copy()
    df_sorted = df_selection.sort_values(by=['validation'], ascending=True).copy().reset_index(drop=True)
    df_sorted_new = df_sorted.drop("mean_std", axis='columns')
    df_sorted_new = df_sorted_new.drop("option", axis='columns')
    df_sorted_new = df_sorted_new.drop("loss", axis='columns')
    df_sorted_new = df_sorted_new.drop("normalization_method", axis='columns')

    df_sorted_mse = df_selection.sort_values(by=['MSE'], ascending=True).copy().reset_index(drop=True)
    print('dfsortedval', df_sorted_new)
    print("\n, dfsortedmse", df_sorted_mse)
    modelnames_hyperparam1 = df_sorted['modelname'].tolist()

    MSE_best = math.inf
    best_model_hyperparam1 = 'None'
    best_validation_index = None
    best_mse_index = None
    predictions_hyperparam1 = None

    print('fitting started')
    for i, modelname in enumerate(modelnames_hyperparam1):
        model = retrieve_model_from_name(modelname)

        # load data of nullclines in phasespace
        amount_of_points = 500
        axis_values_for_nullcline, exact_nullcline_values = nullcline_and_boundary(option, amount_of_points)

        # Predict normalized data 
        mean_std = df_selection[(df_selection['modelname'] == modelname)]['mean_std'].iloc[0]
        input_prediction, reverse_norm_mean_std = normalize_axis_values(axis_values_for_nullcline, mean_std, option)
        prediction_output_normalized = model.predict(input_prediction)
        # Reverse normalize to 'normal' data
        prediction_output_column = reverse_normalization(prediction_output_normalized, reverse_norm_mean_std)
        prediction_output = prediction_output_column.reshape(-1)

        MSE_calculated = calculate_mean_squared_error(mean_prediction, prediction_output)
        if MSE_calculated < MSE_best:
            MSE_best = MSE_calculated
            best_model_hyperparam1 = modelname
            prediction_hyperparam1 = prediction_output
            # best_validation_index = df_sorted[df_sorted['modelname'] == modelname].index[0]  # Get the index of the best model
            # best_mse_index = df_sorted_mse[df_sorted_mse['modelname'] == modelname].index[0]
            best_validation_index = df_sorted.index[df_sorted['modelname'] == modelname]# Get the index of the best model
            best_mse_index = df_sorted_mse.index[df_sorted_mse['modelname'] == modelname]


    print(f'Best model:{best_model_hyperparam1}, with MSE compared between the two predictions:{"{:.2e}".format(MSE_best)} at val_index: {best_validation_index}/39 (starting at 0), and mse_index: {best_mse_index}')

    # plot results:
    print("MSE vs vdot nullcline")
    nullcline_val = nullcline_vdot(axis_value)
    mse_mean = calculate_mean_squared_error(nullcline_val, mean_prediction)
    plt.plot(axis_value, mean_prediction, color='b', label=f'mean prediction {activation_function_2}')
    print(f'{activation_function_2}: mean fit on nullcline has mse: {"{:.2e}".format(mse_mean)}')

    mse_fit_on_mean = calculate_mean_squared_error(nullcline_val, prediction_hyperparam1)
    plt.plot(axis_value, prediction_hyperparam1, color='C1', label=f'prediction {activation_function_1}')
    print(f'{activation_function_1}: fit on mean prediction on nullcline has mse: {"{:.2e}".format(mse_fit_on_mean)}')
    # Now the plotting the limit cycle together with the (real) nullclines
    x_lc, y_lc = limit_cycle()
    plt.plot(x_lc, y_lc, 'r-', label=f'LC = {0}')

    # Plot Nullcines
    # vdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$w=v - (1/3)*v**3 + R * I$"+r" ,$\dot{v}=0$ nullcline")
    # wdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$w=(v + A) / B$"+r" ,$\dot{w}=0$ nullcline")

    plt.xlabel('v (voltage)')
    plt.ylabel('w (recovery variable)')
    plt.title(f'Phase Space: Limit Cycle and Cubic Nullcline with ReLU mean and Sigmoid fit')
    # plt.title(f"Phase Space: Limit Cycle and Cubic Nullcline with Prediction\n{option}, lr{learning_rate}, {nodes}, epoch {max_epochs}\n {normalization_method}, {activation_function}\n{title_extra}")
    plt.grid(True)
    plt.legend()
    plt.show()
    # error through time

    plt.plot(axis_value, np.abs(mean_prediction-nullcline_val), color='b', label=f'mean prediction error {activation_function_2}')

    plt.plot(axis_value, np.abs(prediction_hyperparam1-nullcline_val), color='C1', label=f'prediction {activation_function_1} error')
    plt.title("The absolute error of the nullcline.")


def plot_lc_from_modelname_thesis(modelname, title_extra='', plot_bool=True, df=None):
    """
    Plots the nullcline on the phase space.

    Parameters:
    - modelname (str): Name of the model.
    - title_extra (str): Additional information to add to the title, ex. 'low val, high MSE'.
    - plot_bool (bool): Boolean indicating whether to plot the nullcline or not.
    - df (DataFrame): DataFrame containing model information.

    Returns:
    - tuple: Tuple containing axis values for nullcline, prediction output, and DataFrame.

    Notes:
    - If df is not provided, the function reads the data from a default CSV file.
    """
    
    if df is None:
        start_time = time.time()
        absolute_path = os.path.abspath('')
        # absolute_path = os.path.dirname(__file__)
        relative_path = f"FHN_NN_loss_and_model_{TAU}.csv"
        csv_name = os.path.join(absolute_path, relative_path)
        df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}, engine='c') # literal eval returns [2,2] as list not as str
        
        end_time = time.time()
        print('took seconds:', end_time - start_time)

    option = df[(df['modelname'] == modelname)]['option'].iloc[0]
    mean_std = df[(df['modelname'] == modelname)]['mean_std'].iloc[0]
    learning_rate = df[(df['modelname'] == modelname)]['learning_rate'].iloc[0]
    nodes = df[(df['modelname'] == modelname)]['nodes'].iloc[0]
    layers = df[(df['modelname'] == modelname)]['layers'].iloc[0]
    max_epochs = df[(df['modelname'] == modelname)]['epoch'].iloc[-1]
    normalization_method = df[(df['modelname'] == modelname)]['normalization_method'].iloc[0]
    activation_function = df[(df['modelname'] == modelname)]['activation_function'].iloc[0]

    print(option, mean_std)
    model = retrieve_model_from_name(modelname)

    # load data of nullclines in phasespace
    amount_of_points = 500
    axis_values_for_nullcline, exact_nullcline_values = nullcline_and_boundary(option, amount_of_points)

    # Predict normalized data 
    input_prediction, reverse_norm_mean_std = normalize_axis_values(axis_values_for_nullcline, mean_std, option)
    prediction_output_normalized = model.predict(input_prediction)
    # Reverse normalize to 'normal' data
    prediction_output_column = reverse_normalization(prediction_output_normalized, reverse_norm_mean_std)
    prediction_output = prediction_output_column.reshape(-1)
    
    plt.subplots(figsize=(3, 3))

    if plot_bool:
        # plot normal LC
        x_lc, y_lc = limit_cycle(tau=100)
        # plt.plot(x_lc, y_lc, 'r-', label=f'Trajectory')
        plt.scatter(x_lc, y_lc, color='red', label=f'Trajectory', alpha=0.01, s=2)
        plt.plot([100], [200], color='red', label=f'Limit cycle', zorder=10)


        # Plot Nullcines
        # vdot
        v = np.linspace(-2.5, 2.5, 1000)
        plt.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$\dot{v}=0$ nullcline") #$w=v - (1/3)*v**3 + R * I$"+r" ,
        # wdot
        v = np.linspace(-2.5, 2.5, 1000)
        plt.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$\dot{w}=0$ nullcline") #$w=(v + A) / B$"+r" ,

        if option == 'option_1' or option == 'option_3':
            plt.plot(axis_values_for_nullcline, prediction_output, label = 'prediction', linewidth=2)
        if option == 'option_2' or option == 'option_4':
            plt.plot(prediction_output, axis_values_for_nullcline, label = 'prediction', linewidth=2)
        plt.xlabel(r'$v$ (voltage)')
        plt.ylabel(r'$w$ (recovery variable)')
        # plt.title(f"Phase Space: Limit Cycle and Cubic Nullcline with Prediction\n{option}, lr{learning_rate}, {nodes}, epoch {max_epochs}\n {normalization_method}, {activation_function}\n{title_extra}")
        print(f"Phase Space: Limit Cycle and Cubic Nullcline with Prediction\n{option}, lr{learning_rate}, {nodes}, epoch {max_epochs}\n {normalization_method}, {activation_function}\n{title_extra}")

        xmin = -2.2
        xmax = 2.2
        ymin = -0.2
        ymax = 2.2
        plt.ylim(ymin, ymax)
        plt.xlim(xmin, xmax)

        plt.grid(True)
        plt.tight_layout()

        mpl.rc("savefig", dpi=300)
        plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\Predictions\Horizontal_Predict_100.png")

        

        plt.show()
    return (axis_values_for_nullcline, prediction_output, df)

def proof_sigmoid_smoothly_approximates_piecewise_relu(df=None):
    # (option, learning_rate, max_epochs, nodes, layers, normalization_method, activation_function_1, activation_function_2, df=None):
    """ZOOM IN ON FUNCTION 'fitting_hyperparam1_avg_to_hyperparam2' to proof dat ReLU is piecewise, and Sigmoid makes it smooth
    Function will   1) Zoom in
                    2) Calculate derivative and show smoothness of derivative of Sigmoid.


    >>> Example:
    Using Benchmark ReLU for average and applying Sigmoid for more detail
    """
    option='option_3'
    learning_rate=0.01
    max_epochs=499
    nodes=[8,8]
    layers=2
    normalization_method='min-max'
    activation_function_1='sigmoid'
    activation_function_2='relu'
    assert TAU == 100, "Tau must equal 100 for the example required for this part"
    if df is None:
        assert False, 'please load df with pickle'



    axis_value , mean_prediction = plot_best_avg_param(option, learning_rate, max_epochs, nodes, layers, normalization_method, activation_function_2, df)
    # select from df the hyperparam2
    df = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount=40, normalization_methods=[normalization_method], activation_functions=['relu', 'sigmoid']) # made it work, save seperately otherwise
    df_selection = df[(df['normalization_method'] == normalization_method) & (df['activation_function'] == activation_function_1)].copy()
    df_sorted = df_selection.sort_values(by=['validation'], ascending=True).copy().reset_index(drop=True)
    df_sorted_new = df_sorted.drop("mean_std", axis='columns')
    df_sorted_new = df_sorted_new.drop("option", axis='columns')
    df_sorted_new = df_sorted_new.drop("loss", axis='columns')
    df_sorted_new = df_sorted_new.drop("normalization_method", axis='columns')

    df_sorted_mse = df_selection.sort_values(by=['MSE'], ascending=True).copy().reset_index(drop=True)
    print('dfsortedval', df_sorted_new)
    print("\n, dfsortedmse", df_sorted_mse)
    modelnames_hyperparam1 = df_sorted['modelname'].tolist()

    MSE_best = math.inf
    best_model_hyperparam1 = 'None'
    best_validation_index = None
    best_mse_index = None
    predictions_hyperparam1 = None

    print('fitting started')
    for i, modelname in enumerate(modelnames_hyperparam1):
        model = retrieve_model_from_name(modelname)

        # load data of nullclines in phasespace
        amount_of_points = 500
        axis_values_for_nullcline, exact_nullcline_values = nullcline_and_boundary(option, amount_of_points)

        # Predict normalized data 
        mean_std = df_selection[(df_selection['modelname'] == modelname)]['mean_std'].iloc[0]
        input_prediction, reverse_norm_mean_std = normalize_axis_values(axis_values_for_nullcline, mean_std, option)
        prediction_output_normalized = model.predict(input_prediction)
        # Reverse normalize to 'normal' data
        prediction_output_column = reverse_normalization(prediction_output_normalized, reverse_norm_mean_std)
        prediction_output = prediction_output_column.reshape(-1)

        MSE_calculated = calculate_mean_squared_error(mean_prediction, prediction_output)
        if MSE_calculated < MSE_best:
            MSE_best = MSE_calculated
            best_model_hyperparam1 = modelname
            prediction_hyperparam1 = prediction_output
            # best_validation_index = df_sorted[df_sorted['modelname'] == modelname].index[0]  # Get the index of the best model
            # best_mse_index = df_sorted_mse[df_sorted_mse['modelname'] == modelname].index[0]
            best_validation_index = df_sorted.index[df_sorted['modelname'] == modelname]# Get the index of the best model
            best_mse_index = df_sorted_mse.index[df_sorted_mse['modelname'] == modelname]


    print(f'Best model:{best_model_hyperparam1}, with MSE compared between the two predictions:{"{:.2e}".format(MSE_best)} at val_index: {best_validation_index}/39 (starting at 0), and mse_index: {best_mse_index}')

    # plot results:
    print("MSE vs vdot nullcline")
    nullcline_val = nullcline_vdot(axis_value)
    mse_mean = calculate_mean_squared_error(nullcline_val, mean_prediction)
    plt.plot(axis_value, mean_prediction, color='b', label=f'mean prediction {activation_function_2}')
    print(f'{activation_function_2}: mean fit on nullcline has mse: {"{:.2e}".format(mse_mean)}')

    mse_fit_on_mean = calculate_mean_squared_error(nullcline_val, prediction_hyperparam1)
    plt.plot(axis_value, prediction_hyperparam1, color='C1', label=f'prediction {activation_function_1}')
    print(f'{activation_function_1}: fit on mean prediction on nullcline has mse: {"{:.2e}".format(mse_fit_on_mean)}')
    # Now the plotting the limit cycle together with the (real) nullclines
    x_lc, y_lc = limit_cycle()
    plt.plot(x_lc, y_lc, 'r-', label=f'LC = {0}')

    # Plot Nullcines
    # vdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$w=v - (1/3)*v**3 + R * I$"+r" ,$\dot{v}=0$ nullcline")
    # wdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$w=(v + A) / B$"+r" ,$\dot{w}=0$ nullcline")

    plt.xlabel('v (voltage)')
    plt.ylabel('w (recovery variable)')
    plt.title(f'Phase Space: Limit Cycle and Cubic Nullcline with ReLU mean and Sigmoid fit')
    # plt.title(f"Phase Space: Limit Cycle and Cubic Nullcline with Prediction\n{option}, lr{learning_rate}, {nodes}, epoch {max_epochs}\n {normalization_method}, {activation_function}\n{title_extra}")
    plt.grid(True)
    plt.legend()
    plt.show()
    # error through time

    plt.plot(axis_value, np.abs(mean_prediction-nullcline_val), color='b', label=f'mean prediction error {activation_function_2}')

    plt.plot(axis_value, np.abs(prediction_hyperparam1-nullcline_val), color='C1', label=f'prediction {activation_function_1} error')
    plt.title("The absolute error of the nullcline.")

    return None





if __name__ == "__main__":

    # num_points_list = [600, 800, 1400, 1800, 2200, 2600, 3000, 3400, 3800, 4200, 4600] # DONE
    num_points_list = [100, 200, 300, 400, 500, 700, 900, 1100, 1200, 1300, 1500, 1600, 1700, 1900, 2000, 2100, 2300, 2400, 2500, 2700, 2800, 2900, 3100, 3200, 3300, 3500, 3600, 3700, 3900, 4000, 4100, 4300, 4400, 4500, 4700, 4800, 4900, 6000, 7000, 8000, 9000, 11000, 12000, 13000, 14000]
    for NUM_POINTS in num_points_list:
        data = {}
        
        print("\n \n \n")
        print(f"AT NUMPOINTS={NUM_POINTS}")
        print("\n \n \n")

# 1     Make dataframe where data can be saved
        remake_dataframe(tau=TAU, num_of_points=NUM_POINTS)
# 2     Train neural networks on data
        start_total_time = time.time()
        for seed in range(0,40):
            print(f"\n ROUND NUMBER STARTING {seed} \n")

            start_time = time.time()
            create_neural_network_and_save(normalization_method='min-max', activation_function='relu', nodes_per_layer = [4,4], num_layers=2, learning_rate=0.01, epochs=500, option='option_3', seed=seed)
            # create_neural_network_and_save(normalization_method='min-max', activation_function='sigmoid', nodes_per_layer = [4,4], num_layers=2, learning_rate=0.01, epochs=500, option='option_3', seed=seed)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed time one seed", elapsed_time, "seconds \n\n")

        end_total_time = time.time()
        elapsed_total_time = end_total_time - start_total_time
        print("Elapsed time everything", elapsed_total_time, "seconds \n\n")
        data["elapsed time"] = elapsed_total_time
# 3     Evaluate validation error of training
        data["validation log10"] = plot_loss_and_validation_loss_param(normalization_method='min-max', activation_function='relu', learning_rate=0.01, nodes=[4,4], layers=2, max_epochs=499, option='option_3', average=40)
        plt.clf()

# 4     saving nullcline error and other data
        absolute_path = os.path.dirname(__file__)
        relative_path = f"VAL_vs_MSE_{TAU}_{NUM_POINTS}"
        # Combine the absolute path and relative path to get the full folder path
        folder_path = os.path.join(absolute_path, relative_path)
        # Check if the folder exists, and if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
# 5     calculating PCC
        save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[4,4], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu'], amount_per_parameter=40, save=True)
        data["PCC"] = plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[4,4], layers=2, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu'])
        plt.clf()

# 6     calculating MSE (nullcline error) 
        absolute_path = os.path.dirname(__file__)
        relative_path = f"FHN_NN_loss_and_model_{TAU}_{NUM_POINTS}.csv"
        csv_name = os.path.join(absolute_path, relative_path)
        df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}) # literal eval returns [2,2] as list not as str
        _, mean_prediction = plot_best_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[4,4], layers=2, normalization_method='min-max', activation_function='relu', df=df) #1.42e-3
        axis_values, nullcline_values = nullcline_and_boundary("option_3", len(mean_prediction))
        MSE_calculated = calculate_mean_squared_error(nullcline_values, mean_prediction)
        data["MSE mean relu"] = MSE_calculated
        plt.clf()
# 7     saving PCC, MSE, Validation and elapsed time
        script_dir = os.path.dirname(os.path.abspath(__file__))

        output_file_path = f"results_changing_amount_of_points_{NUM_POINTS}.json"
        file_path = os.path.join(script_dir, 'changing_data_size', output_file_path)

        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
