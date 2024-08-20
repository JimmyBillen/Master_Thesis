# FitzHugh-Nagumo (ps: phasespace)
# Goal to plot the phasespace w(v)
# Nullclines were calculated analytically (see notes)

import numpy as np
import matplotlib.pyplot as plt
# from keras.models import Model
from Time_Series_Excitability import compute_fitzhugh_nagumo_dynamics, find_boundary_nullclines

# Constants
R = 0
I = 0
TAU = 7.5 # /1/10/100
A = 1.717
B = 0.6
# https://www.desmos.com/calculator/wwmgeficva

# Formule is based on: (https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model), variable transformation: 
# CONSTANTS

# functions used for NN_y_FHN.py
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

def limit_cycle():
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

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()

    # own position 
    position = 0
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


def plot_limit_excite(u_nullcline=True, y_ifo_x=True, with_neural_network=False, mean_std=None, plot=True, ax=None):
    """ Plots the excite with the model
    
    This is the old version before optimisation, new version is 'plot_limit_cycle_with_model' """
    if plot==True:
        fig, ax = plt.subplots(figsize=(5, 3))
    
    # Plot Limit Cycle
    x_lc, y_lc = limit_cycle()
    line = ax.plot(x_lc, y_lc, 'r-', label=f'Trajectory', linewidth=2, zorder=3)[0]
    add_arrow(line)

    xmin = -2.5
    xmax = 1.8
    ymin = -1
    ymax = 2.2

    ax.axhline(0, color='black',linewidth=0.5, zorder=0)
    ax.axvline(0, color='black',linewidth=0.5, zorder=0)


    # Nullclines
        # vdot
    v = np.linspace(xmin, xmax, 1000)
    #     # plt.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$w=v - (1/3)*v**3 + R * I$"+r" ,$\dot{v}=0$ Nullcline")
    ax.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$\dot{x}=0$")

    #     # wdot
    v = np.linspace(xmin, xmax, 1000)
    #     # plt.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$w=(v + A) / B$"+r" ,$\dot{w}=0$ Nullcline")
    ax.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$\dot{y}=0$")

    #     # Plotting a dot where the nullclines intersect
    dots = [-1.725]
    dots_null = [(i + A) / B for i in dots]
    ax.scatter(dots, dots_null,color='black', label='Fixed Point', zorder=4)
    ax.scatter([1],[0], color='red', label='Start Point', zorder=5)

    ax.set_xlabel(r'$v$ (voltage)')
    ax.set_ylabel(r'$w$ (recovery variable)')
    ax.grid(True)
    # ax.legend(loc='upper right')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if plot:
        ax.set_title('Phase Space of FitzHugh-Nagumo Model:\n Limit Cycle and Nullclines')
        plt.tight_layout()
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



if __name__ == '__main__':
    # plot_heatmap_line_limit_cycle()
    plot_limit_excite()
    pass