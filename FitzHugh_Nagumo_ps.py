# FitzHugh-Nagumo (ps: phasespace)
# Goal to plot the phasespace w(v)
# Nullclines were calculated analytically (see notes)

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from FitzHugh_Nagumo_t import compute_fitzhugh_nagumo_dynamics, find_boundary_nullclines
from settings import R, I, A, B, TAU, NUM_OF_POINTS

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
        bound_nullcline = nullclines_per_option['option_4']
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

def calculate_derivatives(values, h):
    forward_deriv = forward_difference(values, h, begin=0, end = len(values)-1)
    backward_deriv = backward_difference(values, h, begin=len(values)-1, end=len(values))

    return forward_deriv + backward_deriv

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

def plot_limit_cycle_seed():
    time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics()
    u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
    v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))
    print("amount of points", len(u_t_data))
    seed=0
    train_u, val_u, train_v, val_v, train_u_dot, val_u_dot, train_v_dot, val_v_dot = split_train_validation_data_seed(u_t_data, v_t_data, u_dot_t_data, v_dot_t_data, validation_ratio=0.2, seed=seed)

    plt.scatter(train_v, train_u, s=4, alpha=0.4, color='blue', label='train')
    plt.scatter(val_v, val_u, s=4, alpha=1, color='orange', label='validation')

    plt.legend()
    plt.title(f'Phase Space of FitzHugh-Nagumo Model:\n Limit Cycle and Nullclines\n TAU {TAU}, #points {NUM_OF_POINTS}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_heatmap_line_limit_cycle()
    plot_limit_cycle()
    # plot_limit_cycle_seed()

    pass