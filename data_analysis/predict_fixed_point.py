# This script visualizes the predicted fixed points (cross section of nullclines) with the limit cycle for the FitzHugh-Nagumo model.
# The linear nullcline is always very precise for ReLU.
# For the cubic nullcline, the neural network is trained using ReLU activation function. Then cross section is calculated
# to find the fixed point. Five of its best* predictions are shown in grey. The average prediction of these five is shown in blue.
# The neural network trained with the Sigmoid activation function is shown in orange. The ground truth is shown in black.
# 
# * best indicating the five neural networks (out of forty) with given hyperparameters that have lowest validation error 
#
# Results are analyzed in Section 4.4 of the thesis
#  
# Main function to execute the script: plot_fixed_point_prediction()
# 
# To run this script directly, the main entry point is the plot_fixed_point_prediction() function,
# which is executed when the script is run as a standalone program.
# This is controlled by the following block at the end:
# 
# if __name__ == '__main__':
#     plot_fixed_point_prediction()
# 

import sys
sys.path.append('../../Master_Thesis') # needed to import settings

from ast import literal_eval
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

from data_generation_exploration.FitzHugh_Nagumo_ps import nullcline_and_boundary, nullcline_vdot, nullcline_wdot, limit_cycle
from Nullcline_MSE_plot import open_csv_and_return_all
from NN_model_analysis import retrieve_model_from_name, normalize_axis_values, reverse_normalization, search_5_best_5_worst_modelnames
from settings import TAU, NUM_OF_POINTS

# Newton-Raphson method
def newton_raphson_method(x, F, G):
    """Using numerical Newton Raphson method to find x-value of intersection of two functions."""
    F_interp = interp1d(x, F, kind='linear')
    G_interp = interp1d(x, G, kind='linear')

    # Initial guess for the intersection point
    x0 = 0

    # Max iterations before stopping:
    max_iter = 200

    # Tolerance for convergence:
    tolerance = 1e-6

    # Newton's method iteration:
    for i in range(max_iter):
        # call H the difference between F and G

        # Evaluate H(x0) and H'(x0)
        H_val = F_min_G(x0, F_interp, G_interp)
        H_prime_val = F_min_G_prime(x0, F_interp, G_interp)
        
        # Update x0 using Newton's method
        x1 = x0 - H_val / H_prime_val
        
        # Check for convergence
        if abs(x1 - x0) < tolerance:
            print('Succesfully Acquired Wanted Accuracy')
            break
        
        # Update x0 for the next iteration
        x0 = x1

    print('The fixed point is located at', x0, F_interp(x0))
    return x0, F_interp(x0)

def F_min_G(x, F_interpol, G_interpolate):
    """Help function for 'newton_raphson_method'."""
    return F_interpol(x) - G_interpolate(x)

def F_min_G_prime(x, F_interpol, G_interpol, h=1e-6):
    """Help function for 'newton_raphson_method'."""
    return (F_min_G(x+h, F_interpol, G_interpol) - F_min_G(x, F_interpol, G_interpol)) / h

# => Searching for fixed point <=

#  Extra Functions

def return_partial_nullcline_from_modelname(modelname, title_extra='', plot_bool=True, df=None) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """ Returns the nullcline on the phase space, but in option_1 region of axis value
    df is a dataframe with one row containing everything

    input:
    title_extra:
        Something extra in to put at the end in the title, like 'low val, high MSE'.
    """

    if df is None:
        absolute_path = os.path.dirname(__file__)
        relative_path = f"FHN_NN_loss_and_model_{TAU}.csv"
        csv_name = os.path.join(absolute_path, relative_path)
        df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}) # literal eval returns [2,2] as list not as str

    option = df[(df['modelname'] == modelname)]['option'].iloc[0]
    mean_std = df[(df['modelname'] == modelname)]['mean_std'].iloc[0]

    model = retrieve_model_from_name(modelname)

    # load data of nullclines in phasespace
    amount_of_points = 500
    axis_values_for_nullcline, exact_nullcline_values = nullcline_and_boundary('option_1', amount_of_points)

    # Predict normalized data 
    input_prediction, reverse_norm_mean_std = normalize_axis_values(axis_values_for_nullcline, mean_std, option)
    prediction_output_normalized = model.predict(input_prediction)
    # Reverse normalize to 'normal' data
    prediction_output_column = reverse_normalization(prediction_output_normalized, reverse_norm_mean_std)
    prediction_output = prediction_output_column.reshape(-1)
    
    return (axis_values_for_nullcline, prediction_output, df)

def select_5_best_validation(df, learning_rate, nodes, layers, max_epochs, option, amount):

    df = open_csv_and_return_all(option, learning_rate, max_epochs, nodes, layers, amount=20)

    df_sorted = df.sort_values(by=['validation'], ascending=True)

    cut_off = 5
    best_models = df_sorted.iloc[:5]['modelname'].tolist()

    return best_models

def save_mean_data(array, name):
    absolute_path = os.path.dirname(__file__)
    relative_path = "mean_data_predicted_nullcline"
    folder_path = os.path.join(absolute_path, relative_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    full_path = os.path.join(folder_path, name)
    np.save(full_path, array)

# Main Function:

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

def plot_lc_from_modelname(modelname, title_extra='', plot_bool=True, df=None):
    """
    CHANGED TO ONLY BE APPLICABLE FOR OPTION_1, SUCH THAT AXIS_VALUES OVERLAP

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
        relative_path = f"FHN_NN_loss_and_model_{TAU}_{NUM_OF_POINTS}.csv"
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
    axis_values_for_nullcline, exact_nullcline_values = nullcline_and_boundary('option_1', amount_of_points)

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
    Only for option1!! It cuts the domain to only be allowed for option 1
    
    Parameters:
    - modelnames (list): List of model names.
    - performance (str): Description of the performance.

    Returns:
    - None

    Notes:
    - This function is used by 'search_modelname_of_point', 'plot_best_worst_avg_param'.

    """

    all_predictions = np.zeros((len(modelnames),), dtype=object)
    # df = None
    for i, modelname in enumerate(modelnames):
        (axis_value, all_predictions[i], df) =  plot_lc_from_modelname(modelname, title_extra='', plot_bool=False, df = df)
    
    mean_prediction = np.mean(all_predictions, axis=0)

    axis_values, nullcline_values = nullcline_and_boundary("option_1", len(mean_prediction))
    MSE_calculated = calculate_mean_squared_error(nullcline_values, mean_prediction)
    if args[-1] == 'no plot':
        return axis_value, mean_prediction, df
    std_dev_prediction = np.std(all_predictions, axis=0)

    return axis_value, mean_prediction, df

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


def find_fixed_point_intersection_two_modelname(modelname_param1, modelname_param2, df):
    """Returns the fixed point of the predictions of two models, one for each nullcline.
    
    Args:
        modelname_param1: The modelname of parameter 1, corresponding to nullcline 1.
        modelname_param2: The modelname of parameter 2, corresponding the nullcline 2.
        df: Dataframe including all the training data.
    """

    (axis_value, predictions1, df) =  return_partial_nullcline_from_modelname(modelname_param1, title_extra='', plot_bool=False, df = df)

    (axis_value, predictions2, df) =  return_partial_nullcline_from_modelname(modelname_param2, title_extra='', plot_bool=False, df = df)

    xFP, yFP = newton_raphson_method(axis_value, predictions1, predictions2)

    if xFP < min(axis_value) or xFP > max(axis_value):
        assert False, f'value xFP does not satisfy min(axis value)<xFP<max(axis value): {min(axis_value)} < {xFP} < {max(axis_value)}.'
    return xFP, yFP, df

def all_fixed_point_from_best_models(best_5_modelnames_param1, best_5_modelnames_param2, df):
    """
    Finds the fixed points of the five models for each nullcline: so 25 points in total.

    Args:
        best_5_modelnames_param1: Modelnames of the five best (lowest validation error) networks of nullcline 1.
        best_5_modelnames_param2: Modelnames of the five best (lowest validation error) networks of nullcline 2.
        df: Dataframe all trained data, including the modelnames.
    """

    fixed_point_x_value = []
    fixed_point_y_value = []
    for modelname1 in best_5_modelnames_param1:
        for modelname2 in best_5_modelnames_param2:
            
            x, y_arr, df = find_fixed_point_intersection_two_modelname(modelname1, modelname2, df)
            y = float(y_arr)
            fixed_point_x_value.append(x)
            fixed_point_y_value.append(y)

    return fixed_point_x_value, fixed_point_y_value


def fixed_point_analysis_gaussian_fit(df=None):
    """
    Plot fixed points for linear and cubic nullcline using neural network parameters:
    linear nullcline: option='option_1', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    cubic nullcline: option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    Additionally, the function plots the limit cycle and nullclines for the system.

    For each nullcline the five predictions with lowest validation error are used. Also, the fixed point for the mean of
    these predictions is considered. Also the for the activation function Sigmoid, the fixed point is considered.

    Figure is plotted where the points are plotted over a gaussian distribution.

    Figure used in thesis/defense.
    """
    if df is None:
        absolute_path = os.path.dirname(__file__)
        relative_path = f"FHN_NN_loss_and_model_{TAU}_{NUM_OF_POINTS}.csv"
        csv_name = os.path.join(absolute_path, relative_path)
        df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}) # literal eval returns [2,2] as list not as str

    best_worst_modelnames_param1, _ = search_5_best_5_worst_modelnames(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    best_val_modelnames_param1 = best_worst_modelnames_param1['best models']

    best_worst_modelnames_param2, _ = search_5_best_5_worst_modelnames(option='option_1', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    best_val_modelnames_param2 = best_worst_modelnames_param2['best models']

    print("All Models Found, starting to search for fixed point\n")
    x_values_FP, y_values_FP = all_fixed_point_from_best_models(best_val_modelnames_param1, best_val_modelnames_param2, df)
    print(x_values_FP, y_values_FP)

    """ ALSO FOR THE RELU MEAN PREDICTION"""
    axis_value_mean, mean_prediction = plot_best_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu', df=df.copy())
    (axis_value, predictions2, df) =  return_partial_nullcline_from_modelname(best_val_modelnames_param2[1], title_extra='', plot_bool=False, df = df)
    # print("does it even equal?", axis_value_mean, axis_value)
    # print(len(axis_value_mean), len(axis_value_mean))
    # plt.plot(testing, axis_value_mean)
    # plt.plot(testing, axis_value)
    # plt.show()

    xFP_meanrelu, yFP_meanrelu = newton_raphson_method(axis_value, mean_prediction, predictions2)
    if xFP_meanrelu < min(axis_value) or xFP_meanrelu > max(axis_value):
        assert False, f'value xFP does not satisfy min(axis value)<xFP<max(axis value): {min(axis_value)} < {xFP_meanrelu} < {max(axis_value)}.'
    print("FP for meanrelu", xFP_meanrelu, yFP_meanrelu)

    x_values_sigmoid_FP, y_values_sigmoid_FP = all_fixed_point_from_best_models(['6bc4b512c04a4401b2eb80b1fb160461'], best_val_modelnames_param2, df)

    print("THE PREDICTED FIXED POINTS\n", x_values_FP, y_values_FP, x_values_sigmoid_FP, y_values_sigmoid_FP)
    # => values of print saved below
    # x = [(0.2579361321410763, 1.1974153677670007), (0.2578986349102627, 1.1973898568549572), (0.2579480258962664, 1.1974234595790076), (0.2578934446236969, 1.1973863256890236),
    #     (0.2578982574320299, 1.197389600040953), (0.06747968602243959, 0.9593495252593581), (0.06745663612048293, 0.959331612039192), (0.06748983242644874, 0.9593574105323303),
    #     (0.06743880905589467, 0.9593177577445627), (0.06744678671768216, 0.9593239575805654), (0.4960329062995323, 1.495030779896996), (0.49597090995841725, 1.4949890501079361),
    #     (0.4960500537322715, 1.4950423218483153), (0.495980637494374, 1.4949955977205251), (0.49598330096963045, 1.4949973905079479), (0.21544334146371094, 1.1443004253029967),
    #     (0.2154103272407928, 1.1442787081894892), (0.21545401069428866, 1.1443074436384597), (0.21540408674743075, 1.1442746031253468), (0.21540897416753152, 1.144277818123239),
    #     (0.3706149332741673, 1.3382610442561138), (0.37043596011870455, 1.338066059102687), (0.37066557363020575, 1.3383162151981816), (0.37044627337859726, 1.3380772950478617),
    #     (0.37045823893476604, 1.3380903311135184)]

    # x_values_FP = [xFP for (xFP, yFP) in x]
    # y_values_FP = [yFP for (xFP, yFP) in x]

    data = pd.DataFrame({'X': x_values_FP, 'Y': y_values_FP})
    data['Fixed Point'] = 'Predicted fixed point'

    hue_color = {'Predicted fixed point': 'red'}
    g = sns.jointplot(data=data, x='X', y='Y', hue='Fixed Point', ratio=10, palette=hue_color)
    g.fig.set_size_inches((9.5,6))

    plt.xlabel("Validation")
    plt.ylabel("Mean Squared Error")

    x_mean = np.mean(x_values_FP)
    y_mean = np.mean(y_values_FP)

    plt.scatter([x_mean], [y_mean], color = 'green', marker='o', label='mean fixed point')

    plt.scatter([x_values_sigmoid_FP], [y_values_sigmoid_FP], color= 'pink', marker='o', label='detailed fixed point')

    # Phase Space Standard
    x_real_FP = 0.40886584
    A = 0.7
    B = 0.8
    y_real_FP = (x_real_FP + A) / B
    plt.scatter([x_real_FP], [y_real_FP], color = 'b', marker='o', label='real fixed point')

    # Print Standard Deviation between Predicted FP and real FP
    points = [(x,y) for x,y in zip(x_values_FP, y_values_FP)]
    std_distance = distance_deviation_calculator(x_mean, y_mean, points=points)
    std_distance_mean_from_real = np.sqrt((x_real_FP-x_mean)**2 + (y_real_FP-y_mean)**2) / std_distance
    print(f"\nOur mean point ({round(x_mean,2)}, {round(y_mean,2)}) is distance {round(std_distance_mean_from_real,2)}stds from real point ({round(x_real_FP,2)}, {round(y_real_FP,2)})")

    # Now plotting the limit cycle together with the (real) nullclines
    x_lc, y_lc = limit_cycle()
    plt.plot(x_lc, y_lc, 'r-', label=f'LC')
    # Plot Nullcines
    # vdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$w=v - (1/3)*v**3 + R * I$"+r" ,$\dot{v}=0$")
    # wdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$w=(v + A) / B$"+r" ,$\dot{w}=0$")
    
    plt.xlabel('v (voltage)')
    plt.ylabel('w (recovery variable)')
    # plt.title(f'Predictions!!')
    plt.suptitle(f"Phase Space: Limit Cycle and Nullclines with Predictions.\nBest 5 of validation.\n Option 1, lr0.01, 499, [16,16] relu min-max #40\nOption 3, lr0.01, 499, [16,16] relu min-max #40")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.10, 1.0), loc='upper left')

    plt.subplots_adjust(top=0.85, right=0.7) # Reduce plot to make room 

    plt.show()


def plot_fixed_point_prediction():
    """
    Plot fixed points for linear and cubic nullcline using neural network parameters:
    linear nullcline: option='option_1', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    cubic nullcline: option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    Values come from the function 'fixed_point_analysis_gaussian_fit(), for tau=7.5
    Additionally, the function plots the limit cycle and nullclines for the system.

    For each nullcline the five predictions with lowest validation error are used. Also, the fixed point for the mean of
    these predictions is considered. Also the for the activation function Sigmoid, the fixed point is considered.

    Figure used in thesis/defense.
    """
    x_values_FP = [0.25792583527381774, 0.25792819261505523, 0.2579235334421827, 0.25792005750395175, 0.25794969061985296, 0.06747576469453473, 0.06748005087652252, 0.0674739974865549, 0.06747185671312873, 0.06750865574947167, 0.49601259228050537, 0.49601569216705144, 0.49601026103875184, 0.49600494589369293, 0.49603305153170985, 0.21543490321247138, 0.2154373122942576, 0.2154331921991781, 0.21542987747619022, 0.2154584491016629, 0.37056003616889804, 0.370570300799389, 0.37055130731078995, 0.37053794306270926, 0.37063933108409663]
    y_values_FP = [1.1974084847809543, 1.1974100888132329, 1.1974069185197005, 1.1974045533478568, 1.1974247169430392, 0.959346471647618, 0.9593498025920661, 0.9593450982874512, 0.9593434346162915, 
                    0.9593720324560837, 1.495017112614222, 1.4950191991732724, 1.4950155434362835, 1.4950119657685645, 1.495030883903956, 1.1442949089691996, 1.1442964936693028, 1.1442937834603197, 1.1442916030271393, 1.144310397514772, 1.3382013566173183, 1.3382125387845243, 1.3381918475024046, 1.3381772886484105, 1.3382877395612665]
    x_values_sigmoid_FP = [0.4767745090720008, 0.47677822960123145, 0.47677149927903656, 0.4767650412836927, 0.4768015972086843]
    y_values_sigmoid_FP = [1.4709693559847439, 1.4709723856584043, 1.470966905072548, 1.4709616462458652, 1.4709914141945117]

    x_value_mean_relu_FP = [0.24130567024459346] 
    y_value_mean_relu_FP = [1.1766319489763382]

    plt.figure(figsize=(3,3))
    plt.scatter(x_values_FP, y_values_FP, color='grey', zorder=4, alpha=0.6, edgecolors='none')
    plt.scatter(x_value_mean_relu_FP, y_value_mean_relu_FP, color='blue', marker='o', label='mean', zorder=5, alpha=1)
    plt.scatter([x_values_sigmoid_FP], [y_values_sigmoid_FP], color='C1', marker='o', label='detailed fixed point', zorder=6, alpha=1)

    # Real fixed point
    x_real_FP = 0.40886584
    A = 0.7
    B = 0.8
    y_real_FP = (x_real_FP + A) / B
    plt.scatter([x_real_FP], [y_real_FP], color='black', marker='o', label='Real', zorder=10)

    # Calculate distances (same as before)
    x_mean = x_value_mean_relu_FP[0]
    y_mean = y_value_mean_relu_FP[0]
    dist_to_mean = np.sqrt((x_real_FP - x_mean)**2 + (y_real_FP - y_mean)**2)
    dist_to_sigmoid = np.sqrt((x_real_FP - x_values_sigmoid_FP[0])**2 + (y_real_FP - y_values_sigmoid_FP[0])**2)
    print("Distance real to mean FP", dist_to_mean)
    print("Distance real to fitted FP", dist_to_sigmoid)
    print("Distance shortened of sigmoid by", (1 - (dist_to_sigmoid) / dist_to_mean) * 100, "percent")

    # Plot Limit Cycle and Nullclines
    x_lc, y_lc = limit_cycle()
    plt.plot(x_lc, y_lc, 'r-', label=f'LC', zorder=0)
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_vdot(v), '--', color="lime", label=r"$w=v - (1/3)*v**3 + R * I$"+r" ,$\dot{v}=0$", zorder=2)
    plt.plot(v, nullcline_wdot(v), '--', color="cyan", label=r"$w=(v + A) / B$"+r" ,$\dot{w}=0$", zorder=1)

    plt.xlabel(r'$v$ (voltage)', labelpad=-1)
    plt.ylabel(r'$w$ (recovery variable)')
    plt.xlim(-1.8014541357370093,1.8661846058677982)
    plt.ylim(0.1527082175888098,1.8752590634770199)
    plt.yticks([0.5, 1, 1.5])
    print(f"Phase Space: Limit Cycle and Nullclines with Predictions.\nBest 5 of validation.\n Option 1, lr0.01, 499, [16,16] relu min-max #40\nOption 3, lr0.01, 499, [16,16] relu min-max #40")

    plt.tight_layout()
    plt.subplots_adjust(top=0.925,
    bottom=0.143,
    left=0.19,
    right=0.985,
    hspace=0.2,
    wspace=0.2)
    plt.title("Fixed Point Prediction", pad=-1)

    plt.show()

def distance_deviation_calculator(x_real, y_real, points):
    distance = []
    for (x,y) in points[0::5]:
        print(x,y)
        calculated_distance = np.sqrt((x-x_real)**2 + (y-y_real)**2)
        distance.append(calculated_distance)
    return np.std(distance)

if __name__ == '__main__':
    plot_fixed_point_prediction()
