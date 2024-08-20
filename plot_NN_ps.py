# This file will be used to plot (hardcoding will be seen here a lot)


"""
In this file it is possible to plot validation vs. mse with the option to click on the point to receive the modelname
    -search_modelname_of_point()

Possible to plot the nullcline in phase-space from modelname
    -plot_lc_from_modelname()

Averages over 5 best and 5 worst models given the parameters and for the two cases the mean and deviation on the nullclines:
    -plot_best_worst_avg_param()
Averages over 5 best given the parameters and uses the mean and deviation on the nullclines:
    -plot_best_avg_param()
"""

# First goal is to find a modelname

# Then to use this modelname to plot on the phase space


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from ast import literal_eval
from FitzHugh_Nagumo_ps import nullcline_and_boundary, nullcline_vdot, nullcline_wdot, limit_cycle, calculate_mean_squared_error
from keras.models import load_model, Model
from create_NN_FHN import normalization_with_mean_std, reverse_normalization
from Nullcine_MSE_plot import open_csv_and_return_all
import time
from settings import TAU, NUM_OF_POINTS
import math
import matplotlib as mpl
import pickle

# plotting and clicking on point to visualize

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
    relative_path = f"VAL_vs_MSE_{TAU}_{NUM_OF_POINTS}"
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

def plot_lc_from_modelname_extrapolate_thesis(modelname, title_extra='', plot_bool=True, df=None):
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
    # axis_values_for_nullcline, exact_nullcline_values = nullcline_and_boundary(option, amount_of_points)
    minvalue_axis, max_value_axis = -4, 4
    axis_values_for_nullcline = np.linspace(minvalue_axis, max_value_axis, 1000)

    # Predict normalized data 
    input_prediction, reverse_norm_mean_std = normalize_axis_values(axis_values_for_nullcline, mean_std, option)
    prediction_output_normalized = model.predict(input_prediction)
    # Reverse normalize to 'normal' data
    prediction_output_column = reverse_normalization(prediction_output_normalized, reverse_norm_mean_std)
    prediction_output = prediction_output_column.reshape(-1)
    
    fig, ax = plt.subplots()
    width = 3
    height = 3
    fig.set_size_inches(width, height)

    if plot_bool:
        # plot normal LC
        x_lc, y_lc = limit_cycle()
        plt.plot(x_lc, y_lc, 'r-', label=f'Trajectory')
        # Plot Nullcines
        # vdot
        v = np.linspace(minvalue_axis, max_value_axis, 1000)
        plt.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$\dot{v}=0$ nullcline") #$w=v - (1/3)*v**3 + R * I$"+r" ,
        # wdot
        v = np.linspace(minvalue_axis, max_value_axis, 1000)
        plt.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$\dot{w}=0$ nullcline") #$w=(v + A) / B$"+r" ,

        if option == 'option_1' or option == 'option_3':
            plt.plot(axis_values_for_nullcline, prediction_output, label = 'Prediction')
        if option == 'option_2' or option == 'option_4':
            plt.plot(prediction_output, axis_values_for_nullcline, label = 'Prediction')
        plt.xlim(-4.076173609403142, 4.154202080761771)
        plt.ylim(-10.540196531313729, 7.571200797751855)
        plt.yticks([-10, -5, 0, 5])
        plt.xlabel('v (voltage)', labelpad=-1)
        plt.ylabel('w (recovery variable)', labelpad=-5)
        plt.title(f"Phase Space")
        print(f"Phase Space: Limit Cycle and Cubic Nullcline with Prediction\n{option}, lr{learning_rate}, {nodes}, epoch {max_epochs}\n {normalization_method}, {activation_function}\n{title_extra}")
        # plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.subplots_adjust(top=0.917,
bottom=0.154,
left=0.17,
right=0.98,
hspace=0.2,
wspace=0.2)

        mpl.rc("savefig", dpi=300)
        plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\Uniqueness\ExtrapolateNullcline.png')


        plt.show()
    return (axis_values_for_nullcline, prediction_output, df)


def average_lc_from_modelnames(modelnames:list, performance='',df=None, *args):
    """
    Computes the average prediction from a list of model names and plots it along with standard deviation.
    Only for option3!!
    
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
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\PCCs\{args}, {performance}, {"{:.2e}".format(MSE_calculated)}_{TAU}.png')

    plt.show()

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
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\PCCs\ERROR_{args}, {performance}, {"{:.2e}".format(MSE_calculated)}_{TAU}.png')

    plt.show()


    # seperately
    print('Possibility to plot each of the 5 separately here >>>')
    # """"
    plt.figure(figsize=(3, 3))
    for i,v in enumerate(modelnames):
        plt.plot(axis_value, all_predictions[i], color='grey', alpha=0.5)
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
    # """"
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

def plot_best_avg_param_thesis(option, learning_rate, max_epochs, nodes, layers, normalization_method, activation_function, df=None):
    """
    Plots the phase space with averaged predictions showing deviation from ONLY the 5 best performing models.
    (extra: returns std)

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
    axis_value, mean_prediction, std_prediction,_ = average_lc_from_modelnames_thesis(best_modelnames, performance_modelname, df ,option, nodes, learning_rate, max_epochs, normalization_method, activation_function)
    return axis_value, mean_prediction, std_prediction

def average_lc_from_modelnames_thesis(modelnames:list, performance='',df=None, *args):
    """
    Computes the average prediction from a list of model names and plots it along with standard deviation.
    Only for option3!!
    
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
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\PCCs\{args}, {performance}, {"{:.2e}".format(MSE_calculated)}_{TAU}.png')

    plt.show()

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
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\PCCs\ERROR_{args}, {performance}, {"{:.2e}".format(MSE_calculated)}_{TAU}.png')

    plt.show()


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
    return axis_value, mean_prediction, std_dev_prediction, df

def fitting_hyperparam1_to_avg_hyperparam2(option, learning_rate, max_epochs, nodes, layers, normalization_method, activation_function_1, activation_function_2, df=None):
    """Take the 5 best (lowest validation error) networks with hyperpameter2, average the nullcline prediction
    and search for the network that best fits to this average. Calculate its MSE as well. 

    Only difference in hyperparameter allowed right now is the activation function. 
    
    >>> Example:
    Using Benchmark ReLU for average and applying Sigmoid for more detail
    """
    axis_value , mean_prediction, std_dev_prediction = plot_best_avg_param_thesis(option, learning_rate, max_epochs, nodes, layers, normalization_method, activation_function_2, df)
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

    plt.figure(figsize=(3, 2))

    if TAU == 7.5:
        plt.xlim(-2.05, 2.05)
        plt.ylim(-0.05, 2.2)
        # plt.ylim(-0.05, 2.5) # if legend

    if TAU == 100:
        plt.ylim(-0.05,2.4)
        plt.xlim(-2.3, 2.2)
    # plot results:
    print("MSE vs vdot nullcline")
    nullcline_val = nullcline_vdot(axis_value)
    mse_mean = calculate_mean_squared_error(nullcline_val, mean_prediction)
    plt.plot(axis_value, mean_prediction, color='b', label=f'mean prediction {activation_function_2}')
    plt.fill_between(axis_value, mean_prediction-std_dev_prediction, mean_prediction+std_dev_prediction, color='grey', alpha=0.3, label="Std", zorder=0)
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

    plt.xlabel(r'$v$ (voltage)', labelpad=-2)
    plt.ylabel(r'$w$ (recovery variable)', labelpad=-1)
    # plt.title(f'Phase Space: Limit Cycle and Cubic Nullcline with ReLU mean and Sigmoid fit')
    # plt.title(f"Phase Space: Limit Cycle and Cubic Nullcline with Prediction\n{option}, lr{learning_rate}, {nodes}, epoch {max_epochs}\n {normalization_method}, {activation_function}\n{title_extra}")
    # plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.97,
bottom=0.191,
left=0.185,
right=0.985,
hspace=0.2,
wspace=0.2)

    mpl.rc("savefig", dpi=300)
    # plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\Benchmark\PS_zoomed_smoothing.png')
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\Benchmark\PS_{TAU}{option}{nodes}.png')


    plt.show()
    # error through time

    # +++++++++++++++++++++++++++++ ERROR ++++++++++++++++++
    plt.subplots(figsize=(3,1))

    # plt.plot(axis_value, mean_prediction, color='b', label='Mean')
    plt.plot(axis_value, np.array(mean_prediction)-np.array(nullcline_vdot(axis_value)), '--', color = "b") # r"$w=v - (1/3)*v**3 + R * I$"
    plt.fill_between(axis_value, mean_prediction-std_dev_prediction-np.array(nullcline_vdot(axis_value)), mean_prediction+std_dev_prediction-np.array(nullcline_vdot(axis_value)), color='grey', alpha=0.4, label="Std")
    plt.plot(axis_value, np.array(prediction_hyperparam1)-np.array(nullcline_vdot(axis_value)), '--', color = "C1")
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

    # mpl.rc("savefig", dpi=300)
    # plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\Benchmark\Error_{TAU}{option}{nodes}.png')

    plt.show()



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





if __name__ == '__main__':
    # using cache to reduce load of df7.5 from 2minutes to 10 seconds, for 100 from 50seconds to 10seconds

    """ USE IF NOT USING NUM OF POINTS VARIATION"""
    a = time.time()
    # Check if cache exists
    cache_file = f'data_cache_{TAU}_{NUM_OF_POINTS}.pkl'
    if os.path.exists(cache_file):
        print(f"loading from cache {TAU}")
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
    else:
        print(f"need to make cache {TAU}")
        # Load your CSV file and cache it
        absolute_path = os.path.dirname(__file__)
        relative_path = f"FHN_NN_loss_and_model_{TAU}_{NUM_OF_POINTS}.csv"
        csv_name = os.path.join(absolute_path, relative_path)
        df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}, engine='c') # literal eval returns [2,2] as list not as str
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
    b= time.time()
    print(b-a, 'seconds')
    # tau 100
    # fitting_hyperparam1_to_avg_hyperparam2(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[4,4], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu')
    # tau 7.5
    # fitting_hyperparam1_to_avg_hyperparam2(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu', df=df)

    # fitting_hyperparam1_to_avg_hyperparam2(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu')

    # THESIS: KEY EXAMPLE:
    # fitting_hyperparam1_to_avg_hyperparam2(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu')
    '''
    Tau7.5!
    fitting_hyperparam1_to_avg_hyperparam2(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu')
    1) First 5 MSE values are printed, those ReLU networsk are then averaged (can show that average is less than seperate MSE!!)
    2) ReLU average MSE is shown.
    3) Sigmoid fit on ReLU average is shown. The LOWEST (!!!) nullcline error network corresponds best (9th best on valid (dont care bout this)) 
    4) Sigmoid fit has lower error than ReLU fit!  (6.85e-4) vs (1.98e-3)

    

    idem geld VOLLEDIG voor tau100!
    '''

    '''tau=100
    
    [8,8] (tot 8.89e-5 accuracy)
    fitting_hyperparam1_to_avg_hyperparam2(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu')

    fitting_hyperparam1_to_avg_hyperparam2(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[4,4], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu')
    [4,4]  (tot 9.33e-5(!!!)) accuracy (waarbij het de 5de beste nullcline error gebruikt) terwijl mean MSE relu 'maar' 1.42e-03 is

    fitting_hyperparam1_to_avg_hyperparam2(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu')
    [16,16] (tot 2.00e-3)

    10k points
    fitting_hyperparam1_to_avg_hyperparam2(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[4,4], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu')
    [4,4] 10k points:  (tot 4.54e-4(!!!)) accuracy (waarbij het de 22ste beste nullcline error gebruikt) terwijl mean MSE relu 'maar' 1.6e-03 is

    
    '''
    # fitting_hyperparam1_to_avg_hyperparam2(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu', df=df)
    # fitting_hyperparam1_to_avg_hyperparam2(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu')

    # fitting_hyperparam1_to_avg_hyperparam2(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[4,4], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu')

# =====================================  vb3: PCC effectivness testing by plotting on LC===========================================
    # om aan te tonen dat PCC belangrijk is kiezen we ons beste resultaat PCC:0.46 (met outlier erbij!) [16,16]relu
    # gaan we vergelijken met PCC=-0.31 [8,8,8,8]sigmoid 
    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[4,4], layers=2, normalization_method='min-max', activation_function='relu')
    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, normalization_method='min-max', activation_function='relu')
    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    
    # 4layers, 8nodes, 2 layers:4,8,16nodes

    # import pickle

    # # Load your CSV file
    # absolute_path = os.path.dirname(__file__)
    # relative_path = f"FHN_NN_loss_and_model_{TAU}.csv"
    # csv_name = os.path.join(absolute_path, relative_path)
    # df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}, engine='c') # literal eval returns [2,2] as list not as str

    # # Cache the data
    # with open(f'data_cache_{TAU}.pkl', 'wb') as f:
    #     pickle.dump(df, f)


    # plot_lc_from_modelname_thesis('76da590b9b4c4c41a0fb3a8bc58fad38', df=df) # example 1: 16layers 8 nodes minmax, relu   (tau7.5)

# option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu', df=df)


    # plot_best_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=4*[8], layers=4, normalization_method='min-max', activation_function='relu', df=df)
    # plot_best_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[4,4], layers=2, normalization_method='min-max', activation_function='relu', df=df) #1.42e-3
    # plot_best_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, normalization_method='min-max', activation_function='relu', df=df) #7.99e-4 
    # plot_best_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu', df=df) #2.06e-3
    # plot_best_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[4,4], layers=2, normalization_method='min-max', activation_function='sigmoid', df=df) #2.35e-4
    # plot_best_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, normalization_method='min-max', activation_function='sigmoid', df=df) #2.77e-4
    # plot_best_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='sigmoid', df=df) #1.13e-3

    # df = plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, normalization_method='no-norm', activation_function='relu')
    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, normalization_method='min-max', activation_function='relu', df=df)
    # plot_best_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, normalization_method='no-norm', activation_function='relu')

    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[4,4], layers=2, normalization_method='min-max', activation_function='relu')
    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[4,4], layers=2, normalization_method='min-max', activation_function='sigmoid')

    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, normalization_method='min-max', activation_function='relu')
    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, normalization_method='min-max', activation_function='sigmoid')

    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='sigmoid')

    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[4,4,4,4], layers=4, normalization_method='min-max', activation_function='relu')

    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8,8,8], layers=4, normalization_method='min-max', activation_function='relu')


    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8,8,8], layers=4, normalization_method='min-max', activation_function='sigmoid')
    # plot_best_worst_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16,16,16], layers=4, normalization_method='min-max', activation_function='relu')

    # search_modelname_of_point(option='option_2', learning_rate=0.01, max_epochs=99, nodes=[8,8], layers=2, total=180)
    # search_modelname_of_point(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, total=360)
    # search_modelname_of_point(option='option_3', learning_rate=0.01, max_epochs=499, nodes=2*[4], layers=2, total=80, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    
    # search_modelname_of_point(option='option_3', learning_rate=0.01, max_epochs=499, nodes=16*[16], layers=16, total=80, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    
    # MSE of 0.2145
    # plot_lc_from_modelname('76da590b9b4c4c41a0fb3a8bc58fad38') # example 1: 16layers 8 nodes minmax, relu   (tau7.5)
    # plot_lc_from_modelname('6853072c28984d009d8de8eac3b6c495') # example 2: 16layers 8 nodes minmax, sigmoid(tau7.5)

    # plot_lc_from_modelname('78f82307e814413099d52f5bd2cf4dba') # example 3: 16layers 16 nodes minmax, relu (tau100)

    # Specific MSE values:
    # plot_lc_from_modelname('e8929823831e41239aea64c2f66f18eb') # example 1: nullcline error 10e(-1) no-norm, relu(tau7.5)
    # plot_lc_from_modelname('5e40c7553fe647feabdc2155f7b9f99a') # example 2: nullcline error 10e(-2) z-score, relu(tau7.5)
    # plot_lc_from_modelname('fe52654ee942482191d032e3e498511e')   # example 3: nullcline error 10e(-3) min-max, sigmoid(tau100)
    # plot_lc_from_modelname('a3eddbb0a66e4f2dbe725db6afd0c971')   # example 4: nullcline error 10e(-4) min-max, sigmoid(tau100)



    # plot_lc_from_modelname('d76909ef0c9d41908bf1dc3bbbdb9dff')
    # plot_lc_from_modelname('88ff4b894d55456ea556026d5422acde')
    # plot_lc_from_modelname('872b51d5b0b648478fc3b0d24305d9e2')
    # plot_lc_from_modelname('0dfa2a41cc824b5a807d361bc644063c')
    # plot_lc_from_modelname('0f67c1b218d442148f2ce4f9556213d3')
    # plot_lc_from_modelname('845e2e7b769a4b0591f4fd3ffd969e7c')
    # plot_lc_from_modelname('68347f61830841be8c8c3c5f55b441fa')
    # plot_lc_from_modelname('7a57f0d0b4d14feca1a384a602c94519')
    # plot_lc_from_modelname('4a6117b3dd7c42a09a335bbd7cfe816f')
    # plot_lc_from_modelname('8e3c481c86de46509a243ab4187b6468')

    # To look at whole TAU=7.5 with minmax relu sigmoid
    # plot_val_vs_MSE_extract_modelname_tau('option_3', learning_rate=0.01, max_epochs=499, amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # # =>
    # modelnames = [
    #     '628a4b661c22445a9471abef03c682be',
        # '8144adb218bb43f1b6fbcc1d044ef022',
        # 'afd40432dedb4322a15878c8da5a012c',
        # '59965f2a7f9142f69683dc9f5fa2ecd4',
        # 'c60f176b3a7d4cd2b8121bcb5447b8bc',
        # 'be74a8d8a84a418b9f00a39cb5bc47a2',
        # '7a87132a2f1c46e7bb75842e5d6b196d'
    # ]
    # average_lc_from_modelnames(modelnames)

    # To look at whole TAU=100 with minmax relu sigmoid
    # plot_val_vs_MSE_extract_modelname_tau('option_3', learning_rate=0.01, max_epochs=499, amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # modelnames = [
        # 'df6d14260b15452c8eb545536b3dab89'
        # 'ef88f215cb7e4f2abb24c33d2fd4219d',
        # 'ef88f215cb7e4f2abb24c33d2fd4219d',
        # '5012b85f2e674a1798bfb49a5155b45b',
        # '194cc2df343e4e63b37fe4fdbecdebac',
        # '3eb26ad1487748e795b7f05d6c6cd617'
    # ]
    # average_lc_from_modelnames(modelnames)
    
    pass

# ===========================================vb 1: 3 mogelijkheden getoond hier ===========================================

# search_modelname_of_point(option='option_3', learning_rate=0.01, max_epochs=499, nodes=16*[16], layers=2, total=80)

# plot_lc_from_modelname_extrapolate_thesis('5abe772788dc4faf8860f6f57a52e421')

# lr 0.01, epoch 500, [8,8]
# modelnames_dict = {'low val, low MSE':'e35aab45b22f4b8ba05f37e50e39d40c',
#                    'low val, high MSE':'12136245d29447eb9dd7ff4e357808a5',
#                    'high val, low MSE':'5f7fde69ff9d4d50856a1c26d8bcd942',
#                    'high val, high MSE':'4aa7f465e242460eb4d3edd19f6712a0'}
# modelnames = ['2cdbd6e254c84cf18ee4cd692ff169db', '12136245d29447eb9dd7ff4e357808a5', '4aa7f465e242460eb4d3edd19f6712a0', '5f7fde69ff9d4d50856a1c26d8bcd942']

# lr 0.005, epoch 1000, [8,8]
# modelnames_dict = {'low val, low MSE':'2b79bed8546649ab91589958d992cbae',
#                    'low val, high MSE':'36da3fdf6aa04f04a9527df5b5bea88a',
#                    'high val, low MSE':'429930bf12ab472e87a384d44ca7d5db',
#                    'high val, high MSE':'c964d1956f4640d09b399eb06bbb1f5c'}
# modelnames = ['2b79bed8546649ab91589958d992cbae', '36da3fdf6aa04f04a9527df5b5bea88a', '429930bf12ab472e87a384d44ca7d5db', 'a0d044c5fedd4d779d82e8b7adbf8343']

# lr 0.01, epoch 500, [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8] (16 layers)
    # modelnames_dict = {'low val, low MSE':'003654a309db440296f8993d8176b6d2',
    #                 'low val, high MSE':'ab3b8d2e3677475e9562d94a14e53116',
    #                 'high val, low MSE':'61966c1986c24475b3d411f4ebf6913d',
    #                 'high val, high MSE':'85477d64407e4cc785f967ff888cfd0c'}
# modelnames = ['003654a309db440296f8993d8176b6d2', 'ab3b8d2e3677475e9562d94a14e53116', '61966c1986c24475b3d411f4ebf6913d', '85477d64407e4cc785f967ff888cfd0c']

    # for location, modelname in modelnames_dict.items():
    #     plot_lc_from_modelname(modelname, location)

# of specifiek:
# plot_lc_from_modelname("5f7fde69ff9d4d50856a1c26d8bcd942")


# ========================================== vb 2: 16x [8] nodes, studie van minmix, nonorm, zscore ===========================================================

# in 16 x 8nodes: Bestuderen van sigmoid in minmax, nonorm en zscore
    # sigmoid links: minmax sigmoid
        # 42751f73666c448ab45e0078770fd51f
        # 114dbad064d547959c4143e8ee7a8c02
    # sigmoid midden: nonorm sigmoid
        # 1f7780d9a70d434896fa4457dcc814b2
        # 77d698c367684ed2bafed038bb3385c4
        # 04ef13126f494f7cad2009970f2c79e6
    # sigmoid rechts: z-score sigmoid
        # 0e1f62a2e62e44ee959a39f9e26eba5d
        # 112279ea30f443e6b950dc448c6a4686
        # 08586e0b1a9748e7a937a4f75dff50ad
        # d59fee7cd1e04d3e8bf414fff2f28a41
    
    # plot_lc_from_modelname('42751f73666c448ab45e0078770fd51f', '')
    # plot_lc_from_modelname('114dbad064d547959c4143e8ee7a8c02', '')

    # plot_lc_from_modelname('1f7780d9a70d434896fa4457dcc814b2', '')
    # plot_lc_from_modelname('77d698c367684ed2bafed038bb3385c4', '')
    # plot_lc_from_modelname('04ef13126f494f7cad2009970f2c79e6', '')

    # plot_lc_from_modelname('0e1f62a2e62e44ee959a39f9e26eba5d', '')
    # plot_lc_from_modelname('112279ea30f443e6b950dc448c6a4686', '')
    # plot_lc_from_modelname('08586e0b1a9748e7a937a4f75dff50ad', '')
    # plot_lc_from_modelname('d59fee7cd1e04d3e8bf414fff2f28a41', '')

    # plot_lc_from_modelname('e218fc39425148ac884ebeaf7012f6c4') #option4: 1024 MSE
    # plot_lc_from_modelname('37b4f17364ec4e7696657523cbdf5e75') #option4: 0.07996753175858606 MSE
# ======================================================================================================
#   PLAYING WITH AMOUNT OF POINTS
    # absolute_path = os.path.dirname(__file__)
    # relative_path = f"FHN_NN_loss_and_model_{TAU}_{NUM_OF_POINTS}.csv"
    # csv_name = os.path.join(absolute_path, relative_path)
    # df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}) # literal eval returns [2,2] as list not as str

    # plot_best_avg_param(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu', df=df) #1.42e-3

    # fitting_hyperparam1_to_avg_hyperparam2(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[4,4], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu')
