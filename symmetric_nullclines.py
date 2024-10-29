# This script visualizes the predicted nullclines with the limit cycle for the FitzHugh-Nagumo model.
# The linear and cubic nullclines are trained in different directions, such as w as a function of v, and vice versa.
# These trained nullclines are plotted along with the limit cycles.

# Main function to execute the script: plot_symmetric_nullclines()

# To run this script directly, the main entry point is the plot_symmetric_nullclines() function,
# which is executed when the script is run as a standalone program.
# This is controlled by the following block at the end:
#
# if __name__ == '__main__':
#     plot_symmetric_nullclines()

import os
from ast import literal_eval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
from FitzHugh_Nagumo_ps import nullcline_and_boundary, nullcline_vdot, nullcline_wdot, limit_cycle, calculate_mean_squared_error
from NN_model_analysis import plot_lc_from_modelname
from predict_fixed_point import search_5_best_5_worst_modelnames
from settings import TAU

def plot_symmetric_nullclines():
    """
    Plots symmetric nullclines and limit cycles for the FitzHugh-Nagumo model
    for two nullclines in two directions. Each subplot contains different
    combinations to visualize the predicted nullclines.

    The four subplots represent:
        1. First: input v with a linear nullcline
        2. Second: input v with a cubic nullcline
        3. Third: input u with a linear nullcline
        4. Fourth: input u with a cubic nullcline
    
    The five models with lowest validation error are selected and plotted.

    Parameters of neural network are:
    learning_rate=0.01, epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu'
    """
    # Create 2x2 subplots
    fig, axs = plt.subplots(nrows=2, ncols=2)

    fig.set_size_inches(3, 3)

    # Model data retrieval and plotting
    # Call plot_custom function on each subplot
    best_worst_modelnames_param1, df_param1 = search_5_best_5_worst_modelnames(option='option_1', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    best_val_modelnames_param1 = best_worst_modelnames_param1['best models']
    prepare_plot_lc_from_modelname(axs[0, 0], best_val_modelnames_param1, df_param1) # Plot on first subplot: input v, linear nullcline

    best_worst_modelnames_param3, df_param3 = search_5_best_5_worst_modelnames(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    best_val_modelnames_param3 = best_worst_modelnames_param3['best models']
    prepare_plot_lc_from_modelname(axs[0, 1], best_val_modelnames_param3, df_param3) # Plot on second subplot: input v, cubic nullcline
    # model chosen because best performing of the 360 [8,8] 0.01 499 (found by search_modelname_of_point() in plot_NN_ps.py)

    best_worst_modelnames_param2, df_param2 = search_5_best_5_worst_modelnames(option='option_2', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    best_val_modelnames_param2 = best_worst_modelnames_param2['best models']
    prepare_plot_lc_from_modelname(axs[1,0], best_val_modelnames_param2, df_param2) # plot on third subplot: input u, linear nullcline
    # model chosen because best performing of the 20x90=180 (found by search_modelname_of_point() in plot_NN_ps.py)

    best_worst_modelnames_param4, df_param4 = search_5_best_5_worst_modelnames(option='option_4', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    best_val_modelnames_param4 = best_worst_modelnames_param4['best models']
    handles, labels, _ = prepare_plot_lc_from_modelname(axs[1,1], best_val_modelnames_param4, df_param4) # plot on fourth subplot: input u, cubic nullcline
    # model chosen because best performing of the 20x90=180 (found by search_modelname_of_point in plot_NN_ps.py and testing some, not based on MSE (cause MSE on option_4 are inaccurate [FIXED]))

    axs[0,0].set_xticklabels([])
    axs[0,1].set_xticklabels([])
    axs[0,1].set_yticklabels([])
    axs[1,1].set_yticklabels([])

    axs[0,1].set_ylabel("")
    axs[0,1].set_xlabel("")
    axs[1,1].set_ylabel("")
    axs[0,0].set_xlabel("")

    plt.subplots_adjust(top=0.895,
    bottom=0.13,
    left=0.11,
    right=0.975,
    hspace=0.205,
    wspace=0.09)
    plt.suptitle('Symmetric Nullclines', fontsize=11)
    plt.show()

def prepare_plot_lc_from_modelname(ax, modelnames, df=None):
    """ Prepare the plot of the nullcline on the phase space

    Args:
        ax: The axis to plot on.
        modelnames: List of model names to plot.
    """

    if df is None:
        absolute_path = os.path.dirname(__file__)
        relative_path = f"FHN_NN_loss_and_model_{TAU}.csv"
        csv_name = os.path.join(absolute_path, relative_path)
        df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}, engine='c') # literal eval returns [2,2] as list not as str

    option = df[(df['modelname'] == modelnames[0])]['option'].iloc[0]

    mean_mse_seprate, std_mse_seperate = average_std_MSE_of_models(modelnames, df)

    axis_values_for_nullcline, mean_value, std_value = average_std_models_plot_and_MSE(modelnames, df)
    # load data of nullclines in phasespace
    _, exact_nullcline_values = nullcline_and_boundary(option, len(axis_values_for_nullcline))

    MSE_mean = calculate_mean_squared_error(exact_nullcline_values, mean_value)    

    # set-up for plotting the nullcline
    if option == 'option_1' or option == 'option_3':
        ax.errorbar(x=axis_values_for_nullcline, y=mean_value, yerr=std_value, fmt=' ', color='black', ecolor='lightgrey', capsize=0, alpha=0.7, label='Prediction error')
        predicted_nullcline_line = ax.plot(axis_values_for_nullcline, mean_value, label='Prediction', color='blue')
        plot_title = rf'$w(v)$, All MSE: {"{:.2e}".format(MSE_mean)},'+'\n'+rf'Seperate MSE: {"{:.2e}".format(mean_mse_seprate)} $\pm$ {"{:.2e}".format(std_mse_seperate)}'
        print(option, plot_title)

    if option == 'option_2' or option == 'option_4':
        ax.errorbar(y=axis_values_for_nullcline, x=mean_value, xerr=std_value, fmt=' ', color='purple', ecolor='lightgrey', capsize=0, alpha=0.6, label='Prediction error')
        predicted_nullcline_line = ax.plot(mean_value, axis_values_for_nullcline, label='Prediction', color='blue')
        plot_title = rf'$v(w)$, All MSE: {"{:.2e}".format(MSE_mean)},'+'\n'+rf'Seperate MSE: {"{:.2e}".format(mean_mse_seprate)} $\pm$ {"{:.2e}".format(std_mse_seperate)}'
        print(option, plot_title)

    x_lc, y_lc = limit_cycle()
    limit_cycle_line = ax.plot(x_lc, y_lc, 'r-', label=f'LC')

    # Plot Nullcines
    # vdot
    v = np.linspace(-2.5, 2.5, 1000)
    cubic_nullcline_line = ax.plot(v, nullcline_vdot(v), '--', color = "lime", label = r"$w=v - (1/3)v^3 + RI$"+r" ,$\dot{v}=0$ nullcline", alpha=1)
    # wdot
    v = np.linspace(-2.5, 2.5, 1000)
    linear_nullcline_line = ax.plot(v, nullcline_wdot(v), '--', color = "cyan", label = r"$w=(v + A) / B$"+r" ,$\dot{w}=0$ nullcline", alpha=0.7)

    ax.set_xlim(-2, 2)
    ax.set_ylim(0,2)
    ax.set_xlabel(r'$v$ (voltage)', labelpad=-1, fontsize=9)
    ax.set_ylabel(r'$w$ (recovery var.)', labelpad=-0.5, fontsize=9)

    if option == 'option_1':
        plot_title += 'linear nullcline'
        plot_title_used = 'linear nullcline w(v)'
    if option == 'option_2':
        plot_title_used = 'linear nullcline v(w)'
    if option == 'option_3':
        plot_title_used = 'cubic nullcline w(v)'
    if option == 'option_4':
        plot_title_used = 'cubic nullcline v(w)'

    ax.set_title(plot_title_used, fontsize=9.5, pad=-1)

    # ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    return handles, labels, df

def average_std_MSE_of_models(modelnames, df):
    """Calculates the MSE (nullcline error) average and std (standard deviation) of the models.
    It returns just one value for MSE and std.

    Args:
        modelnames: List of model names to plot.
        df: the dataframe containing the MSE data of each model
    """
    selected_rows = df.loc[df['modelname'].isin(modelnames)]

    MSE_list  = selected_rows['MSE']
    print("lijst van MSE", MSE_list)
    mean_mse = selected_rows['MSE'].mean()
    std_mse = selected_rows['MSE'].std()

    return mean_mse, std_mse

def average_std_models_plot_and_MSE(modelnames, df):
    """Calculates the mean and std of the prediction with its corresponding x-or y-values.
    It returns arrays of the same length.

    Args:
        modelnames: List of model names to plot.
        df: the dataframe containing the MSE data of each model
    """
    # average_lc_from_mmodelnames in plot_NN_ps.py
    all_predictions = np.zeros((len(modelnames),), dtype=object)
    for i, modelname in enumerate(modelnames):
        (axis_value, all_predictions[i], df) =  plot_lc_from_modelname(modelname, title_extra='', plot_bool=False, df = df)

    mean_prediction = np.mean(all_predictions, axis=0)
    std_prediction = np.std(all_predictions, axis=0)

    return axis_value, mean_prediction, std_prediction


if __name__ == '__main__':
    plot_symmetric_nullclines()
    pass
