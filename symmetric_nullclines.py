#  Hier gaan we symmetrische nullclines analyseren

"""
atm: still generating data do fullfill this

not sure which data to compare (lr 0.01, 100, vs 500 ... vs different layers)
"""

# keuze model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from ast import literal_eval
from FitzHugh_Nagumo_ps import nullcline_and_boundary, nullcline_vdot, nullcline_wdot, limit_cycle, calculate_mean_squared_error
# from keras.models import load_model, Model
# from create_NN_FHN import normalization_with_mean_std, reverse_normalization
# from Nullcine_MSE_plot import open_csv_and_return_all
from plot_NN_ps import normalize_axis_values, retrieve_model_from_name, plot_lc_from_modelname
from predict_fixed_point import search_5_best_5_worst_modelnames
from settings import TAU, NUM_OF_POINTS
import matplotlib as mpl

def plot_symmetric_nullclines():
    """
    Best models are chosen, this could be done using the plot_NN_ps.py functions (to find lowest MSE)
    """
    # Create 2x2 subplots
    # fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig, axs = plt.subplots(nrows=2, ncols=2)

    fig.set_size_inches(3, 3)
    # Call plot_custom function on each subplot
    best_worst_modelnames_param1, df_param1 = search_5_best_5_worst_modelnames(option='option_1', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu')
    best_val_modelnames_param1 = best_worst_modelnames_param1['best models']

    _, _, df = prepare_plot_lc_from_modelname(axs[0, 0], best_val_modelnames_param1, df_param1) # Plot on first subplot: input v, linear nullcline
    # model chosen because best performing of the 20x90=180 (ignoring the outlier)


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
    # model chosen because best performing of the 20x90=180 (found by search_modelname_of_point in plot_NN_ps.py and testing some, not based on MSE (cause MSE on option_4 are inaccurate))

    plt.subplots_adjust(left=0.07, right=0.76, hspace=0.375) # Reduce plot to make room 
    # fig.legend(handles, labels, bbox_to_anchor=(1.005, 0.8))

    # axs[0,0].get_xaxis().set_visible(False)
    axs[0,0].set_xticklabels([])
    axs[0,1].set_xticklabels([])
    axs[0,1].set_yticklabels([])
    axs[1,1].set_yticklabels([])
    axs[0,1].set_ylabel("")
    axs[0,1].set_xlabel("")
    axs[1,1].set_ylabel("")
    axs[0,0].set_xlabel("")

    # plt.suptitle(f"Phase Space: Limit Cycle and Nullclines with Prediction")
    # plt.tight_layout()
    plt.subplots_adjust(top=0.94,
bottom=0.13,
left=0.11,
right=0.975,
hspace=0.205,
wspace=0.09)
    
    plt.subplots_adjust(top=0.895,
    bottom=0.13,
    left=0.11,
    right=0.975,
    hspace=0.205,
    wspace=0.09)
    plt.suptitle('Symmetric Nullclines', fontsize=11)
    mpl.rc("savefig", dpi=300)
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\SymmetricNullclinesAndFixedPoint\SymmetricNullclines.png')


    plt.show()

def prepare_plot_lc_from_modelname(ax, modelnames, df=None):
    """ Prepare the plot of the nullcline on the phase space

    input:
    title_extra:
        Something extra in to put at the end in the title, like 'low val, high MSE'.
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
        
    # plot normal LC

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
    "Calculates the MSE average and std of the models."
    selected_rows = df.loc[df['modelname'].isin(modelnames)]

    MSE_list  = selected_rows['MSE']
    print("lijst van MSE", MSE_list)
    mean_mse = selected_rows['MSE'].mean()
    std_mse = selected_rows['MSE'].std()

    return mean_mse, std_mse

def average_std_models_plot_and_MSE(modelnames, df):
    """Calculates the mean and std of the plot, from this function also the MSE"""
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
