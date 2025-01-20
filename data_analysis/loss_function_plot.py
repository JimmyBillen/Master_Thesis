# use data from FHN_NN_loss_and_model.csv to plot loss functions

# only (main)function used is 'validation_data_compare' where we can compare data as below
# 
# format:
# set-up factor 1\ set-up factor 2 | parameter 1-1 |parameter 2-1 | parameter 3-1 |
#                    parameter 1-1 | LOSSFUNC      | LOSSFUNC     | LOSSFUNC      |
#                    parameter 1-2 | LOSSFUNC      | LOSSFUNC     | LOSSFUNC      |
#                    parameter 1-3 | LOSSFUNC      | LOSSFUNC     | LOSSFUNC      |
#                    parameter 1-4 | LOSSFUNC      | LOSSFUNC     | LOSSFUNC      |

# POSSIBLE IMPROVEMENT:
# Ipv direct een functie die alles tegelijk checkte: Maak een functie die alles 1 per 1 doet en op zijn geheel werkt
# Maak dan daarna een programmaetje dat al die data dan samenbundelt en dan een mxn plot maakt 

import sys
sys.path.append('../../Master_Thesis') # needed to import settings

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import seaborn as sns
from model_building.CSV_clean import df_select_max_epoch
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from settings import TAU, NUM_OF_POINTS
import time

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
    Plots the averaged + std training and validation error for specific combination of normalization method and activation function
    """
    # Load DataFrame
    if df is None:
        absolute_path = os.path.dirname(__file__)
        relative_path = f"FHN_NN_loss_and_model_{TAU}_{NUM_OF_POINTS}.csv"
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

if __name__ == '__main__':
    """
    validation_data_compare() :plots the 3x3 validation data for combination normalization method and activation function

    plot_loss_and_validation_one_model() :plots the validation data for one specific trained model

    plot_loss_and_validation_loss_param() :plots the validation data for specific combination of normalization method and activation function
    """


    # 1 (3x3 validation)
    # normalization_method = ["no-norm", "z-score", "min-max"]
    # activation_function = ["relu", "tanh", "sigmoid"]
    # learning_rate = 0.01
    # max_epochs = 499
    # nodes = [8,8]
    # layers = 2
    # average = 40
    # option = 'option_3'

    # validation_data_compare(normalization_method, activation_function, learning_rate, nodes, layers, max_epochs, option, average)

    # 2: plotting of one set of normalization method and activation function (ER WAS EEN FOUT IN DE CODE WAARBIJ SELECT DATA FROM DF norm & act niet werd doorgegeven)
    # df = plot_loss_and_validation_loss_param(normalization_method='min-max', activation_function='relu', learning_rate=0.01, nodes=[4,4], layers=2, max_epochs=499, option='option_3', average=40)
    plot_loss_and_validation_loss_param(normalization_method='min-max', activation_function='relu', learning_rate=0.01, nodes=[4,4], layers=2, max_epochs=499, option='option_3', average=40)
    # plot_loss_and_validation_loss_param(normalization_method='min-max', activation_function='sigmoid', learning_rate=0.005, nodes=[8,8], layers=2, max_epochs=1999, option='option_3', average=1)

    # plot_loss_and_validation_loss_param(normalization_method='min-max', activation_function='relu', learning_rate=0.01, nodes=[16,16], layers=2, max_epochs=499, option='option_3', average=40, df=df)

    # normalization_method = ["no-norm", "z-score", "min-max"]
    # activation_function = ["relu", "tanh" ,"sigmoid"]
    # learning_rate = 0.01
    # max_epochs = 499
    # nodes = [8,8]
    # layers = 2
    # average = 40
    # option = 'option_3'

    # validation_data_compare(normalization_method, activation_function, learning_rate, nodes, layers, max_epochs, option, average)