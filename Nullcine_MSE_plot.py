# In this code we will use data from FHN_NN_loss_and_model.csv to calculate and plot MSES

from loss_function_plot import does_data_exist
import os
import pandas as pd
from ast import literal_eval
from keras.models import load_model, Model
from FitzHugh_Nagumo_ps import nullcline_and_boundary, calculate_mean_squared_error, nullcline_vdot, nullcline_wdot, limit_cycle
from create_NN_FHN import normalization_with_mean_std, reverse_normalization
from CSV_clean import df_select_max_epoch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, shapiro, levene, kruskal, mannwhitneyu
import statsmodels.api as sm
from statsmodels.formula.api import ols
from settings import TAU, NUM_OF_POINTS

"""
Main functions:
1) Saving
    3 normalization x 3 activation
        -save_all_MSE_vs_VAL

    1 normalization x 2 activation
        -save_one_norm_two_act_MSE_vs_VAL

2) Plotting
    3 normalization x 3 activation
        -open_csv_and_plot_all
        -three_by_three_plot
    
    1 normalization x 2 activation
        -plot_validation_vs_mse_one_norm_two_act
        -big_MSE_for_one_norm_two_activation
        -big_MSE_vs_VAL_for_one_norm_two_act
    
    1 normalization x 1 activation
        -Val_vs_MSE_node_norm_act_plot

Note:
    The '1 norm x 2 act' may be focussed around min-max and relu/sigmoid (hardcoded) 
"""



# function that calculate MSEs and return
# function that plots it

# =========================================================================================================================

# => Extra functions <=

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
    relative_path = f"VAL_vs_MSE_{TAU}_{NUM_OF_POINTS}"
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
    relative_path = f"FHN_NN_loss_and_model_{TAU}_{NUM_OF_POINTS}.csv"
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
    relative_path = f"FHN_NN_loss_and_model_{TAU}_{NUM_OF_POINTS}.csv"
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
    # medians = df_plot.groupby(['normalization_method', 'activation_function'])['MSE'].median().reset_index()

    # # Convert to dictionary format for JSON
    # medians_dict = {}
    # for _, row in medians.iterrows():
    #     norm_act = f"{row['normalization_method']} + {row['activation_function']}"
    #     medians_dict[norm_act] = row['MSE']

    # import json
    # # Save the medians to a JSON file
    # with open(f'linear_nullcline_mse_medians_{TAU}.json', 'w') as f:
    #     json.dump(medians_dict, f, indent=4)
    # assert False, "Was only to save the medians"
    
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
    # folder_path = os.path.join(absolute_path, f"VAL_vs_MSE_{TAU}_{NUM_OF_POINTS}")
    # csv_name = os.path.join(folder_path, f"{save_name}.csv")

    # Try the first folder path
    try:
        absolute_path = os.path.dirname(__file__)
        folder_path = os.path.join(absolute_path, f"VAL_vs_MSE_{TAU}_{NUM_OF_POINTS}")
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
    relative_path = f"VAL_vs_MSE_{TAU}_{NUM_OF_POINTS}"
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

    # pcc_dict = {}

    for i, norm_method in enumerate(normalization_methods):
        for j, activ_func in enumerate(activation_functions):
            df_selection = df_plot[(df_plot['normalization_method'] == norm_method) & (df_plot['activation_function'] == activ_func)]

            min_mse = min(min_mse, min(df_selection['MSE']))
            max_mse = max(max_mse, max(df_selection['MSE']))
            min_validation = min(min_validation, min(df_selection['validation']))
            max_validation = max(max_validation, max(df_selection['validation']))

            pcc = scatterplot_setup_provider(df_selection, axs[i,j])
            # pcc_dict[f'{norm_method} + {activ_func}'] = pcc

    # import json
    # with open(f'linear_nullcline_pcc_{TAU}.json', 'w') as f:  for pcc analysis linear nullcline
    #     json.dump(pcc_dict, f, indent=4)    


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

            scatterplot_setup_provider(df_selection, axs[i,j])

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



def linear_nullcline_analysis_mse(ax=None):

    tau_values = [1, 5, 7.5, 100]
    file_names = [f'linear_nullcline_mse_medians_{tau}.json' for tau in tau_values]

    # Initialize a dictionary to hold the combined data
    combined_data = {f'tau={tau}': [] for tau in tau_values}

    import json
    # # Load each JSON file and combine the data
    # for i, file_name in enumerate(file_names):
    #     with open(file_name, 'r') as f:
    #         medians_dict = json.load(f)
        
    #     # Append the median values for each normalization + activation function combination
    #     for norm_act, value in medians_dict.items():
    #         combined_data[f'tau={tau_values[i]}'].append(value)

    # # Convert the combined data into a DataFrame
    # df = pd.DataFrame(combined_data, index=list(medians_dict.keys()))

    # Define the specific order for normalization methods and activation functions
    norm_methods_order = ['no-norm', 'z-score', 'min-max']
    activ_funcs_order = ['relu', 'tanh', 'sigmoid']

    # Create the specific order for the index (rows)
    row_order = [f'{norm} + {act}' for norm in norm_methods_order for act in activ_funcs_order]

    # Initialize a dictionary to hold the combined data
    combined_data = {f'tau={tau}': [] for tau in tau_values}

    # Load each JSON file and combine the data
    for i, file_name in enumerate(file_names):
        with open(file_name, 'r') as f:
            pcc_dict = json.load(f)
        
        # Append the PCC values for each normalization + activation function combination
        for norm_act in row_order:
            combined_data[f'tau={tau_values[i]}'].append(pcc_dict.get(norm_act, None))

    # Convert the combined data into a DataFrame with the specific order
    df = pd.DataFrame(combined_data, index=row_order)

    # Create the heatmap
    import matplotlib.colors as mcolors
    heatmap = sns.heatmap(df, cmap='YlOrBr_r', norm=mcolors.LogNorm(), ax=ax)

    # Customize the colorbar
    colorbar = heatmap.collections[0].colorbar
    # colorbar.set_label('Log Scale Colorbar')
    # colorbar.ax.set_ylabel('Logarithmic Scale', rotation=270, labelpad=20)

    ax.set_xticklabels([1,5,7.5,100], rotation=0)
    ax.set_xlabel(r'$\tau$')


    ax.set_title('Nullcline Error')
    if ax is None:
        plt.show()

def linear_nullcline_analysis_pcc(ax=None):
    import json
    # List of tau values and corresponding filenames
    tau_values = [1, 5, 7.5, 100]
    file_names = [f'linear_nullcline_pcc_{tau}.json' for tau in tau_values]

    # # Initialize a dictionary to hold the combined data
    # combined_data = {f'tau={tau}': [] for tau in tau_values}

    # # Load each JSON file and combine the data
    # for i, file_name in enumerate(file_names):
    #     with open(file_name, 'r') as f:
    #         pcc_dict = json.load(f)
        
    #     # Append the PCC values for each normalization + activation function combination
    #     for norm_act, value in pcc_dict.items():
    #         combined_data[f'tau={tau_values[i]}'].append(value)

    # # Convert the combined data into a DataFrame
    # df = pd.DataFrame(combined_data, index=list(pcc_dict.keys()))

    # Define the specific order for normalization methods and activation functions
    norm_methods_order = ['no-norm', 'z-score', 'min-max']
    activ_funcs_order = ['relu', 'tanh', 'sigmoid']

    # Create the specific order for the index (rows)
    row_order = [f'{norm} + {act}' for norm in norm_methods_order for act in activ_funcs_order]

    # Initialize a dictionary to hold the combined data
    combined_data = {f'tau={tau}': [] for tau in tau_values}

    # Load each JSON file and combine the data
    for i, file_name in enumerate(file_names):
        with open(file_name, 'r') as f:
            pcc_dict = json.load(f)
        
        # Append the PCC values for each normalization + activation function combination
        for norm_act in row_order:
            combined_data[f'tau={tau_values[i]}'].append(pcc_dict.get(norm_act, None))

    # Convert the combined data into a DataFrame with the specific order
    df = pd.DataFrame(combined_data, index=row_order)


    # Create the heatmap
    if ax is None:
        plt.figure(figsize=(6, 3))
    sns.heatmap(df, cmap='Blues', vmax=1, ax=ax)
    ax.set_title('PCC')
    ax.set_yticklabels([])
    ax.set_xticklabels([1,5,7.5,100], rotation=0)
    # ax.set_xticklabels(ax.get_xticks(), rotation = 90)

    # ax.set_xticks
    ax.set_xlabel(r'$\tau$')
    if ax is None:
        plt.show()

def linear_nullcline_analysis():
    """Plots a heatmap showing PCC and Nullcline error for changing Tau"""
    fig, axs = plt.subplots(1,2)
    fig.set_figheight(3)
    fig.set_figwidth(6)

    linear_nullcline_analysis_mse(axs[0])
    linear_nullcline_analysis_pcc(axs[1])

    plt.tight_layout()

    plt.subplots_adjust(top=0.907,
    bottom=0.149,
    left=0.249,
    right=0.936,
    hspace=0.2,
    wspace=0.21)

    import matplotlib as mpl
    mpl.rc("savefig", dpi=300)
    # plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\MSE_vs_VAL_tot_all_88_0.01_499_40_7.5.png")
    plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\LinearNullcline\LinearNullclineHeatmap_tau_[8.8]100epochs.png")


    plt.show()




if __name__ == '__main__':
    # save_all_MSE_vs_VAL(learning_rate = 0.01, nodes = [8,8], layers=2, max_epochs=99, option='option_1', amount_per_parameter=40, save=True)
    # open_csv_and_plot_all(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, total=360, study_lc=False)
    # three_by_three_plot(learning_rate = 0.01, nodes = [8,8], layers=2, max_epochs=99, option='option_1', amount=40)

    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[8,8], layers=2, max_epochs=499, option='option_3', normalization_method=['no-norm'], activation_functions=['relu'], amount_per_parameter=40, save=True)
    
    # save_all_MSE_vs_VAL(learning_rate = 0.01, nodes = [8,8], layers=2, max_epochs=99, option='option_1', amount_per_parameter=40, save=True)
    # open_csv_and_plot_all(option='option_1', learning_rate=0.01, max_epochs=99, nodes=[8,8], layers=2, total=360, study_lc=False)
    # three_by_three_plot(learning_rate = 0.01, nodes = [8,8], layers=2, max_epochs=99, option='option_1', amount=40)

    # linear_nullcline_analysis()
    # linear_nullcline_analysis_pcc()

# Example 1: (full analysis of one specific layer (ex. [8,8]))
    # Firstly save:
    # save_all_MSE_vs_VAL(learning_rate = 0.01, nodes = [8,8], layers=2, max_epochs=99, option='option_2', amount_per_parameter=40, save=True)

    # # Then Plot:
    # three_by_three_plot(learning_rate = 0.01, nodes = [8,8], layers=2, max_epochs=499, option='option_3', amount=40)
    # three_by_three_plot(learning_rate = 0.005, nodes = [8,8], layers=2, max_epochs=999, option='option_3', amount=40)
    

    # or Plot: (When adding Limit Cycle: Must select manually which models to take)
    # open_csv_and_plot_all(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[8,8], layers=2, total=360, study_lc=False)

# Example 2: one normalization method, two activation functions (one act works as well for saving)
    # # Firstly save
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[4,4], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[8,8], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[16,16], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[4,4,4,4], layers=4, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[8,8,8,8], layers=4, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[16,16,16,16], layers=4, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[4,4,4,4,4,4,4,4], layers=8, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[8,8,8,8,8,8,8,8], layers=8, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[16,16,16,16,16,16,16,16], layers=8, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4], layers=16, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8], layers=16, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16], layers=16, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)

    # plot_option_1 for (seperate) in BigMSE
    # big_MSE_vs_VAL_for_one_norm_two_act(option='option_3', learning_rate=0.01, max_epochs=499, amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'], plot_option='0')
    # big_MSE_for_one_norm_two_activation(option='option_3', learning_rate=0.01, max_epochs=499, amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'], plot_option='1')

    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[4,4], layers=2, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[8,8], layers=2, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[16,16], layers=2, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[4,4,4,4], layers=4, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[8,8,8,8], layers=4, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[16,16,16,16], layers=4, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[4,4,4,4,4,4,4,4], layers=8, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[8,8,8,8,8,8,8,8], layers=8, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[16,16,16,16,16,16,16,16], layers=8, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4], layers=16, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8], layers=16, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16], layers=16, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])


    # specific_MSE_for_one_norm_two_activation(option='option_3', learning_rate=0.01, max_epochs=499, amount=40, normalization_methods=['min-max'], activation_functions=['relu'], plot_option='1')

    # specific_MSE_for_one_norm_two_activation(option='option_3', learning_rate=0.01, max_epochs=499, amount=40, normalization_methods=['min-max'], activation_functions=['relu'], plot_option='1')
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[4,4], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['sigmoid'], amount_per_parameter=40, save=True)

    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=2*[4], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=2*[8], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=2*[16], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['sigmoid'], amount_per_parameter=40, save=True)


    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[8,8], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['sigmoid'], amount_per_parameter=40, save=True)

    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[16,16], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['sigmoid'], amount_per_parameter=40, save=True)

    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=16*[4], layers=16, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu'], amount_per_parameter=40, save=True)


    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=16*[4], layers=16, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4], layers=16, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[8,8,8,8], layers=4, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu'], amount_per_parameter=40, save=True)

    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[16,16], layers=2, max_epochs=499, option='option_1', normalization_method=['min-max'], activation_functions=['relu'], amount_per_parameter=40, save=True)
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[16,16], layers=2, max_epochs=499, option='option_1', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # specific_MSE_for_one_norm_two_activation(option='option_1', learning_rate=0.01, max_epochs=499, amount=40, normalization_methods=['min-max'], activation_functions=['relu'], plot_option='1')
    # Val_vs_MSE_node_norm_act_plot(learning_rate=0.01, nodes=[16,16], layers=2, max_epochs=499, option='option_1', amount=40, normalization_method='min-max', activation_function='relu', remove_biggest_validation_outlier=False, search_activation_functions=['relu'])


    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8], layers=16, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16], layers=16, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)


    # Then plot:
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4], layers=16, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8], layers=16, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16], layers=16, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])

    # 
    # Or Plot: (I think better not to use this one, is more focussed for 9 (3x3) parameters)
    # open_csv_and_plot_all(option='option_3', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, total=80, study_lc=False, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])
    # 
    # big_MSE_for_one_norm_two_activation(option='option_3', learning_rate=0.01, max_epochs=499, amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'], plot_option='1')
    # big_MSE_vs_VAL_for_one_norm_two_act(option='option_3', learning_rate=0.01, max_epochs=499, amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'], plot_option='0')
    # big_MSE_for_one_norm_two_activation, big_MSE_vs_VAL_for_one_norm_two_act, plot_validation_vs_mse_one_norm_two_act

# Example 3: for one specific node and specific norm and activation function plot, and remove biggest validation outlier:
    # Val_vs_MSE_node_norm_act_plot(learning_rate=0.01, nodes=[16,16], layers=2, max_epochs=499, option='option_4', amount=40, normalization_method='min-max', activation_function='relu', remove_biggest_validation_outlier=False)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[4,4], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu'], amount_per_parameter=40, save=True)
    # Val_vs_MSE_node_norm_act_plot(learning_rate=0.01, nodes=[4,4], layers=2, max_epochs=499, option='option_3', amount=40, normalization_method='min-max', activation_function='relu', remove_biggest_validation_outlier=False)
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[4,4], layers=2, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu'])


# Example 4: When only running [4,4] [8,8], [16,16] of minmax relu for specific tau:
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[4,4], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[8,8], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu'], amount_per_parameter=40, save=True)
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[16,16], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu'], amount_per_parameter=40, save=True)

    # specific_MSE_for_one_norm_two_activation(option='option_3', learning_rate=0.01, max_epochs=499, amount=40, normalization_methods=['min-max'], activation_functions=['relu'], plot_option='1')
    # plot_validation_vs_mse_one_norm_one_act_one_layer_three_nodes(learning_rate=0.01, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu'])






    # only working with 1 norm 1 act?
    save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[16,16], layers=2, max_epochs=499, option='option_1', normalization_method=['min-max'], activation_functions=['relu'], amount_per_parameter=40, save=True)
    # plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[16,16], layers=2, max_epochs=499, option='option_1', amount=40, normalization_methods=['min-max'], activation_functions=['relu'])

    # voor relu en sigmoid
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[4,4], layers=2, max_epochs=499, option='option_3', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)


    pass