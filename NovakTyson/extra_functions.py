import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import seaborn as sns
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
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
