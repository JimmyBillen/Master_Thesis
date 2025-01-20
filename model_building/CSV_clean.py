# provides overview of what is contained in the CSV 'FHN_NN_loss_and_model*.csv'


import os
from ast import literal_eval
import pandas as pd

import sys
sys.path.append('../../Master_Thesis') # needed to import settings
from settings import TAU, NUM_OF_POINTS

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
   relative_path = f"FHN_NN_loss_and_model_{TAU}_{NUM_OF_POINTS}.csv"
   csv_name = os.path.join(absolute_path, relative_path)

   print('wordt niet gesaved')
  #  new_df.to_csv(csv_name, index=False) # soms uitzetten voor veiligheid

   count_everything(new_df)


if __name__ == "__main__":
  count_everything()


# Ik heb op 'FHN_NN_loss_and_model_old.csv' onderstaande 12 programmas toegepast om alle 1 run's te verwijderen
# 
# # remove_rows('min-max', 'relu', 0.010, "[8,8]", 2, "option_1", max_epochs=999) DONE
# # remove_rows('min-max', 'sigmoid', 0.010, "[8,8]", 2, "option_1", max_epochs=999) DONE
# # remove_rows('min-max', 'tanh', 0.010, "[8,8]", 2, "option_1", max_epochs=999) DONE

# # remove_rows('no-norm', 'relu', 0.005, "[8,8]", 2, 'option_1', max_epochs=99) DONE
# # remove_rows('no-norm', 'tanh', 0.005, "[8,8]", 2, 'option_1', max_epochs=99) DONE
# # remove_rows('no-norm', 'sigmoid', 0.005, "[8,8]", 2, 'option_1', max_epochs=99) DONE

# # remove_rows('z-score', 'relu', 0.005, "[8,8]", 2, 'option_1', max_epochs=99) DONE
# # remove_rows('z-score', 'tanh', 0.005, "[8,8]", 2, 'option_1', max_epochs=99) DONE
# # remove_rows('z-score', 'sigmoid', 0.005, "[8,8]", 2, 'option_1', max_epochs=99) DONE

# # remove_rows('min-max', 'relu', 0.005, "[8,8]", 2, 'option_1', max_epochs=99) DONE
# # remove_rows('min-max', 'tanh', 0.005, "[8,8]", 2, 'option_1', max_epochs=99) DONE
# remove_rows('min-max', 'sigmoid', 0.005, "[8,8]", 2, 'option_1', max_epochs=99) DONE
