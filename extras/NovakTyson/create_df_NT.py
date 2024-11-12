# Program that makes the dataframe with the right column names.
# This program should only be run ONCE, to create it. Do not run it again when a dataframe already exists, this will overwrite it and all data will be lost.

import pandas as pd
import os

remake_dataframe = input(f"Are you sure you want to overwrite the current made dataframe for Novak & Tyson? Type 'yes' ")

if remake_dataframe == "yes":
    print("Rebuilding...")
    columns = ['run', 'normalization_method', 'activation_function', 'learning_rate', 'nodes', 'layers', 'epoch', 'loss', 'validation', 'modelname', 'option', 'mean_std']
    # als je verder zou gaan met een bepaalde epoch, zorg er dan voor dat die dezelfde run number krijgt

    # Create an empty DataFrame
    df = pd.DataFrame(columns=columns)

    # save in right folder
    # relative_folder_path = 'Master-Thesis'
    relative_folder_path = os.path.join('Master-Thesis', 'NovakTyson')


    if not os.path.exists(relative_folder_path):
        assert False, 'This relative folder path  does not exist.'

    name_file = f"FHN_NN_loss_and_model_NT.csv"
    output_path = os.path.join(relative_folder_path, name_file)

    df.to_csv(output_path, index=False)
    print("Done")
