# Neural Network using FitzHugh-Nagumo
# heads-up: previously used v, w -> v, u (x-axis, y-axis) here
# when looking at the picture that was made by the promotor on 13/10 we see that u is on vertical axis and v on horizontal
#   on wikipedia (https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model) w is vertical and v is vertical

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import save_model, load_model
from keras import utils
import matplotlib.pyplot as plt
from novak_tyson_time import compute_novak_dynamics
import pandas as pd
import uuid
import os
import time

# calculating derivatives using finite difference method
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

def calculate_derivatives(values, h):
    forward_deriv = forward_difference(values, h, begin=0, end = len(values)-1)
    backward_deriv = backward_difference(values, h, begin=len(values)-1, end=len(values))

    return forward_deriv + backward_deriv

# normalization_options
def normalization(data: np.ndarray, normalization_method: str='z-score'):
    """
    Perform Z-score normalization (standardization) on the given data.

    Parameters:
    - data: NumPy array or list, the data to be normalized.

    Returns:
    - standardized_data: NumPy array, the Z-score normalized data.
    """
    data = np.array(data)
    if normalization_method == 'z-score':
        mean_val = np.mean(data)
        std_dev = np.std(data)
    elif normalization_method == 'min-max':
        mean_val = np.min(data)
        std_dev = np.max(data) - np.min(data)
    elif normalization_method == 'no-norm':
        mean_val = 0
        std_dev = 1
    else:
        raise ValueError("Invalid normalization method. Please choose 'z-score' or 'min-max' or 'no-norm'.")

    standardized_data = (data - mean_val) / std_dev    
    return standardized_data, mean_val, std_dev

def normalization_with_mean_std(data, mean_std):   # used when we want to normalize data again (see FitzHugh_Nagumo_ps.py)
    """
    Perform normalization of data with already known mean and standard deviation:
    Typically used when we want to feed data into a model, first we need to normalize this data.
    """
    return ( data - mean_std[0] ) / mean_std[1]

def reverse_normalization(standardized_data, mean_std):
    """
    Reverse the normalization (standardization) operation.

    Parameters:
    - standardized_data: NumPy array or list, the Z-score normalized data.
    - mean_val: float, the mean value used for standardization.
    - std_dev: float, the standard deviation used for standardization.

    Returns:
    - original_data: NumPy array, the original data before normalization.
    """
    reversed_data = standardized_data * mean_std[1] + mean_std[0]
    return reversed_data

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

def nullcline_choice(train_u, val_u, train_v, val_v,
              train_u_dot, val_u_dot, train_v_dot, val_v_dot,
              u_nullcline: True, u_ifo_v: True):
    """Chooses from all the data the correct ones for training for the specific nullcline we want to remake. 
    
    Eg. When training option_1 (so u_dot=0 and u(v)) we would like to have as input: udot and v

    Parameters:
    -u_nullcline: Bool which is True if we want to train the NN to reproduce the udot=0 nullcline.
    -u_ifo_v: Bool which is True if we want to calculate the nullcine in as u(v), False for v(u).

    Returns:
        tuple:
        column stack of x_train and x_dot_train,
        column stack of x_validation and x_dot_validation,
        y_train, 
        y_validation
    """
    if u_nullcline:
        x_dot_train = train_u_dot
        x_dot_val = val_u_dot
    else:
        x_dot_train = train_v_dot
        x_dot_val = val_v_dot
    
    if u_ifo_v: # u = f(v)
        x_train = train_v
        x_val = val_v
        y_train = train_u
        y_val = val_u
    else:       # v = g(u) 
        x_train = train_u
        x_val = val_u
        y_train = train_v
        y_val = val_v

    alpha_betadot_data_train = np.column_stack((x_train, x_dot_train))
    alpha_betadot_data_val = np.column_stack((x_val, x_dot_val))

    return alpha_betadot_data_train, alpha_betadot_data_val, y_train, y_val

def train_and_save_losses(model, X_train, X_val, y_train, y_val, save, epochs, nodes, layers, learning_rate, normalization_method, activation_function, option, mean_std, batchsize=32):
    """
    Plots the loss of error for validation and training data.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize, validation_data=(X_val, y_val), verbose=0)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    if save:
        save_data_in_dataframe(train_loss, val_loss, nodes, layers, learning_rate, normalization_method, activation_function, model, option, mean_std)
    return model


def make_dataframe(train_loss, validation_loss, nodes, layers, learning_rate, normalization, activation_function, modelname, option, mean_std):
    length_data = len(train_loss)
    epochs = np.arange(length_data)
    new_data = {'epoch': epochs,
            'normalization_method': [normalization] * length_data,
            'activation_function': [activation_function] * length_data,
            'nodes': [nodes] * length_data,
            'layers': [layers] * length_data,
            'learning_rate': [learning_rate] * length_data,
            'loss': train_loss,
            'validation': validation_loss,
            'modelname': [modelname] * length_data,
            'option': [option] * length_data,
            'mean_std': [mean_std] * length_data
            }
    # we still need to add the 'run' column, this is code in 'concatenating_dataframes' function
    df_new_data = pd.DataFrame(new_data)
    return df_new_data


def save_kerasmodel(model):
    """
    Saves the Neural Network Kerasmodel in a file whose name is a unique ID.
    """
    unique_model_name = uuid.uuid4().hex    # https://stackoverflow.com/questions/2961509/python-how-to-create-a-unique-file-name/44992275#44992275

    # saving in right file
    absolute_path = os.path.dirname(__file__)
    relative_path = "saved_NN_models_NT"
    folder_path = os.path.join(absolute_path, relative_path)
    full_path = os.path.join(folder_path, unique_model_name + '.h5')
    # save_model(model, unique_model_name+'.h5') # alternative: time
    save_model(model, full_path)
    # print("This model is saved with identifier:", unique_model_name) # this print statement is bad, because when printing (and want to break in between, the )
    return unique_model_name

def to_list(val):
    return list(pd.eval(val))

def save_data_in_dataframe(train_loss, validation_loss, nodes, layers, learning_rate, normalization_method, activation_function, model, option, mean_std):
    """
    Saves the newly made data in the already existing dataframe. It also saves the model that was used in another file.
    """
    # load df from right folder
    absolute_folder_path = os.path.dirname(__file__)
    begin_name_file = "FHN_NN_loss_and_model"
    name_file_add_on = f"_NT"
    name_file_extension = ".csv"
    name_file = begin_name_file + name_file_add_on + name_file_extension
    output_path = os.path.join(absolute_folder_path, name_file)
    df = pd.read_csv(output_path, converters={'nodes': to_list}) # converters needed such that list returns as list, not as string (List objects have a string representation, allowing them to be stored as . csv files. Loading the . csv will then yield that string representation.)

    modelname = save_kerasmodel(model)
    new_df = make_dataframe(train_loss, validation_loss, nodes, layers, learning_rate, normalization_method, activation_function, modelname, option, mean_std)

    concatenated_df = concatenate_dataframes(df, new_df, normalization_method, activation_function, nodes, layers, option, learning_rate)

    concatenated_df.to_csv(output_path, index=False)

def new_highest_run_calculator(df):
    """Calculates the 'run' value for in the new dataframe. It looks at what the run number was previously and adds one. If there is no other run it starts at zero."""
    if df.empty:
        highest_run = 0
    else:
        highest_run = max(df["run"]) + 1
    return highest_run


def concatenate_dataframes(existing_df, new_df, normalization_method, activation_function, nodes, layers, option, learning_rate):
    """
    First we add the 'run' column to the newly made dataframe, later we concatenate the exisiting dataframe with the newly made.
    """
    # Here we check if those sets of parameters were chosen already before, this is used to determine the amount of 'runs' with these parameters
    # check same values
    sub_df = existing_df.loc[(existing_df["normalization_method"]==normalization_method)
                        & (existing_df["activation_function"]==activation_function)
                        & (existing_df["layers"]==layers)
                        & (existing_df["option"]==option)
                        & (existing_df["learning_rate"]==learning_rate)
                        ]       # https://www.statology.org/pandas-select-rows-based-on-column-values/
    # to compare lists we use apply
    sub_df = sub_df[sub_df["nodes"].apply(lambda x: x==nodes)]

    # check the 'run' number
    new_highest_run = new_highest_run_calculator(sub_df)
    num_rows = new_df.shape[0]
    new_df["run"] = [new_highest_run] * num_rows

    # Now our new dataframe is fully constructed, we concatenate with already existing.
    concatenated_df = pd.concat([existing_df, new_df], ignore_index=True)

    return concatenated_df


def create_neural_network_and_save(num_layers=1, nodes_per_layer=None, activation_function: str='relu', learning_rate=0.01, option: str='option_1', normalization_method: str='z-score', save: bool=True, epochs: int=100, seed=None, batchsize=32):
    '''
    Program that creates the neural network from FHN data and saves it.

    num_layers:
        Number of hidden layers that is wanted: minimum is 1
    --------
    option: 
        option_1: u_dot=0 and u(v), option 2: u_dot=0 and v(u), option 3: v_dot=0 and u(v), option 4: v_dot=0 and v(u)
    '''
    # Check if input of nodes and layers is correct
    if nodes_per_layer is None: # list is mutable object: https://stackoverflow.com/questions/50501777/why-does-tensorflow-use-none-as-the-default-activation , https://medium.com/@inexturesolutions/top-common-python-programming-mistakes-and-how-to-fix-them-90f0a8bcce43 (mistake 1)
        nodes_per_layer = [10]
    elif type(nodes_per_layer) == int: # if same #nodes for all layers
        nodes_per_layer = np.ones(num_layers) * nodes_per_layer
    if (num_layers < 1) or (type(num_layers) != int):
        assert False, "Please make sure the number of layers is an integer greater than zero."
    if len(nodes_per_layer) != num_layers:
        assert False, f"Please make sure the number of nodes per (hidden)layer (={len(nodes_per_layer)}) are equal to the amount of (hidden) layers (={num_layers})."
    if activation_function != 'relu' and activation_function != 'tanh' and activation_function != 'sigmoid':
        assert False, "Please choose as activation function between 'relu', 'tanh' or 'sigmoid'."
    if seed is None:
        assert False, 'Please use seed, seed is None'

    print("Using:\n",
          f"Number of layers = {num_layers}\n",
          f"Nodes per layer {nodes_per_layer}\n",
          f"Normalization method {normalization_method}\n",
          f"Activation function {activation_function}\n",
          f"Learning rate {learning_rate}\n",
          f"Option {option}\n",
          f"Epochs {epochs}\n"
          )

    # creating the data of the FHN system used for training and validating
    time, v_t_data, u_t_data = compute_novak_dynamics() # assigning v->v, w->v see heads-up above.
    u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
    v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

    # normalizing the data
    u_t_data_norm, mean_u, std_u = normalization(u_t_data, normalization_method)  # mean_u, and std_u equal x_min and (x_max - x_min) respectively when doing min-max normalization
    v_t_data_norm, mean_v, std_v = normalization(v_t_data, normalization_method) 
    u_dot_t_data_norm, mean_u_dot, std_u_dot = normalization(u_dot_t_data, normalization_method)
    v_dot_t_data_norm, mean_v_dot, std_v_dot  = normalization(v_dot_t_data, normalization_method)
    mean_std = {"u_t_data_norm":[mean_u, std_u], "v_t_data_norm":[mean_v, std_v], "u_dot_t_data_norm": [mean_u_dot, std_u_dot], "v_dot_t_data_norm": [mean_v_dot, std_v_dot]}
    print(f"1) {len(u_dot_t_data), len(u_dot_t_data_norm)}")
    # Creating Neural Network (no training yet)
    # Step : Seed selection
    utils.set_random_seed(seed)

    # Step 2: Build the Modelstructure
    model = Sequential()
    model.add(Dense(nodes_per_layer[0], input_dim=2, activation=activation_function)) # 1 input dimension
    for i in range(1, num_layers):
        model.add(Dense(nodes_per_layer[i], activation=activation_function))
    model.add(Dense(1, activation='linear'))

    # Step 3: Compile the model (choose optimizer..)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate)) # default optimizer Adam, learning rate 0.01 common starting point

    # Step 4: Seperate training and validation data
    train_u, val_u, train_v, val_v, train_u_dot, val_u_dot, train_v_dot, val_v_dot = split_train_validation_data_seed(u_t_data_norm, v_t_data_norm, u_dot_t_data_norm, v_dot_t_data_norm, validation_ratio=0.2, seed=seed)
    print(f"2) {len(train_u), len(val_u), len(train_u_dot), len(val_u_dot)}")

    # Step 5: Train Model & Plot
    if option == 'option_1':
        u_nullcline = True  # u_dot = 0
        u_ifo_v = True  # u = f(v)
    elif option == 'option_2':
        u_nullcline = True  # u_dot = 0
        u_ifo_v = False  # v = g(u)
    elif option == 'option_3':
        u_nullcline = False  # v_dot = 0
        u_ifo_v = True  # u(v)
    elif option == 'option_4':
        u_nullcline = False  # v_dot = 0
        u_ifo_v = False  # v(u)        
    
    X_train, X_val, y_train, y_val = nullcline_choice(train_u, val_u, train_v, val_v,
                                        train_u_dot, val_u_dot, train_v_dot, val_v_dot,
                                        u_nullcline, u_ifo_v)

    # now that is chosen with which nullcine and in which way u(x) or v(x) we want to train, the neural network is trained and saved:
    train_and_save_losses(model, X_train, X_val, y_train, y_val, save, epochs, nodes_per_layer, num_layers, learning_rate, normalization_method, activation_function, option, mean_std, batchsize)

if __name__ == '__main__':
# -------------------------------------- NEXT PLANNED: -------------------------------------------------------------------------------------------

# ================================
    start_total_time = time.time()

    for seed in range(20,40):

        print(f"\n ROUND NUMBER STARTING {seed} \n")

        start_time = time.time()

        create_neural_network_and_save(normalization_method='min-max', activation_function='relu', nodes_per_layer = [16,16], num_layers=2, learning_rate=0.01, epochs=500, option='option_3', seed=seed)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time mini part", elapsed_time, "seconds \n\n")

        create_neural_network_and_save(normalization_method='min-max', activation_function='sigmoid', nodes_per_layer = [16,16], num_layers=2, learning_rate=0.01, epochs=500, option='option_3', seed=seed)

        print("Elapsed time second mini part", time.time() - start_time, "seconds \n\n")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time one seed", elapsed_time, "seconds \n\n")

    end_total_time = time.time()

    elapsed_total_time = end_total_time - start_total_time

    print("Elapsed time everything", elapsed_total_time, "seconds \n\n")




"""
Time Management Summary:



1. Neural Network Training: 
    - function: create_neural_network_and_save(normalization_method='no-norm', activation_function='relu', nodes_per_layer = [8,8], num_layers=2, learning_rate=0.01, epochs=100, option='option_3')
    
    # - [100 epochs]: 
    #     - Time per configuration: 
    #     - Total time for 9 configurations:  seconds ( seconds per configuration)
    #     - (for option_4 it took  seconds per configuration,  seconds for 9, and  hours 9x20)
    
    - [500 epochs]:
        - Time per configuration: 170 seconds
        - Total time for 2 configurations: 330 seconds
          2 configurations 20 times = 6612 seconds
    
    # - [1000 epochs]:
    #     - Time per configuration: 450 seconds
    #     - Total time for 9 configurations: 3400 seconds
    
2. Expected Times for Simulation Runs:
    - [1000 epochs]:
        - Expected time: 500 seconds (8.3 minutes)
        - Number of simulations in 8 hours: 57
        
    - [500 epochs]:
        - Expected time: 250 seconds (4 minutes)
        - Number of simulations in 8 hours: 110
        
    - [250 epochs]:
        - Expected time: 125 seconds (2 minutes)
        - Number of simulations in 8 hours: 220
        
    - [100 epochs]:
        - Expected time: 50 seconds (1 minute)
        - Number of simulations in 8 hours: 570
        
3. Summary for Different-Layer Neural Networks:
    - Total time for 1 configuration: relu, no-norm (16 layers, 8 nodes, 500 epochs): 400 seconds
    - Total time for 9 configuration: relu, no-norm (16 layers, 8 nodes, 500 epochs): 3666 seconds

    - Total time for 1 configurations: relu, no-norm (8 layers, 8 nodes 500 epochs]: 320 seconds
    - Total time for 9 configurations: relu, no-norm (8 layers, 8 nodes 500 epochs]: 2600 seconds

    - Total time for 1 configurations: (2 layers, 4 nodes, 500 epochs): 411 seconds

    - Total time for two pairs (2 layers, 16 nodes, 500 epochs) but 40 times: 8 hours
"""

"""
TAU = 100

1) Expected times
    [8,8], 0.01, 500, opt3
    1 configuration: 146 sec
    9 configurations: 1341 sec
    20x9 configurations = 7.86 uur (28300 sec)

    [4,4,4,4], [8,8,8,8], [16,16,16,16]
    40 x 2 (relu/sigmoid) configurations = 16 uur (57_000 sec)

    [4,4,4,4,4,4,4,4], [8,8,8,8,8,8,8,8], [16,16,16,16,16,16,16,16]
    40 x 2 (relu/sigmoid) configurations = 19 uur (70_000sec)

    16 layers van 4 nodes, 8 nodes en 16 nodes:
    40 x 2 (relu/sigmoid) configurations = 30 uur (100_000 sec)

"""

"""
TAU = 80
1) Without seeds
    [8,8] 0.01, 500 opt3
    minmax
    relu: 168 sec, 170 sec
    tanh: 164 sec, 171 sec
    
1) With seeds
    [8,8] 0.01, 500 opt3
    minmax
    relu: 155 sec, 161 sec
    tanh: 157 sec, 162 sec

"""

"""
TAU = 20

relu minmax, 0.01 500
1) [4,4]
    1 time: 163 seconds

2) [8,8]
    1 time: 171 seconds

3) [16,16]
    1 time: 167 seconds

40 times means total:
5.4 uur
"""