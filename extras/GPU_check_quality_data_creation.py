# Quality of GPU and GPU speed will be checked using 1) batchsize and 2) amount of in real life time

"""
Previously done:
checked if validation error evoolves correctly in time. => Names not always accurate
"""


import numpy as np
from data_generation_exploration.FitzHugh_Nagumo_t import compute_fitzhugh_nagumo_dynamics
from model_building.create_NN_FHN import calculate_derivatives, normalization, split_train_validation_data_seed, nullcline_choice
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import save_model, load_model
from keras import utils
import matplotlib.pyplot as plt
import math
import os

import sys
sys.path.append('../../Master_Thesis') # needed to import settings
from settings import TAU

PATIENCE = 200

def save_results(batchsize, time, val_loss, epochs, filename=f"results_GPU_batchsize_patience{PATIENCE}_TAU100_[16,16]_8.json"):
    result = {
        "batchsize": batchsize,
        "time": time,
        "val_loss": val_loss,
        "epochs": epochs
    }
    
    # Read existing data
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []

    # Append new result
    data.append(result)
    
    # Write updated data
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
        
# def save_results(batchsize, time, val_loss, epochs, filename=f"results_GPU_batchsize_patience{PATIENCE}_TAU{TAU}_[16,16].json"):
#     result = {
#         "batchsize": batchsize,
#         "time": time,
#         "val_loss": val_loss,
#         "epochs": epochs
#     }

#     # Split filename into base and extension
#     base, ext = os.path.splitext(filename)

#     # Check if the file exists and modify the name if necessary
#     counter = 1
#     filename = f"{base}_{counter}{ext}"

#     while os.path.exists(filename):
#         counter += 1
#         filename = f"{base}_{counter}{ext}"

#     # Read existing data
#     try:
#         with open(filename, 'r') as file:
#             data = json.load(file)
#     except FileNotFoundError:
#         data = []

#     # Append new result
#     data.append(result)
    
#     # Write updated data
#     with open(filename, 'w') as file:
#         json.dump(data, file, indent=4)

def load_results(filename=f"results_GPU_batchsize_patience{PATIENCE}_[16,16].json"):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        return []


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
    time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics() # assigning v->v, w->v see heads-up above.
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
    val_loss, trained_epochs = train_and_visualize_losses(model, X_train, X_val, y_train, y_val, save, epochs, nodes_per_layer, num_layers, learning_rate, normalization_method, activation_function, option, mean_std, batchsize)
    return val_loss, trained_epochs

def train_and_visualize_losses(model, X_train, X_val, y_train, y_val, save, epochs, nodes, layers, learning_rate, normalization_method, activation_function, option, mean_std, batchsize=32):
    """
    Plots the loss of error for validation and training data.
    """
    # tot optimum
    callbacks = EarlyStopping(monitor='val_loss', patience=PATIENCE)
    epochs = 1_000_000
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize, validation_data=(X_val, y_val), verbose=0, callbacks=callbacks)

    # history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize, validation_data=(X_val, y_val), verbose=0)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    if True:
        plt.plot(epochs, train_loss, color='red', label='train')
        plt.plot(epochs, val_loss, color='blue', label='val')
        plt.scatter(epochs[-101], val_loss[-101], color='blue')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'Error_BS{batch_size}_patience{PATIENCE}_[16,16]_5')
        plt.clf()
        # plt.show()
        # plt.plot(epochs[0:-100], val_loss[0:-100])

        # plt.show()

    return val_loss[-101], epochs[-101]


import time
import json

run_already = False

if not run_already:
    seed = 0
    # batchsizes = [2, 4, 8, 16, 32, 50, 64, 100, 128, 200, 256, 400, 512, 800, 1024, 1600, 2048, 3200, 4096, 6400, 8192, 12000]
    # batchsizes = [8,16, 32, 50, 64, 100, 128, 200, 256, 400, 512, 800, 1024, 1600, 2048, 3200, 4096, 6400, 8192, 12000]
    batchsizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for batch_size in batchsizes:
        start_time = time.time()
        min_val_loss, epochs_min = create_neural_network_and_save(save=False, normalization_method='min-max', activation_function='relu', nodes_per_layer = [16,16], num_layers=2, learning_rate=0.01, epochs=100, option='option_3', seed=seed, batchsize=batch_size)
        end_time = time.time()

        total_time = end_time - start_time
        batchsize = batch_size
        lowest_validation_error = min_val_loss
        epoch_at_low = epochs_min
        print(batchsize, total_time, lowest_validation_error, epoch_at_low)
        save_results(batchsize, total_time, lowest_validation_error, epoch_at_low)

# print("batchsize", batchsize, end_time - start_time)

assert False, "The data processing and visualising part can be done in file: GPU_check_quality_data_analysis.py"
results = load_results()

batchsizes = []
times = []
lowest_val_errors = []
epochs_at_lowest_val = []

for result in results:
    print(result)
    print("\n")
    
    batchsizes.append(result['batchsize'])
    times.append(result['time'])
    lowest_val_errors.append(result['val_loss'])
    epochs_at_lowest_val.append(result['epochs'])

# normalize
max_comp_time = max(times)
times_norm = [comp_time / max(times) for comp_time in times]

log_lowest_val_errors = np.log10(lowest_val_errors)
max_lowest_val_errors_log = max(log_lowest_val_errors)
min_lowest_val_errors_log = min(log_lowest_val_errors)
lowest_val_errors_norm = [ (error - min_lowest_val_errors_log ) / (max_lowest_val_errors_log - min_lowest_val_errors_log )  for error in log_lowest_val_errors] # maybe add log later

max_epochs = max(epochs_at_lowest_val)
epochs_at_lowest_val_norm = [epoch / max_epochs for epoch in epochs_at_lowest_val]

fig, ax = plt.subplots(1, 3)
ax[0].plot(batchsizes, times, color='green', label='time')
ax[1].plot(batchsizes, log_lowest_val_errors, color='blue', label='val error', marker='o')
ax[2].plot(batchsizes, epochs_at_lowest_val, color='red', label='epoch')

plt.legend()
plt.show()

plt.plot(batchsizes, times_norm, color='green', label='time')
plt.plot(batchsizes, lowest_val_errors_norm, color='blue', label='val error', marker='*')
plt.plot(batchsizes, epochs_at_lowest_val_norm, color='red', label='epoch')

mask = [batchsize%2==0 for batchsize in batchsizes]
print(mask)
times_norm_select = times_norm[mask]
print(times_norm_select)

plt.legend()
plt.show()



# print(results)
# print(type(results))
# print(type(results[0]))
# print("\n \n \n")
# print(results[0])