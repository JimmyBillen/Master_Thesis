import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import save_model, load_model
from keras import utils
from ast import literal_eval
import numpy as np
import pandas as pd
import os
import uuid
import time
import seaborn as sns
import matplotlib as mpl

NUM_OF_POINTS = 15000

def to_list(val):
    return list(pd.eval(val))


def udot(u, v):
    return -u**3 + u**2 +u -v

def vdot(u, v):
    return -0.5*(v)**3+0.5*v**2-(1/3)*v+u

def sample_subarray(original_array, new_size):
    # Ensure the new size is less than the original array size
    if new_size >= len(original_array):
        raise ValueError("New size must be less than the original array size")

    # Calculate the size of the inner segment to sample
    inner_size = new_size - 2  # Exclude the first and last elements
    total_size = len(original_array) - 2  # Exclude the first and last elements
    
    # Generate indices for the inner segment
    inner_indices = np.linspace(1, total_size, inner_size, dtype=int)
    
    # Construct the sampled subarray
    sampled_array = [original_array[0]]  # Start with the first element
    sampled_array.extend(original_array[i] for i in inner_indices)  # Add the sampled points
    sampled_array.append(original_array[-1])  # End with the last element
    
    return np.array(sampled_array)

def compute_bicubic_dynamics():
    """
    Compute the dynamics of the FitzHugh-Nagumo model using Euler's method.
    """
    # Initial conditions 
    u0 = 1.0  # Initial value of u
    v0 = 2.0  # Initial value of v

    t0 = 0.0
    t_end = 51.50
    num_steps = NUM_OF_POINTS

    time = np.linspace(t0, t_end, num_steps + 1)
    h = (t_end - t0) / num_steps
    u_values = np.zeros(num_steps + 1)
    v_values = np.zeros(num_steps + 1)

    # Initialize the values at t0
    u_values[0] = u0
    v_values[0] = v0

    # Implement Euler's method
    for i in range(1, num_steps + 1):
        u_values[i] = u_values[i - 1] + h * udot(u_values[i - 1], v_values[i - 1])
        v_values[i] = v_values[i - 1] + h * vdot(u_values[i - 1], v_values[i - 1])
    
    return time, u_values, v_values

def plot_timeseries():
    # Plot the results
    time, v_values, w_values = compute_bicubic_dynamics()

    plt.figure(figsize=(10, 5))
    plt.plot(time, v_values, label='v(t)')
    plt.plot(time, w_values, label='w(t)')
    # print('for v, difference in time between maxima', find_maxima_differences(time, v_values) ,'\n')
    # print('for w, difference in time between maxima', find_maxima_differences(time, w_values))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    # plt.title(rf'FitzHugh-Nagumo in Function of Time for $\tau$={TAU}')
    plt.grid()
    plt.show()

def find_boundary_nullclines():
    # vdot nullcline (cubic): take lowest v-value and high v-value
    time, u, v = compute_bicubic_dynamics() # for bi-cubic: v->u, w->v

    # option 1
    # low_opt1_v, high_opt1_v = inverse_boundary_nullclines(time, u, v)
    low_opt1_v, high_opt1_v = [0], [0]

    # option 2
    low_opt2_w, high_opt2_w = calc_boundary_nullclines(time, u)

    # option 3
    # low_opt3_v, high_opt3_v = calc_boundary_nullclines(time, v)
    low_opt3_v, high_opt3_v= calc_boundary_nullclines(time, v)


    # option 4:
    # solving w=v-v^3/3 + RI => boundary v^2=1 => v=+-1, filling back in w: +-1-+1/3+R*I
    low_opt4_w = None
    high_opt4_w = None

    boundary_nullclines = {"option_1": [low_opt1_v, high_opt1_v],
                           "option_2": [low_opt2_w, high_opt2_w],
                           "option_3": [low_opt3_v, high_opt3_v],
                           "option_4": [low_opt4_w, high_opt4_w]}
    print(boundary_nullclines)
    return boundary_nullclines

def calc_boundary_nullclines(time, y):
    """Calculates the boundary of the nullcline
    
    for v: ifo v
    for w: ifo w
    """
    local_maxima_value = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            local_maxima_value.append(y[i])

    local_minima_value = []
    for j in range(1, len(y) - 1):
        if y[j] < y[j-1] and y[j] < y[j+1]:
            local_minima_value.append(y[j])

    # take second-last value
    low_limit = local_minima_value[-2]
    high_limit = local_maxima_value[-2]
    return low_limit, high_limit

def inverse_boundary_nullclines(time, y1, y2):
    """Calculates the boundary of the nullcline
    
    for v: ifo w (!)
    for w: ifo v (!)
    """
    local_maxima_value = []
    for i in range(1, len(y1) - 1):
        if y1[i] > y1[i - 1] and y1[i] > y1[i + 1]:
            local_maxima_value.append(y2[i])

    local_minima_value = []
    for j in range(1, len(y1) - 1):
        if y1[j] < y1[j-1] and y1[j] < y1[j+1]:
            local_minima_value.append(y2[j])

    # take second-last value
    low_limit = local_minima_value[-2]
    high_limit = local_maxima_value[-2]
    return low_limit, high_limit

def nullcline_udot(u):
    """u (=xcoord)"""
    return -u**3+u**2+u

def nullcline_vdot(v):
    return 0.5*v**3-0.5*v**2+(1/3)*v

def plot_limit_cycle(u_nullcline=True, y_ifo_x=True, model=None, with_neural_network=False, mean_std=None, plot=True):
    """ Plots the limit cycle with the model
    
    This is the old version before optimisation, new version is 'plot_limit_cycle_with_model' """
    # Plot Limit Cycle
    _, u_lc, v_lc = compute_bicubic_dynamics()
    if plot:
        plt.plot(u_lc, v_lc, 'r-', label=f'Limit Cycle')

    # Nullclines
        # vdot
    v = np.linspace(-2.5, 2.5, 1000)

    plt.plot(v, nullcline_udot(v), '--', color = "lime", label = r"$\dot{x}=0$ Nullcline")

        # wdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(nullcline_vdot(v), v, '--', color = "cyan", label = r"$\dot{y}=0$ Nullcline")

    #     # Plotting a dot where the nullclines intersect
    # dots = [0.409]
    # dots_null = [(i + A) / B for i in dots]
    # plt.plot(dots, dots_null, 'bo', label='Fixed Point')

    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.xlabel('u')
    plt.ylabel('v')
    plt.title('Phase Space of FitzHugh-Nagumo Model:\n Limit Cycle and Nullclines')
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.show()
    plt.clf()

    return None

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

    # creating the data of the bicubic system used for training and validating
    time, u_t_data, v_t_data = compute_bicubic_dynamics()
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

    # Step 4: Separate training and validation data
    train_u, val_u, train_v, val_v, train_u_dot, val_u_dot, train_v_dot, val_v_dot = split_train_validation_data_seed(u_t_data_norm, v_t_data_norm, u_dot_t_data_norm, v_dot_t_data_norm, validation_ratio=0.2, seed=seed)
    print(f"2) {len(train_u), len(val_u), len(train_u_dot), len(val_u_dot)}")

    # Step 5: Train Model & Plot
    if option == 'option_1':
        u_nullcline = True  # u_dot = 0
        u_ifo_v = True  # u = f(v)
        assert False, f'{option} not done, only option 2 and 3'
    elif option == 'option_2': # bicubic: udot nullcline
        u_nullcline = True  # u_dot = 0
        u_ifo_v = False  # v = g(u)
    elif option == 'option_3': # bicubic: vdot nullcline
        u_nullcline = False  # v_dot = 0
        u_ifo_v = True  # u(v)
    elif option == 'option_4':
        u_nullcline = False  # v_dot = 0
        u_ifo_v = False  # v(u)
        assert False, f'{option} not done, only option 2 and 3'

    
    X_train, X_val, y_train, y_val = nullcline_choice(train_u, val_u, train_v, val_v,
                                        train_u_dot, val_u_dot, train_v_dot, val_v_dot,
                                        u_nullcline, u_ifo_v)

    # now that is chosen with which nullcine and in which way u(x) or v(x) we want to train, the neural network is trained and saved:
    train_and_save_losses(model, X_train, X_val, y_train, y_val, save, epochs, nodes_per_layer, num_layers, learning_rate, normalization_method, activation_function, option, mean_std, batchsize)

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

def save_data_in_dataframe(train_loss, validation_loss, nodes, layers, learning_rate, normalization_method, activation_function, model, option, mean_std):
    """
    Saves the newly made data in the already existing dataframe. It also saves the model that was used in another file.
    """
    # load df from right folder
    absolute_folder_path = os.path.dirname(__file__)
    begin_name_file = "bi-cubic"
    name_file_add_on = f"_15000points"
    name_file_extension = ".csv"
    name_file = begin_name_file + name_file_add_on + name_file_extension
    output_path = os.path.join(absolute_folder_path, name_file)
    df = pd.read_csv(output_path, converters={'nodes': to_list}) # converters needed such that list returns as list, not as string (List objects have a string representation, allowing them to be stored as . csv files. Loading the . csv will then yield that string representation.)

    modelname = save_kerasmodel(model)
    new_df = make_dataframe(train_loss, validation_loss, nodes, layers, learning_rate, normalization_method, activation_function, modelname, option, mean_std)

    concatenated_df = concatenate_dataframes(df, new_df, normalization_method, activation_function, nodes, layers, option, learning_rate)

    concatenated_df.to_csv(output_path, index=False)

def save_kerasmodel(model):
    """
    Saves the Neural Network Kerasmodel in a file whose name is a unique ID.
    """
    unique_model_name = uuid.uuid4().hex    # https://stackoverflow.com/questions/2961509/python-how-to-create-a-unique-file-name/44992275#44992275

    # saving in right file
    absolute_path = os.path.dirname(__file__)
    relative_path = "saved_NN_models_bicubic"
    folder_path = os.path.join(absolute_path, relative_path)
    full_path = os.path.join(folder_path, unique_model_name + '.h5')
    # save_model(model, unique_model_name+'.h5') # alternative: time
    save_model(model, full_path)
    # print("This model is saved with identifier:", unique_model_name) # this print statement is bad, because when printing (and want to break in between, the )
    return unique_model_name

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
    print('1', sub_df)
    sub_df = sub_df[sub_df["nodes"].apply(lambda x: x==nodes)]
    print('2', sub_df)
    
    # print("DEBUGGING", nodes, type(nodes), sub_df["nodes"], type(sub_df["nodes"]), sub_df["nodes"])
    # sub_df = sub_df[sub_df["nodes"].apply(lambda x: np.array_equal(x, nodes))]

    # check the 'run' number
    new_highest_run = new_highest_run_calculator(sub_df)
    num_rows = new_df.shape[0]
    new_df["run"] = [new_highest_run] * num_rows

    # Now our new dataframe is fully constructed, we concatenate with already existing.
    concatenated_df = pd.concat([existing_df, new_df], ignore_index=True)

    return concatenated_df

def new_highest_run_calculator(df):
    """Calculates the 'run' value for in the new dataframe. It looks at what the run number was previously and adds one. If there is no other run it starts at zero."""
    if df.empty:
        highest_run = 0
    else:
        highest_run = max(df["run"]) + 1
    return highest_run

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
    relative_path = f"bi-cubic_15000points.csv"
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

def retrieve_model_from_name(unique_modelname):
    """Give the modelname and returns the keras.Model"""
    absolute_path = os.path.dirname(__file__)
    relative_path = "saved_NN_models_bicubic"
    folder_path = os.path.join(absolute_path, relative_path)
    full_path = os.path.join(folder_path, unique_modelname + '.h5')
    if not os.path.exists(full_path):
        assert False, f"The model with name {unique_modelname} cannot be found in path {full_path}"
    loaded_model = load_model(full_path)
    return loaded_model

def save_val_mse_df(df: pd.DataFrame, name):
    """
    Saves the dataframe which includes a column named 'MSE'.

    Note:
        This function is not used on its own.
    """
    absolute_path = os.path.dirname(__file__)
    relative_path = f"VAL_vs_MSE_bi-cubic"
    folder_path = os.path.join(absolute_path, relative_path)
    relative_path = f"{name}.csv"
    csv_name = os.path.join(folder_path, relative_path)

    df.to_csv(csv_name, index=False)

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

def normalization_with_mean_std(data, mean_std):   # used when we want to normalize data again (see FitzHugh_Nagumo_ps.py)
    """
    Perform normalization of data with already known mean and standard deviation:
    Typically used when we want to feed data into a model, first we need to normalize this data.
    """
    return ( data - mean_std[0] ) / mean_std[1]

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

def nullcline_and_boundary(option, amount_of_points):
    nullclines_per_option = find_boundary_nullclines()
    if option == 'option_2':
        bound_nullcline = nullclines_per_option['option_2']
        q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline), amount_of_points)
        # input("please check if boundaries nullcline are" + str(np.min(bound_nullcline)) + "," + str( np.max(bound_nullcline)))
        nullcline = nullcline_udot(q)
    if option == "option_3":
        bound_nullcline = nullclines_per_option["option_3"]
        q = np.linspace(np.min(bound_nullcline), np.max(bound_nullcline), amount_of_points)
        # input("please check if boundaries nullcline are" + str(np.min(bound_nullcline)) + "," + str( np.max(bound_nullcline)))
        nullcline = nullcline_vdot(q)
    return q, nullcline

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
    absolute_path = os.path.dirname(__file__)
    folder_path = os.path.join(absolute_path, f"VAL_vs_MSE_bi-cubic")
    csv_name = os.path.join(folder_path, f"{save_name}.csv")
    
    # Check if the folder and file exist, raise an exception if not
    if not os.path.exists(folder_path):
        input(f"Het is misgegaan, de file: {folder_path} wordt niet gevonden, heel zeker dat we verder moeten gaan met andere file?")
        raise FileNotFoundError(f"Directory {folder_path} does not exist.")
    if not os.path.isfile(csv_name):
        input(f"Het is misgegaan, de CSV: {csv_name} wordt niet gevonden, heel zeker dat we verder moeten gaan met andere file?")
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

def average_lc_from_modelnames(modelnames:list, performance='',df=None, *args):
    """
    Computes the average prediction from a list of model names and plots it along with standard deviation.
    Only for option2
    
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

    option = 'option_2'
    print(f"using option {option}")
    axis_values, nullcline_values = nullcline_and_boundary(option, len(mean_prediction))
    MSE_calculated = calculate_mean_squared_error(nullcline_values, mean_prediction)
    if args[-1] == 'no plot':
        return axis_value, mean_prediction, df
    std_dev_prediction = np.std(all_predictions, axis=0)

    plt.figure(figsize=(3, 2))

    # if TAU == 7.5:
    #     plt.xlim(-2.05, 2.05)
    #     plt.ylim(-0.05, 2.2)
    #     # plt.ylim(-0.05, 2.5) # if legend

    # if TAU == 100:
    #     plt.ylim(-0.05,2.4)
    #     plt.xlim(-2.3, 2.2)
    plt.xlim(-2,2)
    plt.ylim(-2,2)

    if option == 'option_2':
        plt.plot(axis_value, mean_prediction, color='b', label='Mean', zorder=5, alpha=0.7)
        plt.fill_between(axis_value, mean_prediction-std_dev_prediction, mean_prediction+std_dev_prediction, color='grey', alpha=0.7, label="Std", zorder=0)
    if option == 'option_3':
        plt.plot(mean_prediction, axis_value)
        plt.errorbar(mean_prediction[::10], axis_value[::10],xerr=std_dev_prediction[::10], linestyle='None', marker='^', alpha=0.4, zorder=0, color='grey')


    # Now the plotting the limit cycle together with the (real) nullclines
    _, x_lc, y_lc = compute_bicubic_dynamics()
    plt.plot(x_lc, y_lc, 'r-', label=f'Trajectory')
    # Plot Nullcines
    # vdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(nullcline_vdot(v), v, '--', color = "lime", label = r"$\dot{v}=0$") # r"$w=v - (1/3)*v**3 + R * I$"
    # wdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_udot(v), '--', color = "cyan", label = r"$\dot{w}=0$") # r"$w=(v + A) / B$"
    
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
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\bicubic\bicubic-{args}, {performance}, {"{:.2e}".format(MSE_calculated)}.png')

    plt.show()

    plt.subplots(figsize=(3,1))

    # plt.plot(axis_value, mean_prediction, color='b', label='Mean')
    if option == 'option_2':
        nullcline_val = nullcline_udot(axis_value)
    if option=='option_3':
        nullcline_val = nullcline_vdot(axis_value)

    plt.plot(axis_value, np.array(mean_prediction)-np.array(nullcline_val), '--', color = "gray") # r"$w=v - (1/3)*v**3 + R * I$"
    plt.fill_between(axis_value, mean_prediction-std_dev_prediction-np.array(nullcline_val), mean_prediction+std_dev_prediction-np.array(nullcline_val), color='grey', alpha=0.4, label="Std")

    ymin, ymax = -0.3, 0.3
    yticks = np.linspace(ymin, ymax, 7)
    plt.ylim(ymin, ymax)
    plt.xticks()
    # plt.yticks(yticks)
    # print(yticks)
    plt.axhline(0, color='black',linewidth=0.5, zorder=0)

    plt.ylabel("Error", labelpad=-3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.939,
bottom=0.228,
left=0.234,
right=0.945,
hspace=0.2,
wspace=0.2)


    mpl.rc("savefig", dpi=300)
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\bicubic\bi-cubic_ERROR_{option}_{args}, {performance}, {"{:.2e}".format(MSE_calculated)}.png')

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
    return axis_value, mean_prediction, df

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
        relative_path = f"bi-cubic_15000points.csv"
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
        _, x_lc, y_lc = compute_bicubic_dynamics()
        plt.plot(x_lc, y_lc, 'r-', label=f'Trajectory')
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        # Plot Nullcines
        # vdot
        v = np.linspace(-2.5, 2.5, 1000)
        plt.plot(nullcline_vdot(v), v, '--', color = "lime", label = r"$\dot{v}=0$ nullcline") #$w=v - (1/3)*v**3 + R * I$"+r" ,
        # wdot
        v = np.linspace(-2.5, 2.5, 1000)
        plt.plot(v, nullcline_udot(v), '--', color = "cyan", label = r"$\dot{w}=0$ nullcline") #$w=(v + A) / B$"+r" ,

        if option=='option_2':
            plt.plot(axis_values_for_nullcline, prediction_output, label = 'prediction')
        if option=='option_3':
            plt.plot(prediction_output, axis_values_for_nullcline, label = 'prediction')

        input(f"Here we use {option}, change accordingly")
        axis_values, nullcline_values = nullcline_and_boundary(option, len(prediction_output))
        MSE_calculated = calculate_mean_squared_error(nullcline_values, prediction_output)
        print("the MSE for this model is", MSE_calculated)

        plt.xlabel('v (voltage)')
        plt.ylabel('w (recovery variable)')
        plt.title(f"Phase Space: Limit Cycle and Cubic Nullcline with Prediction\n{option}, lr{learning_rate}, {nodes}, epoch {max_epochs}\n {normalization_method}, {activation_function}\n{title_extra}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return (axis_values_for_nullcline, prediction_output, df)



def fitting_hyperparam1_to_avg_hyperparam2(option, learning_rate, max_epochs, nodes, layers, normalization_method, activation_function_1, activation_function_2, df=None):
    """Take the 5 best (lowest validation error) networks with hyperpameter2, average the nullcline prediction
    and search for the network that best fits to this average. Calculate its MSE as well. 

    Only difference in hyperparameter allowed right now is the activation function. 
    
    >>> Example:
    Using Benchmark ReLU for average and applying Sigmoid for more detail
    """
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

    import math
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

    plt.subplots(figsize=(3,2))

    # plot results:
    print("MSE vs vdot nullcline")
    if option=='option_3':
        nullcline_val = nullcline_vdot(axis_value)
    if option=='option_2':
        nullcline_val = nullcline_udot(axis_value)
    mse_mean = calculate_mean_squared_error(nullcline_val, mean_prediction)
    if option=='option_3':
        plt.plot(mean_prediction, axis_value, color='b', label=f'mean')
    if option=='option_2':
        plt.plot(axis_value, mean_prediction, color='b', label=f'mean')
    print(f'{activation_function_2}: mean fit on nullcline has mse: {"{:.2e}".format(mse_mean)}')

    mse_fit_on_mean = calculate_mean_squared_error(nullcline_val, prediction_hyperparam1)
    if option=='option_3':
        plt.plot(prediction_hyperparam1, axis_value, color='C1', label=f'fit')
    if option=='option_2':
        plt.plot(axis_value, prediction_hyperparam1, color='C1', label=f'fit')
        
    print(f'{activation_function_1}: fit on mean prediction on nullcline has mse: {"{:.2e}".format(mse_fit_on_mean)}')
    # Now the plotting the limit cycle together with the (real) nullclines
    _, x_lc, y_lc = compute_bicubic_dynamics()
    plt.plot(x_lc, y_lc, 'r-', label=f'LC')

    # Plot Nullcines
    # vdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(v, nullcline_udot(v), '--', color = "lime", label = r"$\dot{u}=0$")
    # wdot
    v = np.linspace(-2.5, 2.5, 1000)
    plt.plot(nullcline_vdot(v), v, '--', color = "cyan", label = r"$\dot{v}=0$")


    plt.xlim(-1, 1.25)
    plt.ylim(-0.6875, 2.0994)
    plt.xticks([-0.75, 0, 0.75])
    plt.yticks([0, 1, 2])

    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=7)

    plt.xlabel(r'$u$', labelpad=-3)
    plt.ylabel(r'$v$')
    # plt.title(f'Phase Space: Limit Cycle and Cubic Nullcline with ReLU mean and Sigmoid fit')
    # plt.title(f"Phase Space: Limit Cycle and Cubic Nullcline with Prediction\n{option}, lr{learning_rate}, {nodes}, epoch {max_epochs}\n {normalization_method}, {activation_function}\n{title_extra}")
    # plt.grid(True)
    # plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=[0,1.15], ncol=5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.921,
bottom=0.185,
left=0.128,
right=0.975,
hspace=0.2,
wspace=0.2)


    mpl.rc("savefig", dpi=300)
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\bicubic\{option}_phasespace_predict_relu_sigmoid.png')


    plt.show()
    # error through time

    plt.plot(axis_value, np.abs(mean_prediction-nullcline_val), color='b', label=f'mean prediction error {activation_function_2}')

    plt.plot(axis_value, np.abs(prediction_hyperparam1-nullcline_val), color='C1', label=f'prediction {activation_function_1} error')
    plt.title("The absolute error of the nullcline.")
    plt.show()


def thesis_timeseries_phase_space():
    time, u_values, v_values = compute_bicubic_dynamics()

    fig, axs = plt.subplots(1,2)
    fig.set_figheight(2)
    fig.set_figwidth(6)

    axs[0].plot(time, u_values, label=r'$u$')
    axs[0].plot(time, v_values, label=r'$v$')
    # print('for v, difference in time between maxima', find_maxima_differences(time, v_values) ,'\n')
    # print('for w, difference in time between maxima', find_maxima_differences(time, w_values))
    axs[0].set_xlabel('Time', labelpad=-3)
    axs[0].set_ylabel('Amplitude')
    axs[0].legend(loc='upper right', ncols=2, frameon=False)
    axs[0].set_xticks([0, 40])
# Plot Limit Cycle
    _, u_lc, v_lc = compute_bicubic_dynamics()

    axs[1].plot(u_lc, v_lc, 'r-', label=f'Limit Cycle')

    # Nullclines
        # vdot
    v = np.linspace(-2.5, 2.5, 1000)

    axs[1].plot(v, nullcline_udot(v), '--', color = "lime", label = r"$\dot{x}=0$ Nullcline")

        # wdot
    v = np.linspace(-2.5, 2.5, 1000)
    axs[1].plot(nullcline_vdot(v), v, '--', color = "cyan", label = r"$\dot{y}=0$ Nullcline")

    axs[1].set_xlim(-2,2)
    axs[1].set_ylim(-2,2)
    axs[1].set_xlabel(r'$u$', labelpad=-1)
    axs[1].set_ylabel(r'$v$', labelpad=-2)

    xmin = -2.2
    xmax = 2.2
    ymin = -2.2
    ymax = 2.2
    x = np.linspace(xmin, xmax, 20)
    y = np.linspace(ymin, ymax, 20)


    X, Y = np.meshgrid(x, y)

    U = -X**3 + X**2 + X - Y  # Example vector dependent on x and y
    V = -0.5*Y**3 + 0.5*Y**2 - (1/3)*Y + X  # Example vector dependent on x and y

    DU = U / np.sqrt((U**2+V**2))
    DV = V / np.sqrt((U**2 + V**2))

    import matplotlib.patches

    # Plot the phase space
    # plt.quiver(X, Y, DU, DV, scale=28, color='grey', alpha=0.6)
    ''' Zet quiver aan om alle pijltjes te zien'''
    c = axs[1].streamplot(X,Y,U,V, density=0.4, linewidth=None, color='grey', minlength=0.1, zorder=0) 
    c.lines.set_alpha(0.6)
    for x in axs[1].get_children():
        if type(x)==matplotlib.patches.FancyArrowPatch:
            x.set_alpha(0.7) # or x.set_visible(False)


    plt.subplots_adjust(top=0.96,
bottom=0.205,
left=0.07,
right=0.98,
hspace=0.2,
wspace=0.305)
    
    mpl.rc("savefig", dpi=300)
    plt.savefig(rf'C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\bicubic\time series and phase space.png')


    plt.show()


if __name__ == '__main__':

    # 1616 laagste MSE dus deze pakken? Doet denken aan middle-range tau

    # plot_timeseries()

    # plot_limit_cycle()

    # thesis_timeseries_phase_space()

    # neural network training
    # start_total_time = time.time()

    # for seed in range(0,40):

    #     print(f"\n ROUND NUMBER STARTING {seed} \n")

    #     start_time = time.time()

    #     create_neural_network_and_save(normalization_method='min-max', activation_function='sigmoid', nodes_per_layer = [16,16], num_layers=2, learning_rate=0.01, epochs=500, option='option_2', seed=seed)

    #     end_time = time.time()

    #     elapsed_time = end_time - start_time

    #     print("Elapsed time seed", elapsed_time, "seconds \n\n")


    # end_total_time = time.time()

    # elapsed_total_time = end_total_time - start_total_time

    # print("Elapsed time everything", elapsed_total_time, "seconds \n\n")
    
    # save_one_norm_two_act_MSE_vs_VAL(learning_rate=0.01, nodes=[16,16], layers=2, max_epochs=499, option='option_2', normalization_method=['min-max'], activation_functions=['relu', 'sigmoid'], amount_per_parameter=40, save=True)
    plot_validation_vs_mse_one_norm_two_act(learning_rate=0.01, nodes=[16,16], layers=2, max_epochs=499, option='option_3', amount=40, normalization_methods=['min-max'], activation_functions=['relu', 'sigmoid'])

    # option = 'option_3'
    # absolute_path = os.path.dirname(__file__)
    # relative_path = f"bi-cubic_15000points.csv"
    # csv_name = os.path.join(absolute_path, relative_path)
    # df = pd.read_csv(csv_name, converters={"nodes": literal_eval, "mean_std": literal_eval}) # literal eval returns [2,2] as list not as str
    # _, mean_prediction = plot_best_avg_param(option=option, learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function='relu', df=df)
    # axis_values, nullcline_values = nullcline_and_boundary(option, len(mean_prediction))
    # MSE_calculated = calculate_mean_squared_error(nullcline_values, mean_prediction)

    # best MSE modelname option2: 40e92b1d0cf34683af20c1852836a5dd
    # plot_lc_from_modelname("40e92b1d0cf34683af20c1852836a5dd") (first debug this before debugging plot_best_avg_param)
    # best MSE modelname option3: 505fa002167e49119b6c91d8a56f5baa
    # plot_lc_from_modelname("505fa002167e49119b6c91d8a56f5baa")

    # fitting_hyperparam1_to_avg_hyperparam2(option='option_2', learning_rate=0.01, max_epochs=499, nodes=[16,16], layers=2, normalization_method='min-max', activation_function_1='sigmoid', activation_function_2='relu')


    # 
    # 1 done: option2 relu
    # 2 done: option3 relu
    # 3 now: option3 sigmoid
    # 