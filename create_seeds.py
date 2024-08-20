from FitzHugh_Nagumo_t import compute_fitzhugh_nagumo_dynamics
from create_NN_FHN import calculate_derivatives, normalization
import numpy as np
import os
from settings import TAU

""""
ipv eerst normalization en dan split=> Eerst split dan norm
=> Nakijken of dit hetzelfde geeft!!

"""

def split_train_validation_data_seed(data_1, data_2, data_3, data_4, validation_ratio=0.2, seed=None):
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

    # Make random number generator
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

for seed in range(40):
    time, v_t_data, u_t_data = compute_fitzhugh_nagumo_dynamics() # assigning v->v, w->v see heads-up above.
    u_dot_t_data = np.array(calculate_derivatives(time, u_t_data))
    v_dot_t_data = np.array(calculate_derivatives(time, v_t_data))

    # normalizing the data
    normalization_methods = ['no-norm', 'z-score', 'min-max']
    for normalization_method in normalization_methods:
        u_t_data_norm, mean_u, std_u = normalization(u_t_data, normalization_method)  # mean_u, and std_u equal x_min and (x_max - x_min) respectively when doing min-max normalization
        v_t_data_norm, mean_v, std_v = normalization(v_t_data, normalization_method) 
        u_dot_t_data_norm, mean_u_dot, std_u_dot = normalization(u_dot_t_data, normalization_method)
        v_dot_t_data_norm, mean_v_dot, std_v_dot  = normalization(v_dot_t_data, normalization_method)
        mean_std = {"u_t_data_norm":[mean_u, std_u], "v_t_data_norm":[mean_v, std_v], "u_dot_t_data_norm": [mean_u_dot, std_u_dot], "v_dot_t_data_norm": [mean_v_dot, std_v_dot]}

        # Step 4: Seperate training and validation data
        train_u, val_u, train_v, val_v, train_u_dot, val_u_dot, train_v_dot, val_v_dot = split_train_validation_data_seed(u_t_data_norm, v_t_data_norm, u_dot_t_data_norm, v_dot_t_data_norm, validation_ratio=0.2, seed=seed)

        savename = f'Tau{TAU}_seed{seed}_{normalization_method}'
        absolute_path = os.path.dirname(__file__)
        relative_path = "seeds"
        folder_path = os.path.join(absolute_path, relative_path)
        full_path = os.path.join(folder_path, savename)
        np.savez(full_path, train_u=train_u, val_u=val_u, train_v=train_v, val_v=val_v, train_u_dot=train_u_dot, val_u_dot=val_u_dot, train_v_dot=train_v_dot, val_v_dot=val_v_dot, mean_std=mean_std)

# 1)
# savename = f'Tau{TAU}_seed21_no-norm'
# absolute_path = os.path.dirname(__file__)
# relative_path = "seeds"
# folder_path = os.path.join(absolute_path, relative_path)
# full_path = os.path.join(folder_path, savename)
# loaded_data = np.load(full_path+'.npz', allow_pickle=True)
# 
# print(loaded_data.files)
# train_u = loaded_data['train_u']
# val_u = loaded_data['val_u']
# train_v = loaded_data['train_v']
# val_v = loaded_data['val_v']
# train_u_dot = loaded_data['train_u_dot']
# val_u_dot = loaded_data['val_u_dot']
# train_v_dot = loaded_data['train_v_dot']
# val_v_dot = loaded_data['val_v_dot']
# mean_std = loaded_data['mean_std']
# print(mean_std)

# 2)
# savename = f'ATau{TAU}_seed21_no-norm'
# absolute_path = os.path.dirname(__file__)
# relative_path = "seeds"
# folder_path = os.path.join(absolute_path, relative_path)
# full_path = os.path.join(folder_path, savename)
# loaded_data = np.load(full_path+'.npz', allow_pickle=True)
# 
# Atrain_u = loaded_data['train_u']
# Aval_u = loaded_data['val_u']
# Atrain_v = loaded_data['train_v']
# Aval_v = loaded_data['val_v']
# Atrain_u_dot = loaded_data['train_u_dot']
# Aval_u_dot = loaded_data['val_u_dot']
# Atrain_v_dot = loaded_data['train_v_dot']
# Aval_v_dot = loaded_data['val_v_dot']
# Amean_std = loaded_data['mean_std']

# are_equal1 = np.array_equal(train_u, Atrain_u)
# are_equal2 = np.array_equal(val_u, Aval_u)
# are_equal3 = np.array_equal(train_v, Atrain_v)
# are_equal4 = np.array_equal(val_v, Aval_v)
# are_equal5 = np.array_equal(train_u_dot, Atrain_u_dot)
# are_equal6 = np.array_equal(val_u_dot, Aval_u_dot)
# are_equal7 = np.array_equal(train_v_dot, Atrain_v_dot)
# are_equal8 = np.array_equal(val_v_dot, Aval_v_dot)
# are_equal9 = mean_std == Amean_std

# print("train_u:", "Equal" if are_equal1 else "Not equal")
# print("val_u:", "Equal" if are_equal2 else "Not equal")
# print("train_v:", "Equal" if are_equal3 else "Not equal")
# print("val_v:", "Equal" if are_equal4 else "Not equal")
# print("train_u_dot:", "Equal" if are_equal5 else "Not equal")
# print("val_u_dot:", "Equal" if are_equal6 else "Not equal")
# print("train_v_dot:", "Equal" if are_equal7 else "Not equal")
# print("val_v_dot:", "Equal" if are_equal8 else "Not equal")
# print("mean_std:", "Equal" if are_equal9 else "Not equal")






