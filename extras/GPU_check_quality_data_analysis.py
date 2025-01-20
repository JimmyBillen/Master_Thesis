# Quality of GPU and GPU speed will be checked using 1) batchsize and 2) amount of in real life time

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
import time
import json

import sys
sys.path.append('../../Master_Thesis') # needed to import settings
from settings import TAU
import pandas as pd
# import seaborn as sns

PATIENCE = 200

def load_results(filename=f"results_GPU_batchsize_patience{PATIENCE}_[16,16].json"):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        return []

def results_of_one_run(filename=None):
    """Shows the result of time, epochs, validation error in function of batchsize for one run"""

    results = load_results(filename=filename)

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

def results_10_runs(basename_file, runs=10):
    """
    
    """
    batchsizes_runs = [] # must be the same
    time_runs = []
    val_runs = []
    epochs_runs = []

    counter = 1
    for i in range(runs):
        namefile = f'{basename_file}_{counter}.json'
        results = load_results(namefile)

        batchsizes = []
        times = []
        lowest_val_errors = []
        epochs_at_lowest_val = []

        for result in results:
            print(result)
            print("\n")

            batchsizes.append(result['batchsize'])
            times.append(result['time'])
            lowest_val_errors.append(np.log10(result['val_loss']))
            epochs_at_lowest_val.append(result['epochs'])
        counter += 1
        batchsizes_runs.append(batchsizes)

        # batchsizes_runs = batchsizes
        time_runs.append(times)
        val_runs.append(lowest_val_errors)
        epochs_runs.append(epochs_at_lowest_val)

    fig, ax2 = plt.subplots(1,3)
    for i in range(runs):
        print(batchsizes)
        # ax2[0].plot(batchsizes, time_runs[i], color='green', label='time')
        # ax2[1].plot(batchsizes, val_runs[i], color='blue', label='val error')
        # ax2[2].plot(batchsizes, epochs_runs[i], color='red', label='epoch')

        powers_of_2 = [2**n for n in range(1,18)]
        time_select = [time_runs[i][j] for j in range(len(batchsizes)) if batchsizes[j] in powers_of_2]
        val_select = [val_runs[i][j] for j in range(len(batchsizes)) if batchsizes[j] in powers_of_2]
        epoch_select = [epochs_runs[i][j] for j in range(len(batchsizes)) if batchsizes[j] in powers_of_2]

        batchsize_select = [batchsize for batchsize in batchsizes if batchsize in powers_of_2]
        print(batchsize_select)
        print(time_select)
        ax2[0].plot(batchsize_select, time_select, color='purple', alpha=0.8)
        ax2[1].plot(batchsize_select, val_select, color='purple', alpha=0.8)
        ax2[2].plot(batchsize_select, epoch_select, color='purple', alpha=0.8)

    plt.show()

    fig, ax3 = plt.subplots(1,3)
    fig.set_figheight(4)
    fig.set_figwidth(6)
    for i in range(runs):
        # ax3[0].plot(batchsizes, time_runs[i], color='green', label='time')
        # ax3[1].plot(batchsizes, val_runs[i], color='blue', label='val error')
        # ax3[2].plot(batchsizes, epochs_runs[i], color='red', label='epoch')
        batchsizes = batchsizes_runs[i]

        log_of_2 = [i for i in range(3,14)]
        powers_of_2 = [2**n for n in range(1,18)]

        time_select = [time_runs[i][j] for j in range(len(batchsizes)) if batchsizes[j] in powers_of_2]
        val_select = [val_runs[i][j] for j in range(len(batchsizes)) if batchsizes[j] in powers_of_2]
        epoch_select = [epochs_runs[i][j] for j in range(len(batchsizes)) if batchsizes[j] in powers_of_2]

        batchsize_select = [batchsize for batchsize in batchsizes if batchsize in powers_of_2]
        print(batchsize_select)
        print(time_select)
        # ax3[0].plot(log_of_2[-len(time_select)], time_select, color='purple', alpha=0.8)
        # ax3[1].plot(log_of_2[-len(val_select)], val_select, color='purple', alpha=0.8)
        # ax3[2].plot(log_of_2[-len(epoch_select)], epoch_select, color='purple', alpha=0.8)

        ax3[0].plot(log_of_2[-len(time_select):], time_select, color='purple', alpha=0.8)
        ax3[1].plot(log_of_2[-len(val_select):], val_select, color='purple', alpha=0.8)
        ax3[2].plot(log_of_2[-len(epoch_select):], epoch_select, color='purple', alpha=0.8)
        
        ax3[0].axhline(y=142, xmin=0, xmax=1)

        ax3[1].axhline(y=np.log10(3.85*10**(-5)), xmin=0, xmax=1)
        ax3[1].axhline(y=np.log10(0.001728), xmin=0, xmax=1)

        ax3[2].axhline(y=500, xmin=0, xmax=1)


    ax3[0].set_title('Time')
    ax3[1].set_title("Validation Error")
    ax3[2].set_title("Epochs")

    ax3[0].set_xlabel("Batchsize (log2)")
    ax3[1].set_xlabel("Batchsize (log2)")
    ax3[2].set_xlabel("Batchsize (log2)")

    plt.suptitle('Training parameter in function of batchsize')
    print("\n Parameters used were: \nfor tau=100, minmax relu [16,16] 0.01")
    plt.subplots_adjust(top=0.875,
bottom=0.115,
left=0.075,
right=0.98,
hspace=0.215,
wspace=0.31)
    plt.show()


if __name__ == "__main__":
    results_10_runs('results_GPU_batchsize_patience200_TAU100_[16,16]', runs=8)