# FitzHugh-Nagumo (ps: phasespace)
# Goal to plot the phasespace w(v)
# Nullclines were calculated analytically (see notes)

import numpy as np
import matplotlib.pyplot as plt
from time_series import plot_timeseries
from phase_space import plot_limit_cycle

# Constants
R = 0.1
I = 10
A = 0.7
B = 0.8

if __name__ == '__main__':

    # fig, axs = plt.subplots(ncols=3, nrows=2,figsize=(5, 3))
    fig, axs = plt.subplots(ncols=3, nrows=2,figsize=(7, 4))

    plot_timeseries(ax=axs[0,0], TAU=1, plot=False)
    plot_timeseries(ax=axs[0,1], TAU=5, plot=False)
    plot_timeseries(ax=axs[0,2], TAU=100, plot=False)

    plot_limit_cycle(ax=axs[1,0], TAU=1, plot=False)
    plot_limit_cycle(ax=axs[1,1], TAU=5, plot=False)
    plot_limit_cycle(ax=axs[1,2], TAU=100, plot=False)

    axs[0,0].set_title('a', loc='left', pad=10)
    axs[0,1].set_title('b', loc='left', pad=10)
    axs[0,2].set_title('c', loc='left', pad=10)

    # plt.subplots_adjust(right=0.983, left=0.072, wspace=0.137)
    plt.subplots_adjust(right=0.72, left=0.072, wspace=0.137)
    # plt.tight_layout()

    plt.show()

    pass