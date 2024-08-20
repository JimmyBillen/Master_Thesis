import numpy as np
import matplotlib.pyplot as plt

def exponential_solution(times, x0, r):
    positions = x0 * np.exp(times * r)
    return positions

def plot_different_exponential_growths():
    r = 0.5

    fraction = [0.001, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
    fraction = np.linspace(0, 1, 10)
    # x_0s = [r*i for i in fraction]

    times = np.linspace(0, 5, 1000)
    for x0 in fraction:
        plt.plot(times, exponential_solution(times, x0, r), label=x0)
    # plt.title(r"Exponential Growth for Various Initial Values (x0) at Growth Rate (r) of 0.5")
    plt.xlabel('Time')
    plt.ylabel('x')
    # plt.legend(title='x0')
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-0.05, 4.55])
    plt.ylim([-0.05, 7.55])
    plt.show()

plot_different_exponential_growths()