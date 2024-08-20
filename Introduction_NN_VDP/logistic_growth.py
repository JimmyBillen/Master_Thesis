import numpy as np
import matplotlib.pyplot as plt

def logistic_solution(times, x0, K, rate):
    positions = np.divide(K, 1 + np.exp(-rate * times)*(K-x0)/x0)
    return positions

def plot_different_logistic_growths():
    K = 5
    rate = 0.5

    # fraction = [0, 0.001, 0.003, 0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
    fraction = [0.005, 1]
    x_0s = [K*i for i in fraction]

    times = np.linspace(0, 20, 1000)
    for i, x0 in enumerate(x_0s):
        plt.plot(times, logistic_solution(times, x0, K, rate), label=f'{fraction[i]}K')
    plt.title(f"Logistic Growth for Various Initial Values (x0)\nat Carrying Capacity (K) of 5 and Growth Rate (r) of 0.5 ")
    plt.xlabel('Time')
    plt.ylabel('x')
    plt.legend(title='x0', bbox_to_anchor=[1.01, 1])
    plt.ylim(0,7)
    plt.show()

plot_different_logistic_growths()