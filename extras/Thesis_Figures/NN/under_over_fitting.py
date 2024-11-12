import numpy as np
import matplotlib.pyplot as plt

def plot_fitting(ax):
    x = np.linspace(-7, 7, 100)

    def overfit(x):
        return np.sin((x+9.8) * np.sin(0.6*x+9.8))*0.4

    def real_func(x):
        return 2.3*np.sin(0.3*x+3)

    np.random.seed(1)
    X = np.sort(14 * np.random.rand(50, 1), axis=0) - 7
    y = real_func(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

    ax.plot(x, real_func(x), 'blue', label='good fit', linewidth=2)
    ax.plot(x, overfit(x) + real_func(x), color='red', label='overfit', linestyle='--')
    ax.plot(x, -0.4*x + 0.3, label='underfit', color='green', linestyle='--')
    ax.scatter(X, overfit(X) + real_func(X), color='grey', alpha=0.5, edgecolor='None', label='data')

    x_extra = np.array([-6.01, -3.6, -7, -4.4, -4.39, -1.84, -1.8, -0.37, 0.133,1.99, 3.97, 4.5, 4.74, 4.8, 5.14, 5.22,5.85, 7, 0.82])
    ax.scatter(x_extra, overfit(x_extra) + real_func(x_extra), color='grey', alpha=0.5, edgecolor='None')

    ax.scatter(X, y, color='grey', alpha=0.5, edgecolor='None')

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_ylabel(r"$y$")
    ax.set_xlabel(r"$x$")

    ax.legend()

