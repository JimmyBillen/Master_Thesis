import matplotlib.pyplot as plt
import numpy as np

def plot_loss(ax):
    def loss_train(x):
        return np.exp(-(x-2))+1

    def loss_val(x):
        return np.exp(-(x-2))+1+0.1*np.exp(x-5)+0.1*x

    x_val = np.linspace(0, 7)

    ax.plot(x_val, loss_train(x_val), color='blue', label='Training loss', zorder=1, alpha=0.8)
    ax.plot(x_val, loss_val(x_val), color='orange', label='Validation loss', zorder=0)

    ax.axvline(3.992, color='black', linewidth=1, linestyle='--', alpha=0.5)

    ax.legend(framealpha=1)
    ax.set_xticks([0])
    ax.set_yticks([0])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()


