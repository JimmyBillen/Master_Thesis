import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def sqrt(r):
    r_vals = np.sqrt(r)
    return r_vals

def plot_different_exponential_growths():
    xlim = 2
    r = np.linspace(-3,3, 100000)

    size = 0.5
    fig, ax = plt.subplots(figsize=(6.4*0.8*size, 4.8*0.7*0.8*size))

    plt.plot(r, sqrt(-r), '--', color='C0', label='Unstable')
    plt.plot(r, -sqrt(-r), color='C0', label='Stable')
    # plt.title(r"Exponential Growth for Various Initial Values (x0) at Growth Rate (r) of 0.5")µ
    ax.axhline(0, color='black',linewidth=0.5, zorder=0)
    ax.axvline(0, color='black',linewidth=0.5, zorder=0)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)        
    plt.xlabel(r'$x$')
    ax.xaxis.set_label_position('top')

    plt.ylabel(r'$s$', rotation='horizontal')
    # plt.legend(title='Stability', bbox_to_anchor = [0.6, 1.2], loc='center left')
    plt.legend(bbox_to_anchor = [0.52, 1], loc='center left')

    plt.xticks([])
    plt.yticks([])
    plt.xlim([-0.05-xlim, xlim + 0.05])
    # plt.ylim([-0.05, 5])
    # plt.title("Bifurcation Diagram")

    plt.tight_layout()
    mpl.rc("savefig", dpi=300)
    plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\BasicsToDynamicalSystems\BifurcationDiagram.png")
    plt.show()

plot_different_exponential_growths()
