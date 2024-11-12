import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.markers import MarkerStyle
import matplotlib as mpl

def plot_function(x, r):
    positions = np.square(x)+r
    return positions

def plot_different_quadratics():
    r = 0.5

    xlow = -1.5
    xhigh = 1.5
    x = np.linspace(xlow+0.05, xhigh-0.05, 1000)

    fig, axes = plt.subplots(nrows=1, ncols=3)
    size = 0.5
    fig.set_figheight(3*size)
    fig.set_figwidth(5*size)

    r = [-0.5, 0, 0.5]
    index = [0, 1, 2]
    for ax, r0, index in zip(axes, r, index):
        ax.plot(x, plot_function(x, r0), zorder=1)
        ax.set_xlim(xlow, xhigh)
        ax.set_ylim(-0.8, 0.8)
        ax.axhline(0, color='black',linewidth=0.5, zorder=0)
        ax.axvline(0, color='black',linewidth=0.5, zorder=0)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        # ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if index == 0:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)        
            ax.set_xlabel(r'$s<0$')
            ax.set_ylabel(r'$x$', rotation='horizontal')
            ax.set_title(r'$\dot{x}$')
            ax.scatter(math.sqrt(-r0), 0, s=60, edgecolor='C0', c='w', zorder=2)
            ax.scatter(-math.sqrt(-r0), 0, s=60, edgecolor='None', zorder=2)

            head_with = 0.1
            dx = 0.0001
            ax.arrow(xlow+0.25, 0, dx, 0, head_width=0.1, head_length=2*head_with,color='black')
            ax.arrow(-0.2, 0, -dx, 0, head_width=0.1, head_length=2*head_with,color='black')
            ax.arrow(0.32, 0, -dx, 0, head_width=0.1, head_length=2*head_with,color='black')
            ax.arrow(xhigh-0.38, 0, dx, 0, head_width=0.1, head_length=2*head_with,color='black')

        if index == 1:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.set_xlabel(r'$s=0$')
            ax.set_ylabel(r'$x$', rotation='horizontal')
            ax.set_title(r'$\dot{x}$')

            ax.scatter(0, 0, s=60, c='w', edgecolor="C0", marker=MarkerStyle("o", fillstyle="right"))
            ax.scatter(0, 0, s=60, marker=MarkerStyle("o", fillstyle="left"))

            head_with = 0.1
            dx = 0.0001
            ax.arrow(-0.85, 0, dx, 0, head_width=0.1, head_length=2*head_with,color='black')
            ax.arrow(0.65, 0, dx, 0, head_width=0.1, head_length=2*head_with,color='black')


        if index == 2:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.set_xlabel(r'$s>0$')
            ax.set_ylabel(r'$x$', rotation='horizontal')
            ax.set_title(r'$\dot{x}$')

            head_with = 0.1
            dx = 0.0001
            ax.arrow(-0.85, 0, dx, 0, head_width=0.1, head_length=2*head_with,color='black')
            ax.arrow(0.65, 0, dx, 0, head_width=0.1, head_length=2*head_with,color='black')



    # plt.title(r"Exponential Growth for Various Initial Values (x0) at Growth Rate (r) of 0.5")
    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$\dot{x}$', rotation='horizontal')
    plt.tight_layout()
    # plt.grid()
    # plt.suptitle('Phase Portrait')
    mpl.rc("savefig", dpi=300)
    plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\BasicsToDynamicalSystems\phaseportrait_rtight2.png")
    plt.show()

plot_different_quadratics()
