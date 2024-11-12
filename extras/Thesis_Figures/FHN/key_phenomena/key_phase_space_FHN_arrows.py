# Task 4.3 of PMCS using Lienard plane and mu>>1
# Van der Pol Oscillator
# (Is an introductory to understanding the FitzHugh–Nagumo model)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

R = 0.1
I = 10
TAU = 7.5
A = 0.7
B = 0.8


def limit_cycle(num_points=10000, t_end=50, x_init=0, y_init=0):
    dt = t_end / num_points
    t = np.linspace(0, t_end, num_points)

    x = np.zeros(num_points)
    y = np.zeros(num_points)

    x[0] = x_init  # Initial condition for x
    y[0] = y_init    # Initial condition for y

    for i in range(1, num_points):
        x_dot = x[i-1] - ((x[i-1])**3)/3 - y[i-1] + R * I
        y_dot = (x[i-1]+A-B*y[i-1]) / TAU

        x[i] = x[i-1] + x_dot * dt
        y[i] = y[i-1] + y_dot * dt

    return x, y

def streamlined_phase_space_FHN(ax=None, plot=True):

    if plot or ax is None:
        fig, ax = plt.figure(figsize=(6, 6))

    # x_inits = [0] 
    # y_inits = [1.767]
    x_inits = [1]
    y_inits = [2]

    # # Plot the limit cycle for each value of mu
    for x,y in zip(x_inits, y_inits):
        x_lc, y_lc = limit_cycle(num_points=10000, t_end=50, x_init=x, y_init=y)
        # ax.plot(x_lc, y_lc, '-', color='red', zorder=3, label='Trajectory')
        # ax.scatter(x_lc, y_lc, color='red', zorder=3, label='Trajectory', alpha=0.05, s=0.5)
        ax.scatter(x_lc, y_lc, color='red', zorder=3, alpha=0.05, s=0.5)

    ax.plot([100], [100], '-', color='red', zorder=3, label='Trajectory')

    
    x = np.linspace(-4,4,1001)
    ax.plot(x, x-(x**3)/3 + R*I, '--', color = "lime", label = r"$\dot{v}=0$", zorder=2)
    ax.plot(x, (x+A)/B, '--', color = "cyan", label = r"$\dot{w}=0$", zorder=1)
    # plt.scatter(x_inits, y_inits, color='grey', alpha=0.6, label='Start point')

    # plt.scatter(0,0, color='black', label='Fixed point')

    x_real_FP = 0.40886584
    # A = 0.7
    # B = 0.8
    y_real_FP = (x_real_FP + A) / B
    ax.scatter([x_real_FP], [y_real_FP], color='black', label='Fixed point', zorder=5)
    # dots = [-2,2]
    # dots_null = [((i**3)/3) - i for i in dots]
    # plt.scatter(dots, dots_null, 'bo', )

    xmin = -2.2
    xmax = 2.2
    ymin = -0.2
    ymax = 2.3
    x = np.linspace(xmin, xmax, 20)
    y = np.linspace(ymin, ymax, 20)

    # Create a grid of x and y values
    X, Y = np.meshgrid(x, y)

    U = X-(X**3)/3 - Y + R*I  # Example vector dependent on x and y
    V = (X + A - B*Y)/TAU  # Example vector dependent on x and y

    DU = U / np.sqrt((U**2+V**2))
    DV = V / np.sqrt((U**2 + V**2))

    # Plot the phase space
    # plt.quiver(X, Y, DU, DV, scale=28, color='grey', alpha=0.6)
    ''' Zet quiver aan om alle pijltjes te zien'''
    # c = ax.streamplot(X,Y,U,V, density=1.5, linewidth=None, color='grey', minlength=0.1, zorder=0) 
    c = ax.streamplot(X,Y,U,V, density=1, linewidth=None, color='grey', minlength=0.1, zorder=0) 

    c.lines.set_alpha(0.4)
    for x in ax.get_children():
        if type(x)==matplotlib.patches.FancyArrowPatch:
            x.set_alpha(0.5) # or x.set_visible(False)

    # plt.axvline(0, color='black', linewidth=1, linestyle='--')
    # plt.axhline(0, color='black', linewidth=1, linestyle='--')


    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title(r'Two-Dimensional Phase Space of Van Der Pol Oscillator for $\mu$=1')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([0, 1 ,2])
    ax.set_xticks([-2, -1, 0, 1 , 2])

    # ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.set_xlabel('v (voltage)')
    ax.set_ylabel('w (recovery variable)')

    # plt.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    order = [0,3,1,2]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='center left', ncols=4, columnspacing=0.4, handlelength=1.2, markerscale=0.6, borderpad=0.2, frameon=False, bbox_to_anchor=[0.02, 1.05]) #columnspacing
    # ax.legend(['Limit cycle', 'Fixed point',r"$\dot{v}=0$",r"$\dot{w}=0$"],loc='upper left', ncols=2)
    if plot:
        plt.show()
