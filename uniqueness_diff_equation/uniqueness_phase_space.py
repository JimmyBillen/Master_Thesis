import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

# Define the differential equations system template
def system(t, z, f, g):
    x, y = z
    dxdt = 1 + (y-x**2)*f(x, y)
    dydt = 2 * x + (y-x**2)*g(x, y)
    return [dxdt, dydt]

# Define the functions f and g
def f1(x, y):
    return 0

def g1(x, y):
    return 0

def f2(x, y):
    return 0

def g2(x, y):
    return 1

def f3(x, y):
    return 0

def g3(x, y):
    return x

# List of functions and initial conditions
functions = [(f1, g1), (f2, g2), (f3, g3)]
initial_conditions = [-10, 100]
titles=[r'$\dot{x}=1$'+'\n'+r'$\dot{y}=2x$', r'$\dot{x}=1$'+'\n'+r'$\dot{y}=2x+(y-x^2)$', r'$\dot{x}=1$'+'\n'+r'$\dot{y}=2x+(y-x^2)x$']

# Time span for the solution
t_span = (0, 25)
t_eval = np.linspace(*t_span, 2000)

# Solver options for higher accuracy
solver_options = {
    'rtol': 1e-10,
    'atol': 1e-13
}


# Prepare the subplots
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

# Solve the system and plot the results
for i, (f, g) in enumerate(functions):
    if i == 0 or i == 1:
        # Solve the system
        solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval, args=(f, g), **solver_options)

    if i == 2:
        solver_options = {
        'rtol': 1e-10,
        'atol': 1e-13
        }
        # Solve the system
        solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval, args=(f, g), **solver_options)


    # Extract the results
    x = solution.y[0]
    y = solution.y[1]

    xmin, xmax, ymin, ymax = -10.5, 10.5, -5, 105

    xmin, xmax, ymin, ymax = -40, 40, -40, 160
    xmin, xmax, ymin, ymax = -2, 2, -1, 4


    # Plot the trajectory
    ax = axes[i]
    # ax.plot(x, y, label='Trajectory', color='green')
    xvals = np.linspace(xmin, xmax,100)
    ax.plot(xvals, xvals**2, color='red', zorder=10)
    ax.scatter([initial_conditions[0]], [initial_conditions[1]], color='red', zorder=5, label='Initial Condition')
    # ax.set_title(f'Phase Space Plot {i+1}')
    ax.set_title(titles[i])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.legend()
    # ax.grid()

    start = time.time()
    # Plot the streamlines
    Y, X = np.mgrid[ymin:ymax:10j, xmin:xmax:10j]
    U = 1 + (Y-X**2)*f(X, Y)
    V = 2 * X + (Y-X**2)*g(X, Y)
    c = ax.streamplot(X, Y, U, V, color='blue', density=0.5, zorder=0, broken_streamlines=False)
    c.lines.set_alpha(0.3)

    ax.set_xticks([])
    ax.set_yticks([])

    end = time.time()
    print(end-start)

    if i == 0:
        xvals = np.zeros(100)
        yvals = np.linspace(-1000, 1000, 100)
        ax.plot(xvals, yvals, color='black', linestyle='--')
    if i == 1:
        xvals = np.linspace(-100, 100, 10000)
        yvals = xvals**2 - 2 * xvals
        ax.plot(xvals, yvals, color='black', linestyle='--')
    if i == 2:
        xvals = np.linspace(-100, 100, 10000)
        yvals = xvals**2 - 2
        ax.plot(xvals, yvals, color='black', linestyle='--')
        xvals = np.zeros(100)
        yvals = np.linspace(-1000, 1000, 100)
        ax.plot(xvals, yvals, color='black', linestyle='--')
    



# plt.axis('off')
plt.suptitle('Phase Space')
plt.tight_layout()

import matplotlib as mpl
mpl.rc("savefig", dpi=300)
# plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\MSE_vs_VAL_tot_all_88_0.01_499_40_7.5.png")
plt.savefig(r"C:\Users\jimmy\OneDrive\Documents\Universiteit\KULeuven\Masterproef\Thesis_Fig\Results\Uniqueness\PhaseSpaceXSQUARED.png")

plt.show()
